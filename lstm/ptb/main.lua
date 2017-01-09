local lapp = require 'pl.lapp'
lapp.slack = true
local colors = sys.COLORS
paths = require 'paths'

require('cunn')
LookupTable = nn.LookupTable

require('nngraph')
require('base')
require '../../dnn/entropyoptim'
require '../../dnn/exptutils'
local ptb = require('data')

opt = lapp[[
--output            (default "/local2/pratikac/results/")
--input             (default 'ptb')
--model             (default 'lstm')            Used inside entropy-optim
-g,--gpu            (default 2)                 GPU idx
-e,--max_epochs     (default 13)
--lr                (default 1)
--lclr              (default 1)
--lrstep            (default 3)
--lrratio           (default 0.5)
--beta1             (default 0.5)
--grad_clip         (default 5)
--L                 (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1e-2)              Langevin gamma coefficient
--scoping           (default 0)                 Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 2)                 GPU idx
-f,--full                                       Use all data
-s,--seed           (default 1)
-l,--log                                        Log statistics
-v,--verbose                                    Show gradient statistics
-h,--help                                       Print this message
]]

--[[
-- Train 1 day and gives 82 perplexity.
local params = {batch_size=20,
T=35,
hdim=1500,
dropout=0.65,
init_weight=0.04,
vocab_size=10000,
max_epoch=14,
max_max_epoch=55,
}
--]]

-- Trains 1h and gives test 115 perplexity.
local params = {batch_size=20,
T=20,
hdim=200,
dropout=0,
init_weight=0.1,
max_epoch=4,
max_max_epoch=13,
}

for k,v in pairs(params) do opt[k] = v end
print(opt)
if opt.help then
    os.exit()
end
opt.vocab_size = 10000
opt.layers = 2

logger = nil
local symbols = {'tv', 'epoch', 'batch', 'loss'}
local blacklist = {'hdim', 'init_weight', 'max_epoch', 'max_max_epoch', 'grad_clip'}
if opt.log then
    logger, logfname = setup_logger(opt, symbols, blacklist)
end

local state_train, state_valid, state_test
local model = {}
local w, dw

local function setup()
    local core_network = create_network()
    w, dw = core_network:getParameters()
    print('Num parameters: ' .. w:size(1))
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, opt.T do
        model.s[j] = {}
        for d = 1, 2 * opt.layers do
            model.s[j][d] = transfer_data(torch.zeros(opt.batch_size, opt.hdim))
        end
    end
    for d = 1, 2 * opt.layers do
        model.start_s[d] = transfer_data(torch.zeros(opt.batch_size, opt.hdim))
        model.ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.hdim))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, opt.T)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(opt.T))
end

local function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * opt.layers do
            model.start_s[d]:zero()
        end
    end
end

local function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

local function fp(state)
    g_replace_table(model.s[0], model.start_s)
    if state.pos + opt.T > state.data:size(1) then
        reset_state(state)
    end
    for i = 1,opt.T do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    g_replace_table(model.start_s, model.s[opt.T])
    return model.err:mean()
end

local function bp(state)
    dw:zero()
    reset_ds()
    for i = opt.T, 1, -1 do
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        local derr = transfer_data(torch.ones(1))
        local tmp = model.rnns[i]:backward({x, y, s},
        {derr, model.ds})[3]
        g_replace_table(model.ds, tmp)
        cutorch.synchronize()
    end
    state.pos = state.pos + opt.T

    model.norm_dw = dw:norm()
    if model.norm_dw > opt.grad_clip then
        local shrink_factor = opt.grad_clip / model.norm_dw
        dw:mul(shrink_factor)
    end
end

local function run_valid()
    reset_state(state_valid)
    g_disable_dropout(model.rnns)
    local len = (state_valid.data:size(1) - 1) / (opt.T)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    g_enable_dropout(model.rnns)
    return torch.exp(perp/len)
end

local function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    g_enable_dropout(model.rnns)
    return torch.exp(perp / (len - 1))
end

function main()
    g_init_gpu()

    print('Loading PTB')
    state_train = {data=transfer_data(ptb.traindataset(opt.batch_size))}
    state_valid =  {data=transfer_data(ptb.validdataset(opt.batch_size))}
    state_test =  {data=transfer_data(ptb.testdataset(opt.batch_size))}

    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do
        reset_state(state)
    end
    setup()

    optim_state = { learningRate= opt.lr,
    beta1=opt.beta1,
    momentum = 0,
    nesterov = false,
    dampening = 0,
    gamma=opt.gamma,
    scoping=opt.scoping,
    rho = opt.rho,
    L=opt.L,
    noise = opt.noise,
    lclr=opt.lclr}

    local timer = torch.Timer()
    local num_train = torch.floor(state_train.data:size(1) / opt.T)
    local num_iterations = num_train*opt.max_max_epoch
    print("Starting training, num_train: " .. num_train)

    local train_loss = 0
    for i=1,num_iterations do
        collectgarbage()
        -- closure for optim
        local function feval(_w, dry)
            local dry = dry or false
            if w ~= _w then w:copy(_w) end

            -- compute fp and bp
            local f = fp(state_train)
            bp(state_train)
            cutorch.synchronize()

            if dry == false then
                train_loss = train_loss + f
            end
            return f, dw
        end
        optim.sgd(feval, w, optim_state)
        
        local epoch = math.ceil(i/num_train)
        local b = i%num_train
        
        local s = {tv=1, batch=b, epoch=epoch, loss=torch.exp(train_loss/b)}
        logger_add(logger, s)

        if b == 0 then
            train_loss = torch.exp(train_loss/num_train)
            print((colors.blue .. '++[%d] %.3f [%.3fs]'):format(epoch, train_loss, timer:time().real))
            local s = {tv=1, batch=0, epoch=epoch, loss=train_loss}
            logger_add(logger, s)
            train_loss = 0

            local val_loss = run_valid()
            print((colors.red .. '**[%d] %.3f'):format(epoch, val_loss))
            local s = {tv=0, batch=0, epoch=epoch, loss=val_loss}
            logger_add(logger, s)
        
            timer:reset()

            if epoch > opt.max_epoch then
                optim_state.learningRate = optim_state.learningRate*opt.lrratio
                print(('[LR] %.5f'):format(optim_state.learningRate))
            end
        elseif b % 50 == 0 then
            print((colors.blue .. '[%2d][%3d/%3d] %.2f'):format(epoch,b,num_train,torch.exp(train_loss/b)))
        end
    end
    local test_loss = run_test()
    print("Training is over.")
end

main()
