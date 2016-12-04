require 'torch'
require 'exptutils'
require 'image'
local pretty = require 'pl.pretty'
local lapp = require 'pl.lapp'
lapp.slack = true
local colors = sys.COLORS

require 'entropyoptim'

opt = lapp[[
--output            (default "/local2/pratikac")
-m,--model          (default 'mnistconv')         
--retrain           (default '')
-b,--batch_size     (default 64)                Batch size
--LR                (default 0.1)               Learning rate
--optim             (default 'adam')            Optimization algorithm
--LRstep            (default 10)                Drop LR after x epochs
--LRratio           (default 0.2)               LR drop factor
--langevin          (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1)                 Langevin gamma coefficient
--langevin_noise    (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 0)                 GPU id
-f,--full                                       Use all data
-d,--dropout        (default 0.5)
--L2                (default 1e-3)              L2 regularization
-s,--seed           (default 42)
-e,--max_epochs     (default 15)
--augment                                       Augment data with flips and mirrors
-l,--log                                        Log statistics
-v,--verbose                                    Show gradient statistics
-h,--help                                       Print this message
]]

if opt.help then
    print(opt)
    os.exit()
end

if string.find(opt.model, 'mnist') then
    opt.dataset = 'mnist'
elseif string.find(opt.model, 'cifar') then
    opt.dataset = 'cifar'
end

dofile('utils.lua')
set_gpu()
local models = require 'models'

local dataset = mnist
opt.output = opt.output .. '/results/'
if opt.dataset == 'mnist' then
    dataset = mnist
elseif opt.dataset == 'cifar' then
    dataset = require 'cifarload'
end

function augment(xc)
    if opt.dataset == 'cifar' and opt.augment == true then
        local r = torch.Tensor(xc:size(1)):uniform()
        for i=1,xc:size(1) do
            if r[i] < 0.33 then xc[i] = image.hflip(xc[i])
            elseif r[i] < 0.66 then xc[i] = image.vflip(xc[i]) end
        end
    end
    return xc
end

function trainer(d)
    local x, y = d.data, d.labels
    model:training()
    local w, dw = model:getParameters()

    local num_batches = x:size(1)/opt.batch_size
    local bs = opt.batch_size
    local timer = torch.Timer()

    local loss = 0
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        timer:reset()

        local feval = function(_w, dry)
            local dry = dry or false
            if _w ~= w then w:copy(_w) end
            dw:zero()

            local idx = torch.Tensor(bs):random(1, d.size):type('torch.LongTensor')
            local xc, yc = x:index(1, idx), y:index(1, idx):cuda()
            xc = augment(xc):cuda()

            local yh = model:forward(xc)
            local f = cost:forward(yh, yc)
            local dfdy = cost:backward(yh, yc)
            model:backward(xc, dfdy)
            cutorch.synchronize()

            if dry == false then
                loss = loss + f
                confusion:batchAdd(yh, yc)
                confusion:updateValids()

                if b > 1 then
                    local stats = { tv=1,
                    epoch=epoch,
                    batch=b,
                    iter=epoch*num_batches + b,
                    loss= f,
                    dF = torch.norm(optim_state.lparams.w)*opt.gamma,
                    lx = torch.mean(optim_state.lparams.lx),
                    miss = (1-confusion.totalValid)*100,
                    mu = torch.mean(w),
                    stddev = torch.std(w),
                    gmax = torch.max(torch.abs(dw)),
                    gmin = torch.min(torch.abs(dw))}
                    logger_add(logger, stats)
                end
            end

            return f, dw
        end

        if opt.optim == 'sgd' then
            optim.entropysgd(feval, w, optim_state)
        elseif opt.optim == 'adam' then
            optim.entropyadam(feval, w, optim_state)
        else
            assert(false, 'opt.optim: ' .. opt.optim)
        end

        if b % 50 == 0 then
            print( (colors.blue .. '+[%2d][%3d/%3d] %.5f %.3f%% [%.2fs]'):format(epoch, b, num_batches, loss/b, (1 - confusion.totalValid)*100, timer:time().real))
        end
    end
    loss = loss/num_batches
    print( (colors.blue .. '*[%2d] %.5f %.3f%%'):format(epoch, loss, (1 - confusion.totalValid)*100))

    local stats = { tv=1,
    epoch=epoch,
    loss=loss,
    miss = (1-confusion.totalValid)*100}
logger_add(logger, stats)

end

function tester(d)
    local x, y = d.data, d.labels
    model:evaluate()

    local num_batches = math.ceil(x:size(1)/opt.batch_size)
    local bs = opt.batch_size

    local loss = 0
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        local sidx,eidx = (b-1)*bs, math.min(b*bs, x:size(1))
        local xc, yc = x:narrow(1, sidx + 1, eidx-sidx):cuda(),
        y:narrow(1, sidx + 1, eidx-sidx):cuda()

        local yh = model:forward(xc)
        local f = cost:forward(yh, yc)
        cutorch.synchronize()

        loss = loss + f
        confusion:batchAdd(yh, yc)
        confusion:updateValids()

        local stats = { tv=0,
        epoch=epoch,
        batch=b,
        loss= f,
        miss = (1-confusion.totalValid)*100}
    logger_add(logger, stats)

    if b % 50 == 0 then
        print( ( colors.red .. '++[%2d][%3d/%3d] %.5f %.3f%%'):format(epoch, b, num_batches, loss/b, (1 - confusion.totalValid)*100))
    end
end
loss = loss/num_batches
print( (colors.red .. '**[%2d] %.5f %.3f%%'):format(epoch, loss, (1 - confusion.totalValid)*100))

local stats = { tv=0,
epoch=epoch,
miss = (1-confusion.totalValid)*100,
        loss=loss}
    logger_add(logger, stats)
end

function save_model()
    if opt.log then
        --torch.save(opt.output .. logfname .. '.model.t7', model:clearState())
        --torch.save(opt.output .. logfname .. '.optim_state.t7', optim_state)
    end
end

function learning_rate_schedule()
    local s = math.floor(epoch/opt.LRstep)
    local lr = opt.LR*opt.LRratio^s
    print(('[LR] %.5f'):format(lr))
    return lr
end

function main()
    model, cost, params = models.build()
    if opt.retrain ~= '' then
        print('Loading model: ' .. opt.retrain)
        model = torch.load(opt.retrain)
    end

    confusion = optim.ConfusionMatrix(params.classes)
    optim_state = { learningRate= opt.LR,
    weightDecay = opt.L2,
    learningRateDecay = 0,
    momentum = 0.9,
    nesterov = true,
    dampening = 0,
    rho=opt.rho,
    gamma=opt.gamma,
    langevin=opt.langevin,
    langevin_noise = opt.langevin_noise}

    local train, val, test = dataset.split( 1e-4,
    (opt.full and 1) or 0.05)

    local symbols = {'tv', 'epoch', 'batch', 'iter', 'loss', 'dF', 'lx', 'miss', 'mu', 'stddev', 'gmax', 'gmin'}
    logger = nil
    if opt.log then
        logger, logfname = setup_logger(opt, symbols)
    end

    epoch = 1
    while epoch <= opt.max_epochs do
        trainer(train)
        tester(test)

        optim_state.learningRate = learning_rate_schedule()
        --save_model()
        
        epoch = epoch + 1
        print('')
    end
end

main()
