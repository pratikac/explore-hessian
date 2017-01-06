require 'torch'
require 'exptutils'
require 'image'
local lapp = require 'pl.lapp'
lapp.slack = true
local colors = sys.COLORS
paths = require 'paths'

require 'entropyoptim'

opt = lapp[[
--output            (default "/local2/pratikac")
-m,--model          (default 'cifarconv')
--retrain           (default '')
-F,--estimateF      (default '')
-b,--batch_size     (default 128)               Batch size
--LR                (default 0)                 Learning rate
--LRD               (default 0)                 Learning rate decay
--optim             (default 'sgd')             Optimization algorithm
--LRstep            (default 4)                 Drop LR after x epochs
--LRratio           (default 0.2)               LR drop factor
--L                 (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1e-4)              Langevin gamma coefficient
--scoping           (default 1e-3)              Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 2)                 GPU id
-f,--full                                       Use all data
-d,--dropout        (default 0.15)              Dropout
--L2                (default 0)                 L2 regularization
-s,--seed           (default 42)
-e,--max_epochs     (default 10)
--augment                                       Augment data with flips and mirrors
-l,--log                                        Log statistics
-v,--verbose                                    Show gradient statistics
-h,--help                                       Print this message
]]

print(opt)
if opt.help then
    os.exit()
end

if string.find(opt.model, 'mnist') then
    opt.dataset = 'mnist'
elseif string.find(opt.model, 'cifar') then
    opt.dataset = 'cifar'
elseif  string.find(opt.model, 'rnn') or
        string.find(opt.model, 'lstm') then
    opt.dataset = 'ptb'
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
elseif opt.dataset == 'ptb' then
    dataset = require 'ptbload'
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
    local w, dw = model:getParameters()
    model:training()

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
                    dF = torch.norm(optim_state.lparams.w),
                    dfdF = optim_state.lparams.dfdF,
                    lx = torch.mean(optim_state.lparams.lx),
                    xxpd = optim_state.lparams.xxpd,
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
        elseif opt.optim == 'sgld' then
            optim.sgdld(feval, w, optim_state)
        else
            assert(false, 'opt.optim: ' .. opt.optim)
        end

        if b % 25 == 0 then
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

    local mc = 1
    local loss = 0
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        local sidx,eidx = (b-1)*bs, math.min(b*bs, x:size(1))
        local xc, yc = x:narrow(1, sidx + 1, eidx-sidx):cuda(),
        y:narrow(1, sidx + 1, eidx-sidx):cuda()

        local f = 0
        for m=1,mc do
            local yh = model:forward(xc)
            f = f + cost:forward(yh, yc)
            cutorch.synchronize()

            confusion:batchAdd(yh, yc)
            confusion:updateValids()
        end
        f = f/mc
        loss = loss + f

        local stats = { tv=0,
        epoch=epoch,
        batch=b,
        loss= f,
        miss = (1-confusion.totalValid)*100}
        logger_add(logger, stats)

        if b % 25 == 0 then
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
        local res = {}
        if true then
            res = {model=model:clearState(),
                    optim_state=optim_state,
                    gitrev=get_gitrev(),
                    epoch=epoch
                }
        else
            res = {gitrev=get_gitrev()}
        end
        torch.save(opt.output .. logfname .. '.t7', res)
    end
end

function learning_rate_schedule()
    local lr = opt.LR
    if opt.LRD > 0 then
        lr = opt.LR*(1-opt.LRD)^epoch
    elseif opt.LR > 0 then
        lr = opt.LR
    else
        --[[
        -- all-cnn-bn on cifar10
        local regimes = {
            {1,4,1},
            {5,6,0.2},
            {7,12,0.01}
        }
        --]]
        --[[
        -- lenet
        local regimes = {
            {1,2,1},
            {3,7,0.1},
            {8,15,0.01}
        }
        --]]
        -- mnistfc
        local regimes = {
            {1,2,1},
            {3,7,0.1},
            {8,15,0.01}
        }

        for _,row in ipairs(regimes) do
            if epoch >= row[1] and epoch <= row[2] then
                lr = row[3]
                break
            end
        end
    end
    print(('[LR] %.5f'):format(lr))
    return lr
end

function estimate_local_entropy(d)
    local x, y = d.data, d.labels
    local modelc = model:clone()
    local wc,dwc = modelc:getParameters()

    local w, dw = model:getParameters()
    --  this is hack because cudnn does not like backprop
    --  if evaluate() is being used
    model:training()

    local num_batches = math.ceil(x:size(1)/opt.batch_size)
    local bs = opt.batch_size
    local feval = function(_w)
        if w ~= _w then w:copy(_w) end
        dw:zero()

        local idx = torch.Tensor(bs):random(1, d.size):type('torch.LongTensor')
        local xc, yc = x:index(1, idx):cuda(), y:index(1, idx):cuda()

        local yh = model:forward(xc)
        local f = cost:forward(yh, yc)
        local dfdy = cost:backward(yh, yc)
        model:backward(xc, dfdy)
        cutorch.synchronize()
        return f, dw
    end

    local res = {}
    local eta, l2 = 0.1, 1e-3
    local N = 200*num_batches
    local e = w.new(dw:size()):zero()
    for i=1,N do
        local f,df = feval(w)
        table.insert(res, {i,torch.norm(w-wc),f})

        e:normal()
        local noise_term = e*opt.noise/math.sqrt(eta)
        df:add(l2,w):add(noise_term)
        w:add(-eta, df)

        if opt.verbose then
            if i % 1000 == 0 then
                print(f, torch.norm(w-wc), torch.norm(df), torch.norm(noise_term))
            end
        end
    end

    model = modelc:clone()
    res = torch.Tensor(res)
    torch.save('F_' .. paths.basename(opt.estimateF), res)
    torch.save(opt.estimateF:sub(1,-10) .. '.F.t7', res)
end

function main()
    model, cost, params = models.build()
    local fp = ''
    if opt.retrain ~= '' then fp = opt.retrain end
    if opt.estimateF ~= '' then fp = opt.estimateF end
    if fp ~= '' then
        print('Loading model: ' .. fp)
        local f = torch.load(fp)
        model, optim_state = f.model, f.optim_state
        optim_state.learningRate = opt.LR
        if f.epoch then epoch = f.epoch + 1 else epoch = 1 end
        print('Will stop logging...')
        opt.log = false
    end
    local train, val, test = dataset.split(0, (opt.full and 1) or 0.05)

    confusion = optim.ConfusionMatrix(params.classes)
    optim_state = optim_state or { learningRate= opt.LR,
    learningRateDecay = 0,
    weightDecay = opt.L2,
    momentum = 0.9,
    nesterov = true,
    dampening = 0,
    rho=opt.rho,
    gamma=opt.gamma,
    scoping=opt.scoping,
    L=opt.L,
    noise = opt.noise}

    if opt.estimateF == '' then
        local symbols = {   'tv', 'epoch', 'batch', 'iter', 'loss', 'dF', 'lx', 'xxpd',
        'miss', 'mu', 'stddev', 'gmax', 'gmin','dfdF'}
        logger = nil
        if opt.log then
            logger, logfname = setup_logger(opt, symbols)
        end

        epoch = epoch or 1
        while epoch <= opt.max_epochs do
            optim_state.learningRate = learning_rate_schedule()
            trainer(train)
            tester(test)

            save_model()

            epoch = epoch + 1
            print('')
        end
    else
        estimate_local_entropy(train)
    end
end

main()
