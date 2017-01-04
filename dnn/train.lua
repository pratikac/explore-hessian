require 'torch'
require 'exptutils'
require 'image'
local lapp = require 'pl.lapp'
lapp.slack = true
local colors = sys.COLORS

require 'entropyoptim'

opt = lapp[[
--output            (default "/local2/pratikac")
-m,--model          (default 'cifarconv')
-F,--estimateF      (default '')
-b,--batch_size     (default 64)                Batch size
--LR                (default 1)                 Learning rate
--LRD               (default 0)                 Learning rate decay
--optim             (default 'sgd')             Optimization algorithm
--LRstep            (default 3)                 Drop LR after x epochs
--LRratio           (default 0.2)               LR drop factor
--L                 (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1e-5)              Langevin gamma coefficient
--scoping           (default 1e-3)              Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 0)                 GPU id
-f,--full                                       Use all data
-d,--dropout        (default 0.5)               Dropout
--L2                (default 1e-3)              L2 regularization
-s,--seed           (default 42)
-e,--max_epochs     (default 20)
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
                    gitrev=get_gitrev()
                }
        else
            res = {gitrev=get_gitrev()}
        end
        torch.save(opt.output .. logfname .. '.t7', res)
    end
end

function learning_rate_schedule()
    local s = math.floor(epoch/opt.LRstep)
    local lr = opt.LR*opt.LRratio^s
    print(('[LR] %.5f'):format(lr))
    return lr
end

function estimate_local_entropy(_model, cost, d)
    local x, y = d.data, d.labels
    local modelc = _model:clone()
    local wc,dwc = modelc:getParameters()

    local res = {}
    --local gamma_array = torch.logspace(-10,2,13)
    local gamma_array = torch.Tensor(1):zero()
    for gi,g in ipairs(torch.totable(gamma_array)) do

        local model = modelc:clone()
        local w, dw = model:getParameters()
        model:training()        --  this is hack because cudnn does not like backprop
        --  if evaluate() is being used

        local num_batches = x:size(1)/opt.batch_size
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

        local eta_denom = 0
        local Z,Zsq = 0,0
        local num_iter = 1e5
        opt.LRD = 100/num_iter
        local e = w.new(dw:size()):zero()
        for i=1,num_iter do
            local f,df = feval(w)
            e:normal()

            local eta = opt.LR/(1 + i*opt.LRD)
            local noise_term = e*opt.noise/math.sqrt(opt.LR)
            df:add(-g, wc-w):add(opt.L2,w):add(noise_term)
            w:add(-eta, df)

            local dZ = math.exp(-f - opt.L2/2*torch.norm(w)^2 - opt.gamma/2*torch.norm(w-wc)^2)
            Z = Z + dZ*eta
            Zsq = Zsq + dZ^2*eta
            eta_denom = eta_denom + eta

            if opt.verbose then
                print(eta,f,torch.norm(df),torch.norm(noise_term),dZ)
            end
            if i % 1000 == 0 then
                print(('[%d][%.5e] %.5f +- %.5e'):format(i, g, Z/eta_denom, Zsq/eta_denom))
            end
        end
        Z = Z/eta_denom
        Zsq = Zsq/eta_denom
        print('Final Z: ' .. Z .. ' +- ' .. math.sqrt(Zsq - Z^2))

        table.insert(res, {g, Z, math.sqrt(Zsq - Z^2)})
        --torch.save(opt.estimateF:sub(1,-10) .. '.Z.t7', res)
    end
    print(res)
    --torch.save(opt.estimateF:sub(1,-10) .. '.Z.t7', res)
end

function main()
    model, cost, params = models.build()
    if opt.estimateF ~= '' then
        print('Loading model: ' .. opt.estimateF)
        model = torch.load(opt.estimateF)
    end
    local train, val, test = dataset.split(0, (opt.full and 1) or 0.05)

    -- setup scoping here
    if opt.scoping > 1e16 then
        opt.scoping = 5/(opt.max_epochs*train.data:size(1)/opt.batch_size)
        print('scoping: ' .. opt.scoping)
    end

    confusion = optim.ConfusionMatrix(params.classes)
    optim_state = { learningRate= opt.LR,
    learningRateDecay = opt.LRD,
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

        epoch = 1
        while epoch <= opt.max_epochs do
            trainer(train)
            tester(test)

            optim_state.learningRate = learning_rate_schedule()
            save_model()

            epoch = epoch + 1
            print('')
        end
    else
        estimate_local_entropy(model, cost, train)
    end
end

main()
