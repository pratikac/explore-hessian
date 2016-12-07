require 'resnet'
local optnet = require 'optnet'

local models = {}
local b = cudnn
local b_name = 'cudnn'

local function shapetest(m)
    local x = torch.randn(1,3,32,32):cuda()
    local y = m:cuda():forward(x)
    print(#y)
    os.exit(1)
end

local function mnistconv()
    local m = nn:Sequential()

    local c1, c2, c3 = 20, 50, 500
    m:add(b.SpatialConvolution(1, c1, 5, 5))
    m:add(b.ReLU())
    m:add(b.SpatialMaxPooling(3,3,3,3))
    m:add(b.SpatialBatchNormalization(c1))
    m:add(nn.Dropout(opt.dropout))

    m:add(b.SpatialConvolution(c1, c2, 5, 5))
    m:add(b.ReLU())
    m:add(b.SpatialMaxPooling(2,2,2,2))
    m:add(b.SpatialBatchNormalization(c2))
    m:add(nn.Dropout(opt.dropout))

    m:add(nn.View(c2*2*2))
    m:add(nn.Linear(c2*2*2, c3))
    m:add(b.ReLU())
    m:add(nn.Dropout(opt.dropout))

    m:add(nn.Linear(c3, 10))
    m:add(b.LogSoftMax())

    return m, {p=3, name='mnistconv'}
end

local function mnistfc()
    local m = nn:Sequential()
    local c = 1024
    local p = 1

    m:add(nn.View(784))
    m:add(nn.Linear(784, c))
    m:add(nn.ReLU(true))
    m:add(nn.Dropout(opt.dropout))

    for i=1,p do
        m:add(nn.Linear(c, c))
        m:add(nn.ReLU(true))
        m:add(nn.Dropout(opt.dropout))
    end

    m:add(nn.Linear(c, 10))
    m:add(b.LogSoftMax())

    return m, {p=p+1, name='mnistfc'}
end

function models.mnist()
    local m, p
    if opt.model == 'mnistfc' then
        m, p = mnistfc()
    elseif opt.model == 'mnistconv' then
        m, p = mnistconv()
    else
        assert(false, 'Unknown model: ' .. opt.model)
    end

    local w, dw = m:getParameters()
    p.n = math.sqrt(w:numel()/p.p)

    p.classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    local cost_function = nn.ClassNLLCriterion()
    return m:cuda(), cost_function:cuda(), p
end

local function cifarfc()
    local m = nn.Sequential()
    local c = 1024
    local p = 4

    --m:add(nn.Dropout(0.25))
    m:add(nn.View(3072))
    m:add(nn.Linear(3072, c))
    m:add(nn.ReLU(true))

    for i=1,p do
        m:add(nn.Linear(c, c))
        m:add(nn.ReLU(true))
        m:add(nn.Dropout(0.1))
    end

    m:add(nn.Linear(c, 10))
    m:add(b.LogSoftMax())

    return m, {p=p+1, name='cifarfc'}
end

local function cifarconv()
    -- ALL-CNN-C
    local function convbn(...)
        local arg = {...}
        return nn.Sequential()
        :add(b.SpatialConvolution(...))
        :add(b.SpatialBatchNormalization(arg[2]))
        :add(b.ReLU(true))
    end

    local m = nn.Sequential()
    :add(nn.Dropout(opt.dropout*0.4))
    :add(convbn(3,96,3,3,1,1,1,1))
    :add(convbn(96,96,3,3,1,1,1,1))
    :add(convbn(96,96,3,3,2,2,1,1))
    :add(nn.Dropout(opt.dropout))
    :add(convbn(96,192,3,3,1,1,1,1))
    :add(convbn(192,192,3,3,1,1,1,1))
    :add(convbn(192,192,3,3,2,2,1,1))
    :add(nn.Dropout(opt.dropout))
    :add(convbn(192,192,3,3,1,1,1,1,1,1))
    :add(convbn(192,192,3,3,1,1,1,1))
    :add(convbn(192,10,1,1,1,1))
    :add(nn.SpatialAveragePooling(8,8))
    :add(nn.View(10))
    :add(b.LogSoftMax())

    --shapetest(m)
    return m, {p=9, name='cifarconv'}
end

local function cifarnin()
    local m = nn.Sequential()
    local c = 192

    local function block(...)
        local arg = {...}
        m:add(b.SpatialConvolution(...))
        m:add(b.SpatialBatchNormalization(arg[2]))
        m:add(b.ReLU(true))
        return m
    end

    block(3, c, 5,5,1,1,2,2)
    block(c, c,1,1)
    block(c, c,1,1)
    m:add(b.SpatialMaxPooling(3,3,2,2))
    block(c, c,5,5,1,1,2,2)
    block(c, c,1,1)
    block(c, c,1,1)
    m:add(b.SpatialMaxPooling(3,3,2,2))
    block(c, c,3,3,1,1,1,1)
    block(c, c,1,1)
    block(c,10,1,1)
    m:add(b.SpatialAveragePooling(7,7))
    m:add(nn.View(10))
    m:add(b.LogSoftMax())

    return m, {p=9, name='cifarnin'}
end

local function cifarinception()
    local function convbn(...)
        local args = {...}
        local c = nn.Sequential()
        :add(b.SpatialConvolution(...))
        :add(b.SpatialBatchNormalization(args[2]))
        :add(b.ReLU(true))
        return c
    end

    local function convpool(ip, op)
        local concat = nn.DepthConcat(2)
        :add(b.SpatialConvolution(ip, op, 3, 3, 2, 2, 1, 1))
        :add(b.SpatialMaxPooling(3, 3, 2, 2))
        return concat
    end

    local function block(ip, ch)    
        local concat = nn.DepthConcat(2)
        :add(convbn(ip, ch[1], 1, 1))
        :add(convbn(ip, ch[2], 3, 3, 1, 1, 1, 1))
        return concat
    end

    local m = nn.Sequential()

    m:add(convbn(3, 96, 3, 3, 1, 1, 1, 1))
    :add(block(96, {32, 32}))
    :add(block(64, {48, 48}))
    :add(convpool(96, 96))
    :add(nn.Dropout(opt.dropout))
    :add(block(192, {128, 64}))
    :add(block(192, {96, 96}))
    :add(block(192, {96, 96}))
    :add(block(192, {64, 128}))
    :add(convpool(192, 192))
    :add(nn.Dropout(opt.dropout))
    :add(block(384, {192, 192}))
    :add(block(384, {192, 192}))
    :add(b.SpatialAveragePooling(8,8))

    m:add(nn.View(384))
    m:add(nn.Linear(384, 10))
    m:add(b.LogSoftMax())

    return m, {p=11, name='cifarinception'}
end

local function cifarresnet()
    local m = resnet({ depth=20,
    shortcutType = 'A',
    dataset = 'cifar10'})
    m:add(b.LogSoftMax())

    return m, {p=20, name='cifarresnet'}
end

local function cifarwideresnet()
    local m = wideresnet({  depth=40,
    widen_factor = 4,
    dropout = 0,
    num_classes = 10})
    m:add(b.LogSoftMax())

    return m, {p=40, name='cifarwideresnet'}
end


function models.cifar()
    local m, p

    if opt.model == 'cifarfc' then
        m, p = cifarfc()
    elseif opt.model == 'cifarconv' then
        m, p = cifarconv()
    elseif opt.model == 'cifarnin' then
        m, p = cifarnin()
    elseif opt.model == 'cifarinception' then
        m, p = cifarinception()
    elseif opt.model == 'cifarresnet' then
        m, p = cifarresnet()
    elseif opt.model == 'cifarwideresnet' then
        m, p = cifarwideresnet()
    else
        assert('Unknown model: ' .. opt.model)
    end

    if opt.cifar100 then
        p.classes = torch.totable(torch.range(1,100))
    else
        p.classes = {'airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    end

    local w, dw = m:getParameters()
    p.n = math.sqrt(w:numel()/p.p)

    local cost_function = nn.ClassNLLCriterion()
    return m:cuda(), cost_function:cuda(), p
end

function models.build()
    if opt.dataset == 'mnist' then
        return models.mnist()
    elseif opt.dataset == 'cifar' then
        return models.cifar()
    else
        assert('Unknown opt.dataset: ' .. opt.dataset)
    end
end
return models
