mnist = require 'mnist'

function mnist.split(val_frac, small_frac)
    local Ntrain, Ntest = 60000, 10000
    local val_frac = val_frac or 0.12
    local small_frac = small_frac or 0.05
    local strain, num_train = 1, Ntrain*(1-val_frac)*small_frac
    local sval, num_val = Ntrain*(1-val_frac), Ntrain*val_frac*small_frac
    local stest, num_test = 1, Ntest*small_frac

    local train, test = mnist.traindataset(), mnist.testdataset()
    local shuffle = torch.randperm(Ntrain):type('torch.LongTensor')
    train.data, train.label = train.data:index(1, shuffle), train.label:index(1, shuffle)
    train.data = train.data:reshape(Ntrain, 1, 28, 28):float()
    train.data:add(-126):div(126)
    train.label:add(1)

    local shuffle = torch.randperm(Ntest):type('torch.LongTensor')
    test.data, test.label = test.data:index(1, shuffle), test.label:index(1, shuffle)
    test.data = test.data:reshape(Ntest, 1, 28, 28):float()
    test.data:add(-126):div(126)
    test.label:add(1)

    local X, Y = train.data:narrow(1, strain, num_train),
    train.label:narrow(1, strain, num_train)
    local Xval, Yval = train.data:narrow(1, sval, num_val),
    train.label:narrow(1, sval, num_val)

    local Xtest, Ytest = test.data:narrow(1, stest, num_test),
    test.label:narrow(1, stest, num_test)

    return  {data=X, labels=Y, size=num_train}, 
    {data=Xval, labels=Yval, size=num_val},
    {data=Xtest, labels=Ytest, size=num_test}
end

function set_gpu()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(opt.seed)
    torch.setnumthreads(4)
    print('Using GPU')
    require 'cutorch'
    require 'nn'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeedAll(opt.seed)
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.fastest = true
end


function makeDataParallelTable(m, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(m, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      m = dpt:cuda()
   end
   return m
end
