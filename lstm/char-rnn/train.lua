require 'torch'
require 'nn'
require 'optim'
local colors = sys.COLORS

require 'LanguageModel'
require 'util.DataLoader'

require '../../dnn/entropyoptim'
require '../../dnn/exptutils'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local lapp = require 'pl.lapp'
lapp.slack = true

local cmd = torch.CmdLine()

opt = lapp[[
--output            (default "/local2/pratikac/results/")
--input             (default 'tiny-shakespeare')
--batch_size        (default 50)
--seq_length        (default 50)
--max_epochs        (default 50)
--model_type        (default 'lstm')
--wordvec_size      (default 64)
--rnn_size          (default 128)
--num_layers        (default 2)
--dropout           (default 0)
--lr                (default 1e-2)
--lrstep            (default 2)
--lrratio           (default 0.5)
--grad_clip         (default 5)
--L                 (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1e-3)              Langevin gamma coefficient
--scoping           (default 0)                 Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 2)                 GPU id
-f,--full                                       Use all data
-s,--seed           (default 42)
-l,--log                                        Log statistics
-v,--verbose                                    Show gradient statistics
-h,--help                                       Print this message
]]

opt.input_h5 = 'data/' .. opt.input .. '.h5'
opt.batchnorm = 1
print(opt)

logger = nil
local symbols = {'tv', 'epoch', 'iter', 'loss'}
local blacklist = { 'input', 'rnn_size', 'seq_length',
                    'model_type', 'num_layers', 'wordvec_size', 'grad_clip',
                    'input_h5', 'batchnorm'}
if opt.log then
    logger, logfname = setup_logger(opt, symbols, blacklist)
end

-- Set up GPU stuff
require 'cutorch'
require 'cunn'
cutorch.setDevice(opt.gpu)
local dtype = 'torch.CudaTensor'
print(string.format('Running with CUDA on GPU %d', opt.gpu))

-- setup seed
torch.manualSeed(opt.seed)
cutorch.manualSeedAll(opt.seed)

-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json('data/' .. opt.input .. '.json')
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
    idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nil
model = nn.LanguageModel(opt_clone):type(dtype)

local params, grad_params = model:getParameters()
print('Num params: ' .. params:size(1))
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length

local function feval(w)
    if params ~= w then params:copy(w) end
    grad_params:zero()

    -- Get a minibatch and run the model forward, maybe timing it
    local x, y = loader:nextBatch('train')
    x, y = x:type(dtype), y:type(dtype)
    local scores = model:forward(x)

    -- Use the Criterion to compute loss; we need to reshape the scores to be
    -- two-dimensional before doing so. Annoying.
    local scores_view = scores:view(N * T, -1)
    local y_view = y:view(N * T)
    local loss = crit:forward(scores_view, y_view)

    -- Run the Criterion and model backward to compute gradients, maybe timing it
    local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
    model:backward(x, grad_scores)

    if opt.grad_clip > 0 then
        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    end

    return loss, grad_params
end

-- Train the model!
print('Start training...')
local optim_config = {learningRate = opt.lr,
L=opt.L,
scoping=opt.scoping,
noise=opt.noise,
gamma=opt.gamma,
rho = 0,
lclr=0.1}


local num_train = loader.split_sizes['train']
local num_val = loader.split_sizes['val']
print('Num train: ' .. num_train)

local num_iterations = opt.max_epochs * num_train
model:training()

for i = 1, num_iterations do
    local epoch = math.ceil(i / num_train)

    -- check if we are at the end of an epoch
    if i % num_train == 0 then

        if epoch % opt.lrstep == 0 then
            optim_config.learningRate = optim_config.learningRate * opt.lrratio
            print(('[LR] %.3f'):format(optim_config.learningRate))
        end

        -- evaluate model
        model:evaluate()
        model:resetStates()
        local val_loss = 0
        for j=1,num_val do
            local x,y = loader:nextBatch('val')
            x,y = x:type(dtype), y:type(dtype)
            local f = model:forward(x):view(N*T, -1)
            val_loss = val_loss + crit:forward(f, y:view(N*T))
        end
        val_loss = val_loss/num_val
        print((colors.red .. '[%d] %.3f'):format(epoch, val_loss))
        local s = {tv=0, iter=0, epoch=epoch, loss=val_loss}
        logger_add(logger, s)

        model:training()
    end

    -- take a gradient step and maybe print
    local _, loss = optim.entropyadam(feval, params, optim_config)
    local s = {tv=1, iter=i, epoch=epoch, loss=loss[1]}
    logger_add(logger, s)
    
    if i % 10 == 0 then
        print((colors.blue .. '[%2d][%3d/%3d] %.2f'):format(epoch,i,num_train,loss[1]))
    end
end
