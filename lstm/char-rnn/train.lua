require 'torch'
require 'nn'
require 'optim'

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
--input             (default 'tiny-shakespeare')
--batch_size        (default 50)
--seq_length        (default 50)
--model_type        (default 'lstm')
--wordvec_size      (default 64)
--rnn_size          (default 128)
--num_layers        (default 2)
--dropout           (default 0)
--batchnorm         (default 1)
--max_epochs        (default 50)
--learning_rate     (default 2e-3)
--grad_clip         (default 5)
--lr_decay_every    (default 5)
--lr_decay_factor   (default 0.5)
--L                 (default 0)                 Num. Langevin iterations
-r,--rho            (default 0)                 Coefficient rho*f(x) - F(x,gamma)
--gamma             (default 1e-4)              Langevin gamma coefficient
--scoping           (default 1e-3)              Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 2)                 GPU id
-f,--full                                       Use all data
-s,--seed           (default 42)
-v,--verbose                                    Show gradient statistics
-h,--help                                       Print this message
]]

opt.input_h5 = 'data/' .. opt.input .. '.h5'
print(opt)

-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    dtype = 'torch.CudaTensor'
    print(string.format('Running with CUDA on GPU %d', opt.gpu))
end


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
local start_i = 0
model = nn.LanguageModel(opt_clone):type(dtype)

local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}

-- Loss function that we pass to an optim method
local function f(w)
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
local optim_config = {learningRate = opt.learning_rate,
L=opt.L,
scoping=opt.scoping,
noise=opt.noise,
gamma=opt.gamma,
rho = 0,
lclr=0.1}


local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
model:training()
for i = start_i + 1, num_iterations do
    local epoch = math.floor(i / num_train) + 1

    -- Check if we are at the end of an epoch
    if i % num_train == 0 then
        model:resetStates() -- Reset hidden states

        -- Maybe decay learning rate
        if epoch % opt.lr_decay_every == 0 then
            optim_config.learningRate = optim_config.learningRate * opt.lr_decay_factor
            print(('[LR] %.3f'):format(optim_config.learningRate))
        end
    end

    -- Take a gradient step and maybe print
    local _, loss = optim.entropyadam(f, params, optim_config)
    table.insert(train_loss_history, loss[1])
    if i % 10 == 0 then
        local float_epoch = i / num_train + 1
        local msg = '[%.2f/%d][%4d/%4d] loss = %2.2f'
        local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
        print(string.format(unpack(args)))
    end

    --[[
    -- Maybe save a checkpoint
    local check_every = opt.checkpoint_every
    if (check_every > 0 and i % check_every == 0) or i == num_iterations then
        -- Evaluate loss on the validation set. Note that we reset the state of
        -- the model; this might happen in the middle of an epoch, but that
        -- shouldn't cause too much trouble.
        model:evaluate()
        model:resetStates()
        local num_val = loader.split_sizes['val']
        local val_loss = 0
        for j = 1, num_val do
            local xv, yv = loader:nextBatch('val')
            xv = xv:type(dtype)
            yv = yv:type(dtype):view(N * T)
            local scores = model:forward(xv):view(N * T, -1)
            val_loss = val_loss + crit:forward(scores, yv)
        end
        val_loss = val_loss / num_val
        print('val_loss = ', val_loss)
        table.insert(val_loss_history, val_loss)
        table.insert(val_loss_history_it, i)
        model:resetStates()
        model:training()

        -- First save a JSON checkpoint, excluding the model
        local checkpoint = {
            opt = opt,
            train_loss_history = train_loss_history,
            val_loss_history = val_loss_history,
            val_loss_history_it = val_loss_history_it,
            forward_backward_times = forward_backward_times,
            memory_usage = memory_usage,
            i = i
        }
        local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
        -- Make sure the output directory exists before we try to write it
        paths.mkdir(paths.dirname(filename))
        utils.write_json(filename, checkpoint)

        -- Now save a torch checkpoint with the model
        -- Cast the model to float before saving so it can be used on CPU
        model:clearState()
        model:float()
        checkpoint.model = model
        local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
        paths.mkdir(paths.dirname(filename))
        torch.save(filename, checkpoint)
        model:type(dtype)
        params, grad_params = model:getParameters()
        collectgarbage()
    end
    --]]
end
