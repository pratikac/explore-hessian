require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

require '../../dnn/entropyoptim'
require '../../dnn/exptutils'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/shakespeare.h5')
cmd:option('-input_json', 'data/shakespeare.json')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 256)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 1)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- local entropy
cmd:option('-L',0)
cmd:option('-gamma',1e-3)
cmd:option('-scoping',0)
cmd:option('-noise',2e-6)

-- Output options
cmd:option('-verbose', 0)
cmd:option('-checkpoint_every', 20000)

-- Backend options
cmd:option('-gpu', 1)

opt = cmd:parse(arg)

--[[
local fname = build_file_name(opt, {'checkpoint_every','grad_clip',
                'lr_decay_factor','lr_decay_every',
                'wordvec_size','rnn_size','num_layers','batchnorm',
                'input_h5','input_json','batch_size','seq_length'})
--]]

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
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
    idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nil
local start_i = 0
if opt.init_from ~= '' then
    print('Initializing from ', opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    model = checkpoint.model:type(dtype)
    if opt.reset_iterations == 0 then
        start_i = checkpoint.i
    end
else
    model = nn.LanguageModel(opt_clone):type(dtype)
end
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
gamma=opt.gamma}
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
        end
    end

    -- Take a gradient step and maybe print
    local _, loss = optim.entropyadam(f, params, optim_config)
    table.insert(train_loss_history, loss[1])
    if opt.verbose > 0 and i % 100 == 0 then
        local float_epoch = i / num_train + 1
        local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
        local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
        print(string.format(unpack(args)))
    end

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
end
