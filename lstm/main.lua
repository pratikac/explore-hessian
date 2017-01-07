--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local arg = {2}

require('cunn')
LookupTable = nn.LookupTable

require('nngraph')
require('base')
local ptb = require('data')
require '../dnn/entropyoptim'

--[[
-- Train 1 day and gives 82 perplexity.
local opt = {batch_size=20,
T=35,
layers=2,
decay=1.15,
hdim=1500,
dropout=0.65,
init_weight=0.04,
lr=1,
vocab_size=10000,
max_epoch=14,
max_max_epoch=55,
max_grad_norm=10,
L=20,
rho=0,
gamma=0.1,
scoping=1e-3,
noise=1e-4,
verbose=true}
--]]

-- Trains 1h and gives test 115 perplexity.
opt = {batch_size=20,
T=20,
layers=2,
decay=2,
hdim=200,
dropout=0,
init_weight=0.1,
lr=1,
vocab_size=10000,
max_epoch=4,
max_max_epoch=13,
max_grad_norm=5,
L=20,
rho=0,
gamma=0.1,
scoping=1e-3,
noise=1e-4,
verbose=true}

local function transfer_data(x)
    return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local w, dw

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h = nn.Linear(opt.hdim, 4*opt.hdim)(x)
    local h2h = nn.Linear(opt.hdim, 4*opt.hdim)(prev_h)
    local gates = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates =  nn.Reshape(4,opt.hdim)(gates)
    local sliced_gates = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

local function create_network()
    local x                = nn.Identity()()
    local y                = nn.Identity()()
    local prev_s           = nn.Identity()()
    local i                = {[0] = LookupTable(opt.vocab_size,
    opt.hdim)(x)}
    local next_s           = {}
    local split         = {prev_s:split(2 * opt.layers)}
    for layer_idx = 1, opt.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(opt.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y              = nn.Linear(opt.hdim, opt.vocab_size)
    local dropped          = nn.Dropout(opt.dropout)(i[opt.layers])
    local pred             = nn.LogSoftMax()(h2y(dropped))
    local err              = nn.ClassNLLCriterion()({pred, y})
    local module           = nn.gModule({x, y, prev_s},
    {err, nn.Identity()(next_s)})
    module:getParameters():uniform(-opt.init_weight, opt.init_weight)
    return transfer_data(module)
end

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
    if model.norm_dw > opt.max_grad_norm then
        local shrink_factor = opt.max_grad_norm / model.norm_dw
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
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
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
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end


function main()
    g_init_gpu(arg)

    print('Loading PTB')
    state_train = {data=transfer_data(ptb.traindataset(opt.batch_size))}
    state_valid =  {data=transfer_data(ptb.validdataset(opt.batch_size))}
    state_test =  {data=transfer_data(ptb.testdataset(opt.batch_size))}

    print('opt:')
    print(opt)

    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do
        reset_state(state)
    end
    setup()

    optim_state = optim_state or { learningRate= opt.lr,
    momentum = 0.9,
    nesterov = true,
    dampening = 0,
    rho=opt.rho,
    gamma=opt.gamma,
    scoping=opt.scoping,
    L=opt.L,
    noise = opt.noise}

    local step = 0
    local epoch = 0
    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()

    print("Starting training.")
    local epoch_size = torch.floor(state_train.data:size(1) / opt.T)
    local perps
    while epoch < opt.max_max_epoch do

        -- compute fp and bp
        local perp = fp(state_train)
        if perps == nil then
            perps = torch.zeros(epoch_size):add(perp)
        end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(state_train)

        -- closure for optim
        local function feval(_w)
            if w ~= _w then w:copy(_w) end
            return perp, dw
        end
        optim.sgd(feval, w, optim_state)

        total_cases = total_cases + opt.T * opt.batch_size
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 10) == 10 then
            local since_beginning = g_d(torch.toc(beginning_time) / 60)
            print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(optim_state.learningRate) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
        end

        if step % epoch_size == 0 then
            run_valid()
            if epoch > opt.max_epoch then
                optim_state.learningRate = optim_state.learningRate / opt.decay
            end
        end

        cutorch.synchronize()
        collectgarbage()
    end
    run_test()
    print("Training is over.")
end

main()
