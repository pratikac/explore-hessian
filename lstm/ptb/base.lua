--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the Apache 2 license found in the
--  LICENSE file in the root directory of this source tree. 
--

function g_disable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_disable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = false
    end
end

function g_enable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_enable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = true
    end
end

function g_cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function g_init_gpu()
    print(string.format("Using %s-th gpu", opt.gpu))
    cutorch.setDevice(opt.gpu)
    g_make_deterministic(opt.seed)
end

function g_make_deterministic(seed)
    torch.manualSeed(seed)
    cutorch.manualSeed(seed)
    torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
    assert(#to == #from)
    for i = 1, #to do
        to[i]:copy(from[i])
    end
end

function g_f3(f)
    return string.format("%.3f", f)
end

function g_d(f)
    return string.format("%d", torch.round(f))
end

function transfer_data(x)
    return x:cuda()
end

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

function create_network()
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
