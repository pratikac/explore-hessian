local cjson = require 'cjson'
require 'optim'

function optim.entropyadam(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001
    local lrd = config.learningRateDecay or 0

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8
    local wd = config.weightDecay or 0

    local rho = config.rho or 0
    local gamma = config.gamma or 0
    local scoping = config.scoping or 1e32
    local langevin_noise = config.langevin_noise or 1e-3

    state.lparams = state.lparams or {beta1=0.25}
    local lparams = state.lparams

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) weight decay
    if wd ~= 0 then
        dfdx:add(wd, x)
    end
    local xc = x:clone()

    -- Initialization
    state.t = state.t or 0
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    -- (3) learning rate decay (annealing)
    local clr = lr / (1 + state.t*lrd)

    state.t = state.t + 1

    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local stepSize = clr * math.sqrt(biasCorrection2)/biasCorrection1

    -- (x-<x>) that is added to SGD from Langevin
    lparams.lx = xc:clone()
    lparams.lmx = lparams.lx:clone()
    lparams.eta = lparams.eta or x.new(dfdx:size()):zero()
    lparams.xxpd = 0
    lparams.w = lparams.w or x.new(dfdx:size()):zero()
    lparams.w:zero()

    if config.langevin > 0 then
        local lx = lparams.lx
        local lmx = lparams.lmx
        local eta = lparams.eta

        lparams.cgamma = gamma*(1-math.exp(-scoping*state.t))
        local lstepSize = stepSize

        for i=1,config.langevin do
            local lfx,ldfdx = opfunc(lx, true)

            -- bias term
            eta:normal()
            ldfdx:add(-lparams.cgamma, xc-lx):add(langevin_noise/math.sqrt(0.5*lstepSize), eta)

            -- update and average
            lx:add(-lstepSize, ldfdx)
            lmx:mul(lparams.beta1):add(1-lparams.beta1, lx)

            -- collect statistics
            lparams.xxpd = lparams.xxpd + torch.norm(xc-lx)
        end
        lparams.xxpd = lparams.xxpd/config.langevin

        lparams.w:copy(xc-lmx)

        -- also multiply dfdx by rho
        dfdx:mul(rho)
    end

    if opt.verbose and state.t % 10 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        dF=torch.norm(lparams.w),
        --dfdF = torch.dot(dfdx/torch.norm(dfdx), lparams.w/torch.norm(lparams.w)),
        --etanorm = torch.norm(lparams.eta*langevin_noise/math.sqrt(0.5*stepSize)),
        xxpd = lparams.xxpd,
        cgamma = lparams.cgamma}
        print(cjson.encode(debug_stats))
    end

    x:copy(xc)
    dfdx:add(lparams.w)
    --print('df: ' .. torch.norm(dfdx) .. ' dF: ', torch.norm(lparams.w))

    -- Decay the first and second moment running average coefficient
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
    state.denom:copy(state.v):sqrt():add(epsilon)

    -- (4) update x
    x:addcdiv(-stepSize, state.m, state.denom)

    -- return x*, f(x) before optimization
    return x, {fx}
end

function optim.entropysgd(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local lrd = config.learningRateDecay or 0
    local wd = config.weightDecay or 0
    local mom = config.momentum or 0
    local damp = config.dampening or mom
    local nesterov = config.nesterov or false
    local lrs = config.learningRates
    local wds = config.weightDecays

    local rho = config.rho or 0
    local gamma = config.gamma or 0
    local scoping = config.scoping or 1e32
    local langevin_noise = config.langevin_noise or 1e-3
    state.lparams = state.lparams or {beta1=0.25}
    local lparams = state.lparams

    state.evalCounter = state.evalCounter or 0
    local nevals = state.evalCounter
    assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    -- (1) evaluate f(x) and df/dx
    local xc = x:clone()
    local fx,dfdx = opfunc(x)
    state.evalCounter = state.evalCounter + 1

    -- (2) weight decay with single or individual parameters
    if wd ~= 0 then
        dfdx:add(wd, x)
    elseif wds then
        if not state.decayParameters then
            state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end
        state.decayParameters:copy(wds):cmul(x)
        dfdx:add(state.decayParameters)
    end

    -- (3) apply momentum
    if mom ~= 0 then
        if not state.dfdx then
            state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
        else
            state.dfdx:mul(mom):add(1-damp, dfdx)
        end
        if nesterov then
            dfdx:add(mom, state.dfdx)
        else
            dfdx = state.dfdx
        end
    end

    -- (4) learning rate decay (annealing)
    local clr = lr / (1 + nevals*lrd)

    -- (x-<x>) that is added to SGD from Langevin
    lparams.lx = xc:clone()
    lparams.lmx = lparams.lx:clone()
    lparams.mdfdx = lparams.mdfdx or xc:clone():zero()
    lparams.xxpd = 0
    lparams.cgamma = gamma*(1-math.exp(-scoping*state.t))

    lparams.eta = lparams.eta or x.new(dfdx:size()):zero()
    lparams.w = lparams.w or x.new(dfdx:size()):zero()
    lparams.w:zero()

    if config.langevin > 0 then
        local lx = lparams.lx
        local lmx = lparams.lmx
        local eta = lparams.eta
        local mdfdx = lparams.mdfdx:zero()
        local cgamma = lparams.cgamma

        local lclr = clr

        local debug_states = {}
        for i=1,config.langevin do
            local lfx,ldfdx = opfunc(lx, true)

            --[[
            if mom ~= 0 then
            mdfdx:mul(mom):add(1-damp, ldfdx)
            end
            if nesterov then
            ldfdx:add(mom, mdfdx)
            else
            ldfdx = mdfdx
            end
            end
            --]]

            -- bias term
            eta:normal()
            ldfdx:add(-cgamma, xc-lx):add(wd,lx):add(langevin_noise/math.sqrt(0.5*lclr), eta)

            -- update and average
            lx:add(-lclr, ldfdx)

            lmx:mul(lparams.beta1):add(1-lparams.beta1, lx)

            -- collect statistics
            lparams.xxpd = lparams.xxpd + torch.norm(xc-lx)
        end
        lparams.xxpd = lparams.xxpd/config.langevin
        lparams.w:copy(xc-lmx)

        -- also multiply dfdx by rho
        dfdx:mul(rho)
    end

    if opt.verbose and state.evalCounter % 10 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        dF=torch.norm(lparams.w),
        --dfdF = torch.dot(dfdx/torch.norm(dfdx), lparams.w/torch.norm(lparams.w)),
        --etanorm = torch.norm(lparams.eta*langevin_noise/math.sqrt(0.5*clr)),
        xxpd = lparams.xxpd,
        cgamma = lparams.cgamma}
        print(cjson.encode(debug_stats))
    end

    x:copy(xc)
    dfdx:add(lparams.w)

    -- (5) parameter update with single or individual learning rates
    if lrs then
        if not state.deltaParameters then
            state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end
        state.deltaParameters:copy(lrs):cmul(dfdx)
        x:add(-clr, state.deltaParameters)
    else
        x:add(-clr, dfdx)
    end


    -- return x*, f(x) before optimization
    return x,{fx}
end


function estimate_local_entropy(opfunc, x, config, state)
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local wd = config.weightDecay or 0
    local mom = config.momentum or 0
    local damp = config.dampening or mom
    local nesterov = config.nesterov or false

    local gamma = config.gamma or 0
    local langevin_noise = config.langevin_noise or 1e-3

    local xc = x:clone()
    local lx = xc:clone()
    local mdfdx = xc:clone():zero()
    local Z = 0

    local num_iter = config.langevin*100
    for i=1,num_iter do
        local lfx,ldfdx = opfunc(lx,true)

        if mom ~= 0 then
            mdfdx:mul(mom):add(1-damp, ldfdx)
        end
        if nesterov then
            ldfdx:add(mom, mdfdx)
        else
            ldfdx = mdfdx
        end

        -- bias term
        eta:normal()
        ldfdx:add(-gamma, xc-lx):add(wd,lx):add(langevin_noise/math.sqrt(lr), eta)

        lx:add(-lr, ldfdx)

        Z = Z + math.exp(-lfx - wd/2*torch.norm(lx) - gamma/2*torch.norm(xc-lx))
    end
    x:copy(xc)

    Z = Z/num_iter
    return Z
end
