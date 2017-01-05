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
    local noise = config.noise or 1e-3

    state.lparams = state.lparams or {beta1=0.9}
    local lparams = state.lparams

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x, false)

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

    if config.L > 0 then
        local lx = lparams.lx
        local lmx = lparams.lmx
        local eta = lparams.eta

        --lparams.cgamma = gamma*(1 - math.exp(-state.t*scoping))
        lparams.cgamma = gamma*(1+scoping)^state.t

        local lstepSize = 0.1

        for i=1,config.L do
            local lfx,ldfdx = opfunc(lx, true)

            -- bias term
            eta:normal()
            ldfdx:add(-lparams.cgamma, xc-lx):add(noise/math.sqrt(0.5*lstepSize), eta)

            -- update and average
            lx:add(-lstepSize, ldfdx)
            lmx:mul(lparams.beta1):add(1-lparams.beta1, lx)

            -- collect statistics
            lparams.xxpd = lparams.xxpd + torch.norm(xc-lx)
        end
        lparams.xxpd = lparams.xxpd/config.L

        lparams.w:copy(xc-lmx)
    end

    if opt.verbose and state.t % 10 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        dF=torch.norm(lparams.w),
        dfdF = torch.dot(dfdx/torch.norm(dfdx), lparams.w/(torch.norm(lparams.w)+1e-6)),
        eta = torch.norm(lparams.eta*noise/math.sqrt(0.5*stepSize)),
        xxpd = lparams.xxpd,
        g = lparams.cgamma}
        print(cjson.encode(debug_stats))

        lparams.dfdF = debug_stats.dfdF
    end

    if opt.L > 0 then
        -- also multiply dfdx by rho
        dfdx:mul(rho)
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

    -- averaging parameters
    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    local rho = config.rho or 0
    local gamma = config.gamma or 0
    local scoping = config.scoping or 1e32
    local noise = config.noise or 1e-3
    state.lparams = state.lparams or {beta1=0.9}
    local lparams = state.lparams

    state.evalCounter = state.evalCounter or 0
    local nevals = state.evalCounter
    assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    -- (1) evaluate f(x) and df/dx
    local xc = x:clone()
    local fx,dfdx = opfunc(x, false)
    state.evalCounter = state.evalCounter + 1

    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()

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

    --lparams.cgamma = gamma*(1 - math.exp(-scoping*state.evalCounter))
    lparams.cgamma = gamma*(1+scoping)^state.evalCounter

    lparams.eta = lparams.eta or x.new(dfdx:size()):zero()
    lparams.w = lparams.w or x.new(dfdx:size()):zero()
    lparams.w:zero()

    if config.L > 0 then
        local lx = lparams.lx
        local lmx = lparams.lmx
        local eta = lparams.eta
        local mdfdx = lparams.mdfdx:zero()
        local cgamma = lparams.cgamma

        local lclr = 0.1

        local debug_states = {}
        for i=1,config.L do
            local lfx,ldfdx = opfunc(lx, true)

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
            ldfdx:add(-cgamma, xc-lx):add(wd,lx):add(noise/math.sqrt(0.5*lclr), eta)

            -- update and average
            lx:add(-lclr, ldfdx)

            lmx:mul(lparams.beta1):add(1-lparams.beta1, lx)

            -- collect statistics
            lparams.xxpd = lparams.xxpd + torch.norm(xc-lx)
        end
        lparams.xxpd = lparams.xxpd/config.L
        lparams.w:copy(xc-lmx)
    end

    if opt.verbose and state.evalCounter % 50 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        dF=torch.norm(lparams.w),
        dfdF = torch.dot(dfdx/torch.norm(dfdx), lparams.w/(torch.norm(lparams.w)+1e-6)),
        eta = torch.norm(lparams.eta*noise/math.sqrt(0.5*clr)),
        xxpd = lparams.xxpd,
        g = lparams.cgamma}
        print(cjson.encode(debug_stats))

        lparams.dfdF = debug_stats.dfdF
    end

    if opt.L > 0 then
        -- also multiply dfdx by rho
        dfdx:mul(rho)
    end

    x:copy(xc)
    dfdx:add(lparams.w)

    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
    x:add(-clr, dfdx)

    return x,{fx}
end

function optim.sgdld(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-1
    local lrd = config.learningRateDecay or 0
    local wd = config.weightDecay or 0
    local noise = config.noise or 1e-3

    local b = config.b or 0.25
    state.t = state.t or 0

    -- (1) evaluate f(x) and df/dx
    local fx,dfdx = opfunc(x, false)
    state.t = state.t + 1
    state.eta = state.eta or x.new(dfdx:size()):zero()

    -- so that the logging scripts do not break
    state.lparams = state.lparams or {}
    local lparams = state.lparams
    lparams.lx = x:clone()
    lparams.lmx = lparams.lx:clone()
    lparams.eta = lparams.eta or x.new(dfdx:size()):zero()
    lparams.xxpd = 0
    lparams.w = lparams.w or x.new(dfdx:size()):zero()
    lparams.w:zero()

    -- lr annealing
    local clr = lr / (1 + state.t)^(b)

    -- weight decay
    if wd ~= 0 then
        dfdx:add(wd, x)
    end

    -- update gradient
    state.eta:normal()

    if opt.verbose and state.t % 10 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        eta = torch.norm(state.eta*noise/math.sqrt(0.5*clr))}
        print(cjson.encode(debug_stats))
    end

    dfdx:add(noise/math.sqrt(0.5*clr), state.eta)
    x:add(-clr, dfdx)

    return x, {fx}
end
