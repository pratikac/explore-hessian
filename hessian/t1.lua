t = require 'torch'
require 'nn'
autograd = require 'autograd'
autograd.optimize(true)

function f(p, x, y)
  local w, b = t.reshape(p[{ {1, 100}}], 10,10), p[{ {101, 110}}]
  local h = w*x + b
  return t.sum(t.pow(h-y, 2))/2.
end

function flat_jacobian(f)
  return
  function(...)
    local args = {...}
    local x, y = args[1], f(...)
    y = t.Tensor(y)

    local jac = t.zeros(y:numel(), x:numel())
    for yi=1,y:numel() do
      local function c(z) return f(z)[yi] end
      local dc = autograd(c)(...)
      jac[{yi, {}}] = dc
    end
    return jac
  end
end

function flat_hessian(f)
  return
  function(...)
      local args = {...}
      local df = autograd(f)
      local nw = args[1]:numel()
      local h = t.zeros(nw, nw)
      for i=1,nw do
        local function c(z) return df(z)[i] end
        h[{i,{}}] = autograd(c)(...)
      end
      return h
  end
end

w = t.randn(110)
x = t.randn(10)
y = t.randn(10)
df = flat_hessian(f)

--[[
w = t.randn(100, 10)

local x = t.randn(1, 100)
local y = t.randn(1, 10)

function f(w, x, y)
  local yh = x*w
  return t.sum(t.pow(yh-y, 2))
end
local df = autograd(f)

local function f2(w, x, y)
  local g = df(w, x, y)
  return t.sum(g)
end

local ddf = autograd(f2)
local gg = ddf(w, x, y)
print(gg)
--]]
