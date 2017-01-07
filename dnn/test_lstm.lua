require 'cunn'
require 'cudnn'
m = require 'models'

T,b = 35, 16
opt = {model='lstm'}
model, cost, p = m.ptb()
x = torch.Tensor(T,b):random(1,10000):cuda()
y = torch.Tensor(T,b):random(1,10000):cuda()

yh = model:forward(x)
f = cost:forward(yh,y)
dfdy = cost:backward(yh,y)

print('starting backprop')
dx = model:backward(x,dfdy)
cutorch:synchronize()
