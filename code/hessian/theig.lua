t = require 'torch'
npy4th = require 'npy4th'
require 'paths'

fp = arg[1]
base = paths.basename(arg[1])
dir = paths.dirname(arg[1])
outfp = dir .. '/e_' .. base
print(outfp)

print('[loading]')
h = npy4th.loadnpy(fp)
h = h:float()

print('[eig]')
e = t.eig(h)

print('[saving]')
npy4th.savenpy(outfp, e)
