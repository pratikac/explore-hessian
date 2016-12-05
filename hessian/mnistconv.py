from __future__ import absolute_import, division

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad, grad, hessian
from autograd.scipy.misc import logsumexp
import autograd.scipy.signal

from estimate_hessian import *

from autograd.util import flatten, flatten_func
from optimizers import adam, sgd
from data import load_mnist

import argparse, os, time, pdb, gc

parser = argparse.ArgumentParser(description='Hessian of MNIST (conv)')
parser.add_argument('-s', '--seed',      help='Random seed',            type=int, required=True)
parser.add_argument('-o', '--output',    help='Save eigenvalues here',   type=str, default='/dev/null')
parser.add_argument('--hessian_num_batches',    help='Hessian batches',   type=int, default = 32)
parser.add_argument('--max_epochs',    help='Max. epochs',   type=int, default = 1)
args = vars(parser.parse_args())
dtype = np.float32

convolve = autograd.scipy.signal.convolve

opt = {
    'batch_size' : 32,
    'lr' : 1e-3,
    'lrd' : 0.98,
    'scale' : 1e-1,
    'max_epochs' : args['max_epochs'],
    'full' : True,
    'cnn' : True,
    'width': 28
}
opt.update(args)

def init_params(scale, rs = npr.RandomState(0)):
    w = range(4)

    if True:
        # LeNet:    20-50-500
        #           10-20-(320)-128-10
        w[0] = (scale*rs.randn(1,10, 5,5).astype(dtype),
                scale*rs.randn(1,10,1,1).astype(dtype))
        w[1] = (scale*rs.randn(10,20, 5,5).astype(dtype),
                scale*rs.randn(1,20,1,1).astype(dtype))
        w[2] = (scale*rs.randn(320, 128).astype(dtype),
                scale*rs.randn(128).astype(dtype))
        w[3] = (scale*rs.randn(128, 10).astype(dtype),
                scale*rs.randn(10).astype(dtype))
    else:
        w[0] = (scale*rs.randn(1,8, 5,5).astype(dtype),
                scale*rs.randn(1,8,1,1).astype(dtype))
        w[1] = (scale*rs.randn(8,8, 5,5).astype(dtype),
                scale*rs.randn(1,8,1,1).astype(dtype))
        w[2] = (scale*rs.randn(128, 64).astype(dtype),
                scale*rs.randn(64).astype(dtype))
        w[3] = (scale*rs.randn(64,10).astype(dtype),
                scale*rs.randn(10).astype(dtype))

    t1,_ = flatten(w)
    print '[size]: ', t1.shape
    return w

def maxpool(x, k):
    newsz = x.shape[:2]
    sz = x.shape[2:]
    newsz += (k[0], sz[0]//k[0])
    newsz += (k[1], sz[1]//k[1])
    r = x.reshape(newsz)
    return np.max(np.max(r, axis=2), axis=3)

def predict(p, x):
    relu = lambda _x: np.maximum(_x, 0.)

    x = relu(convolve(x, p[0][0], axes=([2,3],[2,3]), dot_axes=([1], [0]), mode='valid') + p[0][1])
    x = maxpool(x, (2,2))
    x = relu(convolve(x, p[1][0], axes=([2,3],[2,3]), dot_axes=([1], [0]), mode='valid') + p[1][1])
    x = maxpool(x, (2,2))
    x = x.reshape(x.shape[0], -1)

    for w, b in p[2:]:
        yh = np.dot(x, w) + b
        x = relu(yh)
    return yh - logsumexp(yh, axis=1, keepdims=True)

def log_posterior(p, x, y):
    return np.sum(predict(p, x)*y)

def accuracy(p, x, y):
    c = np.argmax(y, axis=1)
    ch = np.argmax(predict(p, x), axis=1)
    return np.mean(c == ch)*100

def objective(p, i):
    idx = i % opt['num_batches']
    b = slice(idx*opt['batch_size'], (idx+1)*opt['batch_size'])
    print i, accuracy(p, tx[b], ty[b])
    return -log_posterior(p, tx[b], ty[b])
objective_grad = grad(objective)

n, tx, ty, vx, vy = load_mnist('/local2/pratikac/mnist/raw', opt)
p = init_params(opt['scale'], npr.RandomState(opt['seed']))
opt['num_batches'] = int(np.ceil(n / opt['batch_size']))

s = time.time()
for e in xrange(opt['max_epochs']):
    lr = opt['lr']*opt['lrd']**e

    def stats(p, i, g):
        if i % opt['num_batches'] == 0:
            te = accuracy(p, tx, ty)
            ve = accuracy(p, vx, vy)
            print '{:15}|{:20}|{:20}|'.format(e, te, ve)
        print ('[%03d][%03d/%03d]')%(e, i%opt['num_batches'], opt['num_batches'])
        gc.collect()

    p = adam(objective_grad, p,
                step_size = lr, num_iters=opt['num_batches'] // 10,
                callback=stats)
print '[opt] ', time.time()-s
params = p

print '[flat params] ...'
flat_f, unflatten, flat_params = flatten_func(objective, params)

print '[flat hess] ...'
flat_hess = estimate_hessian(flat_f, 100)

h = None
print '[compute hess] ...'
for i in np.random.permutation(np.arange(opt['num_batches']))[:opt['hessian_num_batches']]:
    if h is None:
        h = flat_hess(flat_params, i)
    else:
        np.add(h, flat_hess(flat_params, i), h)
    print '[progress] ', i, ' dt: ', time.time()-s
    gc.collect()
h = h.squeeze()/float(opt['hessian_num_batches']*opt['batch_size'])
print '[hessian] ', time.time() -s

if not opt['output'] == '/dev/null':
    np.save(opt['output']+'.hes', h)

e = np.linalg.eigvals(h)
if not opt['output'] == '/dev/null':
    np.save(opt['output']+'.eig', e)
