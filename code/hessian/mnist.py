from __future__ import absolute_import, division

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad, grad, hessian
from autograd.scipy.misc import logsumexp

from estimate_hessian import *

from autograd.util import flatten, flatten_func
from optimizers import adam, sgd
from data import load_mnist

import argparse, os, time

parser = argparse.ArgumentParser(description='Hessian of MNIST')
parser.add_argument('-s', '--seed',      help='Random seed',            type=int, required=True)
parser.add_argument('-o', '--output',    help='Save eigenvalues here',   type=str, default='/dev/null')
parser.add_argument('-n', '--num_hidden',    help='Size of hidden layer',   type=int, default = 32)
parser.add_argument('--hessian_num_batches',    help='Hessian batches',   type=int, default = 32)
parser.add_argument('--max_epochs',    help='Max. epochs',   type=int, default = 50)
args = vars(parser.parse_args())

dtype = np.float16

opt = {
    'batch_size' : 128,
    'lr' : 1e-2,
    'lrd' : 0.95,
    'l2' : 1e-16,
    'scale' : 1e-1,
    'network' : [14*14, args['num_hidden'], 10],
    'max_epochs' : args['max_epochs'],
    'full' : True,
    'cnn': False,
    'width': 14
}
opt.update(args)

def init_params(scale, sz, rs = npr.RandomState(0)):
    return [(   scale*rs.randn(m,n).astype(dtype),
                scale*rs.randn(n).astype(dtype))
    for m, n in zip(sz[:-1], sz[1:])]

def predict(p, x):
    relu = lambda _x: np.maximum(_x, 0.)

    for w, b in p:
        yh = np.dot(x, w) + b
        x = relu(yh)
    return yh - logsumexp(yh, axis=1, keepdims=True)

def log_posterior(p, x, y, l2reg):
    t1,_ = flatten(p)

    log_prior = -l2reg*np.dot(t1, t1)
    log_lik = np.sum(predict(p, x)*y)
    return log_prior/float(opt['num_batches']) + log_lik

def accuracy(p, x, y):
    c = np.argmax(y, axis=1)
    ch = np.argmax(predict(p, x), axis=1)
    return np.mean(c == ch)*100

def objective(p, i):
    idx = i % opt['num_batches']
    b = slice(idx*opt['batch_size'], (idx+1)*opt['batch_size'])
    return -log_posterior(p, tx[b], ty[b],
            opt['l2'])
objective_grad = grad(objective)

n, tx, ty, vx, vy = load_mnist('/local2/pratikac/mnist/raw', opt)
p = init_params(opt['scale'], opt['network'], npr.RandomState(opt['seed']))
opt['num_batches'] = int(np.ceil(n / opt['batch_size']))

s = time.time()
for e in xrange(opt['max_epochs']):
    lr = opt['lr']*opt['lrd']**e

    def stats(p, i, g):
        if i % opt['num_batches'] == 0:
            te = accuracy(p, tx, ty)
            ve = accuracy(p, vx, vy)
            print '{:15}|{:20}|{:20}|'.format(e, te, ve)

    p = sgd(objective_grad, p,
                step_size = lr, num_iters=opt['num_batches'], mass=0,
                callback=stats)
print '[opt] ', time.time()-s
params = p

flat_f, unflatten, flat_params = flatten_func(objective, params)
flat_hess = hessian(flat_f)
h = np.zeros((flat_params.shape[0], flat_params.shape[0]))
for i in np.random.permutation(np.arange(opt['num_batches']))[:opt['hessian_num_batches']]:
    h += flat_hess(flat_params, i).squeeze()
    print '[progress] ', i, ' dt: ', time.time()-s
h = h.squeeze()/float(opt['hessian_num_batches']*opt['batch_size'])
print '[hessian] ', time.time() -s

e = np.linalg.eigvals(h)
if not opt['output'] == '/dev/null':
    np.save(opt['output'], e)
