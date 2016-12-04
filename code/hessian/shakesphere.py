from __future__ import absolute_import, division

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad, grad, hessian
from autograd.scipy.misc import logsumexp
from numba import double, jit, autojit, vectorize

from autograd.util import flatten, flatten_func
from optimizers import adam, sgd
from data import load_lstm_data, one_hot_to_string

import argparse, os, time

parser = argparse.ArgumentParser(description='Hessian of LSTM')
parser.add_argument('-s', '--seed',      help='Random seed',            type=int, required=True)
parser.add_argument('-i', '--input',    help='Input file name',   type=str, default='tiny-shakesphere.txt')
parser.add_argument('-o', '--output',    help='Save eigenvalues here',   type=str, default='/dev/null')
parser.add_argument('-n', '--num_hidden',    help='Size of hidden layer',   type=int, default = 48)
parser.add_argument('--max_epochs',    help='Max. epochs',   type=int, default = 256)
args = vars(parser.parse_args())
dtype = np.float16

opt = {
    'batch_size' : 32,
    'lr' : 1e-1,
    'lrd' : 0.98,
    'l2' : 1e-10,
    'scale' : 1e-1,
    'network' : [96, args['num_hidden'], 96],
    'T' : 32,
    'max_epochs' : args['max_epochs'],
    'full' : False,
    'max_lines' : 32
}
opt.update(args)

param_idx = {   'initc':0, 'inith':1, 'change':2, 'forget':3,
                'ingate':4, 'outgate':5, 'predict':6
            }

def init_params(scale, sz, rs = npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape).astype(dtype) * scale

    x, h, y = sz[0], sz[1], sz[2]
    return (rp(1, h),
            rp(1, h),
            rp(x + h + 1, h),
            rp(x + h + 1, h),
            rp(x + h + 1, h),
            rp(x + h + 1, h),
            rp(h + 1, y))

def predict(p, x):
    def sigmoid(x):
        return (np.tanh(x) + 1.)/2.

    def catmult(weights, *args):
        cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
        return np.dot(cat_state, weights)

    def update(_x, h, c):
        change = np.tanh(catmult(p[param_idx['change']], _x, h))
        forget = sigmoid(catmult(p[param_idx['forget']], _x, h))
        ingate = sigmoid(catmult(p[param_idx['ingate']], _x, h))
        outgate = sigmoid(catmult(p[param_idx['outgate']], _x, h))
        c = outgate*forget + ingate*change
        h = outgate*np.tanh(c)
        return h, c

    def htoprob(h):
        y = catmult(p[param_idx['predict']], h)
        return y - logsumexp(y, axis=1, keepdims=True)

    nx = x.shape[1]
    h = np.repeat(p[param_idx['inith']], nx, axis=0)
    c = np.repeat(p[param_idx['initc']], nx, axis=0)
    y = [htoprob(h)]
    for _x in x:
        h, c = update(_x, h, c)
        y.append(htoprob(h))
    return y    

def log_likelihood(p, x, y):
    yh = predict(p, x)
    ll = 0
    T, nx, _ = x.shape
    for t in xrange(T):
        ll += np.sum(yh[t]*y[t])

    t1,_ = flatten(p)
    l2term = opt['l2']*np.dot(t1, t1)

    return ll/float(T*nx) - l2term

def objective(p, i):
    idx = i % opt['num_batches']
    b = slice(idx*opt['batch_size'], (idx+1)*opt['batch_size'])
    return -log_likelihood(p, tx[:,b,:], tx[:,b,:])

objective_grad = grad(objective)

tx = load_lstm_data(opt['input'], opt)
p = init_params(opt['scale'], opt['network'], npr.RandomState(opt['seed']))
opt['num_batches'] = int(np.ceil(tx.shape[1] / opt['batch_size']))

s = time.time()
for e in xrange(opt['max_epochs']):
    lr = opt['lr']*opt['lrd']**e

    def stats(p, i, g):
        if i % (opt['num_batches']/10) == 0:
            yh = np.asarray(predict(p, tx))
            for j in np.random.permutation(range(yh.shape[1]))[:8]:
                x = one_hot_to_string(tx[:,j,:])
                xh = one_hot_to_string(yh[:,j,:])
                print(x.replace('\n', ' ') + '|' + xh.replace('\n', ' '))
        #print '{:15}|{:20}|'.format(e+i/float(opt['num_batches']), objective(p,i))

    p = adam(objective_grad, p,
                step_size = lr, num_iters=opt['num_batches'],
                callback=stats)
    
    loss = np.mean([objective(p, i) for i in xrange(opt['num_batches'])])
    print '{:15}|{:20}|'.format(e, loss)
    print('---------------------------')

print '[opt] ', time.time()-s
params = p

flat_f, unflatten, flat_params = flatten_func(objective, params)
flat_hess = hessian(flat_f)
h = flat_hess(flat_params, i).squeeze()
h = h.squeeze()/float(opt['batch_size'])
print '[hessian] ', time.time() -s

if not opt['output'] == '/dev/null':
    np.save(opt['output'], h)
