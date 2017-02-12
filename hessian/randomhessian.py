from __future__ import absolute_import, division

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad, grad, hessian
from autograd.scipy.misc import logsumexp

from autograd.util import flatten, flatten_func
from optimizers import adam, sgd
from data import load_mnist

import argparse, os, time

parser = argparse.ArgumentParser(description='Hessian of random data')
parser.add_argument('-s', '--seed',  default=42,    help='Random seed',  type=int)
args = vars(parser.parse_args())

dtype = np.float16

opt = dict(
    B=100,
    N=10000,
    bsz=128,
    d=196,
    lr=1e-2,
    lrd=0.95,
    l2=1e-1,
    network=[196,256,256,10])

opt.update(args)
opt['nb'] = opt['N'] // opt['bsz']
opt['hnb'] = opt['nb']
np.random.seed(args['seed'])

dataset = dict(data = np.random.randn(opt['N'], opt['d']),
            labels = np.eye(10)[np.random.randint(0, 10, (opt['N'],))])
dataset['data'] /= np.repeat(np.linalg.norm(dataset['data'], axis=1), opt['d']).reshape(opt['N'], opt['d'])

def init_params(sz, rs = npr.RandomState(opt['seed'])):
    scale = 0.1
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
    return log_prior/float(opt['nb']) + log_lik

def accuracy(p, x, y):
    c = np.argmax(y, axis=1)
    ch = np.argmax(predict(p, x), axis=1)
    return np.mean(c == ch)*100

def objective(p, i):
    idx = i % opt['nb']
    b = slice(idx*opt['bsz'], (idx+1)*opt['bsz'])
    return -log_posterior(p, dataset['data'][b], dataset['labels'][b], opt['l2'])
objective_grad = grad(objective)

p = init_params(opt['network'])

s = time.time()
for e in xrange(opt['B']):
    lr = opt['lr']*opt['lrd']**e

    def stats(p, i, g):
        if i % opt['nb'] == 0:
            te = accuracy(p, dataset['data'], dataset['labels'])
            print '{:15}|{:20}|'.format(e, te)

    p = sgd(objective_grad, p,
                step_size = lr, num_iters=opt['nb'], mass=0,
                callback=stats)
print '[opt] ', time.time()-s
params = p

flat_f, unflatten, flat_params = flatten_func(objective, params)
flat_hess = hessian(flat_f)
h = np.zeros((flat_params.shape[0], flat_params.shape[0]))
for i in np.random.permutation(np.arange(opt['nb']))[:opt['hnb']]:
    h += flat_hess(flat_params, i).squeeze()
    print '[progress] ', i, ' dt: ', time.time()-s
h = h.squeeze()/float(opt['hnb']*opt['bsz'])
print '[hessian] ', time.time() -s

e = np.linalg.eigvals(h)
np.save('randomh.npz', e=e,h=h)