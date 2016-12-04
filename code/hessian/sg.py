import numpy as np
import os, argparse, sys, pdb
from joblib import Parallel, delayed
import cPickle as pickle
import pandas as pd
import time
import json

parser = argparse.ArgumentParser(description='Spin glass subdominant clusters')
parser.add_argument('-n', '--num_spins',    help='num spins',   type=int, default = 100)
parser.add_argument('-N', '--num_jobs',     help='num jobs',    type=int, default = 1)
parser.add_argument('-m', '--max_jobs',     help='max jobs',    type=int, default = 1)
parser.add_argument('-s', '--seed',         help='start seed',  type=int, default = 42)
parser.add_argument('--LR',                 help='step size',  type=float, default = 1e-1)

parser.add_argument('--j2',                 help='2-spin glass coefficient',  type=float, default = 0.1)
parser.add_argument('-r', '--rho',          help='Entropy coupling',  type=float, default = 1e-2)
parser.add_argument('-g', '--gamma',        help='Langevin coupling gamma',  type=float, default = 1e-3)
parser.add_argument('--langevin',           help='Langevin iterations',  type=int, default = 0)

parser.add_argument('--eps',                help='norm of grad',  type=float, default = 1e-4)
parser.add_argument('--dry',                help='Dry run',    action='store_true')
opt = vars(parser.parse_args())

np.random.seed(opt['seed'])

n = opt['num_spins']
N = opt['num_jobs']
J = np.random.randn(n,n,n)/n
J2 = opt['j2']*np.random.randn(n,n)/np.sqrt(n)
rho, gamma, langevin = opt['rho'], opt['gamma'], opt['langevin']

def bot_timestamp_now():
    return int(time.time()*1e6)

def proj_grad(x,g):
    return g - np.outer(x,x).dot(g)/float(n)

def proj_hess(x, g, h):
    return h - np.outer(x,x).dot(h)/float(n) - np.sum(x*g)*np.eye(n)/np.sqrt(n)

def feval(x, get_hess = False):
    kx = np.einsum('i,j,k->ijk',x,x,x)
    xxt = np.outer(x, x)

    f = np.sum(J*kx) + np.sum(J2*xxt)

    grad2 = J2.dot(x) + J2.T.dot(x)
    grad =  np.einsum("sjk,jk->s", J, xxt) + \
            np.einsum("jsk,jk->s", J, xxt) + \
            np.einsum("jks,jk->s", J, xxt) + \
            grad2

    if not get_hess:
        return  f, grad, None

    off_diag =  np.einsum("kmn,k->mn", J, x) + \
                np.einsum("knm,k->mn", J, x) + \
                np.einsum("mkn,k->mn", J, x) + \
                np.einsum("nkm,k->mn", J, x) + \
                np.einsum("mnk,k->mn", J, x) + \
                np.einsum("nmk,k->mn", J, x)
    np.fill_diagonal(off_diag, 0)

    diag =  2*(np.einsum("kmm,k->m", J, x) + \
            np.einsum("mkm,k->m", J, x) + \
            np.einsum("mmk,k->m", J, x))

    hess2 = J2 + J2.T
    hess = np.diag(diag) + off_diag + hess2

    return  f, grad, hess

def compute_dF(x):
    """
        f' = f(x) + gamma x.T.dot(x')
    """
    eta, eps = opt['LR'], opt['eps']
    discard = 0.25

    z = x.copy()
    mu = np.zeros(x.shape)
    for i in xrange(langevin):
        f,g,_ = feval(z)
        gn = g - gamma*x + 1e-1*np.random.randn(n)

        gnn = proj_grad(z, gn)
        z = z-gnn*eta

        z = z/np.linalg.norm(z)*np.sqrt(n)

        if i > langevin*discard:
            mu += z
    mu = mu/float(langevin*(1-discard))

    return -gamma*mu

def do_run():
    eta, eps = opt['LR'], opt['eps']

    if langevin > 0:
        eps = np.sqrt(eps)

    np.random.seed()
    x = np.random.randn(n)
    x = x/np.linalg.norm(x)*np.sqrt(n)

    i = 0
    while True:
        try:
            f,g,_ = feval(x, get_hess=False)
            gn = g
            dF = 0*g

            if langevin > 0:
                dF = compute_dF(x)
                gn = rho*g - dF

            gnn = proj_grad(x, gn)
            print 'grad: ', np.linalg.norm(g), np.linalg.norm(dF), \
                            np.linalg.norm(gn), np.linalg.norm(gnn)

            x = x -gnn*eta
            x = x/np.linalg.norm(x)*np.sqrt(n)

            if np.linalg.norm(gnn) < eps or i > 1e5:
                break
            if langevin > 0 and i > 1e3:
                break

            if i%1000 == 0:
                print i,f, np.linalg.norm(gnn)

            print i, f, np.linalg.norm(gnn)
            i += 1

        except KeyboardInterrupt:
            break

    f,g,h = feval(x, get_hess=True)
    h = proj_hess(x,g,h)
    eig = np.linalg.eigvalsh(h)
    ret = [x, f, i, gnn, eig]
    print i, f
    return ret

def main():
    print 'Starting run...'
    d = Parallel(n_jobs=opt['max_jobs'])(delayed(do_run)() for i in range(opt['num_jobs']))
    d = pd.DataFrame(d, columns = ['x','f','i','g','h'])

    print 'Saving ...'
    fname = ('%d_seed_%d')%(bot_timestamp_now(), opt['seed'])
    pickle.dump({'opt': opt, 'J': J, 'd': d}, open(fname+'.p', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)

def test():
    h = do_run()[-1]
    print h
    return h

if __name__=='__main__':
    if opt['dry']:
        test()
    else:
        main()
