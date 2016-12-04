import autograd.numpy as np
from autograd import hessian_vector_product

def estimate_hessian(f, num_samples=100):
    def hfun(*args, **kwargs):
        d = args[0].shape[0]
        ddfv = hessian_vector_product(f)
        n = num_samples
        hv = np.zeros((d,d))
        for i in xrange(n):
            w = np.random.randn(d)
            tmp = args + (w,)
            np.add(hv, np.outer(ddfv(*tmp, **kwargs), w), hv)
        hv /= float(n)
        return hv
    return hfun

def estimate_hessian_eye(f):
    def hfun(*args, **kwargs):
        d = args[0].shape[0]
        ddfv = hessian_vector_product(f)
        e  = np.eye(d)
        hv = np.zeros((d,d))
        for i in xrange(d):
            tmp = args + (e[:,i],)
            hv[:,i] = ddfv(*tmp, **kwargs)
        return hv
    return hfun
