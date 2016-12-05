import os, gzip, struct, array, pdb
import numpy as np
import cPickle as pickle

def bin_ndarray(ndarray, new_shape, operation='mean'):
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

def load_mnist(dir, opt, dtype = np.float32):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)

    def parse_labels(fp):
        with gzip.open(dir + fp, 'rb') as fh:
            _, n = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(fp):
        with gzip.open(dir + fp, 'rb') as fh:
            _, n, r, c = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(n, r, c)

    tx = parse_images('/train-images-idx3-ubyte.gz')
    ty = parse_labels('/train-labels-idx1-ubyte.gz')
    vx  = parse_images('/t10k-images-idx3-ubyte.gz')
    vy  = parse_labels('/t10k-labels-idx1-ubyte.gz')

    w = opt['width']
    tx = bin_ndarray(tx, (tx.shape[0], w,w))
    vx = bin_ndarray(vx, (vx.shape[0], w,w))
    tx = tx.reshape(tx.shape[0], 1, w,w)
    vx = vx.reshape(vx.shape[0], 1, w,w)
    tx /= 255.0
    vx /= 255.0

    if not opt['cnn']:
        tx = partial_flatten(tx)
        vx  = partial_flatten(vx)

    ty = one_hot(ty, 10)
    vy = one_hot(vy, 10)

    frac = (opt['full'] and 1) or 0.01
    tn, vn = int(tx.shape[0]*frac), int(vx.shape[0]*frac)

    idx = np.random.permutation(range(tx.shape[0]))[:tn]
    tx, ty = tx[idx], ty[idx]

    idx = np.random.permutation(range(vx.shape[0]))[:vn]
    vx, vy = vx[idx], vy[idx]

    ret = (tx.shape[0], tx.astype(dtype), ty.astype(dtype), vx.astype(dtype), vy.astype(dtype))
    #pickle.dump(ret, open('mnist.pkl', 'wb'))
    return ret

def load_cifar(dir, opt, dtype=np.float32):
    assert os.path.isdir(dir)

    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)

    tn, vn = 0, 0
    tx, ty, vx, vy = None, None, None, None
    for i in xrange(5):
        fname = dir + '/data_batch_%d' % (i+1)
        with open(fname, 'rb') as fp:
            d = pickle.load(fp)
            a = np.transpose(d['data'].reshape(d['data'].shape[0], 32,32, 3), axes=[0, 3, 1, 2])
            b = one_hot(np.array(d['labels']), 10)
            if tx is not None:
                tx = np.vstack((tx, a))
                ty = np.vstack((ty, b))
            else:
                tx, ty = a, b
            tn += a.shape[0]

    with open(dir + '/test_batch', 'rb') as fp:
        d = pickle.load(fp)
        a = np.transpose(d['data'].reshape(d['data'].shape[0], 32,32, 3), axes=[0, 3, 1, 2])
        b = one_hot(np.array(d['labels']), 10)
        vx, vy = a, b
        vn = a.shape[0]

    w = opt['width']
    tx = bin_ndarray(tx, (tx.shape[0], w,w))
    vx = bin_ndarray(vx, (vx.shape[0], w,w))
    tx = tx.reshape(tx.shape[0], 3, w,w)
    vx = vx.reshape(vx.shape[0], 3, w,w)
    tx /= 255.0
    vx /= 255.0

    if not opt['cnn']:
        tx = partial_flatten(tx)
        vx  = partial_flatten(vx)

    frac = (opt['full'] and 1) or 0.01
    tn, vn = int(tx.shape[0]*frac), int(vx.shape[0]*frac)

    idx = np.random.permutation(range(tx.shape[0]))[:tn]
    tx, ty = tx[idx], ty[idx]

    idx = np.random.permutation(range(vx.shape[0]))[:vn]
    vx, vy = vx[idx], vy[idx]

    ret = (tx.shape[0], tx.astype(dtype), ty.astype(dtype), vx.astype(dtype), vy.astype(dtype))
    return ret

def string_to_one_hot(s, n):
    ascii = np.array([ord(c)-32 for c in s]).T
    return np.array(ascii[:,None] == np.arange(n)[None, :], dtype=int)

def one_hot_to_string(m):
    return "".join([chr(np.argmax(c)+32) for c in m])

def load_lstm_data(fp, opt):
    T = opt['T']
    vocab_sz = opt['network'][0]
    assert opt['network'][0] == opt['network'][2], 'Regurgating input: wrong size'

    with open(fp) as f:
        c = f.readlines()
    c = [l for l in c if len(l) > 2]

    if not opt['full']:
        c = c[: opt['max_lines']]

    d = np.zeros((T, len(c), vocab_sz))
    for i, l in enumerate(c):
        pl = (l + " " * T)[:T]
        d[:, i, :] = string_to_one_hot(pl, vocab_sz)
    return d
