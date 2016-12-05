import os
import numpy as np
import cPickle as pickle

def preprocess_cifar100(d):
    print 'Preprocessing...'
    print '     Loading...'
    d = os.path.join(d, 'cifar-100-python')
    
    def helper(fp):
        t = pickle.load(open(fp, 'rb'))
        return {'x': t['data'].reshape(-1,3,32,32).astype(np.float32),
                'y': t['fine_labels']}
        
    def normalize(data, eps=1e-8):
        data -= data.mean(axis=(1, 2, 3), keepdims=True)
        std = np.sqrt(data.var(axis=(1, 2, 3), ddof=1, keepdims=True))
        std[std < eps] = 1.
        data /= std
        return data

    train_all = helper(os.path.join(d, 'train'))
    test = helper(os.path.join(d, 'test'))

    ntrain_all, ntrain = 50000, 40000
    nval, ntest = 10000, 10000
    train, val = {}, {}
    train['x'], train['y'] = train_all['x'][:ntrain], train_all['y'][:ntrain]
    val['x'], val['y'] = train_all['x'][ntrain:], train_all['y'][ntrain:]

    # Contrast normalize
    print '     Normalizing...'
    for di in [train_all['x'], train['x'], val['x'], test['x']]:
        di = normalize(di)

    # ZCA Whiten
    print '     Computing whitening matrix...'
    train_flat = train['x'].reshape(ntrain, -1).T
    train_all_flat = train_all['x'].reshape(ntrain_all, -1).T
    val_flat = val['x'].reshape(nval, -1).T
    test_flat = test['x'].reshape(ntest, -1).T

    pca = PCA(D=train_flat, n_components=train_flat.shape[1])
    pca_all = PCA(D=train_all_flat, n_components=train_all_flat.shape[1])

    print '   Whitening data...'
    train_flat = pca.transform(D=train_flat, whiten=True, ZCA=True)
    train['x'] = train_flat.T.reshape(train['x'].shape)
    
    train_all_flat = pca_all.transform(D=train_all_flat, whiten=True, ZCA=True)
    train_all['x'] = train_all_flat.T.reshape(train_all['x'].shape)
    
    val_flat = pca.transform(D=val_flat, whiten=True, ZCA=True)
    val['x'] = val_flat.T.reshape(val['x'].shape)
    
    test_flat = pca_all.transform(D=test_flat, whiten=True, ZCA=True)
    test['x'] = test_flat.T.reshape(test['x'].shape)

    print '   Saving...'
    opd = os.path.join(d, 'preprocessed')
    if not os.path.exists(opd):
        os.mkdir(opd)

    np.savez(os.path.join(opd, 'cifar-100-train.npz'),
             data=train['x'],
             labels=train['y'])
    np.savez(os.path.join(opd, 'cifar-100-train_all.npz'),
             data=train_all['x'],
             labels=train_all['y'])
    np.savez(os.path.join(opd, 'cifar-100-val.npz'),
             data=val['x'],
             labels=val['y'])
    np.savez(os.path.join(opd, 'cifar-100-test.npz'),
             data=test['x'],
             labels=test['y'])

    print 'Preprocessing complete'


class PCA(object):

    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)
