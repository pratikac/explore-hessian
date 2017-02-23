import numpy as np
import numpy.random as npr
from scipy.signal import correlate, convolve, gaussian, exponential
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import seaborn as sns
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
opt = vars(parser.parse_args())

plt.ion()
sns.set_style('white')

npr.seed(42)
fontsize = 24

plt.rc('font', size=fontsize)
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)
plt.rc('figure', titlesize=fontsize)

def plot_clusters():
    ds = [1, 0.5, 0.1, 0.01]
    n = 200
    x = np.random.rand(n,2)*2 - 1
    x = x[np.linalg.norm(x,axis=1) < 1]

    xc1 = np.random.multivariate_normal([0.5,0.5], np.eye(2)*0.01, n/4)
    xc2 = np.random.multivariate_normal([-0.25,0.75], np.eye(2)*0.005, n/4)
    xc3 = np.random.multivariate_normal([0,-0.5], np.eye(2)*0.005, n/4)
    xc = np.vstack((xc1,xc2,xc3))

    for i in xrange(len(ds)):
        _x = x[::int(1/ds[i])]
        _xc = xc[::int(1/ds[i])]

        fig = plt.figure(1, figsize=(6,6))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.plot(_x[:,0], _x[:,1], 'r.', marker='.', ms=20)
        plt.plot(_xc[:,0], _xc[:,1], 'b.', marker='.', ms=20)
        plt.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),
            'k-', lw=1)

        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        if opt['save']:
            plt.savefig('../doc/fig/landscapefig%01d.pdf'%i, bbox_inches='tight')

# #def plot_golfcourse():
# t = np.arange(-10, 10, 0.1)
# x = np.sin()

plot_clusters()