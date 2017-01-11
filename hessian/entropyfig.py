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

if opt['save']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

plt.ion()
sns.set_style('white')

npr.seed(42)
n = 1000
fontsize = 24

plt.rc('font', size=fontsize)
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)
plt.rc('figure', titlesize=fontsize)

def create_landscape():
    beta = 5e-4
    f0 = 100
    f = -np.abs(np.sqrt(f0)*npr.random(n))

    f[200:400] = -80
    f[800:820] = -200
    f[npr.random(n) < 0.1] -= -100

    s = 10
    f = beta*convolve(f, gaussian(10*s, s), mode='same')
    return f

def smooth(rho, gamma):
    f = create_landscape()

    gibbs = lambda f: np.exp(-f)/np.sum(np.exp(-f))
    g = lambda f,x: np.array([f[i] + gamma/2.*(x-i)**2 for i in xrange(n)])

    S = lambda f,x: -np.sum(gibbs(g(f,x))*(-g(f,x) - logsumexp(-g(f,x))))
    F = lambda f,x: logsumexp(-g(f,x))

    Fi = np.array([F(f,i) for i in xrange(n)])
    scope = rho*f - Fi
    return f, scope, Fi

f, s,_ = smooth(0, 1e-3)
f = f + 1
s = s - s.min()

f2, s2,_ = smooth(0, 5e-5)
s2 = s2 - s2.min()

fig = plt.figure(1, figsize=(9,8))
plt.clf()
ax = fig.add_subplot(111)
plt.plot(f, 'k-', label=r'$\mathrm{Original\ landscape}$', lw=1)
plt.plot(s, 'indianred', label=r'$\mathrm{Negative\ local\ entropy}:\ \gamma = 0.001$', lw=2)
plt.plot(s2, 'indianred', ls='--', label=r'$\mathrm{Negative\ local\ entropy}:\ \gamma = 0.00005$', lw=2)
plt.legend(loc=2)
#plt.title(r'$\mathrm{Local\ entropy\ energy\ landscape}$')
plt.xticks([], [])
plt.plot([576.73,340.52,811.8], [0.938, 0.005, 0.228], 'o', c='indianred', markersize=10)

ax.text(575.73, 0.82, r'$x_{\mathrm{candidate}}$', fontsize=fontsize,
        verticalalignment='center')
ax.text(338, -0.1, r'$x_{\mathrm{robust}}$', fontsize=fontsize,
        verticalalignment='center')
ax.text(835.8, 0.2, r'$x_{\mathrm{non}\textrm{-}\mathrm{robust}}$', fontsize=fontsize,
        verticalalignment='center')

if opt['save']:
    plt.savefig('../doc/fig/entropyfig.pdf', bbox_inches='tight')