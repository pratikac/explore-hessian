import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import torchfile as thf
from scipy.stats import norm, gumbel_r, cauchy
from ggplot import *
import matplotlib

plt.ion()
fontsize = 30

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
opt = vars(parser.parse_args())

if opt['save']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

plt.rc('font', size=fontsize)
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)
plt.rc('figure', titlesize=fontsize)

def load_data(n):
    e = []
    if n == 'lenet':
        for i in glob.glob('../../results/hessian/lenet/*.npy'):
            e.append(np.real(np.load(i)))
    elif n == 'mnistfc':
        for i in glob.glob('../../results/hessian/mnistfc2/*.npy'):
            e.append(np.load(i)[:,0]*128)
    elif n == 'charlstm':
        for i in glob.glob('../../results/hessian/char_lstm/*.npy'):
            e.append(np.load(i)[:,0]*32)
    elif n == 'cifarfc':
        for i in glob.glob('../../results/hessian/cifarfc/*.npy'):
            e.append(np.load(i)[:,0])
    elif n == 'lenet_nonconvg':
        for i in glob.glob('../../results/hessian/lenet_nonconvg/*.npy'):
            e.append(np.real(np.load(i)))
    elif n == 'cifarconv':
        for i in glob.glob('../../results/hessian/allcnn_fisher/*.optim_state.t7'):
            t = thf.load(i)
            e.append( (t['v'] - t['m']**2)*256)
    else:
        raise AssertionError('data: ', opt['data'])

    return pd.DataFrame(np.hstack(e), columns=['e'])

def plot_lenet():
    d = load_data('lenet')
    t = 1e-4
    dm, dc, dp = (d[d<-t].dropna(), \
                d[(d>-t) & (d<t)].dropna(), \
                d[d>t].dropna())

    plt.figure(1)
    plt.clf()
    p = ggplot(dp, aes(x='e')) + \
        xlab(' ') + \
        scale_color_manual('black') + \
        geom_histogram(binwidth=1) + \
        scale_x_continuous(breaks=[0, 10, 20, 30, 40]) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=10)
    plt.title(r'Long positive tail')
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.savefig('../../doc/fig/lenet_dp.pdf', bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    p = ggplot(dm, aes(x='e')) + \
        xlab(' ') + \
        scale_color_manual('black') + \
        geom_histogram(binwidth=0.025) + \
        theme_bw()
    print p
    plt.xlim([-0.5,0.025])
    plt.yscale('symlog', linthreshy=100)
    plt.title(r'Short negative tail')
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.savefig('../../doc/fig/lenet_dm.pdf', bbox_inches='tight')

    plt.figure(3)
    plt.clf()
    p = ggplot(dc, aes(x='e')) + \
        xlab(' ') + \
        scale_color_manual('black') + \
        geom_histogram(bins=20) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=100)
    plt.title(r'small-LeNet: Eigenspectrum of the Hessian at local minimum')
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        xticks = [-1e-4, -5e-5, 0, 5e-5, 1e-4]
        yticks=[0,1e2, 1e3, 1e4]
        plt.xticks(xticks, [r'$-10$', r'$-5$', r'$0$', r'$5$', r'$10\\ (\times 10^{-4})$'])
        plt.yticks(yticks, [r'$0$', r'$10^2$', r'$10^3$', r'$10^4$'])
        plt.savefig('../../doc/fig/lenet_dc.pdf', bbox_inches='tight')

def plot_allcnn():
    d = load_data('cifarconv')

    plt.figure(1,figsize=(8,7))
    t = 1e-4
    plt.clf()
    p = ggplot(d[d>t].dropna(), aes(x='e')) + \
        xlab(' ') + \
        geom_histogram(bins=50) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=1e2)
    plt.title(r'Positive tail')
    plt.ylim([0,1e4])
    plt.xlim([t,2])
    xticks = [1e-3, 0.5, 1.0, 1.5, 2.0]
    yticks = [0, 1e2, 1e3, 1e4]
    plt.yticks(yticks, [r'$0$', r'$10^2$', r'$10^3$', r'$10^4$'])
    plt.xticks(xticks, [r'$0.02$', r'$0.5$', r'$1.0$', r'$1.5$', r'$2.0$'])
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.savefig('../../doc/fig/allcnn_dp.pdf', bbox_inches='tight')

    plt.figure(2,figsize=(8,7))
    t = 1e-6
    plt.clf()
    p = ggplot(d[d<-t].dropna(), aes(x='e')) + \
        xlab(' ') + \
        geom_histogram(bins=25) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=10)
    plt.title(r'Negative tail')
    plt.ylim([0,1e2])
    plt.xlim([-3e-3,t])
    xticks = [-3e-3, -2e-3, -1e-3, t]
    plt.xticks(xticks, [r'$-300$', r'$-200$', r'$-100$', r'$0\\ (\times 10^{-5})$'])
    yticks = [0, 10, 100]
    plt.yticks(yticks, [r'$0$', r'$10$', r'$10^2$'])
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.savefig('../../doc/fig/allcnn_dm.pdf', bbox_inches='tight')

def plot_mnistfc():
    d = load_data('mnistfc')
    plt.figure(1, figsize=(8,7))
    plt.clf()
    p = ggplot(d, aes(x='e')) + \
        xlab(' ') + \
        geom_histogram(binwidth=0.1) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=10)
    plt.xscale('symlog', linthreshy=1)
    plt.ylim([0, 1e3])
    plt.xlim([-0.25, 50])
    xticks = [-.25, 1, 10, 50]
    plt.xticks(xticks, [r'$-0.25$', r'$1$', r'$10$', r'$50$'])
    yticks = [0, 10, 100]
    plt.yticks(yticks, [r'$0$', r'$10$', r'$10^2$'])
    plt.title(r'small-mnistfc: Hessian eigenspectrum at local minimum')
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.savefig('../../doc/fig/mnistfc_hessian.pdf', bbox_inches='tight')

def plot_charlstm():
    d = load_data('charlstm')
    plt.figure(1, figsize=(8,7))
    plt.clf()
    p = ggplot(d, aes(x='e')) + \
        xlab(' ') + \
        geom_histogram(bins=25) + \
        theme_bw()
    print p
    plt.yscale('symlog', linthreshy=25)
    plt.ylim([0, 1e2])
    plt.xlim([-1e-3, .25])

    xticks = [-1e-3, 5e-2, 15e-2, 25e-2]
    yticks = [0, 10, 50]
    plt.yticks(yticks, [r'$0$', r'$10$', r'$50$'])
    plt.title(r'char-lstm: Hessian eigenspectrum at local minimum')
    plt.xlabel(r'Eigenvalues')
    plt.ylabel(r'Frequency')
    if opt['save']:
        plt.xticks(xticks, [r'$-10$', r'$5$', r'$15$', r'$25\\ (\times 10^{-2})$'])
        plt.savefig('../../doc/fig/charlstm_hessian.pdf', bbox_inches='tight')

d = load_data('cifarfc')
plt.figure(1, figsize=(8,7))
plt.clf()
p = ggplot(d, aes(x='e')) + \
    xlab(' ') + \
    geom_histogram(bins=50) + \
    theme_bw()
print p
plt.yscale('symlog', linthreshy=25)
plt.ylim([0, 1e2])
plt.xlim([-10, 4e3])

xticks = [-1e-3, 5e-2, 15e-2, 25e-2]
yticks = [0, 10, 50]
plt.yticks(yticks, [r'$0$', r'$10$', r'$50$'])
plt.title(r'small-LeNet: Unconverged Hessian')
plt.xlabel(r'Eigenvalues')
plt.ylabel(r'Frequency')
if opt['save']:
    plt.xticks(xticks, [r'$-10$', r'$5$', r'$15$', r'$25\\ (\times 10^{-2})$'])
    plt.savefig('../../doc/fig/cifarfc_hessian.pdf', bbox_inches='tight')


# from fitter import Fitter
# d1 = load_data('lenet')
# d2 = load_data('mnistfc')
# d = pd.concat((d1, d2))
# dp = d[d>0].dropna()
# w = dp['e'].as_matrix()
# np.random.shuffle(w)
# w1 = w[:10000]*1000
# dist = ['foldnorm','skewnorm','halfnorm']
# #f = Fitter(w1, distributions=dist, timeout=50)
# f = Fitter(w1, distributions=dist, timeout=50)
# f.fit()


# plot_lenet()
# plot_allcnn()
# plot_mnistfc()
# plot_charlstm()