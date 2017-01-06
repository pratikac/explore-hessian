import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import torchfile as thf
import seaborn as sns

plt.ion()
sns.set_style('ticks')
sns.set_color_codes()

fsz = 24
com = .01
loss_subsample = 30

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
opt = vars(parser.parse_args())

import matplotlib
if opt['save']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)

smooth = lambda x,c=com: pd.ewma(x, c)
loss = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 1) & (df['batch'] == 2)].loss
train = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 1) & (df['batch'] == 0)].miss
valid = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 0) & (df['batch'] == 0)].miss

def load(fs, m=50):
    r = {}
    Ds = []
    for f in fs:
        Ds.append(pd.read_csv(f, sep=None, engine='python'))
    df = pd.concat(Ds, keys=[i for i in xrange(len(Ds))])

    r['loss'] = loss(df, m)
    r['train'] = train(df, m)
    r['valid'] = valid(df, m)

    return r

# def plot_lenet():
#     r = load(sorted(glob.glob('../results/nov_expts/dnn/lenet/*\"langevin\":0*.log')), \
#         sorted(glob.glob('../results/dnn/nov_expts/lenet/*\"langevin\":20*.log')), 100)

#     fig = plt.figure(1, figsize=(8,7))
#     plt.clf()
#     ax = fig.add_subplot(111)
#     plt.title(r'LeNet: Validation error')

#     sns.tsplot(np.array([smooth(r['valid'][0].ix[i]) for i in range(5)]),
#             condition=r'Adam', rasterized=True, color='k')
#     sns.tsplot(np.array([smooth(r['valid'][1].ix[i]) for i in range(4)]),
#             condition=r'entropy-SGD', rasterized=True, color='r')

#     plt.ylim([0.4, 0.8])
#     plt.xlim([0,100])
#     yticks = [0.45,.55,.65,.75]
#     plt.yticks(yticks, [str(y) for y in yticks])
#     plt.xlabel(r'Epochs')
#     plt.ylabel(r'\% Error')

#     plt.plot([98.89], [0.512], 'o', c='k', markersize=10)
#     plt.plot([81.7], [0.48], 'o', c='r', markersize=10)

#     ax.text(90, 0.54, r'$0.51\%$', fontsize=fsz,
#             verticalalignment='center', color='k')
#     ax.text(75, 0.46, r'$0.48\%$', fontsize=fsz,
#             verticalalignment='center', color='r')

#     plt.grid('on')

#     if opt['save']:
#         plt.savefig('../../doc/fig/lenet_valid.pdf', bbox_inches='tight')

# def plot_mnistfc():
#     r = load(sorted(glob.glob('../results/nov_expts/dnn/mnistfc/*\"langevin\":0*.log')), \
#         sorted(glob.glob('../results/dnn/nov_expts/mnistfc/*\"langevin\":5*.log')), 100)

#     fig = plt.figure(2, figsize=(8,7))
#     plt.clf()
#     ax = fig.add_subplot(111)
#     plt.title(r'mnistfc: Validation error')

#     v1 = np.array([smooth(r['valid'][0].ix[i]) for i in range(5)])
#     v2 = np.array([smooth(r['valid'][1].ix[i]) for i in range(5)])
#     sns.tsplot(v1,
#             condition=r'Adam', rasterized=True, color='k')
#     sns.tsplot(v2,
#             condition=r'entropy-SGD', rasterized=True, color='r')
#     plt.grid('on')

#     plt.ylim([1, 2.5])
#     plt.xlim([0,100])
#     yticks = [1, 1.5, 2, 2.5]
#     plt.yticks(yticks, [str(y) for y in yticks])
#     plt.xlabel(r'Epochs')
#     plt.ylabel(r'\% Error')

#     plt.plot([69], [1.39], 'o', c='k', markersize=10)
#     plt.plot([24], [1.39], 'o', c='r', markersize=10)

#     ax.text(66, 1.3, r'$1.39\%$', fontsize=fsz,
#             verticalalignment='center', color='k')
#     ax.text(20, 1.3, r'$1.39\%$', fontsize=fsz,
#             verticalalignment='center', color='r')

#     if opt['save']:
#         plt.savefig('../../doc/fig/mnistfc_valid.pdf', bbox_inches='tight')

def plot_allcnn():
    r1, r2 = load(sorted(glob.glob('../results/nov_expts/dnn/allcnn/original/*.log')), 200), \
            load(sorted(glob.glob('../results/jan_expts/allcnn10/toplot/*.log')), 10)

    # validation error
    fig = plt.figure(3, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    plt.title(r'All-CNN-BN: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i]) for i in range(5)])
    v2 = np.array([smooth(r2['valid'].ix[i]) for i in range(3)])
    sns.tsplot(v1,
            condition=r'SGD', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(20,220,20),
            condition=r'entropy-SGD', rasterized=True, color='r')
    plt.grid('on')

    plt.ylim([5, 25])
    plt.xlim([0, 200])
    yticks = [5, 10, 15, 20, 25]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'\#backprops/\#batches')
    plt.ylabel(r'\% error')

    plt.plot([190], [8.30], 'o', c='k', markersize=10)
    plt.plot([200], [8.28], 'o', c='r', markersize=10)
    plt.plot(range(200), 8.28*np.ones(200), 'r--', lw=1)

    ax.text(150, 10.0, r'$8.30\%$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(150, 7.0, r'$8.28\%$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/allcnn_valid.pdf', bbox_inches='tight')

    # training loss
    fig = plt.figure(4, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    plt.title(r'All-CNN-BN: Training loss')

    v1 = [smooth(r1['loss'].ix[i], 1e-1) for i in range(5)]
    v2 = [smooth(r2['loss'].ix[i], 1e-1) for i in range(3)]
    sns.tsplot(v1, time = np.arange(0,200),
            condition=r'SGD', rasterized=True, color='k')
    sns.tsplot(v2, time = np.arange(20,220,20),
            condition=r'entropy-SGD', rasterized=True, color='r')
    plt.grid('on')

    plt.ylim([0,0.5])
    plt.xlim([0, 200])
    xticks = [0,50,100,150,200]
    plt.xticks(xticks, [r'$0$', r'$50$', r'$100$', r'$150$', r'$200$'])
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'\#backprops/\#batches')
    plt.ylabel(r'cross-entropy loss')

    plt.plot([200], [0.03465], 'o', c='k', markersize=10)
    plt.plot([200], [0.02681], 'o', c='r', markersize=10)
    plt.plot(range(200), 0.02681*np.ones(200), 'r--', lw=1)

    ax.text(180, 0.06, r'$0.035$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(100, 0.04, r'$0.027$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/allcnn_loss.pdf', bbox_inches='tight')

# plot_lenet()
# plot_mnistfc()
# plot_allcnn()