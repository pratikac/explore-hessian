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
com = 0.001
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
trainloss = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 1) & (df['batch'] == 0)].loss
valid = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 0) & (df['batch'] == 0)].miss
validloss = lambda df, m: df[(df['epoch'] <= m) & (df['tv'] == 0) & (df['batch'] == 0)].loss

def load(fs, m=50, isrnn = False):
    r = {}
    Ds = []
    for f in fs:
        Ds.append(pd.read_csv(f, sep=None, engine='python'))
    df = pd.concat(Ds, keys=[i for i in xrange(len(fs))])

    r['loss'] = loss(df, m)
    if isrnn:
        r['train'] = trainloss(df, m)
    else:
        r['train'] = train(df, m)
    if isrnn:
        r['valid'] = validloss(df, m)
    else:
        r['valid'] = valid(df, m)

    return r

def plot_lenet():
    r1, r2 = load(sorted(glob.glob('../results/nov_expts/dnn/lenet/*\"langevin\":0*.log')), 100), \
        load(sorted(glob.glob('../results/jan_expts/lenet/*.log')), 5)

    fig = plt.figure(1, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'LeNet: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i]) for i in range(5)])
    v2 = np.array([smooth(r2['valid'].ix[i]) for i in range(5)])
    sns.tsplot(v1,
            condition=r'Adam', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(20,120,20),
            condition=r'Entropy-SGD', rasterized=True, color='r')
    sns.tsplot(v2, time=np.arange(20,120,20),
            rasterized=True, color='r',
            err_style="ci_bars", interpolate=False)

    plt.ylim([0.45, 0.75])
    plt.xlim([20,100])
    yticks = [0.45,.55,.65,.75]
    plt.yticks(yticks, [str(y) for y in yticks])
    xticks = [20, 40, 60, 80, 100]
    plt.xticks(xticks, [str(x) for x in xticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')

    plt.plot([98.89], [0.512], 'o', c='k', markersize=10)
    plt.plot([80], [0.502], 'o', c='r', markersize=10)
    plt.plot(range(100), 0.50*np.ones(100), 'r--', lw=1)

    ax.text(90, 0.54, r'$0.51\%$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(75, 0.48, r'$0.50\%$', fontsize=fsz,
            verticalalignment='center', color='r')

    plt.grid('on')

    if opt['save']:
        plt.savefig('../doc/fig/lenet_valid.pdf', bbox_inches='tight')

def plot_mnistfc():
    r1, r2 = load(sorted(glob.glob('../results/nov_expts/dnn/mnistfc/*\"langevin\":0*.log')), 100), \
            load(sorted(glob.glob('../results/jan_expts/mnistfc/*.log')), 10)

    fig = plt.figure(2, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'mnistfc: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i]) for i in range(5)])
    v2 = np.array([smooth(r2['valid'].ix[i]) for i in range(5)])
    sns.tsplot(v1,
            condition=r'Adam', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(20,220,20),
            condition=r'Entropy-SGD', rasterized=True, color='r')
    sns.tsplot(v2, time=np.arange(20,220,20),
        rasterized=True, color='r',
        err_style="ci_bars", interpolate=False)
    plt.grid('on')

    plt.ylim([1.2, 2])
    plt.xlim([20,120])
    yticks = [1.2, 1.4, 1.6, 1.8, 2]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')

    plt.plot([69], [1.39], 'o', c='k', markersize=10)
    plt.plot([120], [1.37], 'o', c='r', markersize=10)
    plt.plot(range(120), 1.37*np.ones(120), 'r--', lw=1)

    ax.text(66, 1.35, r'$1.39\%$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(115, 1.42, r'$1.37\%$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/mnistfc_valid.pdf', bbox_inches='tight')

def plot_allcnn():
    r1, r2 = load(sorted(glob.glob('../results/jan_expts/allcnn10/toplotbn/*\"L\":0*.log')), 200), \
            load(sorted(glob.glob('../results/jan_expts/allcnn10/toplotbn/*\"L\":20*.log')), 10)

    # validation error
    fig = plt.figure(3, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'All-CNN-BN: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i]) for i in range(2)])
    v2 = np.array([smooth(r2['valid'].ix[i]) for i in range(2)])
    sns.tsplot(v1,
            condition=r'SGD', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(20,220,20),
            condition=r'Entropy-SGD', rasterized=True, color='r')
    sns.tsplot(v2, time = np.arange(20,220,20),
            rasterized=True, color='r', err_style='ci_bars',interpolate=False)
    plt.grid('on')

    plt.ylim([5, 20])
    plt.xlim([0, 200])
    yticks = [5, 10, 15, 20]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')

    plt.plot([194], [7.71], 'o', c='k', markersize=10)
    plt.plot([160], [7.81], 'o', c='r', markersize=10)
    plt.plot(range(200), 7.81*np.ones(200), 'r--', lw=1)

    ax.text(180, 9.0, r'$7.71\%$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(115, 6.8, r'$7.81\%$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/allcnn_valid.pdf', bbox_inches='tight')

    # training loss
    fig = plt.figure(4, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'All-CNN-BN: Training loss')

    v1 = [smooth(r1['loss'].ix[i], 1e-1) for i in range(2)]
    v2 = [smooth(r2['loss'].ix[i], 1e-1) for i in range(2)]
    sns.tsplot(v1, time = np.arange(0,200),
            condition=r'SGD', rasterized=True, color='k')
    sns.tsplot(v2, time = np.arange(20,220,20),
            condition=r'Entropy-SGD', rasterized=True, color='r')
    sns.tsplot(v2, time = np.arange(20,220,20),
            rasterized=True, color='r', err_style='ci_bars',interpolate=False)
    plt.grid('on')

    plt.ylim([0,0.6])
    plt.xlim([0, 200])
    xticks = [0,50,100,150,200]
    plt.xticks(xticks, [r'$0$', r'$50$', r'$100$', r'$150$', r'$200$'])
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'Cross-Entropy Loss')

    plt.plot([193], [0.0353], 'o', c='k', markersize=10)
    plt.plot([200], [0.0336], 'o', c='r', markersize=10)
    plt.plot(range(200), 0.0336*np.ones(200), 'r--', lw=1)

    ax.text(190, 0.1, r'$0.0353$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(100, 0.05, r'$0.0336$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/allcnn_loss.pdf', bbox_inches='tight')

def plot_charlstm():
    r1, r2 = load(sorted(glob.glob('../results/jan_expts/lstm/char-rnn/*\"L\":0*.log')), 50, True), \
            load(sorted(glob.glob('../results/jan_expts/lstm/char-rnn/*\"L\":5*.log')), 5, True)

    fig = plt.figure(2, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'char-LSTM: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i]) for i in range(4)])
    v2 = np.array([smooth(r2['valid'].ix[i]) for i in range(4)])
    sns.tsplot(v1,
            condition=r'Adam', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(5,30,5),
            condition=r'Entropy-Adam', rasterized=True, color='r')
    sns.tsplot(v2, time=np.arange(5,30,5),
            rasterized=True, color='r', err_style='ci_bars',interpolate=False)
    plt.grid('on')

    plt.ylim([1.2, 1.35])
    plt.xlim([0,50])
    yticks = [1.2, 1.25, 1.30, 1.35]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'Perplexity')

    plt.plot([25], [1.213], 'o', c='r', markersize=10)
    plt.plot([40], [1.224], 'o', c='k', markersize=10)
    plt.plot(range(50), 1.213*np.ones(50), 'r--', lw=1)

    ax.text(37, 1.25, r'$(\mathrm{Test:} 1.226)$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(38, 1.24, r'$1.224$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(22, 1.23, r'$(\mathrm{Test:} 1.217)$', fontsize=fsz,
            verticalalignment='center', color='r')
    ax.text(23, 1.22, r'$1.213$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/charlstm_valid.pdf', bbox_inches='tight')


def plot_ptb():
    r1, r2 = load(sorted(glob.glob('../results/jan_expts/lstm/ptb/*\"L\":0*.log')), 55, True), \
            load(sorted(glob.glob('../results/jan_expts/lstm/ptb/*\"L\":5*.log')), 5, True)

    fig = plt.figure(2, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)
    #plt.title(r'PTB-LSTM: Validation error')

    v1 = np.array([smooth(r1['valid'].ix[i],0.001) for i in range(2)])
    v2 = np.array([smooth(r2['valid'].ix[i],0.001) for i in range(1)])
    sns.tsplot(v1,
            condition=r'SGD', rasterized=True, color='k')
    sns.tsplot(v2, time=np.arange(5,30,5),
            condition=r'Entropy-SGD', rasterized=True, color='r')
    sns.tsplot(v2, time=np.arange(5,30,5),
            rasterized=True, color='r', err_style='ci_bars',interpolate=False)
    plt.grid('on')

    plt.ylim([75, 115])
    plt.xlim([0,54])
    yticks = [75, 85, 95, 105, 115]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'Perplexity')

    plt.plot([25], [80.13], 'o', c='r', markersize=10)
    plt.plot([54], [81.44], 'o', c='k', markersize=10)
    plt.plot(range(54), 80.13*np.ones(54), 'r--', lw=1)

    ax.text(45, 87, r'$(\mathrm{Test:} 78.6)$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(48, 84, r'$81.43$', fontsize=fsz,
            verticalalignment='center', color='k')
    ax.text(20, 85, r'$(\mathrm{Test:} 77.656)$', fontsize=fsz,
            verticalalignment='center', color='r')
    ax.text(22, 78, r'$80.116$', fontsize=fsz,
            verticalalignment='center', color='r')

    if opt['save']:
        plt.savefig('../doc/fig/ptblstm_valid.pdf', bbox_inches='tight')

# plot_lenet()
# plot_mnistfc()
# plot_allcnn()
# plot_charlstm()
plot_ptb()