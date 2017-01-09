import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns

import os, sys, glob, pdb, re, json
sns.set()

whitelist = set(['seed','model','LR','noise', \
            'gamma','L','scoping','LRstep','rho','LRD', \
            'langevin', 'langevin_noise','lr','input'])

#colors = ["blue", "darkseagreen", "indianred", "purple", "sepia", "black"]
#colors = sns.color_palette("muted")
colors = sns.color_palette("husl", 6)

plt.ion()

def get_params(s):
    t = s[s.rfind('/')+6:s.find('_opt_')]
    _s = s[s.find('_opt_')+5:-4]
    r = json.loads(_s)
    r = {k: v for k,v in r.items() if k in whitelist}
    r['t'] = t
    return r

def load(dir, expr='*'):
    fs = sorted(glob.glob(dir + '/' + expr + '.log'))
    for i in xrange(len(fs)):
        print i, get_params(fs[i])

    D = [pd.read_csv(f, sep=None, engine='python') for f in fs]
    df = pd.concat(D, keys=[i for i in xrange(len(D))])

    return [get_params(f) for f in fs], df

def condition(fs, df, conditions_list):
    idxs = [[] for cli in conditions_list]

    for di in xrange(len(fs)):
        for cli in xrange(len(conditions_list)):
            if conditions_list[cli](fs[di]):
                idxs[cli].append(di)
    for idx in idxs:
        for i in idx:
            print i, fs[i]
        print ''
    return idxs

def plot_idx(fs, df,
    condition_list = None,
    param_dict = None,
    idxs = None, max_epochs = 100, com=5):

    if not idxs and not condition_list and not param_dict:
        return
    if param_dict:
        return
    if condition_list and not idxs:
        idxs = condition(fs,df, condition_list)

    def helper(d, title=''):
        j = 0
        plt.title(title)
        for idx in idxs:
            res = []
            time = None
            for i in idx:
                y = pd.ewma(d[i], com)
                print i, fs[i]
                res.append(y)
                print i, y.min()
            sns.tsplot(np.array(res), color=colors[j % len(colors)],
                    condition='D'+str(idx)+' '+fs[i]['t'])
            j += 1

    # plt.figure(1)
    # plt.clf()
    # plt.title('Training loss')
    # helper(df[(df['epoch'] <= max_epochs) & (df['tv'] == 1)].loss)
    # plt.xlim([0,max_epochs])

    # plt.figure(2)
    # plt.clf()
    # helper(df[(df['epoch'] <= max_epochs) & (df['tv'] == 1) & (df['batch'] == 0)].miss,
    #     'Training error')
    # plt.xlim([0,max_epochs])

    # plt.figure(3)
    # plt.clf()
    # helper(df[(df['epoch'] <= max_epochs) & (df['tv'] == 0) & (df['batch'] == 0)].miss,
    #     'Test error')
    # plt.xlim([0,max_epochs])

    # lstm
    plt.figure(3)
    plt.clf()
    helper(df[(df['epoch'] <= max_epochs) & (df['tv'] == 0) & (df['iter'] == 0)].loss,
        'Test loss')
    plt.xlim([0,max_epochs])