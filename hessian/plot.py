import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ggplot import *
import os, sys, glob, pdb, argparse
import cPickle as pickle

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', nargs='*')
parser.add_argument('-t', '--threshold', default=1e-3, type=float)
opt = vars(parser.parse_args())

e = np.zeros((1,))
for i in opt['input']:
    for ii in glob.glob(i):
        tmp = np.load(i)
        if len(tmp.shape) > 1:
            _e = tmp[:,0]
        else:
            _e = tmp
        e = np.hstack((e, np.real(_e)))
d = pd.DataFrame(e, columns=['e'])

plt.figure(1)
plt.clf()
print ggplot(d[d<-opt['threshold']], aes(x='e')) + geom_histogram(bins=200)
plt.title('Negative eigenvalues')
#plt.savefig('neg.png', bbox_inches='tight')

plt.figure(2)
plt.clf()
print ggplot(d[d>opt['threshold']], aes(x='e')) + geom_histogram(bins=200)
plt.title('Positive eigenvalues')
#plt.savefig('pos.png', bbox_inches='tight')

plt.figure(3)
plt.clf()
print ggplot(d[(d>-opt['threshold']) & (d<opt['threshold'])],
        aes(x='e')) + geom_histogram(bins=200)
plt.title('Near zero')
plt.ylim([0,20000])
# plt.savefig('zero.png', bbox_inches='tight')
