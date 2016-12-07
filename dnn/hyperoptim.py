import os, sys, subprocess, json, argparse
from itertools import product

parser = argparse.ArgumentParser(description='Quick dirty jobber')
parser.add_argument('-c','--command',   help='Main command', type=str, required=True)
parser.add_argument('-p','--params',    help='JSON dict of the hyper-parameters', type=str)
parser.add_argument('-r', '--run',      help='run',  action='store_true')
opt = vars(parser.parse_args())

cmd = opt['command']
params = json.loads(opt['params'])

cmds = []
gs = [1,2]
keys,values = zip(*params.items())
for v in product(*values):
    p = dict(zip(keys,v))
    s = ''
    for k in p:
        s += ' --'+k+' '+str(p[k])

    c = cmd+s
    c = c + (' -f -l -g %d')%(gs[len(cmds)%len(gs)])
    cmds.append(c)
print cmds