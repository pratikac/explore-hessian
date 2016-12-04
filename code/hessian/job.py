import os, sys, time, atexit, subprocess

import argparse
parser = argparse.ArgumentParser(description='Hessian jobber')
parser.add_argument('-d', '--dataset',    help='mnist | mnistconv | cifar | shakesphere',   type=str, required=True)
parser.add_argument('-n', '--num_jobs',    help='num jobs',   type=int, default = 1)
parser.add_argument('-j', '--max_jobs',    help='max jobs',   type=int, default = 1)
args = vars(parser.parse_args())

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def run_cmds(cmds, max_cmds = args['max_jobs']):
    for cs in list(chunks(cmds, max_cmds)):
        ps = []
        for c in cs:
            p = subprocess.Popen(c, shell=True)
            ps.append(p)

        for p in ps:
            p.wait()

def bot_timestamp_now():
    return int(time.time()*1e6)

cmds = []
for m in range(args['num_jobs']):
    s = 42+m
    fp = 'log/%s_%04d_%d'%(args['dataset'], s, bot_timestamp_now())
    c = 'python %s.py -s %d -o %s'%(args['dataset'], s, fp)
    cmds.append(c)
print(cmds)

try:
    run_cmds(cmds)
except KeyboardInterrupt:
    print 'Killling everything'
    sys.exit()
