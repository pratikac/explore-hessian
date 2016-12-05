import os, sys, subprocess

import argparse
parser = argparse.ArgumentParser(description='Local entropy jobber')
parser.add_argument('-m', '--model',    help='mnistfc | mnistconv | cifarconv', \
                    type=str, required=True)
parser.add_argument('-n', '--num_jobs',     help='num jobs',    type=int, default = 1)
parser.add_argument('-j', '--max_jobs',     help='max jobs',    type=int, default = 1)
parser.add_argument('-s', '--seed',         help='start seed',  type=int, default = 42)
parser.add_argument('-g', '--gpu',          help='start gpu',   type=int, default = 0)

parser.add_argument('--max_epochs',         help='max epochs',  type=int, default = -1)
parser.add_argument('--LR',                 help='LR',          type=float, default = 1e-3)
parser.add_argument('--LRstep',             help='LRstep',      type=int, default = 10)
parser.add_argument('--LRratio',            help='LRratio',     type=float, default = 0.2)
parser.add_argument('--optim',              help='optim',       type=str, default = 'adam')
parser.add_argument('--L2',                 help='L2',          type=float, default = 0)
parser.add_argument('--langevin',           help='Langevin',    type=int, default = 0)
parser.add_argument('--langevin_noise',     help='Langevin noise',    type=float, default = 1e-5)
parser.add_argument('--gamma',              help='Langevin gamma',    type=float, default = 1)
parser.add_argument('--rho',                help='Langevin rho',    type=float, default = 0)
parser.add_argument('-d', '--dropout',      help='dropout',     type=float, default = 0.5)

parser.add_argument('-l', '--log',          help='log',         action='store_true')
parser.add_argument('-r', '--run',          help='run',  action='store_true')
opt = vars(parser.parse_args())

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def run_cmds(cmds, max_cmds):
    for cs in list(chunks(cmds, max_cmds)):
        ps = []
        try:
            for c in cs:
                p = subprocess.Popen(c, shell=True)
                ps.append(p)

            for p in ps:
                p.wait()

        except KeyboardInterrupt:
            print 'Killling everything'
            for p in ps:
                p.kill()
            sys.exit()

def setup_defaults():
    if opt['model'].find('mnist') > -1:
        if opt['max_epochs'] < 0:
            opt['max_epochs'] = 50
        opt['max_jobs'] = 4
    elif opt['model'].find('cifar') > -1:
        if opt['max_epochs'] < 0:
            opt['max_epochs'] = 300
        opt['max_jobs'] = 2
    else:
        assert False, 'Unknown model: ' + opt['model']
    
    opt['batch_size'] = 256
    if opt['model'] == 'cifarresnet' or opt['model'] == 'cifarwideresnet':
        opt['batch_size'] = 128
        opt['LRD'] = 1e-5

def build_cmds():
    cmds = []
    c = ('th train.lua -f -m %s -b %d --max_epochs %d --dropout %f --LR %f --LRstep %d --LRratio %f --L2 %f --langevin %d --gamma %f --rho %f --langevin_noise %f  --optim %s ')%\
            (opt['model'], opt['batch_size'], opt['max_epochs'], opt['dropout'],
            opt['LR'], opt['LRstep'], opt['LRratio'], opt['L2'], opt['langevin'], opt['gamma'], opt['rho'], opt['langevin_noise'], opt['optim'])
    if opt['log']:
        c = c + '-l '

    gs = [1,2]
    for i in xrange(opt['num_jobs']):
        if opt['gpu'] > 0:
            _c = c + ('-g %d ')%(opt['gpu'])
        else:
            _c = c + ('-g %d ')%(gs[i%len(gs)])
        _c = _c + ('-s %d ')%(opt['seed'] + i)
        cmds.append(_c)
    return cmds

def main():
    setup_defaults()

    if not opt['run']:
        print '\n'.join(build_cmds())
        sys.exit()
    else:
        run_cmds(build_cmds(), opt['max_jobs'])

if __name__=='__main__':
    main()
