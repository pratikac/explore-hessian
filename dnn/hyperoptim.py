import os, sys, subprocess, json, argparse

parser = argparse.ArgumentParser(description='Quick dirty jobber')
parser.add_argument('-c','--command',   help='Main command', type=str, required=True)
parser.add_argument('-p','--params',    help='JSON dict of the hyper-parameters', type=str)
parser.add_argument('-r', '--run',      help='run',  action='store_true')
opt = vars(parser.parse_args())

c = opt['command']
p = json.loads(opt['params'])

cmds = []
for k in p:
    
