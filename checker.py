#!/usr/bin/env python3

import sys
import os
import re

usage_guild = "Usage: ./checker.py seq/tmp -ghc/-psc"
if len(sys.argv) != 3:
    print(usage_guild)
    exit(-1)

version = sys.argv[1]
machine = sys.argv[2]

valid = True
valid = valid and (version == 'seq' or version == 'tmp')

if not valid:
    print(usage_guild)
    exit(-1)

prog = 'transformer-release-' + version

# set number of workers
# if machine == '-ghc':
#     workers = [4, 8] if version == 'tmp' else [1, 4]
# elif machine == '-psc':
#     workers = [16, 128] if version == 'tmp' else [16,121]
worker = 4

os.system('mkdir -p logs')
os.system('rm -rf logs/*')
print(f'--- running {prog} on {worker} workers ---')
output_file = f'logs/{prog}.txt'
log_file = f'logs/{prog}.log'
cmd = f'mpirun -np {worker} {prog} -o {output_file} > {log_file}' if version == "tmp" \
    else f'./{prog} > {log_file}'
ret = os.system(cmd)
assert ret == 0, 'ERROR -- nbody exited with errors'

print("Execution finished")