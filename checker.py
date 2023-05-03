#!/usr/bin/env python3

import sys
import os
import re

bin_dir = "bin"

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

prog_name = ''
if(version == 'seq'):
    prog_name = 'release-transformer-' + version
else:
    prog_name = 'debug-transformer-' + version + '-cube'

prog = os.path.join(bin_dir, prog_name)

# set number of workers
# if machine == '-ghc':
#     workers = [4, 8] if version == 'tmp' else [1, 4]
# elif machine == '-psc':
#     workers = [16, 128] if version == 'tmp' else [16,121]
workers = [1,2,4,8] if version == 'tmp' else [1]

os.system('mkdir -p logs')
os.system('rm -rf logs/*')

for worker in workers:
    print(f'--- running {prog_name} on {worker} workers ---')
    output_file = f'logs/{prog_name}_{worker}.txt'
    log_file = f'logs/{prog_name}_{worker}.log'
    cmd = f'mpirun -np {worker} {prog} -o {output_file} > {log_file}' if version == "tmp" \
        else f'./{prog} > {log_file}'
    ret = os.system(cmd)
    assert ret == 0, 'ERROR -- nbody exited with errors'

print("Execution finished")