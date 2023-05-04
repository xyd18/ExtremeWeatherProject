#!/usr/bin/env python3

import sys
import os
import re

bin_dir = "bin"

usage_guild = "Usage: ./checker.py seq/tmp/pp/pp_tmp -ghc/-psc debug/release"
if len(sys.argv) != 4:
    print(usage_guild)
    exit(-1)

version = sys.argv[1]
machine = sys.argv[2]
production = sys.argv[3]

valid = True
valid = valid and (version == 'seq' or version == 'tmp' or version == 'pp' or version == 'pp_tmp') \
    and (machine == '-ghc' or machine == '-psc') \
    and (production == 'debug' or production == 'release')

if not valid:
    print(usage_guild)
    exit(-1)

prog_name = ''
if(version == 'seq'):
    prog_name = f'{production}-transformer-cube'
elif(version == 'tmp'):
    prog_name = f'{production}-transformer-tmp-cube'
elif(version == 'pp' or version == 'pp_tmp'):
    prog_name = f'{production}-ViT'

prog = os.path.join(bin_dir, prog_name)

# set number of workers
# if machine == '-ghc':
#     workers = [4, 8] if version == 'tmp' else [1, 4]
# elif machine == '-psc':
#     workers = [16, 128] if version == 'tmp' else [16,121]
workers = [1]
if version == 'tmp':
    workers = [1,2,4,8]
elif version == 'pp' or version == 'pp_tmp':
    workers = [4] #[1,2,4]

os.system('mkdir -p logs')
os.system('rm -rf logs/*')

for worker in workers:
    print(f'--- running {prog_name} | {version} on {worker} workers ---')
    output_file = f'logs/{prog_name}_{version}_{worker}.txt'
    log_file = f'logs/{prog_name}_{version}_{worker}.log'
    cmd = f'mpirun -np {worker} {prog} -o {output_file} > {log_file}' if version == "tmp" \
        else f'./{prog} > {log_file}'
    if(version == 'pp'):
        cmd = f'mpirun -np {worker} {prog} -pip -out Vit_output.bin > {log_file}'
    elif(version == 'pp_tmp'):
        cmd = f'mpirun -np {worker} {prog} -pip -tmp -out Vit_output.bin -micro 32 > {log_file}'
    ret = os.system(cmd)
    assert ret == 0, 'ERROR -- nbody exited with errors'

print("Execution finished")