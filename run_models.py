import sys
import os
import numpy as np
import itertools

list1 = ['multiplicative_embedding', 'additive_embedding', 'concat_embedding', 'single', 'mtl']
for fno, typeexp in enumerate(list1):
    fname = 'runfile_'+str(fno)+'.sh'
    cmd = 'sbatch '+fname
    with open('runfile_'+str(fno)+'.sh', 'w') as model:
        model.write('#!/bin/bash\n')
        model.write('#SBATCH -J ' + typeexp + '_cherry\n')
        model.write('#SBATCH -A eecs\n')
        model.write('#SBATCH -p share\n')
        model.write('#SBATCH -o runfile_logs/' + typeexp+'_cherry.out\n')
        model.write('#SBATCH -e runfile_logs/' + typeexp+'_cherry.err\n')
        model.write('#SBATCH --gres=gpu:1\n')
        model.write('#SBATCH -t 3-0\n')
        model.write('#SBATCH --mem=40G\n')
        print('python main.py --name cherry --experiment ' + typeexp)
        model.write('python main.py --name cherry --experiment ' + typeexp)

    os.popen(cmd).read()
for fno, _ in enumerate(list1):
    os.remove('runfile_'+str(fno)+'.sh')


