import sys
import os
import numpy as np
import itertools

with open('model_selection_script.py','r') as f:
    experiment_list = f.readlines()
for fno, typeexp in enumerate(experiment_list):
    fname = 'runfile_'+str(fno)+'.sh'
    cmd = 'sbatch '+fname
    with open('runfile_'+str(fno)+'.sh', 'w') as model:
        model.write('#!/bin/bash\n')
        model.write('#SBATCH -J ' + typeexp + '_season_selection\n')
        model.write('#SBATCH -A eecs\n')
        model.write('#SBATCH -p share\n')
        model.write('#SBATCH -o runfile_logs/' + typeexp+'_season_selection\n')
        model.write('#SBATCH -e runfile_logs/' + typeexp+'_season_selection\n')
        model.write('#SBATCH --gres=gpu:1\n')
        model.write('#SBATCH -t 7-0\n')
        model.write('#SBATCH --mem=40G\n')
        print(typeexp.strip())
        model.write(typeexp.strip())

    os.popen(cmd).read()
for fno, _ in enumerate(experiment_list):
    os.remove('runfile_'+str(fno)+'.sh')


