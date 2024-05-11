import pickle
import os
valid_cultivars = ['Zinfandel',
                       'Cabernet Franc',
                       'Concord',
                       'Malbec',
                       'Barbera',
                       'Semillon',
                       'Merlot',
                       'Lemberger',
                       'Chenin Blanc',
                       'Riesling',
                       'Nebbiolo',
                        'Cabernet Sauvignon',
                       'Chardonnay',
                       'Viognier',
                       'Gewurztraminer',
                       'Mourvedre',
                       'Pinot Gris',
                       'Grenache',
                       'Syrah',
                       'Sangiovese',
                       'Sauvignon Blanc']
season_selection_list = [2,5,10,20]
experiment_list = ['single','mtl','concat_embedding']
with open('season_len_map.pkl', 'rb') as f:
    season_len_map = pickle.load(f)
fno = 0
for season_selection in season_selection_list:
    for experiment in experiment_list:
        for cultivar in valid_cultivars:
            if season_selection<season_len_map[cultivar]:
                if experiment=='single':
                    runstr = 'python main.py --experiment '+experiment+' --name \''+cultivar+"_no_of_seasons_"+str(season_selection)+"\' --season_selection_cultivar \'"+cultivar+'\' --no_seasons '+str(season_selection)+' --specific_cultivar \''+cultivar+'\''
                else:
                    runstr = 'python main.py --experiment '+experiment+' --name \''+cultivar+"_no_of_seasons_"+str(season_selection)+"\' --season_selection_cultivar \'"+cultivar+'\' --no_seasons '+str(season_selection)
                fname = 'runfile_'+str(fno)+'.sh'
                cmd = 'sbatch '+fname
                with open('runfile_'+str(fno)+'.sh', 'w') as model:
                    model.write('#!/bin/bash\n')
                    model.write('#SBATCH -J ' + "season_selection_cultivar_"+cultivar[-3:]+'_no_seasons_'+str(season_selection) + '\n')
                    model.write('#SBATCH -A eecs\n')
                    model.write('#SBATCH -p share\n')
                    model.write('#SBATCH -o runfile_logs/' + "season_selection_cultivar_"+cultivar[-3:]+'_no_seasons_'+str(season_selection)+'\n')
                    model.write('#SBATCH -e runfile_logs/' + "season_selection_cultivar_"+cultivar[-3:]+'_no_seasons_'+str(season_selection)+'\n')
                    model.write('#SBATCH --gres=gpu:1\n')
                    model.write('#SBATCH -t 1-0\n')
                    model.write('#SBATCH --mem=10G\n')
                    print(runstr.strip())
                    model.write(runstr.strip())

                os.popen(cmd).read()    
                os.remove('runfile_'+str(fno)+'.sh')
                fno+=1
print("no of experiments", fno)                