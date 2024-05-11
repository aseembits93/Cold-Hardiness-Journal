import pickle
import os
import numpy
import pandas as pd
from collections import OrderedDict
import glob 

valid_cultivars = ['Barbera',
                    'Cabernet Franc',
                    'Cabernet Sauvignon',
                    'Chardonnay',
                    'Chenin Blanc',
                    'Concord',
                    'Gewurztraminer',
                    'Grenache',
                    'Lemberger',
                    'Malbec',
                    'Merlot',
                    'Mourvedre',
                    'Nebbiolo',
                    'Pinot Gris',
                    'Riesling',
                    'Sangiovese',
                    'Sauvignon Blanc',
                    'Semillon',
                    'Syrah',
                    'Viognier',
                    'Zinfandel']
results_all_Single = {"Barbera_Single":4.224,
"Cabernet Franc_Single":4.005,
"Cabernet Sauvignon_Single":3.437,
"Chardonnay_Single":1.609,
"Chenin Blanc_Single":2.475,
"Concord_Single":2.610,
"Gewurztraminer_Single":2.705,
"Grenache_Single":2.864,
"Lemberger_Single":3.235,
"Malbec_Single":1.715,
"Merlot_Single":1.660,
"Mourvedre_Single":2.251,
"Nebbiolo_Single":2.483,
"Pinot Gris_Single":2.043,
"Riesling_Single":3.630,
"Sangiovese_Single":1.845,
"Sauvignon Blanc_Single":1.717,
"Semillon_Single":3.588,
"Syrah_Single":1.570,
"Viognier_Single":4.164,
"Zinfandel_Single":2.642
}
results_all_ConcatE = {"Barbera_ConcatE":1.504,
"Cabernet Franc_ConcatE":2.363,
"Cabernet Sauvignon_ConcatE":1.755,
"Chardonnay_ConcatE":1.466,
"Chenin Blanc_ConcatE":1.513,
"Concord_ConcatE":2.422,
"Gewurztraminer_ConcatE":1.405,
"Grenache_ConcatE":1.867,
"Lemberger_ConcatE":1.654,
"Malbec_ConcatE":1.323,
"Merlot_ConcatE":1.535,
"Mourvedre_ConcatE":1.656,
"Nebbiolo_ConcatE":1.582,
"Pinot Gris_ConcatE":1.610,
"Riesling_ConcatE":1.479,
"Sangiovese_ConcatE":1.737,
"Sauvignon Blanc_ConcatE":1.435,
"Semillon_ConcatE":1.677,
"Syrah_ConcatE":1.221,
"Viognier_ConcatE":1.750,
"Zinfandel_ConcatE":1.455
}
results_all_MultiH = {"Barbera_MultiH":1.899,
"Cabernet Franc_MultiH":2.391,
"Cabernet Sauvignon_MultiH":2.279,
"Chardonnay_MultiH":1.403,
"Chenin Blanc_MultiH":1.458,
"Concord_MultiH":1.988,
"Gewurztraminer_MultiH":1.209,
"Grenache_MultiH":1.794,
"Lemberger_MultiH":1.491,
"Malbec_MultiH":0.968,
"Merlot_MultiH":1.539,
"Mourvedre_MultiH":1.561,
"Nebbiolo_MultiH":1.240,
"Pinot Gris_MultiH":1.617,
"Riesling_MultiH":1.975,
"Sangiovese_MultiH":1.400,
"Sauvignon Blanc_MultiH":1.227,
"Semillon_MultiH":1.756,
"Syrah_MultiH":1.294,
"Viognier_MultiH":2.289,
"Zinfandel_MultiH":1.606
}
season_selection_list = [2,5,10,20]
experiment_list = ['single','mtl','concat_embedding']
result_dict = dict()
with open('season_len_map.pkl', 'rb') as f:
    season_len_map = pickle.load(f)
fno = 0
for cultivar in valid_cultivars:
    result_dict[cultivar+'_Single']={'2':0,'5':0,'10':0,'20':0,'all':results_all_Single[cultivar+'_Single']}
    result_dict[cultivar+'_ConcatE']={'2':0,'5':0,'10':0,'20':0,'all':results_all_ConcatE[cultivar+'_ConcatE']}
    result_dict[cultivar+'_MultiH']={'2':0,'5':0,'10':0,'20':0,'all':results_all_MultiH[cultivar+'_MultiH']}
    for idx, season_selection in enumerate(season_selection_list):
        if season_selection<season_len_map[cultivar]:
            with open('models/'+cultivar+"_no_of_seasons_"+str(season_selection) +'/single_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no_losses.pkl','rb') as f:
                temp_data = pickle.load(f)
                avg_lte = (temp_data[cultivar]['trial_0'][cultivar][1]+temp_data[cultivar]['trial_1'][cultivar][1]+temp_data[cultivar]['trial_2'][cultivar][1]) / 3.0
                result_dict[cultivar+'_Single'][str(season_selection)] = avg_lte
            with open('models/'+cultivar+"_no_of_seasons_"+str(season_selection) +'/concat_embedding_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no_losses.pkl','rb') as f:
                temp_data = pickle.load(f)
                avg_lte = (temp_data['concat_embedding']['trial_0'][cultivar][1]+temp_data['concat_embedding']['trial_1'][cultivar][1]+temp_data['concat_embedding']['trial_2'][cultivar][1]) / 3.0
                result_dict[cultivar+'_ConcatE'][str(season_selection)] = avg_lte
            with open('models/'+cultivar+"_no_of_seasons_"+str(season_selection) +'/mtl_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no_losses.pkl','rb') as f:
                temp_data = pickle.load(f)
                avg_lte = (temp_data['mtl']['trial_0'][cultivar][1]+temp_data['mtl']['trial_1'][cultivar][1]+temp_data['mtl']['trial_2'][cultivar][1]) / 3.0
                result_dict[cultivar+'_MultiH'][str(season_selection)] = avg_lte    
        else:
            result_dict[cultivar+'_Single'][str(season_selection)] = result_dict[cultivar+'_Single']['all']
            result_dict[cultivar+'_ConcatE'][str(season_selection)] = result_dict[cultivar+'_ConcatE']['all']
            result_dict[cultivar+'_MultiH'][str(season_selection)] = result_dict[cultivar+'_MultiH']['all']
print(result_dict)         
pd.DataFrame(result_dict).T.to_csv("output.csv")

# import csv

# # assuming this is your dictionary
# data = [
#     {'name': 'John', 'age': 30, 'city': 'New York'},
#     {'name': 'Alex', 'age': 25, 'city': 'London'},
#     {'name': 'Richard', 'age': 35, 'city': 'Chicago'}
# ]

# # names of the columns in the CSV file
# fields = ['name', 'age', 'city']

# # name of the csv file
# filename = 'person.csv'

# # writing to csv file
# with open(filename, 'w') as csvfile:
#     # creating a csv dict writer object
#     writer = csv.DictWriter(csvfile, fieldnames=fields)
    
#     # writing headers (field names)
#     writer.writeheader()
    
#     # writing data rows
#     writer.writerows(data)

# import matplotlib.pyplot as plt
# import numpy as np

# # data
# categories = ['Category1', 'Category2', 'Category3', 'Category4']
# values = np.random.rand(4, 5)  # replace with your actual values

# # setup
# barWidth = 0.15
# r1 = np.arange(len(values[0]))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]

# # plotting
# plt.bar(r1, values[0], color='b', width=barWidth, edgecolor='grey', label=categories[0])
# plt.bar(r2, values[1], color='g', width=barWidth, edgecolor='grey', label=categories[1])
# plt.bar(r3, values[2], color='r', width=barWidth, edgecolor='grey', label=categories[2])
# plt.bar(r4, values[3], color='y', width=barWidth, edgecolor='grey', label=categories[3])

# # Add xticks, ylabel, xlabel, title, and legend
# plt.xlabel('X-axis Title', fontweight='bold')
# plt.ylabel('Y-axis Title', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(values[0]))], ['Bar1', 'Bar2', 'Bar3', 'Bar4', 'Bar5'])
# plt.title('Your Bar Chart Title')
# plt.legend()

# # Show graphic
# plt.show()
