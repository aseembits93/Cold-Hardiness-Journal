import numpy as np
import pandas as pd
import glob
import pickle
season_len_map = dict()
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
def replace_invalid(df):
    for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
        df[column_name] = df[column_name].replace(-100, np.nan)
    return
cultivar_file_dict = {cultivar: pd.read_csv(
    glob.glob('./data/valid/'+'*'+cultivar+'*')[0]) for cultivar in valid_cultivars}
for cultivar_file in valid_cultivars:
    df = cultivar_file_dict[cultivar_file]
    #replace -100 in temp to get accurate mean/std
    replace_invalid(df)
    seasons = [df[(df['SEASON']==season_name) & (df['DORMANT_SEASON']==1)].index.to_list() for season_name in list(df['SEASON'].unique())]
    valid_seasons = list()
    for season in seasons:
        #look at locations where we have valid lte50 values, we will remove those seasons from the data which do not contain any lte values
        # questionable, not sure if it affects rnn training, maybe do unsupervised learning, lets see
        if len(season)==0:
            continue
        missing_temp = df.MIN_AT.iloc[season].isna().sum()
        valid_idx = list(np.array(season)[~np.isnan(df['PREDICTED_LTE50'].iloc[season].to_numpy())])
        valid_lte_readings = list(np.array(season)[~np.isnan(df['LTE50'].iloc[season].to_numpy())])
        # atleast 90% of the season has ferguson readings and missing temps are less than 10% of season length and there is atleast one LTE value
        if (len(valid_idx) >= int(0.9*len(season))) and (missing_temp <= int(0.1*len(season))) and (len(valid_lte_readings)>0):
            valid_seasons.append(season)
    season_len_map[cultivar_file]=len(valid_seasons)-2
print(season_len_map)  
with open('season_len_map.pkl','wb') as f:
    pickle.dump(season_len_map, f)  
