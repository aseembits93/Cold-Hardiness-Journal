#collect stats about dataset
import numpy as np
import pandas as pd
import glob

from collections import OrderedDict

#cultivar, no of seasons, MSE, no of non empty seasons, no of valid reaadings, max season length

# Reading data
all_data_path = "./data/"
cultivars = glob.glob(all_data_path+"*.csv")
data_dict = {'Cultivar':list(),'Dataset_Length':list(),'Seasons':list(),'Valid_Seasons':list(),'Valid_Readings':list(),'Valid_temperature':list(),'Loss_LTE10':list(),'Loss_LTE50':list(),'Loss_LTE90':list(),'RMSE_Loss_LTE10':list(),'RMSE_Loss_LTE50':list(),'RMSE_Loss_LTE90':list()}
label = ['LTE10', 'LTE50', 'LTE90']
temp = ['MEAN_AT']
print('cultivars')
loss_dict = dict()
#
with open('dgx/losses.txt') as f:
    for line in f:
        words = line.strip().split(' ')
        if len(words)==14:
            cultivar_key = ' '.join([words[0],words[1]])
            loss_dict[cultivar_key]=[float(words[5]),float(words[9]),float(words[13])]
        else:
            cultivar_key = words[0]
            loss_dict[cultivar_key]=[float(words[4]),float(words[8]),float(words[12])]
    print(len(words))
for cultivar in cultivars:
    valid_readings = OrderedDict()
    valid_temps = OrderedDict()
    total_valid_readings = 0
    total_valid_temps = 0
    df = pd.read_csv(cultivar)
    #no of seasons
    seasons = np.unique(df['SEASON'].to_numpy())
    valid_seasons = 0
    for season in seasons:
        season_idx = np.where(df['SEASON'].to_numpy()==season)[0]
        dormant_idx = season_idx[np.where(df['DORMANT_SEASON'].iloc[season_idx].to_numpy()==1)[0]]
        season_len = dormant_idx.shape[0]
        valid_readings[season] = min(season_len - df['LTE50'].iloc[dormant_idx].isna().sum(),season_len - df['LTE10'].iloc[dormant_idx].isna().sum(),season_len - df['LTE90'].iloc[dormant_idx].isna().sum())
        valid_temps[season] = np.where(df['MEAN_AT'].iloc[dormant_idx].to_numpy()!=-100)[0].shape[0]
        if valid_readings[season]!=0 and valid_temps[season]!=0:
            valid_seasons+=1
        total_valid_readings+=valid_readings[season]
        total_valid_temps+=valid_temps[season]
        # if season == '2005-2006':
        #
        #max season length
        #no of non empty seasons
        #no of valid readings
    data_dict['Cultivar'].append(cultivar.split('/')[-1].split('.')[0])
    data_dict['Dataset_Length'].append(df.shape[0])
    data_dict['Seasons'].append(seasons.shape[0])
    data_dict['Valid_Seasons'].append(valid_seasons)
    data_dict['Valid_Readings'].append(total_valid_readings)
    data_dict['Valid_temperature'].append(total_valid_temps)
    if cultivar not in loss_dict.keys():
        data_dict['Loss_LTE10'].append(100)
        data_dict['Loss_LTE50'].append(100)
        data_dict['Loss_LTE90'].append(100)
        data_dict['RMSE_Loss_LTE10'].append(100)
        data_dict['RMSE_Loss_LTE50'].append(100)
        data_dict['RMSE_Loss_LTE90'].append(100)
    else :
        data_dict['Loss_LTE10'].append(loss_dict[cultivar][0])
        data_dict['Loss_LTE50'].append(loss_dict[cultivar][1])
        data_dict['Loss_LTE90'].append(loss_dict[cultivar][2])
        data_dict['RMSE_Loss_LTE10'].append(np.sqrt(loss_dict[cultivar][0]))
        data_dict['RMSE_Loss_LTE50'].append(np.sqrt(loss_dict[cultivar][1]))
        data_dict['RMSE_Loss_LTE90'].append(np.sqrt(loss_dict[cultivar][2]))
pd.DataFrame(data_dict).to_csv('myfile.csv', header=True, index=False)

    # seasons = []
    # last_x = 0
    # idx = -1
    # season_max_length = 0
    # for x in df[df["DORMANT_SEASON"] == 1].index.tolist():
    #     if x - last_x > 1:
    #         seasons.append([])
    #         if idx > -1:
    #             season_max_length = max(season_max_length, len(seasons[idx]))
    #         idx += 1
    #     seasons[idx].append(x)
    #     last_x = x

    # season_max_length = max(season_max_length, len(seasons[idx]))

    # # del seasons[19]  # drop season 2007 to 2008 [252 days] because of gap in temperature data, this will be different for different seasons

    # print("len(seasons)", len(seasons))
    # print("Season lengths", [len(x) for x in seasons])
    # print("Max season length", season_max_length)