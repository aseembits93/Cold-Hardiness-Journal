import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset

import glob
import torch.nn as nn
import os
from pathlib import Path

def get_not_nan(y):
    return np.argwhere(np.isnan(y) == False)

def evaluate_ferguson(args):
    dataset = args.dataset
    y_test = dataset['test']['y']
    ferguson_test = dataset['ferguson'][args.current_cultivar][1]
    loss_lt_10 = np.nanmean((y_test[:, :, 0].flatten()-ferguson_test[:, :, 0].flatten())**2)
    loss_lt_50 = np.nanmean((y_test[:, :, 1].flatten()-ferguson_test[:, :, 1].flatten())**2)
    loss_lt_90 = np.nanmean((y_test[:, :, 2].flatten()-ferguson_test[:, :, 2].flatten())**2)
    return {args.current_cultivar:[np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)],'overall':[np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)]}

def graph_residual(plot_dict, title, x, xlabel, ylabel, savepath):
    plt.figure(figsize=(16, 9))

    x = np.array(x)
    width = 1
    pos = 0
    for key, value in plot_dict.items():

        plt.bar(x + pos, value, width, label=key)
        pos += width

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(loc='best')
    # plt.show()
    plt.savefig(savepath + '/'+title+'.png')
    plt.cla()
    plt.clf()
    plt.close('all')
    return

# plots residual LT to ground-truth


def graph_residual_lte(plot_dict, title, x, xlabel, ylabel, savepath):
    plt.figure(figsize=(16, 9))

    x = np.array(x)
    width = 1
    pos = 0
    for key, value in plot_dict.items():

        plt.bar(x + pos, value, width, label=key)
        pos += width

    plt.ylim(-4, 4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(loc='best')
    plt.savefig(savepath + '/'+title+'.png')
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')
    return


class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return self.data_dict['x'].shape[0]

    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['y'][idx], self.data_dict['cultivar_id'][idx], self.data_dict['freq'][idx]


def linear_interp(x, y, missing_indicator, show=False):
    y = np.array(y)
    if (not np.isnan(missing_indicator)):
        missing = np.where(y == missing_indicator)[0]
        not_missing = np.where(y != missing_indicator)[0]
    else:
        # special case for nan values
        missing = np.argwhere(np.isnan(y)).flatten()
        all_idx = np.arange(0, y.shape[0])
        not_missing = np.setdiff1d(all_idx, missing)

    interp = np.interp(x, not_missing, y[not_missing])

    if show == True:
        plt.figure(figsize=(16, 9))
        plt.title("Linear Interp. result where missing = " +
                  str(missing_indicator) + "  Values replaced: " + str(len(missing)))
        plt.plot(x, interp)
        # plt.show()

    return interp


def remove_na(column_name, df):
    ## print("column", column_name)
    total_na = df[column_name].isna().sum()

    df[column_name] = df[column_name].replace(np.nan, -100)
    df[column_name] = linear_interp(
        np.arange(df.shape[0]), df[column_name], -100, False)
    if df[column_name].isna().sum() != 0:
        assert False

    # print("Removed", total_na, "nan from", column_name)

    return

def replace_invalid(df):
    ## print("column", column_name)
    for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
        df[column_name] = df[column_name].replace(-100, np.nan)
    return

# Needs improvement


def split_and_normalize(_df, season_max_length, seasons, features, ferguson_features, label, x_mean=None, x_std=None, ferguson_mean=None, ferguson_std=None):
    x = []
    y = []
    ferguson = []
    for i, season in enumerate(seasons):
        #
        _x = (_df[features].loc[season, :]).to_numpy()

        _x = np.concatenate(
            (_x, np.zeros((season_max_length - len(season), len(features)))), axis=0)

        add_array = np.zeros((season_max_length - len(season), len(label)))
        add_array[:] = np.NaN

        _y = _df.loc[season, :][label].to_numpy()
        _y = np.concatenate((_y, add_array), axis=0)

        _phen = np.zeros((season_max_length, 1))
        # set before-after phen as another output dim
        _y = np.concatenate((_y, _phen), axis=1)
        #_y = np.reshape(_y, (_y.shape[0], _y.shape[1], _y.shape[2], 1))
        add_ferguson = np.zeros(
            (season_max_length - len(season), len(ferguson_features)))
        add_ferguson[:] = np.NaN
        _ferguson = _df.loc[season, :][ferguson_features].to_numpy()
        _ferguson = np.concatenate((_ferguson, add_ferguson), axis=0)

        x.append(_x)
        y.append(_y)
        ferguson.append(_ferguson)

    x = np.array(x)
    y = np.array(y)
    ferguson = np.array(ferguson)

    norm_features_idx = np.arange(0, x_mean.shape[0])

    x[:, :, norm_features_idx] = (
        x[:, :, norm_features_idx] - x_mean) / x_std  # normalize
    #
    # ferguson = (ferguson - ferguson_mean) / ferguson_std
    return x, y, ferguson


def data_processing_multiple_cultivars(cultivar_file, cultivar_idx, args):
    # filename = glob.glob(args.data_path+'*'+cultivar_file+'*')[0]
    #cultivar_name = cultivar_file.split('/')[-1].split('.')[0]
    # random no for setting experiment
    df = args.cultivar_file_dict[cultivar_file]
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

    season_lens = [len(season) for season in valid_seasons]
    season_max_length = max(season_lens)
    no_of_seasons = len(valid_seasons)

    #Heres the part where we select the seasons
    if args.trial == 'trial_0':
        test_idx = list([0,1])
    if args.trial == 'trial_1':
        test_idx = list([(no_of_seasons // 2) - 1,(no_of_seasons // 2) + 1])
    if args.trial == 'trial_2':
        test_idx = list([no_of_seasons - 2, no_of_seasons - 1])

    train_seasons = list()
    test_seasons = list()
    for season_idx, season in enumerate(valid_seasons):
        if season_idx in test_idx:
            test_seasons.append(season)
        else:
            train_seasons.append(season)
    if cultivar_file==args.season_selection_cultivar:
        #select only fewer seasons
        train_no_seasons = len(train_seasons)
        print("original length of training seasons", len(train_seasons))
        train_seasons = [train_seasons[select_idx] for select_idx in np.random.choice(train_no_seasons,args.no_seasons,replace=False)]
        print("new training season length", len(train_seasons))
    valid_idx_train = [x for season in train_seasons for x in season]

    x_mean = df[args.features].iloc[valid_idx_train].mean().to_numpy()
    x_std = df[args.features].iloc[valid_idx_train].std().to_numpy()

    ferguson_mean = df[args.ferguson_features].iloc[valid_idx_train].mean().to_numpy()
    ferguson_std = df[args.ferguson_features].iloc[valid_idx_train].std().to_numpy()

    #do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        remove_na(feature_col, df)
    x_train, y_train, ferguson_train = split_and_normalize(
        df, season_max_length, train_seasons, args.features, args.ferguson_features, args.label, x_mean, x_std, ferguson_mean, ferguson_std)

    x_test, y_test, ferguson_test = split_and_normalize(
        df, season_max_length, test_seasons, args.features, args.ferguson_features, args.label, x_mean, x_std, ferguson_mean, ferguson_std)

    cultivar_label_train = torch.ones(
        (x_train.shape[0], x_train.shape[1], 1))*cultivar_idx
    cultivar_label_test = torch.ones(
        (x_test.shape[0], x_test.shape[1], 1))*cultivar_idx
    return x_train, y_train, x_test, y_test, ferguson_train, ferguson_test, cultivar_label_train, cultivar_label_test

def evaluate(cultivar, model, idx, args):
    dataset = args.dataset
    x_test = dataset['test']['x'][2*idx:2*idx+2].cpu().detach()
    y_test = dataset['test']['y'][2*idx:2*idx+2].cpu().detach()
    cultivar_id = dataset['test']['cultivar_id'][2*idx:2*idx+2].cpu().detach()
    ferguson_test = dataset['ferguson'][cultivar][1]
    model.to('cpu')
    criterion = nn.MSELoss()
    with torch.no_grad():
        out_lt_10, out_lt_50, out_lt_90, _, _ = model(
            x_test, cultivar_label=cultivar_id)
    n_nan = get_not_nan(y_test[:, :, 0])  # LT10/50/90 not NAN
    loss_lt_10 = criterion(
        out_lt_10[n_nan[0], n_nan[1]], y_test[:, :, 0][n_nan[0], n_nan[1]][:, None])  # LT10 GT

    n_nan = get_not_nan(y_test[:, :, 1])  # LT10/50/90 not NAN
    loss_lt_50 = criterion(
        out_lt_50[n_nan[0], n_nan[1]], y_test[:, :, 1][n_nan[0], n_nan[1]][:, None])  # LT50 GT

    n_nan = get_not_nan(y_test[:, :, 2])  # LT10/50/90 not NAN
    loss_lt_90 = criterion(
        out_lt_90[n_nan[0], n_nan[1]], y_test[:, :, 2][n_nan[0], n_nan[1]][:, None])  # LT90 GT

    # # n_nan = get_not_nan(y_test[:, :, 1].cpu())  # LT10/50/90 not NAN
    # # pred = out_lt_50[n_nan[0], n_nan[1]]
    # # gt = y_test[:, :, 1][n_nan[0], n_nan[1]][:, None]
    # gt = y_test.cpu().detach().numpy()

    # gt_2020_lt10 = gt[0, :, 0]
    # gt_2021_lt10 = gt[1, :, 0]

    # gt_2020_lt50 = gt[0, :, 1]
    # gt_2021_lt50 = gt[1, :, 1]

    # gt_2020_lt90 = gt[0, :, 2]
    # gt_2021_lt90 = gt[1, :, 2]

    # ferg_model = ferguson_test[:, :, 1]
    # results_2020_lt10 = out_lt_10[0].cpu().numpy().flatten()
    # results_2021_lt10 = out_lt_10[1].cpu().numpy().flatten()
    # # print("results_2021_lt10.shape", results_2021_lt10.shape)

    # results_2020_lt50 = out_lt_50[0].cpu().numpy().flatten()
    # results_2021_lt50 = out_lt_50[1].cpu().numpy().flatten()
    # # print("results_2021_lt50.shape", results_2021_lt50.shape)

    # results_2020_lt90 = out_lt_90[0].cpu().numpy().flatten()
    # results_2021_lt90 = out_lt_90[1].cpu().numpy().flatten()
    # # print("results_2021_lt90.shape", results_2021_lt90.shape)

    # ferg_diff_2020 = gt_2020_lt50 - ferg_model[0].flatten()
    # results_diff_2020_lt50 = gt_2020_lt50 - results_2020_lt50

    # # print(ferg_diff_2020.shape)
    # # print(results_diff_2020_lt50.shape)

    # ferg_diff_2020 = ferg_diff_2020.flatten()
    # results_diff_2020_lt50 = results_diff_2020_lt50.flatten()
    # cultivar_name = cultivar.split('/')[-1].split('.')[0]

    # plot_path = './plots'
    # savepath = os.path.join(plot_path, args.name, cultivar)
    # Path(savepath).mkdir(parents=True, exist_ok=True)

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     cultivar_name: (gt_2020_lt10 - results_2020_lt10).flatten(),
    # }
    # #plot_dict, title, x, xlabel, ylabel, savepath
    # graph_residual(plot_dict, "LT10 Same Day | 2020-2021 Season |  GT - Pred",
    #                days, "Days", "Temperature. (Deg. C)", savepath)

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     cultivar_name: (gt_2021_lt10 - results_2021_lt10).flatten(),
    # }
    # graph_residual(plot_dict, "LT10 Same Day | 2021 - * Season |  GT - Pred",
    #                days, "Days", "Temperature. (Deg. C)", savepath)

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     "Furguson": ferg_diff_2020,
    #     "RNN": results_diff_2020_lt50,
    # }
    # graph_residual(plot_dict, "LT50 Same Day | 2020-2021 Season |  GT - Pred",
    #                days, "Days", "Temperature. (Deg. C)", savepath)

    # ferg_diff_2021 = gt_2021_lt50 - ferg_model[1].flatten()
    # results_diff_2021_lt50 = gt_2021_lt50 - results_2021_lt50

    # # print(ferg_diff_2021.shape)
    # # print(results_diff_2021_lt50.shape)

    # ferg_diff_2021 = ferg_diff_2021.flatten()
    # results_diff_2021_lt50 = results_diff_2021_lt50.flatten()

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     "Furguson": ferg_diff_2021,
    #     "RNN": results_diff_2021_lt50,
    # }
    # graph_residual(
    #     plot_dict, "LT50 Same Day | 2021 - * Season |  GT - Pred", days, "Days", "LT50", savepath)

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     cultivar_name: (gt_2020_lt90 - results_2020_lt90).flatten(),
    # }
    # graph_residual(plot_dict, "LT90 Same Day | 2020-2021 Season |  GT - Pred",
    #                days, "Days", "Temperature. (Deg. C)", savepath)

    # days = np.arange(0, gt.shape[1])
    # plot_dict = {
    #     cultivar_name: (gt_2021_lt90 - results_2021_lt90).flatten(),
    # }
    # graph_residual(plot_dict, "LT90 Same Day | 2021 - * Season |  GT - Pred",
    #                days, "Days", "Temperature. (Deg. C)", savepath)
    return np.sqrt(loss_lt_10.item()), np.sqrt(loss_lt_50.item()), np.sqrt(loss_lt_90.item())