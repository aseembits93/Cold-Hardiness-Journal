import datetime
import glob
import os
import pickle
import random


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from pathlib import Path

# Reading data
all_data_path = "./data/"
cultivars = glob.glob(all_data_path+"*.csv")
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
features = [
    # 'DATE', # date of weather observation
    # 'AWN_STATION', # closest AWN station
    # 'SEASON',
    # 'SEASON_JDAY',
    # 'DORMANT_SEASON',
    # 'YEAR_JDAY',
    # 'PHENOLOGY',
    # 'PREDICTED_LTE50',
    # 'PREDICTED_Budbreak',
    # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
    'MEAN_AT',
    'MIN_AT',  # a
    # 'AVG_AT', # average temp is AgWeather Network
    'MAX_AT',  # a
    'MIN_REL_HUMIDITY',  # a
    'AVG_REL_HUMIDITY',  # a
    'MAX_REL_HUMIDITY',  # a
    'MIN_DEWPT',  # a
    'AVG_DEWPT',  # a
    'MAX_DEWPT',  # a
    'P_INCHES',  # precipitation # a
    'WS_MPH',  # wind speed. if no sensor then value will be na # a
    'MAX_WS_MPH',  # a
    # 'WD_DEGREE', # wind direction, if no sensor then value will be na
    # 'LW_UNITY', # leaf wetness sensor
    # 'SR_WM2', # solar radiation
    # 'MIN_ST2',
    # 'ST2',
    # 'MAX_ST2',
    # 'MIN_ST8',
    # 'ST8', # soil temperature
    # 'MAX_ST8',
    # 'SM8_PCNT', # soil moisture import matplotlib.pyplot as plt@ 8-inch depth # too many missing values for merlot
    # 'SWP8_KPA', # stem water potential @ 8-inch depth # too many missing values for merlot
    # 'MSLP_HPA', # barrometric pressure
    # 'ETO', # evaporation of soil water lost to atmosphere
    # 'ETR' # ???
]
ferguson_features = ['PREDICTED_LTE10', 'PREDICTED_LTE50', 'PREDICTED_LTE90']
label = ['LTE10', 'LTE50', 'LTE90']


def linear_interp(x, y, missing_indicator, show=True):
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
    #print("column", column_name)
    total_na = df[column_name].isna().sum()

    df[column_name] = df[column_name].replace(np.nan, -100)
    df[column_name] = linear_interp(
        np.arange(df.shape[0]), df[column_name], -100, False)
    if df[column_name].isna().sum() != 0:
        assert False

    print("Removed", total_na, "nan from", column_name)

    return


def get_before_after_phenology(season_max_length, season, df):
    #print(season[0], season[-1])

    _y = np.zeros((season_max_length, 1))
    _y[:] = np.NaN

    season_df = df["PHENOLOGY"].iloc[season[0]:season[-1]]

    budbreak_index = season_df.loc[season_df ==
                                   "Budburst/Budbreak"].index.values

    if (len(budbreak_index) == 1):
        budbreak_index = budbreak_index[-1] - season[0]
        #print("budbreak idx:",  budbreak_index)
        #print("_y before", list(_y.flatten()))

        _y[0:budbreak_index] = 0
        _y[budbreak_index:] = 1

        #print("_y after", list(_y.flatten()))
        #print("budbreak_index", _y[budbreak_index])

    return _y


def split_and_normalize(_df, season_max_length, seasons, x_mean=None, x_std=None, ferguson_mean=None, ferguson_std=None):
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

        _phen = get_before_after_phenology(
            season_max_length, season, _df)  # get before-after phen
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

# For budbreak, gets % gt < pred (higher is better)


def _get_under(pred, gt):
    total = 0
    under = 0
    for i in range(gt.shape[0]):
        if (np.isnan(pred[i]) == False and np.isnan(gt[i]) == False):
            #print(pred[i], gt[i])
            total += 1

            if (gt[i] < pred[i]):
                under += 1

    print(under, total, under / total * 100)
    return under / total * 100

# For budbreak


def plot_budbreak(title, season, _y):
    season_idx = np.arange(0, len(season))

    add_array = np.zeros((season_max_length - len(season)))
    add_array[:] = np.NaN

    _y = np.concatenate((_y, add_array), axis=0)

    plt.title(title)

    plt.scatter(x=season_idx, y=_y[:len(season)], label="Prediction")

    _phen = get_before_after_phenology(
        252, season, df)  # get before-after phen

    # draw vertical red line marking true budbreak if it exists
    try:
        true_budbreak_idx = np.where(_phen == 1)[0][0]
        print("true_budbreak_idx:", true_budbreak_idx)
        print("predicted budbreak idx", np.where(_y > 0.5)[0][0])
        plt.vlines(x=true_budbreak_idx, ymin=0, ymax=5,
                   colors='red', label='True Budbreak')
    except:
        pass

    plt.xlabel("Season Day")
    plt.ylabel("Budbreak")

    plt.ylim(0, 1.1)

    plt.legend(loc="upper left")

    # plt.show()
    return


def get_not_nan(y):
    return np.argwhere(np.isnan(y) == False)


def data_processing(cultivar_file):
    cultivar_name = cultivar_file.split('/')[-1].split('.')[0]
    # random no for setting experiment
    runID = cultivar_name + "_" + str(random.randint(11111, 99999))
    print("training ", cultivar_name)
    numberOfEpochs = 400
    batchSize = 12
    # Reading data
    log_dir = os.path.join("./tensorboard/", runID)
    df = pd.read_csv(cultivar_file)
    for feature_col in features:  # remove nan and do linear interp.
        remove_na(feature_col, df)

    seasons = []
    last_x = 0
    idx = -1
    season_max_length = 0
    for x in df[df["DORMANT_SEASON"] == 1].index.tolist():
        if x - last_x > 1:
            seasons.append([])
            if idx > -1:
                season_max_length = max(season_max_length, len(seasons[idx]))
            idx += 1
        seasons[idx].append(x)
        last_x = x

    #
    season_max_length = max(season_max_length, len(seasons[idx]))

    # del seasons[19]  # drop season 2007 to 2008 [252 days] because of gap in temperature data, this will be different for different seasons
    valid_seasons = list()
    for season in seasons:
        #look at locations where we have valid lte50 values, we will remove those seasons from the data which do not contain any lte values
        # questionable, not sure if it affects rnn training, maybe do unsupervised learning, lets see
        valid_idx = list(np.array(season)[~np.isnan(df['LTE50'].iloc[season].to_numpy())])

        if len(valid_idx)!=0:
            valid_seasons.append(season)
    #
    #
    print("len(seasons)", len(seasons))
    print("Season lengths", [len(x) for x in valid_seasons])
    print("Max season length", season_max_length)

    print("Not using normalizing_constants.pkl")
    x_mean = df[features].mean().to_numpy()
    x_std = df[features].std().to_numpy()

    print("Normalizing Constants:")
    print("x_mean", x_mean)
    print("x_std", x_std)
    #
    # not normalizing ferguson predictions anyway
    ferguson_mean = df[ferguson_features].mean().to_numpy()
    ferguson_std = df[ferguson_features].std().to_numpy()

    print("Normalizing Constants:")
    print("ferguson_mean", ferguson_mean)
    print("ferguson_std", ferguson_std)

    x_train, y_train, ferguson_train = split_and_normalize(
        df, season_max_length, valid_seasons[:-2 or None], x_mean, x_std, ferguson_mean, ferguson_std)

    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)
    print("length of sequence", len(x_train[0]))
    print("length of sequence", len(y_train[0]))

    x_test, y_test, ferguson_test = split_and_normalize(
        df, season_max_length, valid_seasons[-2:], x_mean, x_std, ferguson_mean, ferguson_std)

    print("shape of test dataset", x_test.shape)
    print("shape of test dataset", y_test.shape)
    print("length of sequence", len(x_test[0]))
    print("length of sequence", len(y_test[0]))
    cultivar_label_train = torch.ones((x_train.shape[0],x_train.shape[1],1))*cultivar_dict[cultivar_file]
    cultivar_label_test = torch.ones((x_test.shape[0],x_test.shape[1],1))*cultivar_dict[cultivar_file]
    return x_train, y_train, x_test, y_test, log_dir, batchSize, numberOfEpochs, runID, ferguson_train, ferguson_test, cultivar_label_train, cultivar_label_test

# For training the RNN


class embedding_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size):
        super(embedding_net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Embedding(6, 10)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)

        #
        #concat x, embedding_out
        x = torch.cat((x,embedding_out),axis=-1)

        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

# plots residual LT to ground-truth


def graph_residual(plot_dict, title, x, xlabel, ylabel, cultivar):
    plt.close('all')
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
    plt.savefig('plots/embedding/'+cultivar+'/'+title+'.png')
    return

# plots residual LT to ground-truth


def graph_residual_lte(plot_dict, title, x, xlabel, ylabel, cultivar):
    plt.close('all')
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
    plt.savefig('plots/embedding/'+cultivar+'/'+title+'.png')
    plt.show()
    return


# def graph_residual_ferg(title, ferg_diff, results_diff, cultivar):
#     ferg_diff = ferg_diff.flatten()
#     results_diff = results_diff.flatten()
#
#     days = np.arange(0, gt.shape[1])
#     plot_dict = {
#         "Furguson": ferg_diff,
#         "knn": results_diff,
#     }
#     graph_residual_lte(plot_dict, title, days, "Days",
#                        "Temperature. (Deg. C)", cultivar)
#     return


# def graph_residual_tempdiff(title, results_diff, cultivar):
#     results_diff = results_diff.flatten()
#
#     days = np.arange(0, gt.shape[1])
#     plot_dict = {
#         "knn": results_diff,
#     }
#     graph_residual_lte(plot_dict, title, days, "Days",
#                        "Temperature. (Deg. C)", cultivar)
#     return


def training_loop(train_data_dict, test_data_dict, log_dir, batchSize, numberOfEpochs, runID):
    #
    model = embedding_net(train_data_dict['x'].shape[-1]+10)
    model.to(device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    criterion.to(device)
    criterion2 = nn.BCELoss()
    criterion2.to(device)
    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(train_data_dict)
    trainLoader = DataLoader(train_dataset, batch_size=batchSize)
    val_dataset = MyDataset(test_data_dict)
    valLoader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    for epoch in range(numberOfEpochs):

        # Training Loop
        with tqdm(trainLoader, unit="batch") as tepoch:
            model.train()

            tepoch.set_description(f"Epoch {epoch + 1}/{numberOfEpochs} [T]")
            total_loss = 0
            count = 0
            for i, (x, y, cultivar_id) in enumerate(trainLoader):
                x_torch = x.to(device)
                y_torch = y.to(device)
                cultivar_id_torch = cultivar_id.to(device)
                count += 1
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)

                optimizer.zero_grad()       # zero the parameter gradients

                n_nan = get_not_nan(y[:, :, 0])  # LT10/50/90 not NAN
                loss_lt_10 = criterion(
                    out_lt_10[n_nan[0], n_nan[1]], y_torch[:, :, 0][n_nan[0], n_nan[1]][:, None])  # LT10 GT

                n_nan = get_not_nan(y[:, :, 1])  # LT10/50/90 not NAN
                loss_lt_50 = criterion(
                    out_lt_50[n_nan[0], n_nan[1]], y_torch[:, :, 1][n_nan[0], n_nan[1]][:, None])  # LT50 GT

                n_nan = get_not_nan(y[:, :, 2])  # LT10/50/90 not NAN
                loss_lt_90 = criterion(
                    out_lt_90[n_nan[0], n_nan[1]], y_torch[:, :, 2][n_nan[0], n_nan[1]][:, None])  # LT90 GT

                n_nan = get_not_nan(y[:, :, 3])  # budbreak not NAN
                loss_ph = criterion2(
                    out_ph[n_nan[0], n_nan[1]], y_torch[:, :, 3][n_nan[0], n_nan[1]][:, None])  # Budbreak GT

                #loss = loss_lt_10 + loss_lt_50 + loss_lt_90 + loss_ph
                loss = loss_lt_10 + loss_lt_50 + loss_lt_90

                loss.backward()             # backward +
                optimizer.step()            # optimize

                total_loss += loss.item()

                tepoch.set_postfix(Train_Loss=total_loss / count)
                tepoch.update(1)

            writer.add_scalar('Train_Loss', total_loss / count, epoch)

        # Validation Loop
        with torch.no_grad():
            with tqdm(valLoader, unit="batch") as tepoch:

                model.eval()

                tepoch.set_description(
                    f"Epoch {epoch + 1}/{numberOfEpochs} [V]")
                total_loss = 0
                count = 0
                for i, (x, y, cultivar_id) in enumerate(valLoader):
                    x_torch = x.to(device)
                    y_torch = y.to(device)
                    cultivar_id_torch = cultivar_id.to(device)
                    count += 1
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                    # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                    n_nan = get_not_nan(y[:, :, 0])  # LT10/50/90 not NAN
                    loss_lt_10 = criterion(
                        out_lt_10[n_nan[0], n_nan[1]], y_torch[:, :, 0][n_nan[0], n_nan[1]][:, None])  # LT10 GT

                    n_nan = get_not_nan(y[:, :, 1])  # LT10/50/90 not NAN
                    loss_lt_50 = criterion(
                        out_lt_50[n_nan[0], n_nan[1]], y_torch[:, :, 1][n_nan[0], n_nan[1]][:, None])  # LT50 GT

                    n_nan = get_not_nan(y[:, :, 2])  # LT10/50/90 not NAN
                    loss_lt_90 = criterion(
                        out_lt_90[n_nan[0], n_nan[1]], y_torch[:, :, 2][n_nan[0], n_nan[1]][:, None])  # LT90 GT

                    n_nan = get_not_nan(y[:, :, 3])  # budbreak not NAN
                    loss_ph = criterion2(
                        out_ph[n_nan[0], n_nan[1]], y_torch[:, :, 3][n_nan[0], n_nan[1]][:, None])  # Budbreak GT

                    #loss = loss_lt_10 + loss_lt_50 + loss_lt_90 + loss_ph
                    loss = loss_lt_10 + loss_lt_50 + loss_lt_90
                    total_loss += loss.item()

                    tepoch.set_postfix(Val_Loss=total_loss / count)
                    tepoch.update(1)

                writer.add_scalar('Val_Loss', total_loss / count, epoch)
    modelSavePath = "./models/"
    torch.save(model.state_dict(), modelSavePath + runID + ".pt")
    return loss_lt_10.item(), loss_lt_50.item(), loss_lt_90.item()

class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def __len__(self):
        return self.data_dict['x'].shape[0]
    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['y'][idx], self.data_dict['cultivar_id'][idx]

def evaluate(cultivar, x_test, y_test, ferguson_test, cultivar_id, file_handle):
    model = embedding_net(x_test.shape[-1]+10)
    # model_filename = glob.glob(
    #     'models/'+cultivar.split('/')[-1].split('.')[0]+'*')[0]
    model_filename = 'dgx/models/embedding_net2.pt'
    model.load_state_dict(torch.load(model_filename))
    criterion = nn.MSELoss()
    # x_test = torch.FloatTensor(x_test)
    # y_test = torch.FloatTensor(y_test)
    with torch.no_grad():
        out_lt_10, out_lt_50, out_lt_90, _, _ = model(x_test, cultivar_label=cultivar_id)
    n_nan = get_not_nan(y_test[:, :, 0])  # LT10/50/90 not NAN
    loss_lt_10 = criterion(
        out_lt_10[n_nan[0], n_nan[1]], y_test[:, :, 0][n_nan[0], n_nan[1]][:, None])  # LT10 GT

    n_nan = get_not_nan(y_test[:, :, 1])  # LT10/50/90 not NAN
    loss_lt_50 = criterion(
        out_lt_50[n_nan[0], n_nan[1]], y_test[:, :, 1][n_nan[0], n_nan[1]][:, None])  # LT50 GT

    n_nan = get_not_nan(y_test[:, :, 2])  # LT10/50/90 not NAN
    loss_lt_90 = criterion(
        out_lt_90[n_nan[0], n_nan[1]], y_test[:, :, 2][n_nan[0], n_nan[1]][:, None])  # LT90 GT

    file_handle.write(cultivar+' lte10 '+str(loss_lt_10.item())+' lte50 '+str(loss_lt_50.item())+' lte90 '+str(loss_lt_90.item())+'\n')
    # n_nan = get_not_nan(y_test[:, :, 1].cpu())  # LT10/50/90 not NAN
    # pred = out_lt_50[n_nan[0], n_nan[1]]
    # gt = y_test[:, :, 1][n_nan[0], n_nan[1]][:, None]
    gt = y_test.cpu().detach().numpy()

    gt_2020_lt10 = gt[0, :, 0]
    gt_2021_lt10 = gt[1, :, 0]

    gt_2020_lt50 = gt[0, :, 1]
    gt_2021_lt50 = gt[1, :, 1]

    gt_2020_lt90 = gt[0, :, 2]
    gt_2021_lt90 = gt[1, :, 2]

    ferg_model = ferguson_test[:, :, 1]
    results_2020_lt10 = out_lt_10[0].cpu().numpy().flatten()
    results_2021_lt10 = out_lt_10[1].cpu().numpy().flatten()
    print("results_2021_lt10.shape", results_2021_lt10.shape)

    results_2020_lt50 = out_lt_50[0].cpu().numpy().flatten()
    results_2021_lt50 = out_lt_50[1].cpu().numpy().flatten()
    print("results_2021_lt50.shape", results_2021_lt50.shape)

    results_2020_lt90 = out_lt_90[0].cpu().numpy().flatten()
    results_2021_lt90 = out_lt_90[1].cpu().numpy().flatten()
    print("results_2021_lt90.shape", results_2021_lt90.shape)

    ferg_diff_2020 = gt_2020_lt50 - ferg_model[0].flatten()
    results_diff_2020_lt50 = gt_2020_lt50 - results_2020_lt50

    print(ferg_diff_2020.shape)
    print(results_diff_2020_lt50.shape)

    ferg_diff_2020 = ferg_diff_2020.flatten()
    results_diff_2020_lt50 = results_diff_2020_lt50.flatten()
    cultivar_name = cultivar.split('/')[-1].split('.')[0]
    Path("./plots/embedding/"+cultivar_name).mkdir(parents=True, exist_ok=True)

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        cultivar_name: (gt_2020_lt10 - results_2020_lt10).flatten(),
    }
    #
    graph_residual(plot_dict, "LT10 Same Day | 2020-2021 Season |  GT - Pred",
                   days, "Days", "Temperature. (Deg. C)", cultivar_name)

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        cultivar_name: (gt_2021_lt10 - results_2021_lt10).flatten(),
    }
    graph_residual(plot_dict, "LT10 Same Day | 2021 - * Season |  GT - Pred",
                   days, "Days", "Temperature. (Deg. C)", cultivar_name)

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        "Furguson": ferg_diff_2020,
        "RNN": results_diff_2020_lt50,
    }
    graph_residual(plot_dict, "LT50 Same Day | 2020-2021 Season |  GT - Pred",
                   days, "Days", "Temperature. (Deg. C)", cultivar_name)

    ferg_diff_2021 = gt_2021_lt50 - ferg_model[1].flatten()
    results_diff_2021_lt50 = gt_2021_lt50 - results_2021_lt50

    # print(ferg_diff_2021.shape)
    # print(results_diff_2021_lt50.shape)

    ferg_diff_2021 = ferg_diff_2021.flatten()
    results_diff_2021_lt50 = results_diff_2021_lt50.flatten()

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        "Furguson": ferg_diff_2021,
        "RNN": results_diff_2021_lt50,
    }
    graph_residual(
        plot_dict, "LT50 Same Day | 2021 - * Season |  GT - Pred", days, "Days", "LT50", cultivar_name)

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        cultivar_name: (gt_2020_lt90 - results_2020_lt90).flatten(),
    }
    graph_residual(plot_dict, "LT90 Same Day | 2020-2021 Season |  GT - Pred",
                   days, "Days", "Temperature. (Deg. C)", cultivar_name)

    days = np.arange(0, gt.shape[1])
    plot_dict = {
        cultivar_name: (gt_2021_lt90 - results_2021_lt90).flatten(),
    }
    graph_residual(plot_dict, "LT90 Same Day | 2021 - * Season |  GT - Pred",
                   days, "Days", "Temperature. (Deg. C)", cultivar_name)
    #

cultivar_dict = {'./data/ColdHardiness_Grape_Merlot.csv':0,'./data/ColdHardiness_Grape_Cabernet Sauvignon.csv':1,'./data/ColdHardiness_Grape_Chardonnay.csv':2,'./data/ColdHardiness_Grape_Cabernet Franc.csv':3,'./data/ColdHardiness_Grape_Lemberger.csv':4,'./data/ColdHardiness_Grape_Gewurztraminer.csv':5}
embedding_x_train_list, embedding_y_train_list, embedding_x_test_list, embedding_y_test_list, embedding_cultivar_label_train_list, embedding_cultivar_label_test_list = list(), list(), list(), list(), list(), list()
ferguson_dict = dict()
for cultivar in ['./data/ColdHardiness_Grape_Merlot.csv','./data/ColdHardiness_Grape_Cabernet Sauvignon.csv','./data/ColdHardiness_Grape_Chardonnay.csv','./data/ColdHardiness_Grape_Cabernet Franc.csv','./data/ColdHardiness_Grape_Lemberger.csv','./data/ColdHardiness_Grape_Gewurztraminer.csv']:
    x_train, y_train, x_test, y_test, log_dir, batchSize, numberOfEpochs, runID, ferguson_train, ferguson_test, cultivar_label_train, cultivar_label_test = data_processing(
        cultivar_file=cultivar)
    embedding_x_train_list.append(x_train)
    embedding_x_test_list.append(x_test)
    embedding_y_train_list.append(y_train)
    embedding_y_test_list.append(y_test)
    embedding_cultivar_label_train_list.append(cultivar_label_train)
    embedding_cultivar_label_test_list.append(cultivar_label_test)
    ferguson_dict[cultivar]=[ferguson_train, ferguson_test]
    #concat data in a single
    # final_mse_lte_10, final_mse_lte_50, final_mse_lte_90 = training_loop(x_train, y_train, x_test,
    #                           y_test, log_dir, batchSize, numberOfEpochs, runID)
    # with open('losses.txt', 'a') as f:
    #     f.write('\n'+cultivar+' final loss lte10 '+str(final_mse_lte_10)+' final loss lte50 '+str(final_mse_lte_50)+' final loss lte90 '+str(final_mse_lte_90))
    #evaluate(cultivar, x_test, y_test, ferguson_test)

train_dataset = {'x':torch.Tensor(np.concatenate(embedding_x_train_list)),'y':torch.Tensor(np.concatenate(embedding_y_train_list)),'cultivar_id':torch.squeeze(torch.Tensor(np.concatenate(embedding_cultivar_label_train_list)).long())}
test_dataset = {'x':torch.Tensor(np.concatenate(embedding_x_test_list)),'y':torch.Tensor(np.concatenate(embedding_y_test_list)),'cultivar_id':torch.squeeze(torch.Tensor(np.concatenate(embedding_cultivar_label_test_list)).long())}
# embedding_x_train =
# embedding_x_test =
# embedding_y_train =
# embedding_y_test =
# embedding_cultivar_label_train =
# embedding_cultivar_label_test =
runID = 'embedding_net2'
log_dir = os.path.join("./tensorboard/", runID)
#final_mse_lte_10, final_mse_lte_50, final_mse_lte_90 = training_loop(train_dataset, test_dataset, log_dir, batchSize, numberOfEpochs, runID)
#cultivar wise MSE
# with open('embedding_losses.txt', 'w') as f:
#     f.write('final loss lte10 '+str(final_mse_lte_10)+' final loss lte50 '+str(final_mse_lte_50)+' final loss lte90 '+str(final_mse_lte_90))
with open('embedding_cultiver_wise_loss.txt','w') as fh:
    for i, cultivar in enumerate(['./data/ColdHardiness_Grape_Merlot.csv','./data/ColdHardiness_Grape_Cabernet Sauvignon.csv','./data/ColdHardiness_Grape_Chardonnay.csv','./data/ColdHardiness_Grape_Cabernet Franc.csv','./data/ColdHardiness_Grape_Lemberger.csv','./data/ColdHardiness_Grape_Gewurztraminer.csv']):
        evaluate(cultivar,test_dataset['x'][2*i:2*i+2],test_dataset['y'][2*i:2*i+2],ferguson_dict[cultivar][1], test_dataset['cultivar_id'][2*i:2*i+2], fh)

