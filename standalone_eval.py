import argparse
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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from pathlib import Path

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


def data_processing(cultivar_file, args):
    cultivar_name = cultivar_file.split('/')[-1].split('.')[0]
    print("training ", cultivar_name)
    # Reading data
    log_dir = os.path.join("./tensorboard/", args.name,cultivar_name)
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
    return x_train, y_train, x_test, y_test, log_dir, ferguson_train, ferguson_test, valid_seasons[-2:]

# For training the RNN


class net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size):
        super(net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

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

    def forward(self, x, h=None):
        batch_dim, time_dim, state_dim = x.shape

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


def graph_residual(plot_dict, title, x, xlabel, ylabel, cultivar, savepath):
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
    plt.savefig(savepath + '/'+title+'.png')
    return

# plots residual LT to ground-truth


def graph_residual_lte(plot_dict, title, x, xlabel, ylabel, cultivar, savepath):
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
    plt.savefig(savepath + '/'+title+'.png')
    # plt.show()
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


def training_loop(x_train, y_train, x_test, y_test, args, log_dir, device, cultivar_file):
    cultivar_name = cultivar_file.split('/')[-1].split('.')[0]
    model = net(np.array(x_train).shape[-1])
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
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)

    print(x_train.shape, y_train.shape)

    train_dataset = TensorDataset(x_train, y_train)
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    print(x_test.shape, y_test.shape)

    val_dataset = TensorDataset(x_test, y_test)
    valLoader = DataLoader(
        val_dataset, batch_size=x_test.shape[0], shuffle=False)
    #
    for epoch in range(args.epochs):

        # Training Loop
        with tqdm(trainLoader, unit="batch") as tepoch:
            model.train()

            tepoch.set_description(f"Epoch {epoch + 1}/{args.epochs} [T]")
            total_loss = 0
            count = 0
            for i, (x, y) in enumerate(trainLoader):
                x_torch = x.to(device)
                y_torch = y.to(device)

                count += 1

                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch)

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
                    f"Epoch {epoch + 1}/{args.epochs} [V]")
                total_loss = 0
                count = 0
                for i, (x, y) in enumerate(valLoader):
                    x_torch = x.to(device)
                    y_torch = y.to(device)
                    count += 1
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch)
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
    torch.save(model.state_dict(), "Merlot.pt")
    return loss_lt_10.item(), loss_lt_50.item(), loss_lt_90.item()


def evaluate(cultivar, x_test, y_test, ferguson_test, seasons, evalpath, args):
    model = net(np.array(x_test).shape[-1])
    model.load_state_dict(torch.load('Merlot.pt'))
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    gt = y_test.cpu().detach().numpy()
    with torch.no_grad():
        out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_test)
    out_lt_10, out_lt_50, out_lt_90 = out_lt_10.cpu().numpy()[:,:,0], out_lt_50.cpu().numpy()[:,:,0], out_lt_90.cpu().numpy()[:,:,0]
    write_data = dict()
    column_names = ("season_idx","RNN_LTE10","RNN_LTE50","RNN_LTE90","LTE10","LTE50","LTE90")
    for season_no, season in enumerate(seasons):
        season_range = np.arange(len(season))
        for idx, season_idx in enumerate(season):
            write_data[idx]={"season_idx":season_idx,"RNN_LTE10":out_lt_10[season_no,idx],"RNN_LTE50":out_lt_50[season_no,idx],"RNN_LTE90":out_lt_90[season_no,idx],"LTE10":gt[season_no, idx, 0],"LTE50":gt[season_no, idx, 1],"LTE90":gt[season_no, idx, 2]}
        df = pd.DataFrame(data=write_data, index=column_names).T
        df.to_csv("preds"+str(season_no)+".csv",index=False)
        write_data = dict()


def evaluate_ferguson(cultivar, y_test, ferguson_test):
    loss_lt_10 = np.nanmean((y_test[:, :, 0].flatten()-ferguson_test[:, :, 0].flatten())**2)
    loss_lt_50 = np.nanmean((y_test[:, :, 1].flatten()-ferguson_test[:, :, 1].flatten())**2)
    loss_lt_90 = np.nanmean((y_test[:, :, 2].flatten()-ferguson_test[:, :, 2].flatten())**2)
    return loss_lt_10, loss_lt_50, loss_lt_90



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Running Single Model Training for Grape Cultivars")
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
    parser.add_argument('--epochs', type=int, default=400, help='No of epochs to run the model for')
    parser.add_argument('--lr', type=float, default=1e-4, help = "Learning Rate")
    parser.add_argument('--batch_size', type=int, default=12, help = "Batch size")
    parser.add_argument('--evalpath', type=int, default=None, help = "Evaluation Path")
    args = parser.parse_args()

    # Reading data
    all_data_path = "./data/valid/"
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
    for cultivar in ['ColdHardiness_Grape_Merlot.csv']:
        x_train, y_train, x_test, y_test, log_dir, ferguson_train, ferguson_test, seasons = data_processing(args = args, cultivar_file=cultivar)
        final_mse_lte_10, final_mse_lte_50, final_mse_lte_90 = training_loop(x_train, y_train, x_test, y_test, args, log_dir, device, cultivar)
        with open('losses.txt', 'a') as f:
            f.write('\n'+cultivar+' final loss lte10 '+str(final_mse_lte_10)+' final loss lte50 '+str(final_mse_lte_50)+' final loss lte90 '+str(final_mse_lte_90))
        evaluate(cultivar, x_test, y_test, ferguson_test,seasons, args.name, args)
    # final_mse_lte_10, final_mse_lte_50, final_mse_lte_90 = evaluate_ferguson(cultivar, y_test[:,:,:3], ferguson_test)
    # with open('ferguson_losses.txt', 'a') as f:
    #     f.write('\n'+cultivar+' final loss lte10 '+str(final_mse_lte_10)+' final loss lte50 '+str(final_mse_lte_50)+' final loss lte90 '+str(final_mse_lte_90))
