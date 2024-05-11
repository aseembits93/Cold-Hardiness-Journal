import argparse
import datetime
from util.create_dataset import create_dataset_multiple_cultivars
import torch
from pathlib import Path
import os
import pickle
import glob
import pandas as pd
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running all experiments from here, preventing code duplication")
    parser.add_argument('--experiment', type=str, default="multiplicative_embedding", choices=['model_selection', 'model_selection_comb', 'aggregate_single','multiplicative_embedding', 'mtl', 'additive_embedding', 'concat_embedding', 'single', 'ferguson'], help='type of the experiment')
    #arg for freeze/unfreeze, all/leaveoneout, afterL1, L2,etc, linear/non linear embedding, scratch/linear combination for finetune, task weighting
    parser.add_argument('--setting', type=str, default="all", choices=['all','leaveoneout','allfinetune'], help='experiment setting')
    parser.add_argument('--variant', type=str, default='none',choices=['none','afterL1','afterL2','afterL3','afterL4'])
    parser.add_argument('--unfreeze', type=str, default='no', choices=['yes','no'], help="unfreeze weights during finetune")
    #todo
    parser.add_argument('--nonlinear', type=str, default='no', choices=['yes','no'],help='try non linear embedding/prediction head')
    #todo
    parser.add_argument('--scratch', type=str, default='no', choices=['yes','no'],help='try learning embedding from scratch')
    parser.add_argument('--weighting', type=str, default='none', choices=['none', 'inverse_freq', 'uncertainty'],
                        help="loss weighting strategy")
    parser.add_argument('--name', type=str, default=datetime.datetime.now(
    ).strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
    parser.add_argument('--epochs', type=int, default=400,
                        help='No of epochs to run the model for')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--season_selection_cultivar', type=str, default=None)
    parser.add_argument('--no_seasons', type=int, default=-1, help="no of seasons to select for the Riesling Cultivar")
    parser.add_argument('--batch_size', type=int,
                        default=12, help="Batch size")
    parser.add_argument('--evalpath', type=int,
                        default=None, help="Evaluation Path")
    parser.add_argument('--data_path', type=str,
                        default='./data/grapes/', help="csv Path")
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help="pretrained model to load for finetuning")
    parser.add_argument('--specific_cultivar', type=str, default=None,
                        help="specific cultivar to train for")
    parser.add_argument('--evaluation', action='store_true',
                        help="evaluation mode")
    args = parser.parse_args()
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
    args.cultivar_file_dict = {cultivar: pd.read_csv(
    glob.glob(args.data_path+'*'+cultivar+'*')[0]) for cultivar in valid_cultivars}

    args.features = [
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
    args.ferguson_features = ['PREDICTED_LTE10',
                              'PREDICTED_LTE50', 'PREDICTED_LTE90']
    args.label = ['LTE10', 'LTE50', 'LTE90']
    args.device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.evaluation:
        from experiments.unified_api import run_eval as run_experiment
    else:
        from experiments.unified_api import run_experiment    
    if args.experiment=='ferguson':
        pass
    else:
        if args.variant=='none':
            exec("from nn.models import "+args.experiment+"_net as nn_model")
            exec("from nn.models import "+args.experiment+"_net_finetune as nn_model_finetune")
        else:
            exec("from nn.models import "+args.experiment+"_net_"+args.variant+" as nn_model")
            exec("from nn.models import "+args.experiment+"_net_finetune_"+args.variant+" as nn_model_finetune")
    
    overall_loss = dict()
    #, model_selection
    if args.experiment in ['model_selection']:
        gc.collect()
        args.nn_model = nn_model
        args.valid_cultivars = valid_cultivars
        overall_loss = run_experiment(args)
    #single model training
    elif args.experiment in ['single']:
        valid_cultivars = valid_cultivars if args.specific_cultivar is None else list([
                                                                                      args.specific_cultivar])
        args.nn_model = nn_model
        for left_out in valid_cultivars:
            gc.collect()
            loss_dicts = dict()
            other_cultivars = list([left_out])
            args.current_cultivar = left_out
            args.no_of_cultivars = len(other_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(other_cultivars)
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = run_experiment(args)
            overall_loss[left_out] = loss_dicts
    #ferguson model evaluation
    elif args.experiment in ['ferguson']:
        from util.data_processing import evaluate_ferguson
        for left_out in valid_cultivars:
            gc.collect()
            loss_dicts = dict()
            other_cultivars = list([left_out])
            args.current_cultivar = left_out
            args.no_of_cultivars = len(other_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(other_cultivars)
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = evaluate_ferguson(args)
            overall_loss[left_out] = loss_dicts
    else:
        #leave one out setting
        if args.setting in ['leaveoneout']:
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
            for left_out in valid_cultivars:
                gc.collect()
                loss_dicts = dict()
                finetune_loss_dicts = dict()
                other_cultivars = list(set(valid_cultivars) - set([left_out]))
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
                # similar for all experiments
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts[args.trial] = run_experiment(args)
                # get dataset by selecting features
                args.cultivar_list = list([left_out])
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    args.pretrained_path = os.path.join(
                        './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                    # similar for all experiments
                    finetune_loss_dicts[args.trial] = run_experiment(args, finetune=True)
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    loss_dicts[args.trial].update(finetune_loss_dicts[args.trial])
                overall_loss[left_out] = loss_dicts
        #all setting
        elif args.setting in ['all']:
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            # similar for all experiments
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = run_experiment(args)
            overall_loss[args.experiment] = loss_dicts
        elif args.setting in ['allfinetune']:
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
            # # similar for all experiments
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = run_experiment(args)
            # similar for all experiments
            for left_out in valid_cultivars:
                gc.collect()
                args.cultivar_list = list([left_out])
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    args.pretrained_path = os.path.join(
                        './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                    # similar for all experiments
                    finetune_loss_dicts[args.trial] = run_experiment(args, finetune=True)
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    loss_dicts[args.trial].update(finetune_loss_dicts[args.trial])
            overall_loss[args.experiment] = loss_dicts
    Path(os.path.join('./models', args.name)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join('./models', args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_losses.pkl"), 'wb') as f:
        pickle.dump(overall_loss, f)
    if not os.path.isfile('main_results.csv'):
        output_dict = {'Cultivar':list(sorted(valid_cultivars)), 'ConcatE':[float("nan")]*len(valid_cultivars), 'MultiH':[float("nan")]*len(valid_cultivars), 'Single':[float("nan")]*len(valid_cultivars), 'Ferguson':[float("nan")]*len(valid_cultivars)}  
        with open('output_dict.pkl','wb') as f:
            pickle.dump(output_dict,f)
        pd.DataFrame.from_dict(output_dict).to_csv('main_results.csv',index=False)    
    name_mapping = {'single':'Single', 'mtl':'MultiH', 'ferguson':'Ferguson', 'concat_embedding':'ConcatE'}
    with open('output_dict.pkl','rb') as f:
        output_dict = pickle.load(f)
    for cidx, cultivar in enumerate(sorted(valid_cultivars)):
        average_loss = 0
        if args.experiment in ['single','ferguson']:
            for trial in range(3):
                average_loss += overall_loss[cultivar]['trial_'+str(trial)][cultivar][1]
        else:
            for trial in range(3):
                average_loss += overall_loss[args.experiment]['trial_'+str(trial)][cultivar][1]
        average_loss /= 3
        output_dict[name_mapping[args.experiment]][cidx] = average_loss
    with open('output_dict.pkl','wb') as f:
        pickle.dump(output_dict,f)
    pd.DataFrame.from_dict(output_dict).to_csv('main_results.csv',index=False)
        
        