import torch
from util.create_dataset import MyDataset, get_not_nan
from util.data_processing import evaluate
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import os
from pathlib import Path

def run_experiment(args, finetune=False):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    if finetune:
        model = args.nn_model_finetune(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    else:
        model = args.nn_model(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    if finetune:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    if finetune:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune", args.trial, args.current_cultivar)
    else:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    for epoch in range(args.epochs):
        # Training Loop
        model.train()
        total_loss = 0
        count = 0
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            freq = freq.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            optimizer.zero_grad()       # zero the parameter gradients
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            #assuming lte values are present together
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
            freq_lt_10 = freq[~nan_locs_lt_10]
            freq_lt_50 = freq[~nan_locs_lt_50]
            freq_lt_90 = freq[~nan_locs_lt_90]
            if args.weighting=='inverse_freq':
                loss = torch.mul(loss_lt_10, freq_lt_10).mean() + torch.mul(loss_lt_50, freq_lt_50).mean() + torch.mul(loss_lt_90, freq_lt_90).mean()
            elif args.weighting=='uncertainty':
                pass
            else:
                loss = loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()
            loss.backward()             # backward +
            optimizer.step()            # optimize
            total_loss += loss.item()
        writer.add_scalar('Train_Loss', total_loss / count, epoch)
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_loss = 0
            count = 0
            for i, (x, y, cultivar_id, freq) in enumerate(valLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                #replace nan in gt with 0s, replace corresponding values in pred with 0s
                nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                out_lt_10[:,:,0][nan_locs_lt_10] = 0
                out_lt_50[:,:,0][nan_locs_lt_50] = 0
                out_lt_90[:,:,0][nan_locs_lt_90] = 0
                y_torch = torch.nan_to_num(y_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
                loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
                loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
                loss = loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()
                total_loss += loss.mean().item()
            writer.add_scalar('Val_Loss', total_loss / count, epoch)
    loss_dict = dict()
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name, args.current_cultivar, args.trial)).mkdir(parents=True, exist_ok=True)
    if finetune:
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune.pt"))
    else:
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt"))
    # Validation Loop
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90 = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
            total_loss_lt_10+=loss_lt_10
            total_loss_lt_50+=loss_lt_50
            total_loss_lt_90+=loss_lt_90
            loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)])
    loss_dict['overall'] = list([np.sqrt(total_loss_lt_10), np.sqrt(total_loss_lt_50), np.sqrt(total_loss_lt_90)])
    return loss_dict

def run_eval(args, finetune=False):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    model = args.nn_model(feature_len, no_of_cultivars, args.nonlinear)
    #model.load_state_dict(torch.load('./eval/mtl_all.pt'))
    model.load_state_dict(torch.load('./eval/single/'+args.current_cultivar+'/'+args.trial+'/single.pt'))
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    # Validation Loop
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90 = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            print('i',i, 'cultivar', cultivar)
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            #print(y_torch[1,:,1])
            cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
            #first dim is model, 2nd dim is eval cultivar
            args.loss_array[args.cultivar_dict[args.current_cultivar],args.cultivar_dict[cultivar]]+=np.sqrt(loss_lt_50)

def run_model_selection(args, finetune=False):
    # from util.create_dataset import create_dataset_multiple_cultivars
    # import matplotlib.pyplot as plt
    # dataset_dict = {"trial_0":dict(),"trial_1":dict(),"trial_2":dict()}
    # #Load all datasets in memory
    # for trial in range(3):
    #     for cultivar in args.valid_cultivars:
    #         args.cultivar_list = list([cultivar])
    #         args.trial = 'trial_'+str(trial)
    #         dataset_dict[args.trial][cultivar] = create_dataset_multiple_cultivars(args)
    # #do the eval now 
    # loss_array = np.zeros((len(args.valid_cultivars),len(args.valid_cultivars)))
    # loss_array_1 = np.zeros((len(args.valid_cultivars),len(args.valid_cultivars)))
    # for trial in range(3):
    #     for source_idx, cultivar in enumerate(args.valid_cultivars):
    #         #left_out = [x for jdx, x in enumerate(args.valid_cultivars) if jdx != idx]
    #         feature_len = dataset_dict[args.trial][cultivar]['train']['x'].shape[-1]    
    #         no_of_cultivars = 1
    #         model = args.nn_model(feature_len, no_of_cultivars, args.nonlinear)
    #         model.load_state_dict(torch.load('./models/modelselection/'+cultivar+'/'+args.trial+'/single_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no.pt'))
    #         model.to(args.device)
    #         trainable_params = sum([np.prod(p.size()) for p in filter(
    #             lambda p: p.requires_grad, model.parameters())])
    #         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #         criterion = nn.MSELoss(reduction='none')
    #         criterion.to(args.device)
    #         for eval_idx, eval_cultivar in enumerate(args.valid_cultivars):
    #             val_dataset = MyDataset(dataset_dict[args.trial][eval_cultivar]['train'])
    #             valLoader = DataLoader(val_dataset, batch_size=dataset_dict[args.trial][eval_cultivar]['train']['x'].shape[0], shuffle=False)
    #             val_dataset_1 = MyDataset(dataset_dict[args.trial][eval_cultivar]['test'])
    #             valLoader_1 = DataLoader(val_dataset_1, batch_size=dataset_dict[args.trial][eval_cultivar]['test']['x'].shape[0], shuffle=False)
    #             # Validation Loop
    #             total_loss_lt_10, total_loss_lt_50, total_loss_lt_90 = 0, 0, 0
    #             with torch.no_grad():
    #                 model.eval()
    #                 #Training set
    #                 for validx, (x, y, cultivar_id, freq) in enumerate(valLoader):
    #                     #print("validx", validx)
    #                     x_torch = x.to(args.device)
    #                     y_torch = y.to(args.device)
    #                     cultivar_id_torch = cultivar_id.to(args.device)
    #                     out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
    #                     #replace nan in gt with 0s, replace corresponding values in pred with 0s
    #                     nan_locs_lt_10 = y_torch[:, :, 0].isnan()
    #                     nan_locs_lt_50 = y_torch[:, :, 1].isnan()
    #                     nan_locs_lt_90 = y_torch[:, :, 2].isnan()
    #                     out_lt_10[:,:,0][nan_locs_lt_10] = 0
    #                     out_lt_50[:,:,0][nan_locs_lt_50] = 0
    #                     out_lt_90[:,:,0][nan_locs_lt_90] = 0
    #                     y_torch = torch.nan_to_num(y_torch)
    #                     loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
    #                     loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
    #                     loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
    #                     #first dim is model, 2nd dim is eval cultivar
    #                     loss_array[source_idx, eval_idx]+=np.sqrt(loss_lt_50)
                        
    #                 #Testing Set
    #                 for validx, (x, y, cultivar_id, freq) in enumerate(valLoader_1):
    #                     #print("validx", validx)
    #                     x_torch = x.to(args.device)
    #                     y_torch = y.to(args.device)
    #                     cultivar_id_torch = cultivar_id.to(args.device)
    #                     out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
    #                     #replace nan in gt with 0s, replace corresponding values in pred with 0s
    #                     nan_locs_lt_10 = y_torch[:, :, 0].isnan()
    #                     nan_locs_lt_50 = y_torch[:, :, 1].isnan()
    #                     nan_locs_lt_90 = y_torch[:, :, 2].isnan()
    #                     out_lt_10[:,:,0][nan_locs_lt_10] = 0
    #                     out_lt_50[:,:,0][nan_locs_lt_50] = 0
    #                     out_lt_90[:,:,0][nan_locs_lt_90] = 0
    #                     y_torch = torch.nan_to_num(y_torch)
    #                     loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
    #                     loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
    #                     loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
    #                     #first dim is model, 2nd dim is eval cultivar
    #                     loss_array_1[source_idx, eval_idx]+=np.sqrt(loss_lt_50)        
    # loss_array/=3.0 #Training Set
    # loss_array_1/=3.0 #Testing Set
    
    # np.savetxt("train.csv", loss_array, delimiter=",")
    # np.savetxt("test.csv", loss_array_1, delimiter=",")
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name)).mkdir(parents=True, exist_ok=True)
    loss_array = np.loadtxt("train.csv", delimiter=",")
    loss_array_1 = np.loadtxt("test.csv", delimiter=",")
    print_dict = dict()
    for eval_idx, eval_cultivar in enumerate(args.valid_cultivars):   
        arr = loss_array[:,eval_idx]
        exclude_idx = eval_idx
        m_arr = np.zeros(arr.size, dtype=bool)
        m_arr[exclude_idx]=True
        best_train_idx = np.argmin(np.ma.array(arr,mask=m_arr))
        print("best train error for eval", eval_cultivar, args.valid_cultivars[best_train_idx])
        print("test error for that cultivar", np.round(loss_array_1[best_train_idx,eval_idx],2))
        print_dict[eval_cultivar]=list([loss_array_1[best_train_idx,eval_idx],args.valid_cultivars[best_train_idx]])
    for key in sorted(print_dict.keys()):
        print(print_dict[key][0])
    for key in sorted(print_dict.keys()):
        print(print_dict[key][1])
    # plt.close("all")
    # fig, ax = plt.subplots(figsize=(15,15))
    # im = ax.imshow(loss_array)
    # ax.set_xticks(np.arange(21),labels=args.valid_cultivars)
    # ax.set_yticks(np.arange(21),labels=args.valid_cultivars)
    # ax.set_ylabel("Loaded Model")
    # ax.set_xlabel("Evaluation Cultivar")
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(21):
    #     for j in range(21):
    #         text = ax.text(j, i, np.round(loss_array[i, j],2),
    #                        ha="center", va="center", color="w")

    # plt.savefig("train.png")
    # plt.close("all")
    # fig, ax = plt.subplots(figsize=(15,15))
    # im = ax.imshow(loss_array_1)
    # ax.set_xticks(np.arange(21),labels=args.valid_cultivars)
    # ax.set_yticks(np.arange(21),labels=args.valid_cultivars)
    # ax.set_ylabel("Loaded Model")
    # ax.set_xlabel("Evaluation Cultivar")
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(21):
    #     for j in range(21):
    #         text = ax.text(j, i, np.round(loss_array_1[i, j],2),
    #                        ha="center", va="center", color="w")

    # plt.savefig("test.png")
    return {0:0}

def run_model_selection_comb(args, finetune=False):
    from util.create_dataset import create_dataset_multiple_cultivars
    import matplotlib.pyplot as plt
    dataset_dict = {"trial_0":dict(),"trial_1":dict(),"trial_2":dict()}
    #Load all datasets in memory
    for trial in range(3):
        for cultivar in args.valid_cultivars:
            args.cultivar_list = list([cultivar])
            args.trial = 'trial_'+str(trial)
            dataset_dict[args.trial][cultivar] = create_dataset_multiple_cultivars(args)
    #do the eval now 
    loss_array = np.zeros((len(args.valid_cultivars),len(args.valid_cultivars)))
    loss_array_1 = np.zeros((len(args.valid_cultivars),len(args.valid_cultivars)))
    for trial in range(3):
        for source_idx, cultivar in enumerate(args.valid_cultivars):
            #left_out = [x for jdx, x in enumerate(args.valid_cultivars) if jdx != idx]
            feature_len = dataset_dict[args.trial][cultivar]['train']['x'].shape[-1]    
            no_of_cultivars = 1
            model = args.nn_model(feature_len, no_of_cultivars, args.nonlinear)
            model.load_state_dict(torch.load('./models/modelselection/'+cultivar+'/'+args.trial+'/single_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no.pt'))
            model.to(args.device)
            trainable_params = sum([np.prod(p.size()) for p in filter(
                lambda p: p.requires_grad, model.parameters())])
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss(reduction='none')
            criterion.to(args.device)
            for eval_idx, eval_cultivar in enumerate(args.valid_cultivars):
                val_dataset = MyDataset(dataset_dict[args.trial][eval_cultivar]['train'])
                valLoader = DataLoader(val_dataset, batch_size=dataset_dict[args.trial][eval_cultivar]['train']['x'].shape[0], shuffle=False)
                val_dataset_1 = MyDataset(dataset_dict[args.trial][eval_cultivar]['test'])
                valLoader_1 = DataLoader(val_dataset_1, batch_size=dataset_dict[args.trial][eval_cultivar]['test']['x'].shape[0], shuffle=False)
                # Validation Loop
                total_loss_lt_10, total_loss_lt_50, total_loss_lt_90 = 0, 0, 0
                with torch.no_grad():
                    model.eval()
                    #Training set
                    for validx, (x, y, cultivar_id, freq) in enumerate(valLoader):
                        #print("validx", validx)
                        x_torch = x.to(args.device)
                        y_torch = y.to(args.device)
                        cultivar_id_torch = cultivar_id.to(args.device)
                        out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        #replace nan in gt with 0s, replace corresponding values in pred with 0s
                        nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                        nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                        nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        y_torch = torch.nan_to_num(y_torch)
                        loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                        loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                        loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                        #first dim is model, 2nd dim is eval cultivar
                        loss_array[source_idx, eval_idx]+=np.sqrt(loss_lt_50)
                        
                    #Testing Set
                    for validx, (x, y, cultivar_id, freq) in enumerate(valLoader_1):
                        #print("validx", validx)
                        x_torch = x.to(args.device)
                        y_torch = y.to(args.device)
                        cultivar_id_torch = cultivar_id.to(args.device)
                        out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        #replace nan in gt with 0s, replace corresponding values in pred with 0s
                        nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                        nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                        nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        y_torch = torch.nan_to_num(y_torch)
                        loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                        loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                        loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                        #first dim is model, 2nd dim is eval cultivar
                        loss_array_1[source_idx, eval_idx]+=np.sqrt(loss_lt_50)        
    loss_array/=3.0 #Training Set
    loss_array_1/=3.0 #Testing Set
    
    np.savetxt("train.csv", loss_array, delimiter=",")
    np.savetxt("test.csv", loss_array_1, delimiter=",")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(loss_array)
    ax.set_xticks(np.arange(21),labels=args.valid_cultivars)
    ax.set_yticks(np.arange(21),labels=args.valid_cultivars)
    ax.set_ylabel("Loaded Model")
    ax.set_xlabel("Evaluation Cultivar")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(21):
        for j in range(21):
            text = ax.text(j, i, np.round(loss_array[i, j],2),
                           ha="center", va="center", color="w")

    plt.savefig("train.png")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(loss_array_1)
    ax.set_xticks(np.arange(21),labels=args.valid_cultivars)
    ax.set_yticks(np.arange(21),labels=args.valid_cultivars)
    ax.set_ylabel("Loaded Model")
    ax.set_xlabel("Evaluation Cultivar")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(21):
        for j in range(21):
            text = ax.text(j, i, np.round(loss_array_1[i, j],2),
                           ha="center", va="center", color="w")

    plt.savefig("test.png")
    return {0:0}