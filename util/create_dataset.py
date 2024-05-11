import numpy as np
import torch
from .data_processing import *
from collections import Counter
def create_dataset_multiple_cultivars(args):
    embedding_x_train_list, embedding_y_train_list, embedding_x_test_list, embedding_y_test_list, embedding_cultivar_label_train_list, embedding_cultivar_label_test_list = list(), list(), list(), list(), list(), list()
    ferguson_dict = dict()
    for cultivar_idx, cultivar in enumerate(args.cultivar_list):
        x_train, y_train, x_test, y_test, ferguson_train, ferguson_test, cultivar_label_train, cultivar_label_test = data_processing_multiple_cultivars(
            cultivar, cultivar_idx, args)
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
    cultivar_id_arr = train_dataset['cultivar_id']
    train_freq = dict(Counter(cultivar_id_arr[:,0].numpy()))
    train_freq = {key:1/value for key, value in train_freq.items()}
    train_freq_sum = sum(train_freq.values())
    train_freq = {key:value/train_freq_sum for key, value in train_freq.items()}
    train_freq_array = torch.Tensor([train_freq[key] for key in train_dataset['cultivar_id'][:,0].numpy()])
    train_freq_array = train_freq_array.unsqueeze(dim=1).repeat((1,cultivar_id_arr.shape[1]))
    test_freq_array = torch.zeros_like(test_dataset['cultivar_id'])
    train_dataset.update({'freq':train_freq_array})
    test_dataset.update({'freq':test_freq_array})
    return {'train':train_dataset, 'test':test_dataset, 'ferguson':ferguson_dict}