import os
import random
import multiprocessing as mp

import numpy as np
# import geopandas as gpd

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, seed=22222, normalize=2):#, weather= True

        Data_path = "dataset/"
        

        self.dataset = self.read_npy_data(train, valid,Data_path) # day_nums, neighbor_nums, tract_nums, flow_nums
        
        self._split(train, valid, seed)
        
        self.train_max, self.train_min = self._max_min()
        self.train, self.test, self.valid = self._normalized()

        random.seed(seed,)
        # random.shuffle(self.dataset)
        

    def read_npy_data(self, train, valid, Data_path):
        dg_data = []
        test_data = []
        for i in tqdm([1,9]):

            dg_data_o = np.load(Data_path+str(i)+".npy")
            # merged_arr = []
            # for n in dg_data_o:
            #     merged_arr.append(n)

            # dg_data_o = np.concatenate(merged_arr)
            # dg_data_o = dg_data_o.reshape((dg_data_o.shape[0]*dg_data_o.shape[1], dg_data_o.shape[2], dg_data_o.shape[3]))
            # indices_of_zeros = np.where(dg_data_o[:,-1] == 0)[0]



            if i == 9:
                self.test_org=dg_data_o
                self.test_o = dg_data_o[:,:,24]
                self.test_d = dg_data_o[:,:,25]
            else:
                dg_data.append(dg_data_o)
        
        if len(dg_data)>1:
            return np.concatenate([dg_data[i] for i in range(len(dg_data))])
        else:
            # print(dg_data[0].shape)
            return dg_data[0]

#         self.test_org = np.array(test_data)
#         self.test_data_indices = selected_indices


    def _max_min(self):

        train_dataset = self.train_org
        # print(train_dataset[0][0].shape)
        feature_num = train_dataset[0][0].shape[0]
        # train_reshape = []
        max_by_column = np.array([0 for i in range(feature_num-1)], dtype=float)
        max_by_column = np.append(max_by_column,1)
        min_by_column = np.array([9999 for i in range(feature_num-1)], dtype=float)
        min_by_column = np.append(min_by_column,0)

        for i in range(len(train_dataset)):
    
            max_one_o_by_column = train_dataset[i].max(axis=0)
            min_one_o_by_column = train_dataset[i].min(axis=0)

            for j in range(feature_num-1):

                if max_one_o_by_column[j] > max_by_column[j]:
                    max_by_column[j]=max_one_o_by_column[j]

                if min_by_column[j] > min_one_o_by_column[j]:
                    min_by_column[j]=min_one_o_by_column[j]
        
        self.train_max_flow = max_by_column[-2]
        self.train_min_flow = min_by_column[-2]
        return max_by_column, min_by_column

    def _normalized(self):
        
        norm_train=[]
        for i in range(len(self.train_org)):
            sub_result = (self.train_org[i]-self.train_min)/(self.train_max-self.train_min)
            sub_result[np.isnan(sub_result)] = 0
            norm_train.append(sub_result)
            
        norm_test=[]
        for i in range(len(self.test_org)):
            sub_result = (self.test_org[i]-self.train_min)/(self.train_max-self.train_min)
            sub_result[np.isnan(sub_result)] = 0
            norm_test.append(sub_result)
            
        norm_valid=[]
        for i in range(len(self.valid_org)):
            sub_result = (self.valid_org[i]-self.train_min)/(self.train_max-self.train_min)
            sub_result[np.isnan(sub_result)] = 0
            norm_valid.append(sub_result)
        
        
        return torch.from_numpy(np.array(norm_train)), torch.from_numpy(np.array(norm_test)), torch.from_numpy(np.array(norm_valid))
        
        

    def _split(self, train, valid, seed):

        generator = torch.Generator().manual_seed(seed)
        
        self.train_org, self.valid_org = data.random_split(self.dataset,[train/(train+valid),valid/(train+valid)], generator=generator)


    def get_batches(self, inputs, targets, batch_size):
        length = len(inputs)
        start_idx = 0
        # print(len(inputs))
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            # print(excerpt)
            X = inputs[start_idx:end_idx]
            Y = targets[start_idx:end_idx]
            yield X, Y
            start_idx += batch_size

# Data = Data_utility("weekday", 0.6, 0.2, 2222, 2)
#
# print(Data.train[0][1].size())
