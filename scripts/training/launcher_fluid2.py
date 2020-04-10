# imports
import os
import sys
import torch
import pickle
import numpy as np

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch_cae_multilevel_V4 as net

# paths
map_path = None
data_path = '../../data/npy/channel_flow2.npy'
model_path = '/usr/workspace/wsb/liu73/model/fluid2/'
result_path = '/usr/workspace/wsb/liu73/result/fluid2/'
load_model_path = None  # enable this and use option 2/3 to resume training

# load data, model & train
dataset = net.MultiScaleDynamicsDataSet(data_path, n_levels=4, map_path=map_path)

# option 1
archs = [[1,2,3,4,5],[1,3,5,7,9],[1,4,7,10,13],[1,6,11,16,21]]
tols = [0.02, 0.01, 0.008, 0.005]
net.train_net(archs=archs, dataset=dataset, max_epoch=4000, batch_size=350, tols=tols, activation=torch.nn.Sequential(), w=0.5, model_path=model_path, result_path=result_path, std=0.01, verbose=2)

'''
# option 2
load_model = torch.load(load_model_path)
arch = [1,2,3,4,5]
tol = 0.04
net.train_net_one_level(arch=arch, dataset=dataset, max_epoch=4000, batch_size=350, tol=tol, load_model=load_model, activation=torch.nn.Sequential(), w=0.5, model_path=model_path, result_path=result_path, std=0.01)
'''

'''
# option 3
load_model = torch.load(load_model_path)
mode = 0
n_filters = 5
tol = 0.005
net.train_net_one_stage(mode=mode, n_filters=n_filters, dataset=dataset, max_epoch=4000, batch_size=350, tol=tol, load_model=load_model, activation=torch.nn.Sequential(), w=0.5, model_path=model_path, result_path=result_path, std=0.01)
'''