#%% Imports

import os
import numpy as np
import random
import copy
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import pickle
import argparse

from utils import mlp,train_test_model
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.ewc_utils.sketchEWC import SketchEWC
from utils.mas_utils.mas import MAS
from utils.mas_utils.sketchMAS import SketchMAS
from utils.scp_utils.scp import SCP
from utils.scp_utils.sketchSCP import SketchSCP
from data.permuted_MNIST import get_permuted_mnist
from data.rotated_MNIST import get_rotated_mnist

#%% Input arguments
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--data', '-d', type=str, help='perm_mnist/rotated_mnist', default='perm_mnist')
parser.add_argument('--regularizer', '-r', type=str, help='(Sketch)(EWC/MAS/SCP)', required=True)
parser.add_argument('--id', '-i', type=int, help='experiment id', required=True)
parser.add_argument('--task', '-t', type=int, help='number of tasks', default=10)
parser.add_argument('--importance-power', '-p', nargs="*", type=int,  help='list of power of importance', default=0)
parser.add_argument('--bucket', '-b', nargs="*", type=int,  help='list of bucket number', default=1)
parser.add_argument('--slice', '-s', type=int, help='number of slices in SCP', default=10)
parser.add_argument('--result-folder', type=str, default='5Run')
parser.add_argument('--result-filename', type=str)
args = parser.parse_args()

#%% Folders

results_folder = args.data + '_' + args.result_folder + '/' 

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

#%% Hyperparameters

experiment_id = args.id
seed = experiment_id

input_size = 784
hidden_sizes = [256,256]
output_size = 10
activation='ReLU'
device='cuda:0'

num_task = args.task
epochs = 20

batch_size = 100 
lr = 1e-3
alpha=0.25

importance_power_list = args.importance_power
n_bucket_list = args.bucket
if not type(importance_power_list) is list: importance_power_list = [importance_power_list]
if not type(n_bucket_list) is list: n_bucket_list = [n_bucket_list] 
hyperparameter_list = list(itertools.product(importance_power_list, n_bucket_list))

#%% Load dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if args.data == 'perm_mnist':
    train_loader, test_loader = get_permuted_mnist(num_task,batch_size,num_workers=4)
elif args.data == 'rotated_mnist':
    per_task_rotation = 180.0 / num_task
    train_loader, test_loader = get_rotated_mnist(num_task,batch_size,per_task_rotation=per_task_rotation,num_workers=4)


#%% Model init

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model_init=mlp.MLP(input_size=input_size,output_size=output_size,
                   hidden_size=hidden_sizes,activation=activation,
                   device=device)
model_init.reset()

#%% Compute accuracy function
def get_mean_acc(acc,epochs):
    num_task=len(acc)
    temp=np.zeros((num_task,len(acc[0])))
    for t,v in acc.items():
        temp[t,range(t * epochs, num_task * epochs)]=v
        if t<num_task-1:
            temp[t+1,:]=temp[:t+1,:].mean(0)
    return temp.mean(0)

#%% Model training
loss_list = []
acc_list = []

for importance_power, n_bucket in hyperparameter_list:
    importance = 10 ** importance_power
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = copy.deepcopy(model_init).to(device)
    if args.regularizer == 'EWC':
        regularizer = OnlineEWC(model,device=device,alpha=alpha)
        experiment_str = '%s_id_%d_importance_1e%d'%(args.regularizer, args.id, importance_power)
    elif args.regularizer == 'SketchEWC':
        regularizer = SketchEWC(model,device=device,alpha=alpha,n_bucket=n_bucket)
        experiment_str = '%s_id_%d_importance_1e%d_bucket_%d'%(args.regularizer, args.id, importance_power, n_bucket)
    elif args.regularizer == 'MAS':	
        regularizer = MAS(model,device=device,alpha=alpha)
        experiment_str = '%s_id_%d_importance_1e%d'%(args.regularizer, args.id, importance_power)
    elif args.regularizer == 'SketchMAS':
        regularizer = SketchMAS(model,device=device,alpha=alpha,n_bucket=n_bucket)
        experiment_str = '%s_id_%d_importance_1e%d_bucket_%d'%(args.regularizer, args.id, importance_power, n_bucket)
    elif args.regularizer == 'SCP':
        regularizer = SCP(model,device=device,alpha=alpha,n_slices=args.slice)
        experiment_str = '%s_id_%d_importance_1e%d_slice_%d'%(args.regularizer, args.id, importance_power, args.slice)
    elif args.regularizer == 'SketchSCP':
        regularizer = SketchSCP(model,device=device,alpha=alpha,n_slices=args.slice,n_bucket=n_bucket)
        experiment_str = '%s_id_%d_importance_1e%d_slice_%d_bucket_%d'%(args.regularizer, args.id, importance_power, args.slice, n_bucket)
    print('Starting experiment' + experiment_str)

    ## performing training
    loss, acc = {}, {}
    for task in tqdm(range(num_task)):
        loss[task] = []
        acc[task] = []
        for _ in range(epochs):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss[task].append(train_test_model.regularized_train_classifier(regularizer=regularizer,
                                                                    optimizer=optimizer,
                                                                    data_loader=train_loader[task],
                                                                    importance=importance,
                                                                    device=device))                           
            for sub_task in range(task + 1):
                acc[sub_task].append(train_test_model.test_classifier(model=regularizer.model,
                                                                    data_loader=test_loader[sub_task],
                                                                     device=device))
        regularizer.consolidate(train_loader[task]) 

    loss_list.append(loss)
    acc_list.append(acc)
    mean_acc = get_mean_acc(acc,epochs)
    print("Accuracy: " + str(mean_acc[-1]))

#%% Save results
if args.result_filename is None:
    pickle.dump([loss_list, acc_list, hyperparameter_list], open(results_folder+'experiment_%s_id_%d.pkl'%(args.regularizer, args.id),'wb'))
else:
    pickle.dump([loss_list, acc_list, hyperparameter_list], open(results_folder+'experiment_%s_id_%d_%s.pkl'%(args.regularizer, args.id, args.result_filename),'wb'))

