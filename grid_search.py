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

from utils import mlp,train_test_model
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.ewc_utils.sketchEWC import SketchEWC
from utils.mas_utils.mas import MAS
from utils.mas_utils.sketchMAS import SketchMAS
from utils.scp_utils.scp import SCP
from utils.scp_utils.sketchSCP import SketchSCP
from data.permuted_MNIST import get_permuted_mnist


#%% Folders

results_folder='perm_mnist_10Run/'
models_folder='saved_models/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)
if not os.path.isdir(models_folder):
    os.mkdir(models_folder)

#%% Hyperparameters

experiment_id = 0
seed = experiment_id + 40

input_size = 784
hidden_sizes = [1024,512,256]
output_size = 10
activation='ReLU'
device='cuda:0'

num_task = 10
epochs = 20

batch_size = 100 
lr = 1e-4
alpha=0.25

importance_power = 4
importance = 10 ** importance_power
n_bucket = 100
# hyperparameter_list = list(itertools.product(importance_power_list, n_bucket_list))
# iteration = len(hyperparameter_list)

#%% Load dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_loader, test_loader = get_permuted_mnist(num_task,batch_size)


#%% Model init

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model_init=mlp.MLP(input_size=input_size,output_size=output_size,
                   hidden_size=hidden_sizes,activation=activation,
                   device=device)
model_init.reset()

#%% Model training

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = copy.deepcopy(model_init).to(device)
regularizer = OnlineEWC(model,device=device,alpha=alpha)
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
    torch.save(regularizer.model.state_dict(), 'saved_models/ewc_'+str(task)+'.pt') 


#%% Compute accuracy
def get_mean_acc(acc,epochs):
    num_task=len(acc)
    temp=np.zeros((num_task,len(acc[0])))
    for t,v in acc.items():
        temp[t,range(t * epochs, num_task * epochs)]=v
        if t<num_task-1:
            temp[t+1,:]=temp[:t+1,:].mean(0)
    return temp.mean(0)

mean_acc = get_mean_acc(acc,epochs)
print("Accuracy: " + mean_acc[-1])

#%% Save results
pickle.dump([loss, acc], open(results_folder+'experiment_ewc.pkl','wb'))
