# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import random

from utils import mlp,train_test_model
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.ewc_utils.sketchEWC import SketchEWC
from utils.ewc_utils.kfacEWC import KfacEWC

from data.permuted_MNIST import get_permuted_mnist

import os

results_folder='perm_mnist_test/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

#%% set configurations

epochs = 20
lr = 1e-4 
batch_size = 100 
input_size = 784
hidden_sizes = [1024,512,256]
output_size = 10

seed = 0

num_task = 5

activation='ReLU'
device='cuda:0'

#%% load dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_loader, test_loader = get_permuted_mnist(num_task,batch_size,num_workers=0)

fig,ax=plt.subplots(1,num_task,figsize=(num_task*3,3))
for i in range(num_task):
    iter_data=iter(train_loader[i])
    image,label=iter_data.next()
    I=np.reshape(image.data[0,...].numpy(),(28,28))    
    ax[i].imshow(I,cmap='gray')
    ax[i].set_title(label.data[0].numpy())
    ax[i].set_xlabel("Intensity in [%.2f,%.2f]"%(I.min(),I.max()))
plt.show()

#%% train sketched EWC

sketch_ewc_alpha=0.25
sketch_ewc_importance = 1e4
n_sketch = 50

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## define a MLP model
model=mlp.MLP(input_size=input_size,output_size=output_size,
              hidden_size=hidden_sizes,activation=activation,
              device=device).to(device)
model.reset()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

sketch_ewc = SketchEWC(model,device=device,alpha=sketch_ewc_alpha,n_bucket=n_sketch)
## performing training
loss_sketch_ewc, acc_sketch_ewc = {}, {}
for task in tqdm(range(num_task)):
    loss_sketch_ewc[task] = []
    acc_sketch_ewc[task] = []
    for _ in tqdm(range(epochs)):
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        loss_sketch_ewc[task].append(train_test_model.onlineEWC_train_classifier(ewc=sketch_ewc,
                                                                    optimizer=optimizer,
                                                                    data_loader=train_loader[task],
                                                                    importance=sketch_ewc_importance,
                                                                    device=device))
        for sub_task in range(task + 1):
            acc_sketch_ewc[sub_task].append(train_test_model.test_classifier(model=sketch_ewc.model,
                                                                      data_loader=test_loader[sub_task],
                                                                      device=device))
    sketch_ewc.consolidate(train_loader[task])
    
#%% plot sketched EWC
    
fig, ax=plt.subplots(1,2,figsize=(10,5))
for t, v in loss_sketch_ewc.items():
    ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
ax[0].set_xlabel('Epochs',fontsize=14)
ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
for t, v in acc_sketch_ewc.items():
    ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('Epochs',fontsize=14)
ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
fig.suptitle('Using CountSketch on EWC',fontsize=18)
    
#%% train kfac EWC

kfac_ewc_alpha=0.25
kfac_ewc_importance = 1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## define a MLP model
model=mlp.MLP(input_size=input_size,output_size=output_size,
              hidden_size=hidden_sizes,activation=activation,
              device=device).to(device)
model.reset()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

kfac_ewc = KfacEWC(model,device=device,decay=1-kfac_ewc_alpha,weight=kfac_ewc_alpha)
## performing training
loss_kfac_ewc, acc_kfac_ewc = {}, {}
for task in tqdm(range(num_task)):
    loss_kfac_ewc[task] = []
    acc_kfac_ewc[task] = []
    
    kfac_ewc.reset_current_cov()
    for _ in tqdm(range(epochs)):
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        loss_kfac_ewc[task].append(train_test_model.kfac_train_classifier(
            regularizer=kfac_ewc,
            optimizer=optimizer,
            data_loader=train_loader[task],
            importance=kfac_ewc_importance,
            device=device))
        for sub_task in range(task + 1):
            acc_kfac_ewc[sub_task].append(train_test_model.test_classifier(
                model=kfac_ewc.model,
                data_loader=test_loader[sub_task],
                device=device))
    kfac_ewc.consolidate()
    
#%% plot kfac EWC

fig, ax=plt.subplots(1,2,figsize=(10,5))
for t, v in loss_kfac_ewc.items():
    ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
ax[0].set_xlabel('Epochs',fontsize=14)
ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
for t, v in acc_kfac_ewc.items():
    ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('Epochs',fontsize=14)
ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
fig.suptitle('Using KFAC on EWC',fontsize=18)

#%% compare different methods

def get_mean_acc_on_iterations(acc_list):
    acc = {}
    for i in acc_list[0].keys():
        acc[i] = np.zeros(len(acc_list[0][i]))
    for t in range(len(acc_list)):
        for i in acc_list[0].keys():
            acc[i] += np.array(acc_list[t][i]) / len(acc_list)
    return acc

def get_mean_acc(acc,epochs):
    num_task=len(acc)
    temp=np.zeros((num_task,len(acc[0])))
    for t,v in acc.items():
        temp[t,range(t * epochs, num_task * epochs)]=v
        if t<num_task-1:
            temp[t+1,:]=temp[:t+1,:].mean(0)
    return temp.mean(0)

plt.figure(figsize=(10,5))
for t in range(num_task):
    if t%2:
        c='b'
    else:
        c='r'
    plt.axvspan(t*epochs, (t+1)*epochs, facecolor=c, alpha=0.1)
plt.plot(get_mean_acc(acc_sketch_ewc,epochs),linewidth=3)
plt.plot(get_mean_acc(acc_kfac_ewc,epochs),linewidth=3)
plt.xticks(fontsize=18),plt.yticks(fontsize=18)
plt.legend(['Sketched EWC','KFAC EWC'],fontsize=18)
plt.title('Preliminary results for CoreL2M',fontsize=18)
plt.figtext(0.5, 0.01, 'Fig. 1: %d tasks experiment on permuted MNIST'%(num_task), wrap=True, horizontalalignment='center', fontsize=18)
plt.show()

#%% print accuracy
print(get_mean_acc(acc_sketch_ewc,epochs)[-1],get_mean_acc(acc_kfac_ewc,epochs)[-1])

