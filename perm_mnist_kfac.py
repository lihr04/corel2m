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

from data.permuted_MNIST_test1 import get_permuted_mnist

import os

results_folder='perm_mnist_test/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

#%% set configurations

epochs = 20
lr = 1e-4 
batch_size = 100 
input_size = 196
hidden_sizes = [100,100]
output_size = 10

seed = 0

num_task = 2

activation='ReLU'
device='cuda:0'

#%% load dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_loader, test_loader = get_permuted_mnist(num_task,batch_size,num_workers=4)

fig,ax=plt.subplots(1,num_task,figsize=(num_task*3,3))
for i in range(num_task):
    iter_data=iter(train_loader[i])
    image,label=iter_data.next()
    I=np.reshape(image.data[0,...].numpy(),(14,14))    
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

#%% accuracy utilities

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

#%% compare different methods

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

#%% grid search kfac EWC

trial = 11
kfac_ewc_alpha_list = [0.25 for i in range(trial)]
kfac_ewc_importance_list = [1,3,10,30,100,300,1000,3000,10000,30000,100000]
loss_kfac_ewc_list = []
acc_kfac_ewc_list = []

for i in tqdm(range(trial)):
    kfac_ewc_alpha = kfac_ewc_alpha_list[i]
    kfac_ewc_importance = kfac_ewc_importance_list[i]
    
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
        for _ in range(epochs):
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
    loss_kfac_ewc_list.append(loss_kfac_ewc)
    acc_kfac_ewc_list.append(acc_kfac_ewc)
    
#%% compare grid search kfac EWC

plt.figure(figsize=(10,5))
for t in range(num_task):
    if t%2:
        c='b'
    else:
        c='r'
    plt.axvspan(t*epochs, (t+1)*epochs, facecolor=c, alpha=0.1)
plt.plot(get_mean_acc(acc_sketch_ewc,epochs),linewidth=3)
for i in range(len(acc_kfac_ewc_list)):
    plt.plot(get_mean_acc(acc_kfac_ewc_list[i],epochs),linewidth=3)
plt.xticks(fontsize=18),plt.yticks(fontsize=18)
plt.legend(['Sketched EWC','Kfac 1e0','Kfac 3e0','Kfac 1e1','Kfac 3e1','Kfac 1e2','Kfac 3e2','Kfac 1e3','Kfac 3e3','Kfac 1e4','Kfac 3e4','Kfac 1e5'],fontsize=18)
plt.title('Preliminary results for CoreL2M',fontsize=18)
plt.figtext(0.5, 0.01, 'Fig. 1: %d tasks experiment on permuted MNIST'%(num_task), wrap=True, horizontalalignment='center', fontsize=18)
plt.show()

#%% compare best kfac with sketched EWC
acc_kfac_ewc_best = acc_kfac_ewc_list[10]

plt.figure(figsize=(10,5))
for t in range(num_task):
    if t%2:
        c='b'
    else:
        c='r'
    plt.axvspan(t*epochs, (t+1)*epochs, facecolor=c, alpha=0.1)
plt.plot(get_mean_acc(acc_sketch_ewc,epochs),linewidth=3)
plt.plot(get_mean_acc(acc_kfac_ewc_best,epochs),linewidth=3)
plt.xticks(fontsize=18),plt.yticks(fontsize=18)
plt.legend(['Sketched EWC','KFAC EWC, importance=1e5'],fontsize=18)
plt.title('Preliminary results for CoreL2M',fontsize=18)
plt.figtext(0.5, 0.01, 'Fig. 1: %d tasks experiment on permuted MNIST'%(num_task), wrap=True, horizontalalignment='center', fontsize=18)
plt.show()