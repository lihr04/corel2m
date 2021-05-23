# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle

from utils import mlp,train_test_model
from utils.ewc_utils.ToyExampleEWC import EWC
from utils.ewc_utils.ToyExampleEWC import FullEWC, LowRankEWC, MinorDiagonalEWC
from utils.ewc_utils.ToyExampleEWC import BlockDiagonalEWC
from utils.ewc_utils.ToyExampleEWC import SketchEWC

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
hidden_sizes = [64,64]
output_size = 10

num_task = 5
activation='ReLU'
device='cuda:0'

save_plot = False

loss_full_ewc_list = []
acc_full_ewc_list = []
loss_block_diagonal_ewc_list = []
acc_block_diagonal_ewc_list = []

#%% load dataset

for seed in range(5):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_loader, test_loader = get_permuted_mnist(num_task,batch_size,num_workers=0)
    
    fig,ax=plt.subplots(1,num_task,figsize=(num_task*3,3))
    for i in range(num_task):
        iter_data=iter(train_loader[i])
        image,label=iter_data.next()
        I=np.reshape(image.data[0,...].numpy(),(14,14))    
        ax[i].imshow(I,cmap='gray')
        ax[i].set_title(label.data[0].numpy())
        ax[i].set_xlabel("Intensity in [%.2f,%.2f]"%(I.min(),I.max()))
    plt.show()
    
    #%% train full EWC
    
    full_ewc_alpha=0.25
    full_ewc_importance=1e3
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  require_bias=True,device=device).to(device)
    # model.apply(weights_init)
    full_ewc= FullEWC(model,device=device,alpha=full_ewc_alpha)
    
    ## performing training
    loss_full_ewc, acc_full_ewc = {}, {}
    for task in tqdm(range(num_task)):
        loss_full_ewc[task] = []
        acc_full_ewc[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_full_ewc[task].append(train_test_model.onlineEWC_train_classifier(ewc=full_ewc,
                                                                          optimizer=optimizer,
                                                                          data_loader=train_loader[task],
                                                                          importance=full_ewc_importance,
                                                                          device=device))                           
            for sub_task in range(task + 1):
                acc_full_ewc[sub_task].append(train_test_model.test_classifier(model=full_ewc.model,
                                                                     data_loader=test_loader[sub_task],
                                                                     device=device))        
        if not task == num_task - 1: 
            full_ewc.consolidate(train_loader[task])
    loss_full_ewc_list.append(loss_full_ewc)
    acc_full_ewc_list.append(acc_full_ewc)
    
        
    #%% plot full EWC
        
    fig, ax=plt.subplots(1,2,figsize=(10,5))
    for t, v in loss_full_ewc.items():
        ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
    ax[0].set_xlabel('Epochs',fontsize=14)
    ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
    for t, v in acc_full_ewc.items():
        ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Epochs',fontsize=14)
    ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
    fig.suptitle('Using Full IM on EWC',fontsize=18)
    plt.show()
        
    #%% train block-diagonal EWC
    block_diagonal_ewc_alpha=0.25
    block_diagonal_ewc_importance=1e3
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  require_bias=True,device=device).to(device)
    # model.apply(weights_init)
    block_diagonal_ewc= BlockDiagonalEWC(model,device=device,alpha=block_diagonal_ewc_alpha)
    
    ## performing training
    loss_block_diagonal_ewc, acc_block_diagonal_ewc = {}, {}
    for task in tqdm(range(num_task)):
        loss_block_diagonal_ewc[task] = []
        acc_block_diagonal_ewc[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_block_diagonal_ewc[task].append(train_test_model.onlineEWC_train_classifier(ewc=block_diagonal_ewc,
                                                                          optimizer=optimizer,
                                                                          data_loader=train_loader[task],
                                                                          importance=block_diagonal_ewc_importance,
                                                                          device=device))                           
            for sub_task in range(task + 1):
                acc_block_diagonal_ewc[sub_task].append(train_test_model.test_classifier(model=block_diagonal_ewc.model,
                                                                     data_loader=test_loader[sub_task],
                                                                     device=device))        
        if not task == num_task - 1: 
            block_diagonal_ewc.consolidate(train_loader[task])
    loss_block_diagonal_ewc_list.append(loss_block_diagonal_ewc)
    acc_block_diagonal_ewc_list.append(acc_block_diagonal_ewc)

    #%% plot block-diagonal EWC
    
    fig, ax=plt.subplots(1,2,figsize=(10,5))
    for t, v in loss_block_diagonal_ewc.items():
        ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
    ax[0].set_xlabel('Epochs',fontsize=14)
    ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
    for t, v in acc_block_diagonal_ewc.items():
        ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Epochs',fontsize=14)
    ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
    fig.suptitle('Using block-diagonal on EWC',fontsize=18)
    plt.show()
    
    #%% compare different methods

    fig, ax=plt.subplots(1,1,figsize=(5,5))
    ax.plot(list(range(0 * epochs, num_task * epochs)), acc_full_ewc[0],'m',linewidth=3)
    ax.plot(list(range(1 * epochs, num_task * epochs)), acc_full_ewc[1],'m:',linewidth=3)
    ax.plot(list(range(0 * epochs, num_task * epochs)), acc_block_diagonal_ewc[0],'c',linewidth=3)
    ax.plot(list(range(1 * epochs, num_task * epochs)), acc_block_diagonal_ewc[1],'c:',linewidth=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epochs',fontsize=14)
    ax.legend(['Full IM, task 0','Full IM, task 1','Block-diagonal, task 0','Block-diagonal, task 1'],fontsize=14)
    ax.set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
    # fig.suptitle('Using block-diagonal on EWC',fontsize=14)
    plt.show()
    
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
acc_full_ewc = get_mean_acc_on_iterations(acc_full_ewc_list)
acc_block_diagonal_ewc = get_mean_acc_on_iterations(acc_block_diagonal_ewc_list)

fig, ax=plt.subplots(1,1,figsize=(5,5))
ax.plot(list(range(0 * epochs, num_task * epochs)), acc_full_ewc[0],'m',linewidth=3)
ax.plot(list(range(1 * epochs, num_task * epochs)), acc_full_ewc[1],'m:',linewidth=3)
ax.plot(list(range(0 * epochs, num_task * epochs)), acc_block_diagonal_ewc[0],'c',linewidth=3)
ax.plot(list(range(1 * epochs, num_task * epochs)), acc_block_diagonal_ewc[1],'c:',linewidth=3)
ax.set_ylim(0, 1)
ax.set_xlabel('Epochs',fontsize=14)
ax.legend(['Full IM, task 0','Full IM, task 1','Block-diagonal, task 0','Block-diagonal, task 1'],fontsize=14)
ax.set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
# fig.suptitle('Using block-diagonal on EWC',fontsize=14)
plt.show()

plt.figure(figsize=(10,5))
for t in range(num_task):
    if t%2:
        c='b'
    else:
        c='r'
    plt.axvspan(t*epochs, (t+1)*epochs, facecolor=c, alpha=0.1)
plt.plot(get_mean_acc(acc_full_ewc,epochs),linewidth=3)
plt.plot(get_mean_acc(acc_block_diagonal_ewc,epochs),linewidth=3)
plt.xticks(fontsize=18),plt.yticks(fontsize=18)
plt.legend(['Full EWC','KFAC EWC'],fontsize=18)
plt.title('Preliminary results for CoreL2M',fontsize=18)
plt.figtext(0.5, 0.01, 'Fig. 1: %d tasks experiment on permuted MNIST'%(num_task), wrap=True, horizontalalignment='center', fontsize=18)
plt.show()

