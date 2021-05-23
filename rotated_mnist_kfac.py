# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

from utils import mlp,train_test_model
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.ewc_utils.sketchEWC import SketchEWC
from utils.ewc_utils.kfacEWC import KfacEWC
from utils.ewc_utils.ToyExampleEWC import FullEWC, LowRankEWC, MinorDiagonalEWC, BlockDiagonalEWC

from data.rotated_MNIST import get_rotated_mnist

import os

results_folder='perm_mnist_test/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

#%% set configurations

epochs = 20
lr = 1e-3
batch_size = 100 
num_task = 2

seed = 0
device='cuda:0'

#%% load dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

per_task_rotation = 18
train_loader, test_loader = get_rotated_mnist(num_task,batch_size,
                                              per_task_rotation=per_task_rotation,
                                              num_workers=0)

fig,ax=plt.subplots(1,num_task,figsize=(num_task*3,3))
for i in range(num_task):
    iter_data=iter(train_loader[i])
    image,label=iter_data.next()
    I=np.reshape(image.data[0,...].numpy(),(28,28))    
    ax[i].imshow(I,cmap='gray')
    ax[i].set_title(label.data[0].numpy())
    ax[i].set_xlabel("Intensity in [%.2f,%.2f]"%(I.min(),I.max()))
plt.show()

#%% model lenet5
# class LeNet5(nn.Module):
#     """
#     Input - 1x32x32
#     Output - 10
#     Adopted (but slightly different) from https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py
#     """
#     def __init__(self):
#         super(LeNet5, self).__init__()

#         self.c1 = nn.Sequential(OrderedDict([
#             ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)),
#             ('relu1', nn.ReLU()),
#             ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))
#         self.c2 = nn.Sequential(OrderedDict([
#             ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
#             ('relu2', nn.ReLU()),
#             ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))
#         self.c3 = nn.Sequential(OrderedDict([
#             ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
#             ('relu3', nn.ReLU())
#         ]))
#         self.f4 = nn.Sequential(OrderedDict([
#             ('f4', nn.Linear(120, 84)),
#             ('relu4', nn.ReLU())
#         ]))
#         self.f5 = nn.Sequential(OrderedDict([
#             ('f5', nn.Linear(84, 10)),
#             ('sig5', nn.LogSoftmax(dim=-1))
#         ]))

#     def forward(self, img):
#         output = self.c1(img)
#         output = self.c2(output)
#         output = self.c3(output)
#         output = output.view(img.size(0), -1)
#         output = self.f4(output)
#         output = self.f5(output)
#         return output
    
#%% our model

class Model(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(784, 256)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(256, 10))
        ]))

    def forward(self, x):
        x = self.net(x.view(x.size(0), -1))
        return x
    
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def reset(self):
        self.net.apply(self.init_weights)

#%% show model

model = Model()
print(model)

d=0
for n, p in model.named_parameters():
    print(n, p.data.view(-1).shape[0])
    d+=p.data.view(-1).shape[0]
print(d)

#%% train plain MLP

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = Model().to(device)
model.reset()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## Perform training
loss, acc = {}, {}
for task in tqdm(range(num_task)):
    loss[task] = []
    acc[task] = []
    for _ in tqdm(range(epochs)):
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        loss[task].append(train_test_model.train_classifier(model=model,
                                                            optimizer=optimizer,
                                                            data_loader=train_loader[task],
                                                            device=device))            
        for sub_task in range(task + 1):
            acc[sub_task].append(train_test_model.test_classifier(model=model,
                                                                 data_loader=test_loader[sub_task],
                                                                 device=device))

#%% plot plain MLP

fig, ax=plt.subplots(1,2,figsize=(10,5))
for t, v in loss.items():
    ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
ax[0].set_xlabel('Epochs',fontsize=14)
ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
for t, v in acc.items():
    ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('Epochs',fontsize=14)
ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
fig.suptitle('Using plain MLP on EWC',fontsize=18)

#%% train diagonal EWC

ewc_alpha=0.25
ewc_importance = 1e8
n_sketch = 50

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = Model().to(device)
model.reset()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

loss_ewc, acc_ewc = {}, {}
for task in tqdm(range(num_task)):
    loss_ewc[task] = []
    acc_ewc[task] = []
    for _ in tqdm(range(epochs)):
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        ewc = OnlineEWC(model,device=device,alpha=ewc_alpha)
        loss_ewc[task].append(train_test_model.onlineEWC_train_classifier(ewc=ewc,
                                                                      optimizer=optimizer,
                                                                      data_loader=train_loader[task],
                                                                      importance=ewc_importance,
                                                                      device=device))            
        for sub_task in range(task + 1):
            acc_ewc[sub_task].append(train_test_model.test_classifier(model=model,
                                                                 data_loader=test_loader[sub_task],
                                                                 device=device))
    
#%% plot diagonal EWC

fig, ax=plt.subplots(1,2,figsize=(10,5))
for t, v in loss_ewc.items():
    ax[0].plot(list(range(t * epochs, (t + 1) * epochs)), v,linewidth=3)
ax[0].set_xlabel('Epochs',fontsize=14)
ax[0].set_title('Training loss for the %d tasks'%(num_task),fontsize=14)
for t, v in acc_ewc.items():
    ax[1].plot(list(range(t * epochs, num_task * epochs)), v,linewidth=3)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('Epochs',fontsize=14)
ax[1].set_title('Testing accuracy for the %d tasks'%(num_task),fontsize=14)
fig.suptitle('Using CountSketch on EWC',fontsize=18)

#%% train full EWC

#%% plot full EWC

#%% train block-diagonal EWC

#%% plot block-diagonal EWC

#%% train sketched EWC

sketch_ewc_alpha=0.25
sketch_ewc_importance = 1e4
n_sketch = 50

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## define a convnet model
model = LeNet5().to(device)

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
plt.show()
    
#%% train kfac EWC

kfac_ewc_alpha=0.25
kfac_ewc_importance = 1e6

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## define a convnet model
model = LeNet5().to(device)

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
    
    ## define a convnet model
    model = Model(input_size,conv_kernel_sizes)
    
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