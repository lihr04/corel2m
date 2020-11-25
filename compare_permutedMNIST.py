import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from utils import mlp,train_test_model
from utils.scp_utils.scp import SCP
from utils.scp_utils.scp_v2 import SCP as SCP_v2
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.mas_utils.mas import MAS
from data.permuted_MNIST import get_permuted_mnist
import pickle
import os
import getopt, sys

results_folder='perm_mnist_10Run/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

def main(iteration,gpu):

    iteration = int(iteration)
    epochs = 20
    lr = 1e-4
    batch_size = 100
    input_size = 784
    hidden_sizes = [1024,512,256]
    output_size = 10

    num_task = 10
    n_slices = 100

    activation = 'ReLU'
    device = 'cuda:'+gpu

    ewc_alpha = 0.25
    scp_alpha = 0.25
    mas_alpha = 0.25

    # Grid Searched to Provide Best Performance for each algorithm
    scp_importance = 1e+6
    scp_importance_v2 = 1e+2
    mas_importance = 2e+3
    ewc_importance = 1e+3

    # Load Dataset
    train_loader, test_loader = get_permuted_mnist(num_task,batch_size)

    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  device=device).to(device)

    ## Perform training
    loss, acc = {}, {}
    for task in tqdm(range(num_task)):
        loss[task] = []
        acc[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss[task].append(train_test_model.train_classifier(model=model,
                                                                optimizer=optimizer,
                                                                data_loader=train_loader[task],
                                                                device=device))
            for sub_task in range(task + 1):
                acc[sub_task].append(train_test_model.test_classifier(model=model,
                                                                     data_loader=test_loader[sub_task],
                                                                     device=device))
    del model

    model=mlp.MLP(input_size=input_size,output_size=output_size,
              hidden_size=hidden_sizes,activation=activation,
              device=device).to(device)

    ewc= OnlineEWC(model,device=device,alpha=ewc_alpha)

    ## performing training
    loss_ewc, acc_ewc = {}, {}
    for task in tqdm(range(num_task)):
        loss_ewc[task] = []
        acc_ewc[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_ewc[task].append(train_test_model.onlineEWC_train_classifier(ewc=ewc,
                                                                          optimizer=optimizer,
                                                                          data_loader=train_loader[task],
                                                                          importance=ewc_importance,
                                                                          device=device))
            for sub_task in range(task + 1):
                acc_ewc[sub_task].append(train_test_model.test_classifier(model=ewc.model,
                                                                     data_loader=test_loader[sub_task],
                                                                     device=device))
        ewc.consolidate(train_loader[task])

    del model

    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  device=device).to(device)
    mas= MAS(model,device=device,alpha=mas_alpha)
    ## performing training
    loss_mas, acc_mas = {}, {}
    for task in tqdm(range(num_task)):
        loss_mas[task] = []
        acc_mas[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_mas[task].append(train_test_model.mas_train_classifier(mas=mas,
                                                                        optimizer=optimizer,
                                                                        data_loader=train_loader[task],
                                                                        importance=mas_importance,
                                                                        device=device))
            for sub_task in range(task + 1):
                acc_mas[sub_task].append(train_test_model.test_classifier(model=mas.model,
                                                                        data_loader=test_loader[sub_task],
                                                                         device=device))
        mas.consolidate(train_loader[task])

    del model

    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  device=device).to(device)
    scp= SCP(model,device=device,alpha=scp_alpha,n_slices=n_slices)
    ## performing training
    loss_scp, acc_scp = {}, {}
    for task in tqdm(range(num_task)):
        loss_scp[task] = []
        acc_scp[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_scp[task].append(train_test_model.scp_train_classifier(scp=scp,
                                                                        optimizer=optimizer,
                                                                        data_loader=train_loader[task],
                                                                        importance=scp_importance,
                                                                        device=device))
            for sub_task in range(task + 1):
                acc_scp[sub_task].append(train_test_model.test_classifier(model=scp.model,
                                                                         data_loader=test_loader[sub_task],
                                                                         device=device))
        scp.consolidate(train_loader[task])

    del model
    ## define a MLP model
    model=mlp.MLP(input_size=input_size,output_size=output_size,
                  hidden_size=hidden_sizes,activation=activation,
                  device=device).to(device)
    scp_v2= SCP_v2(model,device=device,alpha=scp_alpha,n_slices=1)
    ## performing training
    loss_scp_v2, acc_scp_v2 = {}, {}
    for task in tqdm(range(num_task)):
        loss_scp_v2[task] = []
        acc_scp_v2[task] = []
        for _ in tqdm(range(epochs)):
            optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
            loss_scp_v2[task].append(train_test_model.scp_train_classifier(scp=scp_v2,
                                                                        optimizer=optimizer,
                                                                        data_loader=train_loader[task],
                                                                        importance=scp_importance_v2,
                                                                        device=device))
            for sub_task in range(task + 1):
                acc_scp_v2[sub_task].append(train_test_model.test_classifier(model=scp_v2.model,
                                                                         data_loader=test_loader[sub_task],
                                                                         device=device))
        scp_v2.consolidate(train_loader[task])

    del model

    pickle.dump([[loss,loss_ewc,loss_mas,loss_scp,loss_scp_v2],
                 [acc,acc_ewc,acc_mas,acc_scp,acc_scp_v2]],
                 open(results_folder+'experiment_%d.pkl'%iteration,'wb'))
    
if __name__ == "__main__":
    argumentList = sys.argv[1:]
    optlist, args = getopt.getopt(argumentList, 'p:g:')
    if len(optlist)>0:
        iteration=optlist[0][1]
        gpu=optlist[1][1]
        main(iteration,gpu)
