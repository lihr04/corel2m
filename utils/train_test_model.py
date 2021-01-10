__author__ = "Soheil Kolouri"
__copyright__ = "Copyright (C) 2019 Soheil Kolouri"
__license__ = "Public Domain"
__version__ = "1.0"

import numpy as np
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from utils.ewc_utils.onlineEWC import OnlineEWC
from utils.scp_utils.scp import SCP
from utils.mas_utils.mas import MAS
from utils.csh_utils.countSketch import CountSketch

def train_classifier(model: nn.Module, optimizer: torch.optim,
                     data_loader: torch.utils.data.DataLoader,device='cuda:0',labels=None):
    ''' train_classifier
    Performs an epoch of training on an input model.
    Inputs:
        model: the NN model
        optimizer: the optimizer to be used
        data_loader: the training data
        device (str): the device to run optimization on [default 'cuda:0']
    Outputs:
        average epoch loss
    '''
    model.to(device)
    model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        if labels is None:
            output = model(img)
        else:
            try: 
                output = model.classify(img)[:,labels]            
            except:
                output = model(img)[:,labels]            
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
            
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_loader))

def regularized_train_classifier(regularizer, optimizer: torch.optim,
                               data_loader: torch.utils.data.DataLoader,
                               importance: float, device='cuda:0',labels=None):
    ''' regularized_train
    This function performs an epoch of training with regularized loss.
    Inputs:
        regularizer: the NN model with importance parameters
        optimizer: the optimizer to be used
        data_loader: the training data
        device (str): the device to run optimization on [default 'cuda:0']
        importance (float): the regularizer coefficient
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    regularizer.model.to(device)
    regularizer.model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        
        if labels is None:
            output = regularizer.model(img)
        else:
            try:
                output = regularizer.model.classifier(img)[:,labels]
            except:
                output = regularizer.model(img)[:,labels]
                
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)           
            
        loss1 = criterion(output, target)
        loss2 = regularizer.penalty(regularizer.model)        
        loss = loss1+importance*loss2
        epoch_loss += loss1.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_loader))

def onlineEWC_train_classifier(ewc: OnlineEWC, optimizer: torch.optim,
                               data_loader: torch.utils.data.DataLoader,
                               importance: float, device='cuda:0',labels=None):
    ''' ewc_train
    This function performs an epoch of training with EWC loss.
    Inputs:
        ewc: the NN model with EWC importance parameters
        optimizer: the optimizer to be used
        data_loader: the training data
        device (str): the device to run optimization on [default 'cuda:0']
        importance (float): the EWC regularizer coefficient
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    ewc.model.to(device)
    ewc.model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        
        if labels is None:
            output = ewc.model(img)
        else:
            try:
                output = ewc.model.classifier(img)[:,labels]
            except:
                output = ewc.model(img)[:,labels]
                
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)           
            
        loss1 = criterion(output, target)
        loss2 = ewc.penalty(ewc.model)        
        loss = loss1+importance*loss2
        epoch_loss += loss1.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_loader))


def mas_train_classifier(mas: MAS, optimizer: torch.optim,
                               data_loader: torch.utils.data.DataLoader,
                               importance: float, device='cuda:0',labels=None):
    ''' ewc_train
    This function performs an epoch of training with EWC loss.
    Inputs:
        ewc: the NN model with EWC importance parameters
        optimizer: the optimizer to be used
        data_loader: the training data
        device (str): the device to run optimization on [default 'cuda:0']
        importance (float): the EWC regularizer coefficient
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    mas.model.to(device)
    mas.model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        
        if labels is None:
            output = mas.model(img)
        else:
            try:
                output = mas.model.classifier(img)[:,labels]
            except:
                output = mas.model(img)[:,labels]
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
               
        loss1 = criterion(output, target)
        loss2 = mas.penalty(mas.model)
        loss = loss1+importance*loss2
        epoch_loss += loss1.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_loader))

def scp_train_classifier(scp: SCP, optimizer: torch.optim,
                               data_loader: torch.utils.data.DataLoader,
                               importance: float, device='cuda:0',labels=None):
    ''' scp_train
    This function performs an epoch of training with EWC loss.
    Inputs:
        scp: the NN model with SCP importance parameters
        optimizer: the optimizer to be used
        data_loader: the training data
        importance (float): the EWC regularizer coefficient
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    scp.model.to(device)
    scp.model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        if labels is None:
            output = scp.model(img)
        else:
            try:
                output = scp.model.classifier(img)[:,labels]
            except:
                output = scp.model(img)[:,labels]
                
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
            
        loss1 = criterion(output, target)
        loss2 = scp.penalty(scp.model)
        loss = loss1+importance*loss2
        epoch_loss += loss1.item()
        loss.backward()
        optimizer.step()
    return epoch_loss /float(len(data_loader))

def csh_train_classifier(csh: CountSketch, optimizer: torch.optim,
                               data_loader: torch.utils.data.DataLoader,
                               importance: float, device='cuda:0',labels=None):
    ''' csh_train
    This function performs an epoch of training with EWC loss.
    Inputs:
        csh: the NN model with CountSketch importance parameters
        optimizer: the optimizer to be used
        data_loader: the training data
        importance (float): the EWC regularizer coefficient
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    csh.model.to(device)
    csh.model.train()
    criterion=nn.CrossEntropyLoss()
    epoch_loss = 0
    for img, target in data_loader:
        img, target = img.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        if labels is None:
            output = csh.model(img)
        else:
            try:
                output = csh.model.classifier(img)[:,labels]
            except:
                output = csh.model(img)[:,labels]
                
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
            
        loss1 = criterion(output, target)
        loss2 = csh.penalty(csh.model)
        loss = loss1+importance*loss2
        epoch_loss += loss1.item()
        loss.backward()
        optimizer.step()
    return epoch_loss /float(len(data_loader))

def test_classifier(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                    device='cuda:0',labels=None):
    ''' test_classifier
    This function test the trained model and returns accuracy
    Inputs:
        model: the NN model
        data_loader: The validation or test DataLoader
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    model.to(device)
    model.eval()
    correct = 0
    for img, target  in data_loader:
        img, target  = img.to(device), target.type(torch.LongTensor).to(device)
        if labels is None:
            output = model(img)
        else:
            try:
                output = model.classifier(img)[:,labels]
            except:
                output = model(img)[:,labels]
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
        correct += ((output.argmax(dim=1) == target).sum()).item()
    return correct/float(len(data_loader.dataset))
