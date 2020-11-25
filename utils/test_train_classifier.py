import torch
import torch.utils
import torch.nn
import torch.optim


def train_classifier(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                    device='cuda:0',criterion=torch.nn.CrossEntropyLoss(),optimizer=torch.optim):
    ''' test_classifier
    This function test the trained model and returns accuracy
    Inputs:
        model: the NN model
        data_loader: The validation or test DataLoader
        device (str): the device to run optimization on [default 'cuda:0']
    '''
    model.to(device)
    model.train()
    epoch_loss = 0.
    for img, target  in data_loader:
        img, target  = img.to(device), target.type(torch.LongTensor).to(device)        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()        
    return epoch_loss/float(len(data_loader))



def test_classifier(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
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
        else:# This branch is for multi-head experiments
            try:
                output = model.classifier(img)[:,labels]
            except:
                output = model(img)[:,labels]
            for i,l in enumerate(labels):
                target.data[target.data==l]=i
            target.type(torch.LongTensor).to(device)
        correct += ((output.argmax(dim=1) == target).sum()).item()
    return correct/float(len(data_loader.dataset))



