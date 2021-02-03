from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

    
class FullEWC():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        self._precision_matrices ={}

        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self.fim=torch.zeros((d,d)).to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def calculate_FIM(self,dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        A=torch.zeros_like(self.fim)
        for i,(imgs,labels) in enumerate(dataloader):
#             self.model.zero_grad() # Zero the gradients
#             imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
#             loss=nn.CrossEntropyLoss()(self.model(imgs),labels)
#             loss.backward()
#             grad=[]
#             for n, p in self.params.items():
#                 grad.append(p.grad.data.view(-1))
#             grad=torch.cat(grad)
#             A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))            
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
            loss_list=nn.CrossEntropyLoss(reduction='none')(self.model(imgs),labels)
            for loss in loss_list:
                self.model.zero_grad() # Zero the gradients
                loss.backward(retain_graph=True)
                grad=[]
                for n, p in self.params.items():
                    grad.append(p.grad.data.view(-1))
                grad=torch.cat(grad)
                A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
        A=A/float(len(dataloader.dataset))
        return A


    def consolidate(self,dataloader,labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        self.fim *= self.alpha
        self.fim += (1-self.alpha)*self.calculate_FIM(dataloader)

        for n, p in self.model.named_parameters():
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss=torch.matmul(dtheta.unsqueeze(0),
                          torch.matmul(self.fim,
                                       dtheta.unsqueeze(1)))
        return loss

class LowRankEWC():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_bucket=10):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_bucket=n_bucket

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self.fim=torch.zeros((d,d)).to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def calculate_FIM(self,dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        A=torch.zeros_like(self.fim)
        for i,(imgs,labels) in enumerate(dataloader):
#             self.model.zero_grad() # Zero the gradients
#             imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
#             loss=nn.CrossEntropyLoss()(self.model(imgs),labels)
#             loss.backward()
#             grad=[]
#             for n, p in self.params.items():
#                 grad.append(p.grad.data.view(-1))
#             grad=torch.cat(grad)
#             A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))            
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
            loss_list=nn.CrossEntropyLoss(reduction='none')(self.model(imgs),labels)
            for loss in loss_list:
                self.model.zero_grad() # Zero the gradients
                loss.backward(retain_graph=True)
                grad=[]
                for n, p in self.params.items():
                    grad.append(p.grad.data.view(-1))
                grad=torch.cat(grad)
                A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
        A=A/float(len(dataloader.dataset))
        return A


    def consolidate(self,dataloader,labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''
        
        u, s, v =torch.svd_lowrank(self.calculate_FIM(dataloader), q=self.n_bucket)
        self.fim *= self.alpha
        self.fim += (1-self.alpha)*torch.mm(torch.mm(u, torch.diag(s)), v.t())

        for n, p in self.model.named_parameters():
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss=torch.matmul(dtheta.unsqueeze(0),
                          torch.matmul(self.fim,
                                       dtheta.unsqueeze(1)))
        return loss

class MinorDiagonalEWC():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_bucket=10):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_bucket=n_bucket

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self.fim=torch.zeros((d,d)).to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def calculate_FIM(self,dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        A=torch.zeros_like(self.fim)
        for i,(imgs,labels) in enumerate(dataloader):
#             self.model.zero_grad() # Zero the gradients
#             imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
#             loss=nn.CrossEntropyLoss()(self.model(imgs),labels)
#             loss.backward()
#             grad=[]
#             for n, p in self.params.items():
#                 grad.append(p.grad.data.view(-1))
#             grad=torch.cat(grad)
#             A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))            
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
            loss_list=nn.CrossEntropyLoss(reduction='none')(self.model(imgs),labels)
            for loss in loss_list:
                self.model.zero_grad() # Zero the gradients
                loss.backward(retain_graph=True)
                grad=[]
                for n, p in self.params.items():
                    grad.append(p.grad.data.view(-1))
                grad=torch.cat(grad)
                A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
        A=A/float(len(dataloader.dataset))
        return A


    def consolidate(self,dataloader,labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        fim = self.calculate_FIM(dataloader)
        deviation = int((self.n_bucket-1)/2)
        fim = torch.triu(fim, diagonal=(-deviation)) - torch.triu(fim, diagonal=(self.n_bucket-deviation))
        self.fim *= self.alpha
        self.fim += (1-self.alpha)*fim

        for n, p in self.model.named_parameters():
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss=torch.matmul(dtheta.unsqueeze(0),
                          torch.matmul(self.fim,
                                       dtheta.unsqueeze(1)))
        return loss