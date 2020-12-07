__author__ = "Soheil Kolouri"
__copyright__ = "Copyright (C) 2019 Soheil Kolouri"
__license__ = "Public Domain"
__version__ = "1.0"

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


class OfflineSketchEWC():
    def __init__(self, model: nn.Module, device='cuda:0', n_bucket=10):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
        """
        self.LARGEPRIME = 2**61-1
        
        self.device=device
        self.model = model.to(self.device)
        self.n_bucket = n_bucket
        
        self._jacobian_matrices = []
        self._means = []
        

    def consolidate(self,dataloader,labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''
        # Initialize a temporary jacobian matrix
        jacobian_matrices = {}
        for n, p in self.model.named_parameters():
            jacobian_matrices[n] = torch.zeros(self.n_bucket, p.shape[0], p.shape[1]).to(self.device)

        # Set the model in the evaluation mode
        self.model.eval()
        
        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes

        # do all these computations on the CPU
        hashes = torch.randint(0, self.LARGEPRIME, (6,), dtype=torch.int64, device="cpu")
        
        # tokens are the indices of the vector entries
        n_data = len(dataloader.dataset)
        indices = torch.arange(n_data, dtype=torch.int64, device="cpu")
        
        # computing sign hashes (4 wise independence)
        h1 = hashes[2]
        h2 = hashes[3]
        h3 = hashes[4]
        h4 = hashes[5]
        signs = (((h1 * indices + h2) * indices + h3) * indices + h4)
        signs = ((signs % self.LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)

        # computing bucket hashes (2-wise independence)
        h1 = hashes[0]
        h2 = hashes[1]
        buckets = ((h1 * indices) + h2) % self.LARGEPRIME % self.n_bucket
        buckets = buckets.to(self.device)
        
        # computing sketch matrix
        sketch = torch.zeros(self.n_bucket, n_data).to(self.device)
        sketch[buckets, indices] = signs

        # The diagonal values of the FIM are essentially average  of the norm of
        # the gradients of the negative log likelihood loss.

        # Note that EWC requires labels (so it can only handle the supervised case)

        loss = list()
        for input,label in dataloader:
            self.model.zero_grad() # Zero the gradients
            input,label = input.to(self.device),label.to(self.device)# Get Inout
            if labels is None:
                output = self.model(input)
            else:
                output = self.model(input)[:,labels]
                label = label[labels]
            temp_loss = nn.CrossEntropyLoss(reduction='none')(output, label) #Define loss
            loss.append(temp_loss)         
        loss=torch.cat(loss)
        
        loss_sketch = torch.matmul(sketch, loss)
        
        for r in range(self.n_bucket):
            self.model.zero_grad() # Zero the gradients
            loss_sketch[r].backward(retain_graph=True) #Get gradient
            ### Update the temporary precision matrix
            for n, p in self.model.named_parameters():
                jacobian_matrices[n].data += p.grad.data / math.sqrt(n_data)

        # Here is the offline part of the EWC.
        
        means = {}
        for n, p in self.model.named_parameters():
            means[n] = deepcopy(p.data).to(self.device)
        self._jacobian_matrices.append(jacobian_matrices)
        self._means.append(means)


    def penalty(self, model: nn.Module):
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        num_task = len(self._jacobian_matrices)
        for i in range(num_task):
            vector = torch.zeros(self.n_bucket).to(self.device)
            for n, p in model.named_parameters():
                vector += torch.sum(self._jacobian_matrices[i][n] * (p - self._means[i][n]), dim=(1,2))
            loss += torch.sum(vector ** 2) / num_task
        return loss
