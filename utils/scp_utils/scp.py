__author__ = "Soheil Kolouri"
__copyright__ = "Copyright (C) 2019 Soheil Kolouri"
__license__ = "Public Domain"
__version__ = "1.0"

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class SCP(object):
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5,n_slices=50):
        """ SCP is the class for implementing the Sliced-Cramer-Regularizer
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
                n_slices (int): Number of slices to use for Monte-Carlo approximation
                        of the integration of unit ball.
        """
        self.n_slices=n_slices
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        self._precision_matrices ={}

        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[n] = p.data.to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def consolidate(self,dataloader):
        ''' Consolidate
         This function receives a dataloader, and calculates and updates the Gamma
         Matrix (only the diagonal values) described in the paper. Gamma then preserves
         the distribution of the output of a network.

         To Do: Add the capability to preserve distributions at any layer

         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        # Initialize a temporary precision matrix
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(self.device)

        # Set the model in the evaluation mode
        self.model.eval()

        # Get network outputs
        z=list()
        for x,_ in dataloader:
            z.append(self.model(x.to(self.device)))            
        z=torch.cat(z)
        
        zmean=z.mean(0)
        K= zmean.shape[0]

        # Randomly sample $\mathbb{S}^{K-1}$.
        xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((K,self.n_slices))])

        for l in range(self.n_slices):
            self.model.zero_grad()
            out=torch.matmul(zmean,xi[:,l])
            out.backward(retain_graph=True)

            ### Update the temporary precision matrix
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / float(len(dataloader.dataset)*self.n_slices)

        # Here we follow a similar approach as in the online EWC framework presented in
        # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.
        for n, p in self.model.named_parameters():
            # Update the precision matrix
            self._precision_matrices[n]=self.alpha*self._precision_matrices[n]+(1-self.alpha)*precision_matrices[n]
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    # Now we need to generate the the EWC updated loss
    def penalty(self, model: nn.Module):
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
