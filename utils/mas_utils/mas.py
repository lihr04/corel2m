__author__ = "Soheil Kolouri"
__copyright__ = "Copyright (C) 2019 Soheil Kolouri"
__license__ = "Public Domain"
__version__ = "1.0"

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class MAS(object):
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5,n_slices=50):
        """ MAS is the class for implementing the Memory-Aware-Synapses

                * Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M. and
                Tuytelaars, T., 2018. Memory aware synapses: Learning what (not)
                to forget. In Proceedings of the European Conference on Computer
                Vision (ECCV) (pp. 139-154).

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

        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[n] = p.data.to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def consolidate(self,dataloader):
        ''' Consolidate
         This function receives a dataloader, and calculates and updates the Omega
         Matrix (only the diagonal values) described in the paper.

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

        for input,label in dataloader:
            self.model.zero_grad() # Zero the gradients
            input,label = input.to(self.device),label.to(self.device)# Get Inout
            output = self.model(input).view(1, -1) #Get output (Peculiar)
            loss = ((torch.softmax(output,1)**2).sum(1)).mean()
            loss.backward()#Get gradients
            ### Update the temporary precision matrix
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 /float(len(dataloader.dataset))

        # Here we follow a similar approach as in the online EWC framework presented in
        # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.
        for n, p in self.model.named_parameters():
            # Update the precision matrix
            self._precision_matrices[n]=self.alpha*self._precision_matrices[n]+(1-self.alpha)*precision_matrices[n]
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    # Now we need to generate the the EWC updated loss
    def penalty(self, model: nn.Module):
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        for n, p in model.named_parameters():
            _loss = 0.5 * self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
