from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class OnlineEWC():
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

        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[n] = p.data.to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)


    def consolidate(self,dataloader,labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
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

        # The diagonal values of the FIM are essentially average  of the norm of
        # the gradients of the negative log likelihood loss.

        # Note that EWC requires labels (so it can only handle the supervised case)

        for input,label in dataloader:
            self.model.zero_grad() # Zero the gradients
            input,label = input.to(self.device),label.to(self.device)# Get Inout
            if labels is None:
                output = self.model(input).view(1, -1) #Get output (Peculiar)
            else:
                output = self.model(input)[:,labels].view(1,-1)                                
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label) #Define loss
            loss.backward()#Get gradients

            ### Update the temporary precision matrix
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 /float(len(dataloader))

        # Here is the online part of the EWC. Instead of saving a precision Matrix
        # for each task, we take a running average of them. This is described in
        # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.

        for n, p in self.model.named_parameters():
            # Update the precision matrix
            self._precision_matrices[n]=self.alpha*self._precision_matrices[n]+(1-self.alpha)*precision_matrices[n]
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


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
