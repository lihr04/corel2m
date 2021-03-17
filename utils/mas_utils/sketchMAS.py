from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

class SketchMAS():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_bucket=50):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
            
        self.LARGEPRIME = 2**61-1
        
        self.device = device
        self.alpha = alpha
        self.model = model.to(self.device)
        self.n_bucket = n_bucket

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        
        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self._jacobian_matrices=torch.zeros((n_bucket,d)).to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)
    
    def calculate_jacobian(self, dataloader: torch.utils.data.DataLoader, labels=None):
        self.model.eval()
        
        jacobian_matrices = torch.zeros_like(self._jacobian_matrices).to(self.device)
        
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

        # computing sketched importance matrix

        loss = list()
        for input,_ in dataloader:
            self.model.zero_grad() # Zero the gradients
            output = self.model(input.to(self.device)) # Get output
            temp_loss = (torch.softmax(output,1)**2).sum(1) #Define loss
            loss.append(temp_loss)         
        loss=torch.cat(loss)
        
        loss_sketch = torch.matmul(sketch, loss)
        
        for r in range(self.n_bucket):
            self.model.zero_grad() # Zero the gradients
            loss_sketch[r].backward(retain_graph=True) #Get gradient
            ### Update the temporary precision matrix
#             for n, p in self.model.named_parameters():
#                 self._jacobian_matrices[n].data[r] += (1-self.alpha) / math.sqrt(n_data) * p.grad.data 
            jacobian=[]
            for n, p in self.model.named_parameters():
                jacobian.append(p.grad.data.view(-1))
            jacobian=torch.cat(jacobian)
            jacobian_matrices[r,:] += jacobian
        jacobian_matrices=jacobian_matrices/math.sqrt(float(len(dataloader.dataset)))
        return jacobian_matrices

    def calculate_approximation(self, dataloader: torch.utils.data.DataLoader, labels=None):
        jacobian_matrices = self.calculate_jacobian(dataloader)
        return torch.matmul(jacobian_matrices.t(), jacobian_matrices)

    def consolidate(self, dataloader: torch.utils.data.DataLoader, labels=None):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''
        
        self._jacobian_matrices = math.sqrt(self.alpha) * self._jacobian_matrices + \
                                  math.sqrt(1 - self.alpha) * self.calculate_jacobian(dataloader)

        for n, p in self.model.named_parameters():
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        dtheta = []
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta = torch.cat(dtheta)
        loss = torch.sum(torch.matmul(self._jacobian_matrices, dtheta) ** 2)
        return loss
    
    def grad_penalty(self, model:nn.Module):
        ''' Generate the gradient of sketched MAS penalty.
            This function receives the current model with its weights and its , and calculates
            the the gradient of sketched MAS penalty on the loss.
        '''
        dtheta = []
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta = torch.cat(dtheta)
        grad = torch.matmul(self._jacobian_matrices.t(), torch.matmul(self._jacobian_matrices, dtheta))
        return grad

# class SketchMAS():
#     def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_bucket=10):
#         """ OnlineEWC is the class for implementing the online EWC method.
#             Inputs:
#                 model : a Pytorch NN model
#                 device (string): the device to run the model on
#                 alpha (in [0,1) ): The online learning hyper-parameter
#         """
#         self.LARGEPRIME = 2**61-1
        
#         self.device=device
#         self.alpha=alpha
#         self.model = model.to(self.device)
#         self.n_bucket = n_bucket

#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self._means = {}

#         self._jacobian_matrices = {}

#         for n, p in deepcopy(self.params).items():
#             p.data.zero_()
#             self._jacobian_matrices[n] = torch.stack([p.data for i in range(n_bucket)]).to(self.device)

#         for n, p in deepcopy(self.params).items():
#             self._means[n] = p.data.to(self.device)


#     def consolidate(self,dataloader):
#         ''' Consolidate
#          This function receives a dataloader, it then calculates and updates the
#          Fisher Information Matrix (FIM) to preserve the max log-likelihood for
#          the data in dataloader.
#          input:
#             dataloader : A Pytorch dataloader containing data from the task to be consolidated
#         '''
#         # Here is the online part of the EWC. Instead of saving a precision Matrix
#         # for each task, we take a running average of them. This is described in
#         # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.
        
#         for n, p in self.model.named_parameters():
#             self._jacobian_matrices[n] *= self.alpha

#         # Set the model in the evaluation mode
#         self.model.eval()
        
#         # initialize hashing functions for each row:
#         # 2 random numbers for bucket hashes + 4 random numbers for
#         # sign hashes

#         # do all these computations on the CPU
#         hashes = torch.randint(0, self.LARGEPRIME, (6,), dtype=torch.int64, device="cpu")
        
#         # tokens are the indices of the vector entries
#         n_data = len(dataloader.dataset)
#         indices = torch.arange(n_data, dtype=torch.int64, device="cpu")
        
#         # computing sign hashes (4 wise independence)
#         h1 = hashes[2]
#         h2 = hashes[3]
#         h3 = hashes[4]
#         h4 = hashes[5]
#         signs = (((h1 * indices + h2) * indices + h3) * indices + h4)
#         signs = ((signs % self.LARGEPRIME % 2) * 2 - 1).float()
#         signs = signs.to(self.device)

#         # computing bucket hashes (2-wise independence)
#         h1 = hashes[0]
#         h2 = hashes[1]
#         buckets = ((h1 * indices) + h2) % self.LARGEPRIME % self.n_bucket
#         buckets = buckets.to(self.device)
        
#         # computing sketch matrix
#         sketch = torch.zeros(self.n_bucket, n_data).to(self.device)
#         sketch[buckets, indices] = signs

#         # The diagonal values of the FIM are essentially average  of the norm of
#         # the gradients of the negative log likelihood loss.

#         # Note that EWC requires labels (so it can only handle the supervised case)

#         loss = list()
#         for input,_ in dataloader:
#             self.model.zero_grad() # Zero the gradients
#             output = self.model(input.to(self.device)) # Get output
#             temp_loss = (torch.softmax(output,1)**2).sum(1) #Define loss
#             loss.append(temp_loss)         
#         loss=torch.cat(loss)
        
#         loss_sketch = torch.matmul(sketch, loss)
        
#         for r in range(self.n_bucket):
#             self.model.zero_grad() # Zero the gradients
#             loss_sketch[r].backward(retain_graph=True) #Get gradient
#             ### Update the temporary precision matrix
#             for n, p in self.model.named_parameters():
#                 self._jacobian_matrices[n].data[r] += (1-self.alpha) / math.sqrt(n_data) * p.grad.data

#         # Here is the online part of the EWC. Instead of saving a precision Matrix
#         # for each task, we take a running average of them. This is described in
#         # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.

#         for n, p in self.model.named_parameters():
#             # Update the means
#             self._means[n] = deepcopy(p.data).to(self.device)


#     def penalty(self, model: nn.Module):
#         ''' Generate the online EWC penalty.
#             This function receives the current model with its weights, and calculates
#             the online EWC loss.
#         '''
#         vector = torch.zeros(self.n_bucket).to(self.device)
#         for n, p in model.named_parameters():
#             vector += torch.sum((self._jacobian_matrices[n] * (p - self._means[n])).view(self.n_bucket, -1), dim=1)
#         loss = torch.sum(vector ** 2)
#         return loss
