__author__ = "Haoran Li"
__copyright__ = "Copyright (C) 2020 Haoran Li"
__license__ = "Public Domain"
__version__ = "1.0"

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

class SketchSCP():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=0, n_slices=50, n_bucket=10):
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
        self.n_slices = n_slices
        self.n_bucket = n_bucket

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        
        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self._jacobian_matrices=torch.zeros((n_bucket,d)).to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)
            
    def calculate_sketch(self, n_bucket, n_data, LARGEPRIME=2**61-1, device='cuda:0'):
        """
        Calculate CountSketch matrix.

        Parameters
        ----------
        n_bucket : int
            number of buckets, aka size of sketch.
        n_data : int
            number of data inputs to be sketched.
        LARGEPRIME : int, optional
            large prime for sketch computing. The default is 2**61-1.
        device : string, optional
            device where the output sketch is stored. The default is 'cuda:0'.

        Returns
        -------
        sketch : torch.Tensor
            output sketch with size (n_bucket, n_data).

        """
        
        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes

        # do all these computations on the CPU
        hashes = torch.randint(0, LARGEPRIME, (6,), dtype=torch.int64, device="cpu")
        
        # tokens are the indices of the vector entries
        indices = torch.arange(n_data, dtype=torch.int64, device="cpu")
        
        # computing sign hashes (4 wise independence)
        h1 = hashes[2]
        h2 = hashes[3]
        h3 = hashes[4]
        h4 = hashes[5]
        signs = (((h1 * indices + h2) * indices + h3) * indices + h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(device)

        # computing bucket hashes (2-wise independence)
        h1 = hashes[0]
        h2 = hashes[1]
        buckets = ((h1 * indices) + h2) % LARGEPRIME % n_bucket
        buckets = buckets.to(device)
        
        # computing sketch matrix
        sketch = torch.zeros(n_bucket, n_data).to(device)
        sketch[buckets, indices] = signs
        
        return sketch
    
    def calculate_jacobian(self, dataloader: torch.utils.data.DataLoader, labels=None):
        self.model.eval()
        
        jacobian_matrices = torch.zeros_like(self._jacobian_matrices).to(self.device)
        
        n_data = len(dataloader.dataset)
        
        # Get network output
        output=list()
        for x,_ in dataloader:
            output.append(self.model(x.to(self.device)))            
        output=torch.cat(output)
        K= output.shape[1]
        
        sketch = self.calculate_sketch(self.n_bucket, n_data, self.LARGEPRIME, self.device)
        # Randomly sample $\mathbb{S}^{K-1}$.
        xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((self.n_slices,K))]).t()
        
        output_sketch = torch.matmul(torch.matmul(sketch, output), xi)
        
        for l in range(self.n_slices):
            sketch = self.calculate_sketch(self.n_bucket, n_data, self.LARGEPRIME, self.device)
            output_sketch=torch.matmul(torch.matmul(sketch, output),xi[:,l])
            for r in range(self.n_bucket):
                self.model.zero_grad() # Zero the gradients
                output_sketch[r].backward(retain_graph=True) # Get gradients
                ### Update the temporary precision matrix
                # for n, p in self.model.named_parameters():
                #     self._jacobian_matrices[n].data[l,k] += (1-self.alpha) / math.sqrt(n_data) * p.grad.data
                jacobian = []
                for n, p in self.model.named_parameters():
                    jacobian.append(p.grad.data.view(-1))
                jacobian=torch.cat(jacobian)
                jacobian_matrices[r,:] += jacobian
        jacobian_matrices = jacobian_matrices / math.sqrt(float(len(dataloader.dataset)*self.n_slices))
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
        ''' Generate the online EWC penalty.
            This function receives the current model with its weights, and calculates
            the online EWC loss.
        '''
        loss = 0
        dtheta = []
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta = torch.cat(dtheta)
        loss = 0.5 * torch.sum(torch.matmul(self._jacobian_matrices, dtheta) ** 2)
        return loss
    
    def grad_penalty(self, model:nn.Module):
        ''' Generate the gradient of sketched EWC penalty.
            This function receives the current model with its weights and its , and calculates
            the the gradient of sketched EWC penalty on the loss.
        '''
        dtheta = []
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta = torch.cat(dtheta)
        grad = torch.matmul(self._jacobian_matrices.t(), torch.matmul(self._jacobian_matrices, dtheta))
        return grad

# class SketchSCP_old(object):
#     def __init__(self, model: nn.Module, device='cuda:0', alpha=0, n_slices=50, n_bucket=10):
#         """ CountSketch is the class for implementing the Count Sketch

#             Inputs:
#                 model : a Pytorch NN model
#                 output_size: model output dimension
#                 device (string): the device to run the model on
#                 alpha (in [0,1) ): The online learning hyper-parameter
#                 n_slices (int): Number of buckets in the hash function.

#         """
#         self.LARGEPRIME = 2**61-1
        
#         self.device=device
#         self.alpha=alpha
#         self.model = model.to(self.device)
#         self.n_slices = n_slices
#         self.n_bucket = n_bucket

#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self._means = {}

#         self._jacobian_matrices = {}

#         for n, p in deepcopy(self.params).items():
#             p.data.zero_()
#             self._jacobian_matrices[n] = torch.stack([torch.stack([p.data for i in range(n_slices)])
#                                                       for i in range(n_bucket)]).to(self.device)

#         for n, p in deepcopy(self.params).items():
#             self._means[n] = p.data.to(self.device)


#     def consolidate(self,dataloader):
#         ''' Consolidate
#           This function receives a dataloader, and calculates and updates the Omega
#           Matrix (only the diagonal values) described in the paper.

#           input:
#             dataloader : A Pytorch dataloader containing data from the task to be consolidated
#         '''
        
# #         tic = time.time()

#         # Set the model in the evaluation mode
#         self.model.eval()

#         # Here we follow a similar approach as in the online EWC framework presented in
#         # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.
#         for n, p in self.model.named_parameters():
#             # Update the precision matrix
#             self._jacobian_matrices[n] *= self.alpha
#             # Update the means
#             self._means[n] = deepcopy(p.data).to(self.device)

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
        
#         # Get network output
#         output=list()
#         for x,_ in dataloader:
#             output.append(self.model(x.to(self.device)))            
#         output=torch.cat(output)
#         K= output.shape[1]

#         # Randomly sample $\mathbb{S}^{K-1}$.
#         xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((K,self.n_slices))])
        
#         output_sketch = torch.matmul(torch.matmul(sketch, output), xi)
            
# #         toc = time.time()
# #         print(toc-tic)
        
#         for l in range(self.n_bucket):
#             for k in range(self.n_slices):
# #                 tic = time.time()
#                 self.model.zero_grad() # Zero the gradients
#                 output_sketch[l,k].backward(retain_graph=True) # Get gradients
#                 ### Update the temporary precision matrix
#                 for n, p in self.model.named_parameters():
#                     self._jacobian_matrices[n].data[l,k] += (1-self.alpha) / math.sqrt(n_data) * p.grad.data
# #                 toc = time.time()
# #                 print(toc-tic)


#     # Now we need to generate the the EWC updated loss
#     def penalty(self, model: nn.Module):
#         ''' Generate the online MAS penalty.
#             This function receives the current model with its weights, and calculates
#             the online MAS loss.
#         '''
#         matrix = torch.zeros(self.n_bucket, self.n_slices).to(self.device)
#         for n, p in model.named_parameters():
#             matrix += 0.5 * torch.sum((self._jacobian_matrices[n] * (p - self._means[n])).view(self.n_bucket, self.n_slices, -1), dim=2)
#         loss = torch.sum(matrix ** 2)
#         return loss
    
#     # def grad_penalty(self, model:nn.Module):
#     #     ''' Generate the gradient of sketched SCP penalty.
#     #         This function receives the current model with its weights and its , and calculates
#     #         the the gradient of sketched EWC penalty on the loss.
#     #     '''
#     #     dtheta = []
#     #     for n, p in model.named_parameters():
#     #         dtheta.append((p - self._means[n]).view(-1))
#     #     dtheta = torch.cat(dtheta)
#     #     grad = torch.matmul(self._jacobian_matrices.t(), torch.matmul(self._jacobian_matrices, dtheta))
#     #     return grad
