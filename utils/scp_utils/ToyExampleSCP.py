from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import math

class SCP(object):
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_slices=50):
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
        xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((self.n_slices,K))]).t()

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
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        for n, p in model.named_parameters():
            _loss = 0.5 * self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class SketchSCP():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=0, n_slices=50, n_bucket=10):
        """ OnlineSCP is the class for implementing the online SCP method.
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
        ''' Generate the online SCP penalty.
            This function receives the current model with its weights, and calculates
            the online SCP loss.
        '''
        loss = 0
        dtheta = []
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta = torch.cat(dtheta)
        loss = 0.5 * torch.sum(torch.matmul(self._jacobian_matrices, dtheta) ** 2)
        return loss
    
    
class FullSCP():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_slices=50):
        """ OnlineSCP is the class for implementing the online SCP method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_slices=n_slices

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
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
                
            output_list=torch.softmax(self.model(imgs),1)
            K=output_list.shape[1]
            for output in output_list:
                xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((self.n_slices,K))]).t()
                for l in range(self.n_slices):
                    self.model.zero_grad() # Zero the gradients
                    loss=torch.matmul(output,xi[:,l])
                    loss.backward(retain_graph=True)
                    grad=[]
                    for n, p in self.params.items():
                        grad.append(p.grad.data.view(-1))
                    grad=torch.cat(grad)
                    A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
            
        A=A/float(len(dataloader.dataset)*self.n_slices)
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
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss

    
class BlockDiagonalSCP():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_slices=50, n_bucket=10):
        """ ONLY FOR BIASED NETWORK
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_slices = n_slices
        self.n_bucket = n_bucket

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
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
                
            output_list=torch.softmax(self.model(imgs),1)
            K=output_list.shape[1]
            for output in output_list:
                xi=torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) for xi_ in torch.randn((self.n_slices,K))]).t()
                for l in range(self.n_slices):
                    self.model.zero_grad() # Zero the gradients
                    loss=torch.matmul(output,xi[:,l])
                    loss.backward(retain_graph=True)
                    grad=[]
                    for n, p in self.params.items():
                        grad.append(p.grad.data.view(-1))
                    grad=torch.cat(grad)
                    A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
            
        A=A/float(len(dataloader.dataset)*self.n_slices)
        return A
    
    def calculate_approximation(self,dataloader: torch.utils.data.DataLoader):
        A = self.calculate_FIM(dataloader)
        B = self.zeros_like(self.fim)
        d = self.fim.shape[0]
        for dimension in range(0, d, self.n_bucket):
            next_dimension = min(dimension + self.n_bucket, d)
            B[dimension:next_dimension, dimension:next_dimension] = A[dimension:next_dimension, dimension:next_dimension]
        return B

    def consolidate(self,dataloader):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        self.fim *= self.alpha
        self.fim += (1-self.alpha)*self.calculate_approximation(dataloader)

        for n, p in self.model.named_parameters():
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss
    

class LowRankSCP():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_slices=50, n_bucket=10):
        """ OnlineMAS is the class for implementing the online MAS method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_slices = n_slices
        self.n_bucket = n_bucket

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
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
            loss_list=(torch.softmax(self.model(imgs),1)**2).sum(1)
            for loss in loss_list:
                self.model.zero_grad() # Zero the gradients
                loss.backward(retain_graph=True)
                grad=[]
                for n, p in self.params.items():
                    grad.append(p.grad.data.view(-1))
                grad=torch.cat(grad)
                A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
        A=A/float(len(dataloader.dataset)*self.n_slices)
        return A


    def calculate_approximation(self,dataloader: torch.utils.data.DataLoader):
        u, s, v =torch.svd_lowrank(self.calculate_FIM(dataloader), q=self.n_bucket)
        return torch.mm(torch.mm(u, torch.diag(s)), v.t())

    def consolidate(self,dataloader: torch.utils.data.DataLoader):
        ''' Consolidate
         This function receives a dataloader, it then calculates and updates the
         Fisher Information Matrix (FIM) to preserve the max log-likelihood for
         the data in dataloader.
         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        self.fim *= self.alpha
        self.fim += (1-self.alpha)*self.calculate_approximation(dataloader)

        for n, p in self.model.named_parameters():
            # Update the means
            self._means[n] = deepcopy(p.data).to(self.device)


    def penalty(self, model: nn.Module):
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        dtheta=[]
        for n, p in model.named_parameters():
            dtheta.append((p - self._means[n]).view(-1))
        dtheta=torch.cat(dtheta)
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss
