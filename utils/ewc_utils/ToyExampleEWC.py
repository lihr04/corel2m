from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import math

class EWC():
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
                precision_matrices[n].data += p.grad.data ** 2 /float(len(dataloader.dataset))

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
            _loss = 0.5 * self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class SketchEWC():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_sketch=50):
        """ OnlineEWC is the class for implementing the online EWC method.
            Inputs:
                model : a Pytorch NN model
                device (string): the device to run the model on
                alpha (in [0,1) ): The online learning hyper-parameter
        """
#         self.device=device
#         self.alpha=alpha
#         self.model = model.to(self.device)

#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self._means = {}

#         self._precision_matrices ={}

#         d=0
#         for n, p in deepcopy(self.params).items():
#             d+=p.data.view(-1).shape[0]
#         self.fim=torch.zeros((d,d)).to(self.device)

#         for n, p in deepcopy(self.params).items():
#             self._means[n] = p.data.to(self.device)
            
        self.LARGEPRIME = 2**61-1
        
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
        self.n_sketch = n_sketch

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        self._jacobian_matrices = {}
        
        d=0
        for n, p in deepcopy(self.params).items():
            d+=p.data.view(-1).shape[0]
        self._jacobian_matrices=torch.zeros((n_sketch,d)).to(self.device)
        self.fim=torch.zeros((d,d)).to(self.device)

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
        buckets = ((h1 * indices) + h2) % self.LARGEPRIME % self.n_sketch
        buckets = buckets.to(self.device)
        
        # computing sketch matrix
        sketch = torch.zeros(self.n_sketch, n_data).to(self.device)
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
        
        for r in range(self.n_sketch):
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

        self.fim = self.alpha * self.fim + (1-self.alpha)*self.calculate_approximation(dataloader)
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
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss
    
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
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss
    
class BlockDiagonalEWC():
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5, n_bucket=10):
        """ ONLY FOR BIASED NETWORK
        """
        self.device=device
        self.alpha=alpha
        self.model = model.to(self.device)
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


    def calculate_approximation(self,dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        A=torch.zeros_like(self.fim)
        for i,(imgs,labels) in enumerate(dataloader):          
            imgs,labels = imgs.to(self.device),labels.to(self.device)# Get Inout
            loss_list=nn.CrossEntropyLoss(reduction='none')(self.model(imgs),labels)
            for loss in loss_list:
                self.model.zero_grad() # Zero the gradients
                loss.backward(retain_graph=True)
                
                grad=[]
                dimension=0
                for n, p in self.params.items():
                    grad.append(p.grad.data.view(-1))
                    if 'bias' not in n:
                        last_dimension = dimension
                        dimension += p.grad.data.view(-1).shape[0]
                    else:
                        dimension += p.grad.data.view(-1).shape[0]
                        grad=torch.cat(grad)
                        A[last_dimension:dimension, last_dimension:dimension] += torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
                        grad=[]
                # grad=[]
                # for n, p in self.params.items():
                #     grad.append(p.grad.data.view(-1))
                # grad=torch.cat(grad)
                # d=grad.shape[0]
                # for dimension in range(0, d, self.n_bucket):
                #     next_dimension = min(dimension + self.n_bucket, d)
                #     A[dimension:next_dimension, dimension:next_dimension] += torch.matmul(
                #         grad[dimension:next_dimension].unsqueeze(1),
                #         grad[dimension:next_dimension].unsqueeze(0))
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
        self.fim += (1-self.alpha)*self.calculate_approximation(dataloader)

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
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
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
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
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
        loss = 0.5 * torch.matmul(dtheta.unsqueeze(0),
                                  torch.matmul(self.fim,
                                               dtheta.unsqueeze(1)))
        return loss