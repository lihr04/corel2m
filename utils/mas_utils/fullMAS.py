from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

def calculate_Hessian(model: nn.Module, 
                  data_loader: torch.utils.data.DataLoader):
    
    device=model.device
    model.eval()
    grad=[]
    for n, p in model.named_parameters():
        grad.append(p.grad.data.view(-1))            
    grad=torch.cat(grad)
    d=grad.shape[0]
    A=torch.zeros((d,d)).to(device)
        
    for i,(imgs,labels) in tqdm(enumerate(data_loader)):
        model.zero_grad() # Zero the gradients
        imgs,labels = imgs.to(device),labels.to(device)# Get Inout

        loss=(torch.softmax(model(imgs).to(device),1)**2).sum(1)
        loss.backward()
        
        grad=[]
        for n, p in model.named_parameters():
            grad.append(p.grad.data.view(-1))            
        grad=torch.cat(grad)
        A+=torch.matmul(grad.unsqueeze(1),grad.unsqueeze(0))
    A=A/float(i+1)
    return A

class FullMAS(object):
    def __init__(self, model: nn.Module, device='cuda:0', alpha=.5):
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
        
        self._means = []
        for n, p in self.model.named_parameters():
            self._means.append(deepcopy(p.data).view(-1).to(self.device))          
        self._means=torch.cat(self._means)
        self.d=self._means.shape[0]
        self._precision_matrices=torch.zeros((self.d,self.d)).to(self.device)

    def consolidate(self,dataloader):
        ''' Consolidate
         This function receives a dataloader, and calculates and updates the Omega
         Matrix (only the diagonal values) described in the paper.

         input:
            dataloader : A Pytorch dataloader containing data from the task to be consolidated
        '''

        # Initialize a temporary precision matrix
        self._precision_matrices=self.alpha*self._precision_matrices+(1-self.alpha)*calculate_Hessian(self.model, dataloader)

        # Here we follow a similar approach as in the online EWC framework presented in
        # Chaudhry et al. ECCV2018 and also in Shwarz et al. ICML2018.
        for n, p in self.model.named_parameters():
            self._means.append(deepcopy(p.data).view(-1).to(self.device))          
        self._means=torch.cat(self._means)


    # Now we need to generate the the EWC updated loss
    def penalty(self, model: nn.Module):
        ''' Generate the online MAS penalty.
            This function receives the current model with its weights, and calculates
            the online MAS loss.
        '''
        loss = 0
        parameters = []
        for n, p in self.model.named_parameters():
            parameters.append(p.data.view(-1).to(self.device))          
        parameters=torch.cat(parameters)
        loss = torch.dot((parameters - self._means), torch.matmul(self._precision_matrices, (parameters - self._means)))
        return loss
