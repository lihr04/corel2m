import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VGG_like(nn.Module):
    """ General VGG-like CNN classifier with inputanble conv kernel
    """
    def __init__(self, input_size,conv_kernel_sizes, fc_sizes, nofclasses=10,drop_rate=.2,activation='ReLU'):
        super(VGG_like, self).__init__()
        self.nofclasses = nofclasses
        self.numofchannels = input_size[0]
        self.activation=nn.ReLU(True)
        
        if activation=='leakyReLU':
            self.activation=nn.LeakyReLU(inplace=True)
        elif activation=='ReLU':
            self.activation=nn.ReLU(inplace=True)
        elif activation=='Sigmoid':
            self.activation=nn.Sigmoid()
        elif activation=='TanH':
            self.activation=nn.Tanh()
        else:
            raise Exception('Activation not implemented!')
        
        
        self.drop_rate=drop_rate
            
        self.conv_layers=list()
        for i,kernel_size in enumerate(conv_kernel_sizes):
            if i==0:
                self.conv_layers.append(nn.Conv2d(self.numofchannels,kernel_size,kernel_size=3,padding=2))
                self.conv_layers.append(nn.BatchNorm2d(kernel_size * 1))
                self.conv_layers.append(self.activation)
            else:
                self.conv_layers.append(nn.Conv2d(conv_kernel_sizes[i-1],kernel_size,kernel_size=3,padding=2))
                self.conv_layers.append(nn.BatchNorm2d(kernel_size))
                self.conv_layers.append(self.activation)
                if i%2:
                    self.conv_layers.append(nn.MaxPool2d(2,2))
        # We need to register parameters, this can be easily done with Sequential
        self.features = nn.Sequential(*self.conv_layers)

        dummy=torch.rand(([1]+input_size))
        dummy=self.features(dummy)    
#         print(dummy.shape)
        d=self.num_flat_features(dummy)
#         print(d)
        self.fc_layers=list()
        for i,fc_size in enumerate(fc_sizes):
            if i==0:
                self.fc_layers.append(nn.Linear(d, fc_size))
                self.fc_layers.append(nn.BatchNorm1d(fc_size))
                self.fc_layers.append(self.activation)
                self.fc_layers.append(nn.Dropout(p=self.drop_rate))
            else:
                self.fc_layers.append(nn.Linear(fc_sizes[i-1], fc_size))
                self.fc_layers.append(nn.BatchNorm1d(fc_size))
                self.fc_layers.append(self.activation)
                self.fc_layers.append(nn.Dropout(p=self.drop_rate))        
        self.fc = nn.Sequential(*self.fc_layers)
        
        self.classify=nn.Sequential(nn.Linear(fc_size,self.nofclasses))



    def num_flat_features(self,x):
        return torch.prod(torch.as_tensor(x.size()[1:]))

    def forward(self,x):           
        x=self.features(x)
#         print(x.shape)
        x = x.view(-1,self.num_flat_features(x) )
#         print(x.shape)
        x=self.fc(x)      
        x=self.classify(x)
        return x

#     def classifier(self,x):
#         x=self.forward(x)
#         x=self.classify(x)
#         return x
        