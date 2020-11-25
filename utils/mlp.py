__author__ = "Soheil Kolouri"
__copyright__ = "Copyright (C) 2019 Soheil Kolouri"
__license__ = "Public Domain"
__version__ = "1.0"

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """ Multi-Layer Perceptron
        Args:
            input_size (int)  [default=784]
            output_size (int) [default=10]
            hidden_size (int) [default=500]: Can be scalar or list
            depth (int) [default=3]: is surpassed if hidden size is a list
            activation (string) [default='leakyReLU']: 'leakyReLU', 'ReLU', 'Sigmoid', 'TanH'
            slope (float) [default=.1]: scalar between 0 and 1 that identifies slope of LeakyReLU
            require_bias (bool) [default=False]
            device (string) [default= 'cuda:0']: 'cpu' or 'cuda:k' for k in number of GPUs
    """
    def __init__(self,**kwargs):
        defaults={
            "input_size":784,
            "output_size":10,
            "hidden_size":500,
            "depth":3,
            "activation":'leakyReLU',
            "slope":.1,
            "require_bias":False,
            "device":'cuda:0'
        }

        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)
        super(MLP, self).__init__()

        if np.isscalar(self.hidden_size):
            self.architecture=np.concatenate([[self.input_size],
                                              self.depth*[self.hidden_size],
                                              [self.output_size]])
        else:
            self.architecture=np.concatenate([[self.input_size],
                                              self.hidden_size,
                                              [self.output_size]])

        # Define activation
        if self.activation=='leakyReLU':
            self.nonlinearity=nn.LeakyReLU(negative_slope=self.slope,inplace=False)
        elif self.activation=='ReLU':
            self.nonlinearity=nn.ReLU(inplace=False)
        elif self.activation=='Sigmoid':
            self.nonlinearity=nn.Sigmoid()
        elif self.activation=='TanH':
            self.nonlinearity=nn.Tanh()
        else:
            raise Exception('Activation not implemented!')

        self.layers=list()
        for i in range(len(self.architecture)-1):
            self.layers.append(nn.Linear(self.architecture[i],self.architecture[i+1],bias=self.require_bias))
            if i!=len(self.architecture)-2:
                self.layers.append(self.nonlinearity)
            # Later we can add BatchNorm and Dropout if need be!

        # We need to register parameters, this can be easily done with Sequential
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x=x.to(self.device)
        for module in self.layers:
            x=module(x)
        return x

    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def reset(self):
        self.net.apply(self.init_weights)
