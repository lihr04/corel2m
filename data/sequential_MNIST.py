import random
import torch
import numpy as np
from torchvision import datasets

class SequentialMNIST(datasets.MNIST):
    def __init__(self, root="~/.torch/data/mnist", train=True, label_pair=(0,5)):
        super(SequentialMNIST, self).__init__(root, train, download=True)
        
        ind1=np.argwhere(self.targets==label_pair[0]).squeeze()
        ind2=np.argwhere(self.targets==label_pair[1]).squeeze()
        ind=np.random.permutation(np.concatenate((ind1,ind2)))
                
        self.data = torch.stack([(img.float()/255.0).unsqueeze(0)
                                 for img in self.data[ind,...]])
        self.targets= self.targets[ind].type(torch.LongTensor)
                                   

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]

def get_sequential_mnist(batch_size=100):
    ind1=np.random.permutation(np.arange(10)[:5])
    ind2=np.random.permutation(np.arange(10)[5:])
    
    train_loader = {}
    test_loader = {}
    for i in range(5):
        train_loader[i] = torch.utils.data.DataLoader(SequentialMNIST(train=True, label_pair=(ind1[i],ind2[i])),
                                                      batch_size=batch_size,
                                                      num_workers=4)
        test_loader[i] = torch.utils.data.DataLoader(SequentialMNIST(train=False, label_pair=(ind1[i],ind2[i])),
                                                     batch_size=batch_size)        
    return train_loader, test_loader, np.stack([ind1,ind2])
