import random
import torch
import numpy as np
from torchvision import datasets

class PermutedMNIST(datasets.MNIST):
    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)

        if permute_idx is None:
            permute_idx=np.arange(28*28)
        else:
            assert len(permute_idx) == 28 * 28

        # self.data = torch.stack([((img.float().view(-1))/255.0)
        #                                for img in self.data])
        self.data = torch.stack([((img.float().view(-1)[permute_idx])/255.0)
                                       for img in self.data])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]

def get_permuted_mnist(num_task=3,batch_size=100,num_workers=4):
    train_loader = {}
    test_loader = {}
    idx = list(range(28 * 28))
    for i in range(num_task):
        train_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=True, permute_idx=idx),
                                                      batch_size=batch_size,
                                                      num_workers=num_workers)
        test_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=False, permute_idx=idx),
                                                     batch_size=batch_size)
#         random.shuffle(idx)
    return train_loader, test_loader
