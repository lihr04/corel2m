from copy import deepcopy
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc


def get_permuted_mnist_single_task(batch_size, permute_idx=None, flatten=True, num_workers=4):
    """
    Returns the dataset for a single task of Permuted MNIST dataset
    :param batch_size:
    :return:
    """
    if permute_idx is None:
        permute_idx=np.arange(14*14)
    else:
        assert len(permute_idx) == 14 * 14
        
    if flatten:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.Resize((14,14)),
            torchvision.transforms.Lambda(lambda x: x.view(-1)[permute_idx]),
            ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.Resize((14,14)),
            torchvision.transforms.Lambda(lambda x: x.view(-1)[permute_idx].view(1, 14, 14)),
            ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("~/.torch/data/mnist", 
                                   train=True, 
                                   download=True, 
                                   transform=transforms), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("~/.torch/data/mnist", 
                                   train=False, 
                                   download=True, 
                                   transform=transforms), 
        batch_size=batch_size, 
        shuffle=False)

    return train_loader, test_loader


def get_permuted_mnist(num_task=5, batch_size=100, flatten=True, num_workers=4):
    """
    Returns data loaders for all tasks of rotated MNIST dataset.
    :param num_tasks: number of tasks in the benchmark.
    :param batch_size:
    :return:
    """
    train_loader = {}
    test_loader = {}
    idx = np.arange(14*14)
    for i in range(num_task):
        train_loader[i], test_loader[i] = get_permuted_mnist_single_task(batch_size, deepcopy(idx), flatten, num_workers)
        np.random.shuffle(idx)
    return train_loader, test_loader
