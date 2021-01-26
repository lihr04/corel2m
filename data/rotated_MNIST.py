import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc

class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))
    
def tmp_func(x):
    return x.view(-1)

def get_rotated_mnist_single_task(task_id, batch_size, per_task_rotation=10):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id: (starting from 0, rather than the original 1)
    :param batch_size:
    :return:
    Original code: https://github.com/imirzadeh/stable-continual-learning/blob/master/stable_sgd/data_utils.py#L59-L92
    """
    rotation_degree = task_id*per_task_rotation
    rotation_degree -= (np.random.random()*per_task_rotation)

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.transforms.Lambda(tmp_func),
        ])

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/.torch/data/mnist", train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/.torch/data/mnist", train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_rotated_mnist(num_task=3, batch_size=100):
    """
    Returns data loaders for all tasks of rotation MNIST dataset.
    :param num_tasks: number of tasks in the benchmark.
    :param batch_size:
    :return:
    """
    train_loader = {}
    test_loader = {}
    datasets = {}
    for i in range(num_task):
        train_loader[i], test_loader[i] = get_rotated_mnist_single_task(i, batch_size)
    return train_loader, test_loader
