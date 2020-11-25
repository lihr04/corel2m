import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(image_size=32,svhn_path='~/.torch/data/svhn',mnist_path='~/.torch/data/mnist',batch_size=64,num_workers=4,train=True):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform_svhn=transforms.Compose([  transforms.Grayscale(),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.5],[.5])])

    transform_mnist=transforms.Compose([transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.5],[.5])])

    if train:
        split='train'
    else:
        split='test'
    svhn = datasets.SVHN(root=svhn_path, download=True, transform=transform_svhn,split=split)
    mnist = datasets.MNIST(root=mnist_path, download=True, transform=transform_mnist,train=train)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    return svhn_loader, mnist_loader