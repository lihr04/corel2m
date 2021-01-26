import random
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_circles


def make_circle(n,r=1.,quadrant=0,arclength=.25,noise=.05):
    theta=[0.,np.pi/2.,np.pi,3.*np.pi/2.][quadrant]
    theta=theta+(arclength*np.random.rand(n)*2.*np.pi)
    x=np.stack([r*np.cos(theta),r*np.sin(theta)],1)
    x+=noise*np.random.randn(x.shape[0],x.shape[1])
    return x

class CircleQuad(Dataset):
    def __init__(self,n_samples,factor=.75,noise=None,task=0):        
        assert task<4
        X=np.concatenate([make_circle(n_samples,r=1.,quadrant=task,noise=noise),
                          make_circle(n_samples,r=factor,quadrant=task,noise=noise)])

        y=np.concatenate([np.zeros(n_samples,),np.ones(n_samples)])
        
        self.data = torch.from_numpy(X).type(torch.FloatTensor)
        self.targets= torch.from_numpy(y).type(torch.LongTensor)
                                   
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        sample, target = self.data[index,:], self.targets[index]
        return sample, target
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [sample for sample in self.data[sample_idx]]

def get_quadrant_circles(n_samples=1000,batch_size=100,test_train_ratio=.2,factor=.75,noise=.025):
       
    train_loader = {}
    test_loader = {}        

    for i in range(4):
        train_loader[i] = torch.utils.data.DataLoader(CircleQuad(n_samples,
                                                             factor=factor,
                                                             noise=noise,
                                                             task=i),
                                                      batch_size=batch_size,shuffle=True)
        test_loader[i] = torch.utils.data.DataLoader(CircleQuad(int(test_train_ratio*n_samples),
                                                                factor=factor,
                                                                noise=noise,
                                                                task=i),
                                                                batch_size=batch_size)        
    return train_loader, test_loader


class Circle(Dataset):
    def __init__(self,n_samples,factor=.75,noise=None,scale=1.):        
        if isinstance(factor,list):
            assert len(scale)==len(factor)
            assert len(n_samples)==len(factor)
            X=list()
            y=list()
            for i in range(len(factor)):
                Xt,yt=make_circles(n_samples=n_samples[i],factor=factor[i],noise=noise)
                Xt*=scale[i]
                X.append(Xt)
                y.append(yt)
            X=np.concatenate(X)
            y=np.concatenate(y)
        else:
            X,y=make_circles(n_samples=n_samples,factor=factor,noise=noise)
            X*=scale
        
        
        self.data = torch.from_numpy(X).type(torch.FloatTensor)
        self.targets= torch.from_numpy(y).type(torch.LongTensor)
                                   
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        sample, target = self.data[index,:], self.targets[index]
        return sample, target
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [sample for sample in self.data[sample_idx]]

def get_sequential_circles(n_samples=1000,batch_size=100,test_train_ratio=.2):
       
    train_loader = {}
    test_loader = {}
    
    factors=[.5,.75]
    noise=[.025,.025]
    scales=[.5,1.]

    for i in range(2):
        train_loader[i] = torch.utils.data.DataLoader(Circle(n_samples,
                                                             factors[i],
                                                             noise[i],
                                                             scales[i]),
                                                      batch_size=batch_size,shuffle=True)
        test_loader[i] = torch.utils.data.DataLoader(Circle(int(test_train_ratio*n_samples),
                                                            factors[i],
                                                            noise[i],
                                                            scales[i]),
                                                     batch_size=batch_size)        
    return train_loader, test_loader

def get_joint_circles(n_samples=1000,batch_size=100,test_train_ratio=.2):
       
    train_loader = {}
    test_loader = {}
    
    factors=[.5,.75]
    noise=.025
    scales=[.5,1.]
    nsamples=2*[n_samples]
    
    train_loader = torch.utils.data.DataLoader(Circle(nsamples,
                                                      factors,
                                                      noise,
                                                      scales),
                                                  batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(Circle(2*[int(test_train_ratio*n_samples)],
                                                        factors,
                                                        noise,
                                                        scales),
                                                 batch_size=batch_size)        
    return train_loader, test_loader
