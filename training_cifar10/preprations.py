import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def load_cifar10():

    # load cifar10 dataset
    NUM_TRAIN = 49000

    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by subtracting the mean RGB value and dividing by the
    # standard deviation of each RGB value; we've hardcoded the mean and std.
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # We set up a Dataset object for each split (train / val / test); Datasets load
    # training examples one at a time, so we wrap each Dataset in a DataLoader which
    # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
    # training set into train and val sets by passing a Sampler object to the
    # DataLoader telling how it should sample from the underlying Dataset.
    cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                 transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64,
                              sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                               transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

    return loader_train, loader_val, loader_test


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)