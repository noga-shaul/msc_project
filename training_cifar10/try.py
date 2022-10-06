# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:10:38 2022

@author: Noga
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from train import train_part34
from train import check_accuracy_part34
from preprations import Flatten
from preprations import load_cifar10

# define cpu\gpu use
USE_GPU = False
dtype = torch.float32  # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)
print('dtype:', dtype)

# load cifar10
loader_train, loader_val, loader_test = load_cifar10()

# define model and optimizer
model = None
optimizer = None

print_every = 100
in_channel = 3
channel_1 = 32
channel_2 = 32
channel_3 = 64
channel_4 = 64
learning_rate = 1e-3
num_classes = 10
results = {}

model = nn.Sequential(
        nn.Conv2d(in_channel, channel_1, (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(channel_1, channel_2, (3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(channel_2, channel_3, (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(channel_3, channel_4, (3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        Flatten(),
        nn.Linear(channel_4 * 8 * 8, num_classes),
    )

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
train_part34(model, optimizer, loader_train, loader_val, epochs=1, print_every=print_every, device=device, dtype=dtype)

# test
best_model = model
check_accuracy_part34(loader_test, best_model, device, dtype)