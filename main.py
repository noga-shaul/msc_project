import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import scipy
from ppm_net import Net
from ppm_trainer import train
from ppm_trainer import shift_test
from data_generate import CustomSignalDataset


def main():

    # set rect params:
    params = {'E': 10,
              'delta': 1,
              'beta': 10,
              'res': 0.001,
              'noise': True,
              'std': 1}

    #generate data
    sigma = 5
    epoch_len = 100000
    test_len = 1000
    train_data = CustomSignalDataset(sigma=sigma, epoch_len=epoch_len)
    test_data = CustomSignalDataset(sigma=sigma, epoch_len=test_len)

    learning_rate = 1e-3
    ppm_net = Net(params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ppm_net.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(ppm_net.parameters(), lr=learning_rate, momentum=0.9)
    train(ppm_net, criterion, optimizer, train_data, test_data, epochs=1)
    trained_net = ppm_net
    shift_test(trained_net)
    print()

#main()

#torch.save(ppm_net, 'C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11')
trained_net = torch.load('C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11.pt')
shift_test(trained_net)







