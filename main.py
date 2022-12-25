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
from net_parallel_noise import Net
from ppm_trainer import train
from ppm_trainer import shift_test
from data_generate import CustomSignalDataset


def main():

    ENR = 10
    ENRlin = 10 ** (ENR / 10)
    test_len = int(1e4)  # 2e5

    ENR_opt = 15
    ENR_opt_lin = 10 ** (ENR_opt / 10)
    beta = (13 / 8) ** (1 / 3) * (ENR_opt_lin) ** (-5 / 6) * np.exp(ENR_opt_lin / 6)
    dt = 1 / (350 * beta)
    overload = 6.4

    # set rect params:
    params = {'E': ENRlin,
              'delta': 1,
              'beta': beta,
              'res': dt,
              'overload': overload,
              'noise': True,
              'std': 1}

    #generate data
    sigma = 1
    epoch_len = 500
    test_len = 100
    train_data = CustomSignalDataset(sigma=sigma, epoch_len=epoch_len)
    test_data = CustomSignalDataset(sigma=sigma, epoch_len=test_len)

    learning_rate = 1e-2
    ppm_net = Net(params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ppm_net.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(ppm_net.parameters(), lr=learning_rate, momentum=0.9)
    train(ppm_net, criterion, optimizer, train_data, test_data, epochs=20)
    trained_net = ppm_net
    shift_test(trained_net)
    print()

main()

#torch.save(ppm_net, 'C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11')
#trained_net = torch.load('C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11.pt')
#shift_test(trained_net)







