import numpy as np
import torch
from test_without_learning import simulateGaussianPPM
from ppm_trainer import check_accuracy
from data_generate import CustomSignalDataset
from torch.utils.data import DataLoader


ENR = 10
ENRlin = 10**(ENR/10)
test_len = int(1e4)  # 2e5

ENR_opt = 15
ENR_opt_lin = 10**(ENR_opt/10)
beta = (13/8)**(1/3) * (ENR_opt_lin)**(-5/6) * np.exp(ENR_opt_lin/6)
dt = 1/(350*beta)
overload = 6.4

#trained_net = torch.load('C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11.pt')
#test_data = CustomSignalDataset(sigma=1, epoch_len=test_len)
#test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
#trained_MSE = check_accuracy(test_loader, trained_net, 'test')
linear_MSE = simulateGaussianPPM(beta, ENRlin, test_len, overload, dt)

#print(trained_MSE)
print(linear_MSE)
