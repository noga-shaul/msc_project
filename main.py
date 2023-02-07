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
from ppm_trainer import check_accuracy
from ppm_trainer import shift_test
from data_generate import CustomSignalDataset


def test_distortion(trained_net, test_len, noise=True):
    test_data = CustomSignalDataset(sigma=1, epoch_len=test_len)
    testloader = DataLoader(test_data, batch_size=512, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    distortion = check_accuracy(testloader, trained_net, 'test', device, noise=noise)
    #shift_test(trained_net)
    print(distortion)
    return distortion


def main():

    ENR_vec = torch.arange(4, 16) #torch.arange(4, 16)
    distortion_vec = torch.zeros(ENR_vec.size(0))
    shift_test_out = torch.zeros(ENR_vec.size(0), 1000)
    input_vec = torch.linspace(-6.4, 6.4, 1000) # for shift test after training

    for i in range(ENR_vec.size(0)):

        ENR = ENR_vec[i]
        ENRlin = 10 ** (ENR / 10)
        final_test_len = int(2e5)  # 2e5

        ENR_opt = ENR
        ENR_opt_lin = 10 ** (ENR_opt / 10)
        beta = (13 / 8) ** (1 / 3) * (ENR_opt_lin) ** (-5 / 6) * np.exp(ENR_opt_lin / 6)
        dt = 1 / (351 * beta)
        overload = 6.35
        noise = True

        # set rect params:
        params = {'E': ENRlin,
                  'delta': 1.0,
                  'beta': beta,
                  'res': dt,
                  'overload': overload,
                  'noise': True,
                  'std': 1.0}

        # generate data
        sigma = 1
        epoch_len = 5000
        test_len = 1000
        train_data = CustomSignalDataset(sigma=sigma, epoch_len=epoch_len)
        test_data = CustomSignalDataset(sigma=sigma, epoch_len=test_len)

        # run training
        learning_rate = 1e-2
        epochs_num = 15 #change back epochs to 20

        beta_mult_vec = torch.arange(0.5, 1.8, 0.25)
        best_val = 100
        for n in range(beta_mult_vec.size(0)):
            params['beta'] = beta * beta_mult_vec[n] #* torch.FloatTensor(1).uniform_(0.5, 1.5)

            print('*********************')
            print('ENR index: ', i, '/', ENR_vec.size(0), '. Beta index: ', n, '/', beta_mult_vec.size(0))
            print('ENR: ', ENR, '. beta_mule: ', beta_mult_vec[n], '. Beta: ', params['beta'])

            params['res'] = 1/(351*params['beta'])
            ppm_net = Net(params)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(ppm_net.parameters(), lr=learning_rate)
            # optimizer = optim.SGD(ppm_net.parameters(), lr=learning_rate, momentum=0.9)
            model, distortion_val = train(ppm_net, criterion, optimizer, train_data, test_data, epochs=epochs_num, noise=noise)
            if distortion_val < best_val:
                best_val = distortion_val
                best_model = model
        trained_net = best_model

        # save train results
        torch.save(trained_net, '.\\models\\ppm_net_trained_ENR_'+str(ENR.to('cpu').numpy())+'.pt')
        _, shift_test_out[i, :] = shift_test(trained_net)
        distortion_vec[i] = test_distortion(trained_net, final_test_len)
        print(distortion_vec[i])

    # save final results
    print(distortion_vec)
    torch.save(distortion_vec, '.\\models\\distortion.pt')
    torch.save(shift_test_out, '.\\models\\shift_test_out.pt')
    torch.save(input_vec, '.\\models\\input_vec.pt')


main()

#torch.save(ppm_net, 'C:\\Users\\Noga\\Documents\\git_repositories\\msc_project\\models\\ppm_net_trained_7_11')



#trained_net = torch.load('C:\\Users\\Noga\\Downloads\\msc_project\\msc_project\\models\\net_parallel_noise_trained_02_01.pt')
#test_distortion(trained_net)





