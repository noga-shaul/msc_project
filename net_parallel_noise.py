import torch
import torch.nn as nn
import torch.nn.functional as F
from rect import rect
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.fc1_de = nn.Linear(1, 16)
        self.fc2_de = nn.Linear(16, 32)
        self.fc3_de = nn.Linear(32, 1)
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc4 = nn.Linear(128, 1)  # input size 32*steps/4
        self.params = params

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # x here is scalar

        out_shift = x
        rect_signal, t = rect(torch.zeros_like(x), E=torch.tensor(self.params['E']),
                 delta=torch.tensor(self.params['delta']),
                 beta=torch.tensor(self.params['beta']),
                 dt=torch.tensor(self.params['res']),
                 overload=torch.tensor(self.params['overload']),
                 noise=True,
                 std=torch.tensor(self.params['std']))

        rect_signal = torch.unsqueeze(rect_signal, 1)

        #convolution with ppm pulse
        conv_kernel = torch.ones(350) * torch.sqrt(torch.tensor(self.params['beta']))
        conv_kernel = conv_kernel / torch.sqrt(torch.sum(torch.square(conv_kernel) * self.params['res']))
        conv_kernel = conv_kernel.repeat(1, 1, 1)
        #x = torch.stack((x,t.repeat(x.size(0),1)),1)
        signal_heat_map = F.conv1d(rect_signal, conv_kernel, padding='same')
        #choosing max
        corr_peak_ind = torch.argmax(signal_heat_map, 2)
        t = t.repeat(corr_peak_ind.size())
        t = t[torch.arange(corr_peak_ind.size(0)), torch.squeeze(corr_peak_ind)]
        x = x + t.unsqueeze(1)

        x = (F.relu(self.fc1_de(x))) #if needed self.pool can be add
        x = (F.relu(self.fc2_de(x)))
        x = (self.fc3_de(x))

        return x, out_shift
