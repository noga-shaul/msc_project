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
        x, t = rect(x, E=torch.tensor(self.params['E']),
                 delta=torch.tensor(self.params['delta']),
                 beta=torch.tensor(self.params['beta']),
                 dt=torch.tensor(self.params['res']),
                 overload=torch.tensor(self.params['overload']),
                 noise=True,
                 std=torch.tensor(self.params['std']))

        x = torch.unsqueeze(x, 1)

        #convolution with ppm pulse
        conv_kernel = torch.ones(350) * torch.sqrt(torch.tensor(self.params['beta']))
        conv_kernel = conv_kernel / torch.sqrt(torch.sum(torch.square(conv_kernel) * self.params['res']))
        conv_kernel = conv_kernel.repeat(1, 1, 1)
        #x = torch.stack((x,t.repeat(x.size(0),1)),1)
        x = F.conv1d(x, conv_kernel, padding='same')
        #x.data = Variable(x.data, requires_grad=True)
        #choosing max
        y = torch.argmax(x, 2)
        t = t.repeat(y.size())
        t = t[torch.arange(y.size(0)), torch.squeeze(y)]
        x = x.squeeze()[torch.arange(y.size(0)), torch.squeeze(y)]
        x = torch.stack((x, t), 1)
        x = torch.unsqueeze(x, 1)

        x = (F.relu(self.conv1(x))) #if needed self.pool can be add
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc4(x)

        return x, out_shift
