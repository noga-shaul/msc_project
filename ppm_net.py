import torch
import torch.nn as nn
import torch.nn.functional as F
from rect import rect


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc4 = nn.Linear(32*2500, 1) # input size 32*steps/4
        self.params = params

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # x here is scalar

        x = rect(x, E=torch.tensor(self.params['E']),
                 delta=torch.tensor(self.params['delta']),
                 beta=torch.tensor(self.params['beta']),
                 res=torch.tensor(self.params['res']),
                 noise=True,
                 std=torch.tensor(self.params['std']))
        x = torch.unsqueeze(x, dim=1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc4(x)

        return x
