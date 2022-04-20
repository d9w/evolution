import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)

        self.n_out = n_out

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class RNN(nn.Module):
    """ Simple RNN for reinforcement learning with LSTM """

    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.n_in = n_in
        self.h_size = h_size
        self.n_out = n_out
        self.lstm = nn.LSTM(n_in, h_size, 1).double()
        self.fc = nn.Linear(h_size, n_out).double()
        self.h = None
        self.c = None

    def reset(self):
        self.h = torch.zeros(1, 1, self.h_size).double()
        self.c = torch.zeros(1, 1, self.h_size).double()

    def forward(self, x):
        x = x.double().unsqueeze(0)
        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        x = self.fc(x)
        return x[0, 0]
