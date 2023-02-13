import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, input_channels=8):
        
        super(Policy, self).__init__()

        self.layer1 = nn.Linear(input_channels, 1)
        self.tanh1   = nn.Tanh()
        # self.layer2 = nn.Linear(2 * input_channels, 1)
        # self.tanh2 = nn.Tanh()

    def forward(self, state):

        hidden = self.layer1(state)
        hidden = self.tanh1(hidden)
        # hidden = self.layer2(hidden)
        # action = self.tanh2(hidden)

        return hidden



    