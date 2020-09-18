#!/usr/bin/env python3
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

class actor_critic(nn.Module):
    
    def __init__(self, device):
        super(actor_critic, self).__init__()

        if(device == None):
            raise AssertionError("device cannot be none!")

        self.device = device

        self.general_L1 = nn.Linear(6, 128)

        self.action_L1 = nn.Linear(128, 64)
        self.action_L2 = nn.Linear(64, 3)

        self.value_L1 = nn.Linear(128, 1)

    # feedforward
    def forward(self, state_vector):
        # convert vector to be torch tensor
        state_vector = torch.from_numpy(state_vector).float().to(self.device)
        
        # pass through the general layer
        state_vector = F.relu(self.general_L1(state_vector))

        # generate action
        action_vector = F.relu(self.action_L1(state_vector))
        action_vector = F.softmax(self.action_L2(action_vector), dim=0)
        action_vector = Categorical(action_vector)
        action = action_vector.sample()

        # generate predicted score, this is defined as V(s), or Q(s)
        value = self.value_L1(state_vector)

        return action, value

