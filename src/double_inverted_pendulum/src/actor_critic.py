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
    
    # gamma = reward discount factor
    # alpha = learning rate
    def __init__(self, device, gamma, alpha):
        super(actor_critic, self).__init__()

        if(device == None):
            raise AssertionError("device cannot be none!")

        self.device = device
        self.gamma = gamma

        # network definitions
        self.general_L1 = nn.Linear(6, 128)

        self.action_L1 = nn.Linear(128, 64)
        self.action_L2 = nn.Linear(64, 21)

        self.value_L1 = nn.Linear(128, 1)
        
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    # feedforward
    def forward(self, state_vector):
        # convert vector to be torch tensor
        state_vector = torch.from_numpy(state_vector).float().to(self.device)
        
        # pass through the general layer
        state_vector = F.relu(self.general_L1(state_vector))

        # generate action
        action_vector = F.relu(self.action_L1(state_vector))
        distribution = F.softmax(self.action_L2(action_vector), dim=0)
        action = Categorical(distribution)
        action = action.sample()

        # generate predicted score, this is defined as V(s), or Q(s)
        value = self.value_L1(state_vector)

        # calculate log probability
        log_prob = torch.log(distribution[action])

        return action, value, log_prob

    # calculate and return reward return for each time step
    def calculate_loss(self, rewards):
        # flip the vector from head to tail
        np.flipud(rewards)

        # empty array to store our reward returns
        returns = []

        # calculate the loss
        for i, reward in enumerate(rewards):
            if i == 0:
                returns.append(reward)
            else:
                returns.append(reward + returns[i-1] * self.gamma)

        # flip the returns vector back right side up
        np.flipud(returns)

        return torch.tensor(returns)

    # update policy
    def update_policy(self, log_probs, rewards):
        # calculate the discounted rewards for each time step
        discounted_returns = self.calculate_loss(rewards)

        # normalize discounted returns
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)

        # finally calculate our policy gradient
        policy_gradient = []

        for log_prob, discounted_return in zip(log_probs, discounted_returns):
            policy_gradient.append(-log_prob * discounted_return)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

