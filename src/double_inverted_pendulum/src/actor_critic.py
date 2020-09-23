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

        self.action_L1 = nn.Linear(128, 256)
        self.action_L2 = nn.Linear(256, 3)

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
        action_vector = self.action_L2(action_vector) 
        # generate distribution from logits from action vector output
        distribution = Categorical(logits=action_vector)
        action = distribution.sample()

        # calculate log probability
        log_prob = distribution.log_prob(action)

        # generate predicted return, this is defined as V(s), or Q(s)
        value = self.value_L1(state_vector)

        return action.cpu().numpy(), value, log_prob

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

        return returns

    # update policy
    def update_policy(self, log_probs, rewards, values):
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)

        # calculate the discounted rewards for each time step
        discounted_returns = self.calculate_loss(rewards)

        # calculate advantages function
        discounted_returns = torch.tensor(discounted_returns).to(self.device)
        advantages = values - discounted_returns

        # normalize advantage, to have at least 50/50 split of network up and down
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # calculate policy gradient
        policy_gradient = -log_probs * advantages_normalized
        policy_gradient = policy_gradient.sum()

        # calculate total loss
        total_loss = policy_gradient + advantages.pow(2).sum()

        # backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

