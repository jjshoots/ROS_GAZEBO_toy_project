#!/usr/bin/env python3
import time
import os
import numpy as np
from itertools import count

import rospy
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from cart_controller import cart_controller
from actor_critic import actor_critic

# init cart controller
cart_controller = cart_controller("controller_commander", node_rate=100)

# init DL agent

PATH = './network.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ', device)
AC = actor_critic(device=device, gamma=0.9, alpha=0.0001).to(device)

if os.path.isfile(PATH):
    AC.load_state_dict(torch.load(PATH))


num_episodes = 10000

for i in range(num_episodes):
    # reset our history
    rewards = []
    log_probs = []

    # reset simulation
    cart_controller.reset_simulation()
    
    # control loop rate of the system
    # do it a few times to let the new states permeate through the system
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    
    for j in count():
        # get the cart states
        cart_states = cart_controller.get_state_vector()

        # get action, predicted value, log probability of the action taken
        action, value, log_prob = AC.forward(cart_states * 10.0)

        # take the action and get the reward
        cart_controller.actuate_wheels((action - 10) / 2.0)
        reward = cart_controller.reward()

        # add our reward and log prob to our history
        # reward is function of angle and time
        rewards.append(reward + 1)
        log_probs.append(log_prob)

        # control loop rate of the system
        cart_controller.throttle_Hz()

        # if the pendulum goes out of bounds, stop, relearn, reset sim
        if(abs(cart_states[4]) > 10
        or abs(cart_states[0]) > 0.5
        or abs(cart_states[2]) > 0.5):

            # update policy
            AC.update_policy(log_probs, rewards)

            # reset simulation
            cart_controller.reset_simulation()

            # for data display
            collected_reward = np.sum(rewards)
            print(
                "Number of iterations: " + str(i)
                +
                "; Number of steps: " + str(j)
                +
                "; Total collected reward: " + str(round(collected_reward, 3))
                )

            # save the network
            torch.save(AC.state_dict(), PATH)
            
            # exit the loop
            break
