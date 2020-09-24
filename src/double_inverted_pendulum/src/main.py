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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ', device)
AC = actor_critic(device=device, gamma=0.9, alpha=0.0001).to(device)

# get the latest pth file and use as weights
MARK_NUMBER = 1
PATH = F"src/double_inverted_pendulum/src/network_weights/MK4/network_MK{MARK_NUMBER}.pth"

while os.path.isfile(PATH):
    MARK_NUMBER += 1
    PATH = F"src/double_inverted_pendulum/src/network_weights/MK4/network_MK{MARK_NUMBER}.pth"

MARK_NUMBER -= 1
PATH = F"src/double_inverted_pendulum/src/network_weights/MK4/network_MK{MARK_NUMBER}.pth"

AC.load_state_dict(torch.load(PATH))

# training loop params
num_episodes = 100000
highest_step_count = 0

for i in range(num_episodes):
    # reset our history
    rewards = []
    values = []
    log_probs = []
    torque = 0

    # reset simulation
    cart_controller.reset_simulation()
    
    # control loop rate of the system
    # do it a few times to let the new states permeate through the system
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
    cart_controller.throttle_Hz()
 
    # reset simulation
    cart_controller.reset_simulation()
    
    for j in count():
        # get the cart states
        cart_states = cart_controller.get_state_vector()

        # get action, predicted value, log probability of the action taken
        # multiply by 10 for better resolution
        action, value, log_prob = AC.forward(cart_states * 10.0)

        # take the action and get the reward
        # integrate action as a torque to the wheels
        torque = torque + (action - 1)
        cart_controller.actuate_wheels(torque)
        reward = cart_controller.reward() + (j / 10)


        # add our reward and log prob to our history
        # reward is function of angle and time
        # rewards.append(reward)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)

        # control loop rate of the system
        cart_controller.throttle_Hz()

        # if the pendulum goes out of bounds, stop
        if(abs(cart_states[4]) > 10
        or abs(cart_states[0]) > 1
        or abs(cart_states[2]) > 1):
            # update policy
            loss = AC.update_policy(log_probs, rewards, values).item()

            # for data display
            collected_reward = np.sum(rewards)
            print(
                "Number of iterations:", i,
                "; Steps walked:", j,
                "; Collected reward:", round(collected_reward, 3),
                "; Total loss:",  round(loss, 3)
                )

            highest_step_count = (j if (i == 0) else highest_step_count)

            # save the network
            if(j > highest_step_count + 100):
                highest_step_count = j
                MARK_NUMBER += 1
                PATH = F"src/double_inverted_pendulum/src/network_weights/MK4/network_MK{MARK_NUMBER}.pth"
                torch.save(AC.state_dict(), PATH)

            break