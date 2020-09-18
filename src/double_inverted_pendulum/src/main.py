#!/usr/bin/env python3
import time
import os
import numpy as np

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
cart_controller.reset_simulation()

# init DL agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ', device)
AC = actor_critic(device=device).to(device)

while True:
    cart_controller.actuate_wheels(0)

    cart_states = cart_controller.get_state_vector()

    AC.forward(cart_states)

    # if the pendulum goes out of bounds, reset sim
    if(abs(cart_states[5]) > 10
    or abs(cart_states[1]) > 1.5
    or abs(cart_states[3]) > 1.5):
        cart_controller.reset_simulation()

    exit()

    cart_controller.throttle_Hz()

