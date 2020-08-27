#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from cart_controller import cart_controller
from std_srvs.srv import Empty
import time

first_pendulum_state = JointState()
second_pendulum_state = JointState()

cart_controller = cart_controller("controller_commander", 100)

while True:
    cart_controller.actuate_wheels(0)

    first_pendulum_state = cart_controller.get_joint_state("pendulum_joint_to_first_pendulum")
    second_pendulum_state = cart_controller.get_joint_state("first_pendulum_to_second_pendulum")

    # print(first_pendulum_state.position)
    # print(second_pendulum_state.velocity)

    time.sleep(10)

    cart_controller.reset_simulation()


