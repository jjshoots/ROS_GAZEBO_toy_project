#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from cart_controller import cart_controller

first_pendulum_state = JointState()
second_pendulum_state = JointState()

base_state = ModelStates()

cart_controller = cart_controller("controller_commander", 10)

cart_controller.reset_simulation()

while True:
    cart_controller.actuate_wheels(0)

    # get the pendulum angular states
    # pendulum states, dtype = sensor_msgs/JointState
    first_pendulum_state = cart_controller.get_joint_state("pendulum_joint_to_first_pendulum")
    second_pendulum_state = cart_controller.get_joint_state("first_pendulum_to_second_pendulum")

    # get the model linear states
    # model state, dtype = gazebo_msgs/ModelStates
    base_state = cart_controller.get_model_state("my_robot")

    # if the pendulum goes out of bounds, reset sim
    if(abs(base_state.pose.position.y) > 10
    or abs(first_pendulum_state.position) > 0.2
    or abs(second_pendulum_state.position) > 0.2):
        cart_controller.reset_simulation()
        pass

    cart_controller.throttle_Hz()

    # print("Script is still ok.")

