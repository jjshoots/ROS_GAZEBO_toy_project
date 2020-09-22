#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates

import time
import numpy as np

class cart_controller:

    def __init__(self, node_name, node_rate):
        # initialize variables
        self.__have_checked_joint_index__ = False
        self.__have_checked_model_index__ = False
        
        # this is a dictionary of model_name -> model index, or joint_name -> joint_index
        self.__joint_dictionary__ = dict()
        self.__model_dictionary__ = dict()

        self.__joint_state__ = JointState()
        self.__model_state__ = ModelStates()

        self.__state_vector__ = []   

        # INIT STARTS HERE

        # start node
        rospy.init_node(node_name)
        self.r = rospy.Rate(node_rate, reset=True)

        # service for resetting simulation
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # model and joint state subscriber
        self.joint_state_subscriber = rospy.Subscriber("/pendulum/joint_states", JointState, self.read_joint_states)
        self.model_state_subscriber = rospy.Subscriber("/gazebo/model_states", ModelStates, self.read_model_states)

        # wheel controller
        self.FLwheel_publisher = rospy.Publisher('/pendulum/FLwheel_controller/command', Float64, queue_size=1)
        self.FRwheel_publisher = rospy.Publisher('/pendulum/FRwheel_controller/command', Float64, queue_size=1)
        self.BLwheel_publisher = rospy.Publisher('/pendulum/BLwheel_controller/command', Float64, queue_size=1)
        self.BRwheel_publisher = rospy.Publisher('/pendulum/BRwheel_controller/command', Float64, queue_size=1)

        # don't init anything else when the robot is not available yet
        while len(self.__joint_state__.name) == 0 or len(self.__model_state__.name) == 0:
            pass

    # caller to run the loop at a fixed node_rate
    def throttle_Hz(self):
        self.r.sleep()

    # actuate wheels based on a certain rad/sec
    def actuate_wheels(self, wheel_speed):
        if not rospy.is_shutdown():
            wheel_speed_command = Float64()
            wheel_speed_command.data = wheel_speed

            self.FLwheel_publisher.publish(wheel_speed_command)
            self.FRwheel_publisher.publish(wheel_speed_command)
            self.BLwheel_publisher.publish(wheel_speed_command)
            self.BRwheel_publisher.publish(wheel_speed_command)
            self.r.sleep()

    # read the joint states, this function is called by the subscriber node
    def read_joint_states(self, data):
        self.__joint_state__ = data

        if not self.__have_checked_joint_index__:
            for index, name in enumerate(data.name):
                self.__joint_dictionary__[name] = index
                
            self.__have_checked_joint_index__ = True

    def get_joint_names(self):
        return self.__joint_dictionary__.keys()

    def get_joint_state(self, joint_name):
        selected_joint = JointState()

        if joint_name in self.__joint_dictionary__:
            index = self.__joint_dictionary__[joint_name]

            selected_joint.header = self.__joint_state__.header
            selected_joint.name = joint_name
            selected_joint.position = self.__joint_state__.position[index]
            selected_joint.velocity = self.__joint_state__.velocity[index]

            return selected_joint
        else:
            raise Exception("Can't access joint: " + joint_name)



    # read the model states, this function is called by the subscriber node
    def read_model_states(self, data):
        self.__model_state__ = data

        if not self.__have_checked_model_index__:
            for index, name in enumerate(data.name):
                self.__model_dictionary__[name] = index
                
            self.have_checked_index = True

    def get_model_names(self):
        return self.__model_dictionary__.keys()

    def get_model_state(self, model_name):
        selected_model = ModelStates()

        if model_name in self.__model_dictionary__:
            index = self.__model_dictionary__[model_name]

            selected_model.name = model_name
            selected_model.pose = self.__model_state__.pose[index]
            selected_model.twist = self.__model_state__.twist[index]

            return selected_model
        else:
            raise Exception("Can't access model: " + model_name)


    # get state of everything as a state vector
    def get_state_vector(self):
        self.__state_vector__ = []   

        # pendulum states, dtype = sensor_msgs/JointState
        self.__state_vector__.append(self.get_joint_state("pendulum_joint_to_first_pendulum").position)
        self.__state_vector__.append(self.get_joint_state("pendulum_joint_to_first_pendulum").velocity)
        self.__state_vector__.append(self.get_joint_state("first_pendulum_to_second_pendulum").position)
        self.__state_vector__.append(self.get_joint_state("first_pendulum_to_second_pendulum").velocity)
        
        # model state, dtype = gazebo_msgs/ModelStates
        self.__state_vector__.append(self.get_model_state("my_robot").pose.position.y)
        self.__state_vector__.append(self.get_model_state("my_robot").twist.linear.y)

        self.__state_vector__ = np.array(self.__state_vector__)

        return self.__state_vector__

    # returns the instantaneous reward of being in a state
    def reward(self):
        # value function used is absolute of states multiplied by scaler
        raw_values = np.absolute(self.__state_vector__)
        scaler = np.array([1, 1, 1, 1, 1, 1])

        return -np.dot(raw_values, scaler)

    # reset simulation caller
    def reset_simulation(self):
        # print("Resetting simulation...")
        self.reset()