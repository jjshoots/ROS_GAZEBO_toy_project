#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
import time

class cart_controller:

    have_checked_joint_index = False
    have_checked_model_index = False
    
    # this is a dictionary of model_name -> model index, or joint_name -> joint_index
    joint_dictionary = dict()
    model_dictionary = dict()

    joint_state = JointState()
    model_state = ModelStates()

    def __init__(self, node_name, node_rate):
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
        while len(self.joint_state.name) == 0 or len(self.model_state.name) == 0:
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
        self.joint_state = data

        if not self.have_checked_joint_index:
            for index, name in enumerate(data.name):
                self.joint_dictionary[name] = index
                
            self.have_checked_joint_index = True

    def get_joint_names(self):
        return self.joint_dictionary.keys()

    def get_joint_state(self, joint_name):
        selected_joint = JointState()

        if joint_name in self.joint_dictionary:
            index = self.joint_dictionary[joint_name]

            selected_joint.header = self.joint_state.header
            selected_joint.name = joint_name
            selected_joint.position = self.joint_state.position[index]
            selected_joint.velocity = self.joint_state.velocity[index]

            return selected_joint
        else:
            raise Exception("Can't access joint: " + joint_name)



    # read the model states, this function is called by the subscriber node
    def read_model_states(self, data):
        self.model_state = data

        if not self.have_checked_model_index:
            for index, name in enumerate(data.name):
                self.model_dictionary[name] = index
                
            self.have_checked_index = True

    def get_model_names(self):
        return self.model_dictionary.keys()

    def get_model_state(self, model_name):
        selected_model = ModelStates()

        if model_name in self.model_dictionary:
            index = self.model_dictionary[model_name]

            selected_model.name = model_name
            selected_model.pose = self.model_state.pose[index]
            selected_model.twist = self.model_state.twist[index]

            return selected_model
        else:
            raise Exception("Can't access model: " + model_name)


    # reset simulation caller
    def reset_simulation(self):
        print("Resetting simulation...")
        self.reset()