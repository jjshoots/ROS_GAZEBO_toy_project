#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class cart_controller:

    have_checked_index = False

    joint_dictionary = dict()
    joint_state = JointState()

    def __init__(self, node_name, node_rate):
        rospy.init_node(node_name)
        self.r = rospy.Rate(node_rate)

        self.link_state_subscriber = rospy.Subscriber("/pendulum/joint_states", JointState, self.read_states)
        
        self.FLwheel_publisher = rospy.Publisher('/pendulum/FLwheel_controller/command', Float64, queue_size=1)
        self.FRwheel_publisher = rospy.Publisher('/pendulum/FRwheel_controller/command', Float64, queue_size=1)
        self.BLwheel_publisher = rospy.Publisher('/pendulum/BLwheel_controller/command', Float64, queue_size=1)
        self.BRwheel_publisher = rospy.Publisher('/pendulum/BRwheel_controller/command', Float64, queue_size=1)

        while len(self.joint_state.name) == 0:
            pass

    def actuate_wheels(self, wheel_speed):
        if not rospy.is_shutdown():
            wheel_speed_command = Float64()
            wheel_speed_command.data = wheel_speed

            self.FLwheel_publisher.publish(wheel_speed_command)
            self.FRwheel_publisher.publish(wheel_speed_command)
            self.BLwheel_publisher.publish(wheel_speed_command)
            self.BRwheel_publisher.publish(wheel_speed_command)
            self.r.sleep()

    def read_states(self, data):
        self.joint_state = data

        if not self.have_checked_index:
            for index, name in enumerate(data.name):
                self.joint_dictionary[name] = index
                
            self.have_checked_index = True

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
            selected_joint.effort = self.joint_state.effort[index]

            return selected_joint
        else:
            raise Exception("Can't access joint: " + joint_name)

