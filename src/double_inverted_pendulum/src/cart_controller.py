#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64

class cart_controller:
    def __init__(self, node_name, node_rate):
        rospy.init_node(node_name)
        self.r = rospy.Rate(node_rate)
        
        self.FLwheel_publisher = rospy.Publisher('/pendulum/FLwheel_controller/command', Float64, queue_size=1)
        self.FRwheel_publisher = rospy.Publisher('/pendulum/FRwheel_controller/command', Float64, queue_size=1)
        self.BLwheel_publisher = rospy.Publisher('/pendulum/BLwheel_controller/command', Float64, queue_size=1)
        self.BRwheel_publisher = rospy.Publisher('/pendulum/BRwheel_controller/command', Float64, queue_size=1)

    def actuate_wheels(self, wheel_speed):
        if not rospy.is_shutdown():
            wheel_speed_command = Float64()
            wheel_speed_command.data = wheel_speed

            self.FLwheel_publisher.publish(wheel_speed_command)
            self.FRwheel_publisher.publish(wheel_speed_command)
            self.BLwheel_publisher.publish(wheel_speed_command)
            self.BRwheel_publisher.publish(wheel_speed_command)
            self.r.sleep()
