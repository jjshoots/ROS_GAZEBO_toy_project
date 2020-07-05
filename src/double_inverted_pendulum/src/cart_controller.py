#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64

rospy.init_node("controller")
r = rospy.Rate(100) # 10hz

FLwheel_publisher = rospy.Publisher('/pendulum_cart/FLwheel_controller/command', Float64, queue_size=1)
FRwheel_publisher = rospy.Publisher('/pendulum_cart/FRwheel_controller/command', Float64, queue_size=1)
BLwheel_publisher = rospy.Publisher('/pendulum_cart/BLwheel_controller/command', Float64, queue_size=1)
BRwheel_publisher = rospy.Publisher('/pendulum_cart/BRwheel_controller/command', Float64, queue_size=1)


while not rospy.is_shutdown():

    FLvelocity_command = Float64()
    FLvelocity_command.data = -10.0
    
    FLwheel_publisher.publish(FLvelocity_command)
    FRwheel_publisher.publish(FLvelocity_command)
    BLwheel_publisher.publish(FLvelocity_command)
    BRwheel_publisher.publish(FLvelocity_command)

    r.sleep()