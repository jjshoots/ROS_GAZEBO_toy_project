#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import String
from sensor_msgs.msg import JointState

rospy.init_node("controller")
r = rospy.Rate(100) # 10hz

joint_state_publisher = rospy.Publisher('joint_states', JointState, queue_size=1)

while not rospy.is_shutdown():

    current_time = rospy.get_time()
    angle = math.sin(current_time)

    joint_position = JointState()

    joint_position.header.stamp = rospy.Time.now()
    joint_position.name = (
        "base_to_right_leg", "base_to_left_leg", "left_leg_to_left_wheel", "right_leg_to_right_wheel")
    joint_position.position = (angle, angle, angle, angle)
    
    joint_state_publisher.publish(joint_position)
    r.sleep()