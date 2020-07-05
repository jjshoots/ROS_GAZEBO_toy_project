#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.ros_img_sub = rospy.Subscriber("/pendulum_cart/camera1/image_raw", Image, self.camera_callback)

    def camera_callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        cv2.imshow("Image Window", cv_img)
        cv2.waitKey(3)


print("Hello")
rospy.init_node("camera_listener")
camera_listener = image_converter()

rospy.spin()