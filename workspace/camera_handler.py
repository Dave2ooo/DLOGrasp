#!/usr/bin/env python
import os
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import sensor_msgs.msg


class ImageSubscriber:
    def __init__(self, topic):
        self.topic = topic
        self.image = None
        self.bridge = CvBridge()
        rospy.Subscriber(topic, sensor_msgs.msg.Image, self.callback)

    def callback(self, msg):
        self.msg = msg

    def get_current_image(self):
        self.msg = None
        rospy.sleep(0.2)
        rospy.loginfo("No image received yet. Waiting")
        while self.msg is None and not rospy.is_shutdown():
            print("Message is not None")
            print(self.msg)
            rospy.sleep(0.2)
            print('.', end='')
            try:
                self.image = self.bridge.imgmsg_to_cv2(self.msg, "rgb8")
                print('Image received')
                return self.image
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
