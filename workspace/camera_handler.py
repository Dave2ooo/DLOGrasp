#!/usr/bin/env python
import os
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import sensor_msgs.msg


class ImageSubscriber:
    def __init__(self, topic):
        self.bridge = CvBridge()
        self.msg    = None
        rospy.Subscriber(topic, sensor_msgs.msg.Image, self.callback, queue_size=1)

    def callback(self, msg):
        self.msg = msg            # just store it; no conversion here

    def get_current_image(self, timeout=2.0):
        start = rospy.Time.now()
        while self.msg is None and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                raise RuntimeError("Timed out waiting for image")
            rospy.sleep(0.05)

        # convert the first image we got
        try:
            image = self.bridge.imgmsg_to_cv2(self.msg, "bgr8")  # OpenCV expects BGR
            self.msg = None                                      # optional: clear it
            return image
        except CvBridgeError as e:
            rospy.logerr(e)
            return None
