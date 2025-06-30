#!/usr/bin/env python

from my_uils_new import 

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

    def get_current_image(self, timeout=2.0, show=False):
        print("Waiting for image...")
        self.msg = None
        # start = rospy.Time.now()
        while self.msg is None and not rospy.is_shutdown():
            # if (rospy.Time.now() - start).to_sec() > timeout:
            #     raise RuntimeError("Timed out waiting for image")
            rospy.sleep(0.1)

        # convert the first image we got
        try:
            image = self.bridge.imgmsg_to_cv2(self.msg, "bgr8")  # OpenCV expects BGR
            self.msg = None                                      # optional: clear it
            if show:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 700, 550)
                cv2.imshow("Original Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return image
        except CvBridgeError as e:
            rospy.logerr(e)
            return None
