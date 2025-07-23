# #!/usr/bin/env python
# import os
# import rospy
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np

# import sensor_msgs.msg


# class ImageSubscriber:
#     def __init__(self, topic):
#         self.bridge = CvBridge()
#         self.msg    = None
#         rospy.Subscriber(topic, sensor_msgs.msg.Image, self.callback, queue_size=1)

#     def callback(self, msg):
#         self.msg = msg            # just store it; no conversion here

#     def get_current_image(self, timeout=2.0, show=False):
#         print("Waiting for image...")
#         self.msg = None
#         # start = rospy.Time.now()
#         while self.msg is None and not rospy.is_shutdown():
#             # if (rospy.Time.now() - start).to_sec() > timeout:
#             #     raise RuntimeError("Timed out waiting for image")
#             rospy.sleep(0.1)

#         # convert the first image we got
#         try:
#             image = self.bridge.imgmsg_to_cv2(self.msg, "bgr8")  # OpenCV expects BGR
#             self.msg = None                                      # optional: clear it
#             if show:
#                 title = "Camera Image"
#                 cv2.namedWindow(title, cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow(title, 700, 550)
#                 cv2.imshow("Original Image", image)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#             return image
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return None


#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.msg

class ImageSubscriber:
    def __init__(self, topic):
        self.bridge = CvBridge()
        self.topic  = topic

    def get_current_image(self, timeout=1000000.0, show=False):
        print("Waiting for Image...")
        # This call always returns the next incoming Image,
        # not a buffered one.
        try:
            msg = rospy.wait_for_message(self.topic,
                                         sensor_msgs.msg.Image,
                                         timeout=timeout)
        except rospy.ROSException:
            raise RuntimeError(f"Timed out after {timeout}s waiting for image")

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if show:
                title = "Camera Image"
                cv2.imshow(title, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return image

        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge conversion failed: {e}")
            return None