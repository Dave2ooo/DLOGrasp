import sys
import copy
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, quaternion_slerp, quaternion_matrix, quaternion_from_matrix, quaternion_multiply
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d

class ROSHandler():
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_current_pose(self, target_frame, reference_frame):
        print(f"Waiting for transform from {reference_frame} to {target_frame}...")
        transform = None
        while transform is None and not rospy.is_shutdown():
            rospy.sleep(0.2)
            try:
                transform = self.tf_buffer.lookup_transform(reference_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
                print("Transform received.")
                pose = convert_transform_stamped_to_pose_stamped(transform)
                return pose
            except tf2_ros.LookupException as e:
                rospy.logerr("Transform lookup failed: {}".format(e))
            except tf2_ros.ExtrapolationException as e:
                rospy.logerr("Transform extrapolation error: {}".format(e))
