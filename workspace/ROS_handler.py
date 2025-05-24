import sys
import copy
import rospy
from tf.transformations import quaternion_from_euler
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_slerp, quaternion_matrix, quaternion_from_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d

class ROSHandler():
    def __init__(self):       
        rospy.init_node("ROS Handler", anonymous=True) 
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_stamped_pose(self, target_frame, reference_frame):
        rospy.sleep(1)  # Allow time for buffer to fill
        
        try:
            transform = self.tf_buffer.lookup_transform(reference_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
            return transform
        except tf2_ros.LookupException as e:
            rospy.logerr("Transform lookup failed: {}".format(e))
            return None
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr("Transform extrapolation error: {}".format(e))
            return None
        
    def interpolate_poses(self,
                          pose_start: TransformStamped,
                          pose_end: TransformStamped,
                          num_steps: int = 10) -> list[TransformStamped]:
        """
        Generate a linear‐in‐translation, slerp‐in‐rotation path between two poses.

        Parameters
        ----------
        pose_start : TransformStamped
            The starting gripper pose (camera/world frame).
        pose_end : TransformStamped
            The desired end pose (same frames).
        num_steps : int
            Number of intermediate steps (not including start), so the returned
            list will have num_steps+1 poses including the start at index 0
            and the end at index num_steps.

        Returns
        -------
        List[TransformStamped]
            A list of poses for the gripper to follow.
        """

        # extract start translation & quaternion
        ts = pose_start.transform.translation
        qs = pose_start.transform.rotation
        t_start = np.array([ts.x, ts.y, ts.z], dtype=np.float64)
        q_start = np.array([qs.x, qs.y, qs.z, qs.w], dtype=np.float64)

        # extract end translation & quaternion
        te = pose_end.transform.translation
        qe = pose_end.transform.rotation
        t_end = np.array([te.x, te.y, te.z], dtype=np.float64)
        q_end = np.array([qe.x, qe.y, qe.z, qe.w], dtype=np.float64)

        path = []
        for i in range(num_steps + 1):
            alpha = i / float(num_steps)
            # linear interp translation
            t_i = (1 - alpha) * t_start + alpha * t_end
            # spherical linear interp rotation
            q_i = quaternion_slerp(q_start, q_end, alpha)

            # build TransformStamped
            pose_i = TransformStamped()
            # copy header info from start, update stamp if desired
            pose_i.header = copy.deepcopy(pose_start.header)
            pose_i.child_frame_id = pose_start.child_frame_id

            pose_i.transform.translation.x = float(t_i[0])
            pose_i.transform.translation.y = float(t_i[1])
            pose_i.transform.translation.z = float(t_i[2])

            pose_i.transform.rotation.x = float(q_i[0])
            pose_i.transform.rotation.y = float(q_i[1])
            pose_i.transform.rotation.z = float(q_i[2])
            pose_i.transform.rotation.w = float(q_i[3])

            path.append(pose_i)

        return path

    def create_pointcloud_message(self,
                                  pointcloud: o3d.geometry.PointCloud,
                                  frame: str) -> PointCloud2:
        """
        Convert an Open3D PointCloud into a ROS PointCloud2 message.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The point cloud to convert.
        frame : str
            The ROS frame_id to stamp the message with.

        Returns
        -------
        sensor_msgs.msg.PointCloud2
            A PointCloud2 message ready for publishing (XYZ only).
        """
        # Extract points as a list of (x,y,z) tuples
        pts = np.asarray(pointcloud.points, dtype=np.float32)
        xyz = [(float(x), float(y), float(z)) for x, y, z in pts]

        # Build the header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame

        # Create the PointCloud2 message (XYZ only, 32-bit floats)
        pc2_msg = point_cloud2.create_cloud_xyz32(header, xyz)
        return pc2_msg



if __name__ == '__main__':
    ros_handler = ROSHandler()
    transform = ros_handler.get_stamped_pose("hand_camera_frame", "odom")
    print(f'Transform: {transform}')