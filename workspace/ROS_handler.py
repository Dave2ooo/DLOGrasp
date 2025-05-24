import sys
import copy
import rospy
from tf.transformations import quaternion_from_euler
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_slerp, quaternion_matrix, quaternion_from_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d

class ROSHandler():
    def __init__(self):
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
        
    def interpolate_poses(self, pose_start, pose_end, num_steps: int = 10) -> list:
        """
        Generate a linear‐in‐translation, slerp‐in‐rotation path between two poses,
        where inputs can be either TransformStamped or geometry_msgs.msg.Pose.
        Returns a list of Pose messages.

        Parameters
        ----------
        pose_start : TransformStamped or Pose
            The starting gripper pose.
        pose_end : TransformStamped or Pose
            The desired end gripper pose.
        num_steps : int
            Number of intermediate steps (excluding start), so total poses = num_steps+1.

        Returns
        -------
        List[Pose]
            A list of geometry_msgs.msg.Pose for the interpolated path.
        """
        def extract(p):
            # returns (t: np.array[3], q: np.array[4])
            if isinstance(p, TransformStamped):
                t = p.transform.translation
                q = p.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])
            elif isinstance(p, Pose):
                pos = p.position
                ori = p.orientation
                return np.array([pos.x, pos.y, pos.z]), np.array([ori.x, ori.y, ori.z, ori.w])
            else:
                raise TypeError("pose must be TransformStamped or Pose")

        t_start, q_start = extract(pose_start)
        t_end,   q_end   = extract(pose_end)

        path: list = []
        for i in range(num_steps + 1):
            alpha = i / float(num_steps)
            # lerp translation
            t_i = (1 - alpha) * t_start + alpha * t_end
            # slerp rotation
            q_i = quaternion_slerp(q_start, q_end, alpha)

            # build Pose
            p = Pose()
            p.position.x = float(t_i[0])
            p.position.y = float(t_i[1])
            p.position.z = float(t_i[2])
            p.orientation.x = float(q_i[0])
            p.orientation.y = float(q_i[1])
            p.orientation.z = float(q_i[2])
            p.orientation.w = float(q_i[3])
            path.append(p)

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

    def publish_pose(self, pose, topic: str, frame: str = "map") -> None:
        """
        Publish a Pose or PoseStamped message to a ROS topic.

        Parameters
        ----------
        pose : geometry_msgs.msg.Pose or geometry_msgs.msg.PoseStamped
            The pose to publish.
        topic : str
            The ROS topic name to publish to.
        frame : str
            The frame_id to stamp the message with (used if publishing a bare Pose).
        """
        # Initialize publisher (latch last value)
        pub = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=True)
        rospy.sleep(0.1)  # allow publisher registration

        # Build a PoseStamped if necessary
        if isinstance(pose, PoseStamped):
            ps = pose
        elif isinstance(pose, Pose):
            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = frame
            ps.pose = pose
        else:
            raise TypeError("publish_pose requires a Pose or PoseStamped")

        # Publish once
        pub.publish(ps)

    def publish_path(self, path: list, topic: str, frame: str = "map") -> None:
        """
        Publish a sequence of Pose messages as a nav_msgs/Path on a ROS topic.

        Parameters
        ----------
        path : list of geometry_msgs.msg.Pose
            The ordered list of poses to publish.
        topic : str
            The ROS topic name to publish the Path message on.
        frame : str
            The frame_id to stamp the Path header with.
        """
        # Validate inputs
        if not isinstance(path, (list, tuple)):
            raise TypeError("path must be a list or tuple of Pose messages")
        for p in path:
            if not isinstance(p, Pose):
                raise TypeError("each element of path must be a geometry_msgs.msg.Pose")

        # Create publisher (latched so it stays around)
        pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)
        # wait briefly to register
        rospy.sleep(0.1)

        # Build the Path message
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame

        # Convert each Pose into a stamped Pose
        for pose in path:
            ps = PoseStamped()
            ps.header = path_msg.header  # same timestamp and frame for all
            ps.pose = pose
            path_msg.poses.append(ps)

        # Publish once
        pub.publish(path_msg)


if __name__ == '__main__':
    ros_handler = ROSHandler()
    transform = ros_handler.get_stamped_pose("hand_camera_frame", "map")
    print(f'Transform: {transform}')