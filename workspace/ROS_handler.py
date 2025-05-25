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

    def get_current_pose(self, target_frame, reference_frame):
        print(f"Waiting for transform from {reference_frame} to {target_frame}...")
        transform = None
        while transform is None and not rospy.is_shutdown():
            rospy.sleep(0.2)
            try:
                transform = self.tf_buffer.lookup_transform(reference_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
                return transform
            except tf2_ros.LookupException as e:
                rospy.logerr("Transform lookup failed: {}".format(e))
            except tf2_ros.ExtrapolationException as e:
                rospy.logerr("Transform extrapolation error: {}".format(e))

    def interpolate_poses(self, pose_start, pose_end, num_steps: int) -> list[PoseStamped]:
        """
        Generate a smooth path of PoseStamped messages interpolating between two poses.
        Inputs can be either PoseStamped or TransformStamped. Outputs are PoseStamped.

        Parameters
        ----------
        pose_start : PoseStamped or TransformStamped
            Starting pose.
        pose_end : PoseStamped or TransformStamped
            Ending pose.
        num_steps : int
            Number of intermediate steps (inclusive of end). Returned list length = num_steps+1.

        Returns
        -------
        List[PoseStamped]
            The interpolated sequence from start to end.
        """
        def extract(t):
            # Return (translation: np.ndarray(3), quaternion: np.ndarray(4), header: Header)
            if isinstance(t, TransformStamped):
                trans = t.transform.translation
                rot = t.transform.rotation
                hdr = copy.deepcopy(t.header)
                return (np.array([trans.x, trans.y, trans.z], dtype=float),
                        np.array([rot.x, rot.y, rot.z, rot.w], dtype=float),
                        hdr)
            elif isinstance(t, PoseStamped):
                pos = t.pose.position
                ori = t.pose.orientation
                hdr = copy.deepcopy(t.header)
                return (np.array([pos.x, pos.y, pos.z], dtype=float),
                        np.array([ori.x, ori.y, ori.z, ori.w], dtype=float),
                        hdr)
            else:
                raise TypeError("pose_start and pose_end must be PoseStamped or TransformStamped")

        t0, q0, hdr0 = extract(pose_start)
        t1, q1, hdr1 = extract(pose_end)
        # Use start header frame_id; keep stamp constant (or update if desired)
        frame_id = hdr0.frame_id
        stamp = hdr0.stamp

        path = []
        for i in range(num_steps + 1):
            alpha = i / float(num_steps)
            # linear translation
            ti = (1 - alpha) * t0 + alpha * t1
            # slerp rotation
            qi = quaternion_slerp(q0, q1, alpha)

            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.header.stamp = stamp
            ps.pose.position.x = float(ti[0])
            ps.pose.position.y = float(ti[1])
            ps.pose.position.z = float(ti[2])
            ps.pose.orientation.x = float(qi[0])
            ps.pose.orientation.y = float(qi[1])
            ps.pose.orientation.z = float(qi[2])
            ps.pose.orientation.w = float(qi[3])
            path.append(ps)

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

    def publish_pose(self, pose: PoseStamped, topic: str, frame: str = "map") -> None:
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
            pub.publish(pose)
        else:
            raise TypeError("publish_pose requires a Pose or PoseStamped")

        return

    def publish_path(self,
                     path: list[PoseStamped],
                     topic: str,
                     frame: str = "map") -> None:
        """
        Publish a sequence of PoseStamped messages as a nav_msgs/Path on a ROS topic.

        Parameters
        ----------
        path : list of geometry_msgs.msg.PoseStamped
            The ordered list of stamped poses to publish.
        topic : str
            The ROS topic name to publish the Path message on.
        frame : str
            The frame_id to stamp the Path header with.
        """
        import rospy
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped

        # Validate inputs
        if not isinstance(path, (list, tuple)):
            raise TypeError("path must be a list or tuple of PoseStamped messages")
        for ps in path:
            if not isinstance(ps, PoseStamped):
                raise TypeError("each element of path must be a geometry_msgs.msg.PoseStamped")

        # Create a latched publisher so the path persists
        pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)
        rospy.sleep(0.1)  # allow registration

        # Build Path message
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame

        # Append each stamped pose (overwrite header to unify frame/time if desired)
        for ps in path:
            ps_out = PoseStamped()
            ps_out.header = path_msg.header
            ps_out.pose = ps.pose
            path_msg.poses.append(ps_out)

        pub.publish(path_msg)


    def convert_pose_to_pose_stamped(self, pose: Pose, frame: str = "map") -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = frame
        ps.pose = pose
        return ps
    

if __name__ == '__main__':
    ros_handler = ROSHandler()
    transform = ros_handler.get_stamped_pose("hand_camera_frame", "map")
    print(f'Transform: {transform}')