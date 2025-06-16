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
                # print(transform)
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

    def create_pointcloud_message(self, pointcloud: o3d.geometry.PointCloud, frame: str) -> PointCloud2:
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
        
    def convert_pose_to_pose_stamped(self, pose: Pose, frame: str = "map") -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = frame
        ps.pose = pose
        return ps
    
    def rotate_pose(self, pose: PoseStamped, angle:float) -> PoseStamped:
        """
        Rotate a stamped pose around its own local Z axis by the given angle (radians).

        Parameters
        ----------
        pose : geometry_msgs.msg.PoseStamped
            The input pose to rotate.
        angle : float
            The rotation angle in radians (positive = counter‐clockwise around Z).

        Returns
        -------
        geometry_msgs.msg.PoseStamped
            A new PoseStamped with the same position but rotated orientation.
        """
        if not isinstance(pose, PoseStamped):
            raise TypeError("pose must be a geometry_msgs.msg.PoseStamped")

        # extract the original quaternion [x, y, z, w]
        q = pose.pose.orientation
        q_orig = [q.x, q.y, q.z, q.w]

        # build a quaternion for a rotation about Z by `angle`
        q_delta = quaternion_from_euler(0.0, 0.0, angle)

        # compose: apply the delta in the pose's local frame
        q_new = quaternion_multiply(q_orig, q_delta)

        # construct the new PoseStamped
        new_pose = PoseStamped()
        new_pose.header = copy.deepcopy(pose.header)
        new_pose.pose.position.x = pose.pose.position.x
        new_pose.pose.position.y = pose.pose.position.y
        new_pose.pose.position.z = pose.pose.position.z
        new_pose.pose.orientation.x = float(q_new[0])
        new_pose.pose.orientation.y = float(q_new[1])
        new_pose.pose.orientation.z = float(q_new[2])
        new_pose.pose.orientation.w = float(q_new[3])

        return new_pose
       

if __name__ == '__main__':
    ros_handler = ROSHandler()
    transform = ros_handler.get_stamped_pose("hand_camera_frame", "map")
    print(f'Transform: {transform}')