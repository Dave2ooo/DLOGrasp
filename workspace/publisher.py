import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped
import numpy as np
from std_msgs.msg import String
import tf
from std_msgs.msg import Bool



class PathPublisher:
    def __init__(self, topic: str, frame: str = "map"):
        self.pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)
        self.frame = frame

    def publish(self, path: list[PoseStamped]) -> None:
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
        # Validate inputs
        if not isinstance(path, (list, tuple)):
            raise TypeError("path must be a list or tuple of PoseStamped messages")
        for ps in path:
            if not isinstance(ps, PoseStamped):
                raise TypeError("each element of path must be a geometry_msgs.msg.PoseStamped")
        

        # Build Path message
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.frame

        # Append each stamped pose (overwrite header to unify frame/time if desired)
        for ps in path:
            ps_out = PoseStamped()
            ps_out.header = path_msg.header
            ps_out.pose = ps.pose
            path_msg.poses.append(ps_out)

        self.pub.publish(path_msg)
        print("Published path")


class PosePublisher:
    def __init__(self, topic: str):
        self.pub = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=True)
        
    def publish(self, pose: PoseStamped) -> None:
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
        # Build a PoseStamped if necessary
        if isinstance(pose, PoseStamped):
            self.pub.publish(pose)
            print("Published Pose")
        else:
            raise TypeError("publish_pose requires a Pose or PoseStamped")

        return
    
    

class CommandPublisher:
    """Publish simple command strings on a ROS topic."""

    def __init__(self, topic: str = "command_word", latch: bool = True):
        """
        Parameters
        ----------
        topic : str
            Name of the ROS topic to publish to.
        latch : bool, optional
            If *True* the last message is latched so late subscribers still receive it.
        """
        self.pub = rospy.Publisher(topic, String, queue_size=1, latch=latch)
        rospy.loginfo(f"CommandPublisher set up on /{topic} (latch={latch})")

    def publish(self, command: str) -> None:
        """Publish *command* immediately.
        
        Raises
        ------
        TypeError
            If *command* is not a plain string.
        """
        if not isinstance(command, str):
            raise TypeError("CommandPublisher.publish expects a string")
        self.pub.publish(String(data=command))
        rospy.loginfo(f"Published command: '{command}'")



class PointcloudPublisher:
    def __init__(self, topic: str, frame_id: str = "map"):
        """
        Initialize a ROS publisher for PointCloud2 messages.

        Parameters
        ----------
        topic : str
            The name of the ROS topic to publish the point cloud on.
        frame_id : str
            The frame_id to stamp each message with (default: "map").
        """
        # Publisher for PointCloud2
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
        self.frame_id = frame_id

    def publish(self, pointcloud: o3d.geometry.PointCloud) -> None:
        """
        Convert an Open3D PointCloud into a ROS PointCloud2 and publish it.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The Open3D point cloud to publish (XYZ only).
        """
        # Extract XYZ coordinates to a list of tuples
        pts = np.asarray(pointcloud.points, dtype=np.float32)
        xyz = [(float(x), float(y), float(z)) for x, y, z in pts]

        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        # Build PointCloud2 message (XYZ as float32)
        pc2_msg = point_cloud2.create_cloud_xyz32(header, xyz)

        # Publish
        self.pub.publish(pc2_msg)


class PointStampedPublisher:
    def __init__(self, topic: str, frame_id: str = "map"):
        """
        Initialize a ROS publisher for PointStamped messages.

        Parameters
        ----------
        topic : str
            The name of the ROS topic to publish the PointStamped on.
        frame_id : str
            The frame_id to stamp each message with (default: "map").
        """
        self.pub = rospy.Publisher(topic, PointStamped, queue_size=1, latch=True)
        self.frame_id = frame_id

    def publish(self, pt):
        """
        Publish a 3D point as a PointStamped, accepting:
          - numpy array of shape (3,) or (3,1)
          - geometry_msgs.msg.Point
          - geometry_msgs.msg.PointStamped

        Parameters
        ----------
        pt : numpy.ndarray or Point or PointStamped
            The point to publish.
        """
        # Build the stamped message
        if isinstance(pt, PointStamped):
            # Update header only
            msg = pt
        else:
            # Convert to Point if needed
            if isinstance(pt, Point):
                p = pt
            elif isinstance(pt, np.ndarray):
                arr = pt.flatten()
                if arr.size != 3:
                    raise ValueError("NumPy array must have 3 elements")
                p = Point(float(arr[0]), float(arr[1]), float(arr[2]))
            else:
                raise TypeError("publish() accepts numpy.ndarray, Point, or PointStamped")

            # Wrap into a fresh PointStamped
            msg = PointStamped()
            msg.point = p

        # Stamp the header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        msg.header = header

        # Publish
        self.pub.publish(msg)


class TFBroadcaster:
    def __init__(self):
        self.br = tf.TransformBroadcaster()
        self.finished_sub = rospy.Subscriber('/finished_cleanly', Bool, self.finished_callback)
        self.finished = False

    def finished_callback(self, msg):
        self.finished = msg.data

    def execute(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.finished:
            # self.br.sendTransform((0.517, -0.967, 0.819), (-0.01075, 0.7118, -0.702, 0.01994), rospy.Time.now(), "ar_marker/17fix", "map")
            self.br.sendTransform((20, -0.967, 0.819), (-0.01075, 0.7118, -0.702, 0.01994), rospy.Time.now(), "ar_marker/17fix", "map")
            rospy.loginfo("Broadcasting tf")
            rate.sleep()
        rospy.loginfo("Stopping the tf broadcaster.")



# "position": (0.517, -0.967, 0.819),
# "orientation": (-0.01075, 0.7118, -0.702, 0.01994)


LOCATION = "TUW"

MARKER_POSITIONS = {
    "TUW": {
        "ar_marker/15fix": {
            "position": (-1.100, -0.55666, 0.41058),
            "orientation": (-0.0129, 0.92591, -0.3777, 0.00639)
        },
        "ar_marker/17fix": {
            "position": (0.517, -0.967, 0.819),
            "orientation": (-0.01075, 0.7118, -0.702, 0.01994)
        },
        "ar_marker/20fix": {
            "position": (-1.6413, -0.0723, 0.80751),
            "orientation": (0.4989, -0.49854, 0.5005, -0.50194)
        },
    } 
}



class TFBroadcaster:
    def __init__(self):
        self.br = tf.TransformBroadcaster()
        self.finished_sub = rospy.Subscriber('/finished_cleanly', Bool, self.finished_callback)
        self.finished = False
        # self.marker_position = marker_positions

    def finished_callback(self, msg):
        self.finished = msg.data

    def execute(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown() and not self.finished:
            # for frame_id, pose in self.marker_position.items():
            #     self.br.sendTransform(pose["position"], pose["orientation"], rospy.Time.now(), frame_id, "map")
            self.br.sendTransform((5.17, -0.967, 0.819), (-0.01075, 0.7118, -0.702, 0.01994), rospy.Time.now(), "ar_marker/17fix", "map")
            rate.sleep()
        rospy.loginfo("Stopping the tf broadcaster.")

# def main():
#     rospy.init_node('fixed_tf_broadcaster', anonymous=True)
#     rospy.loginfo("Starting tf broadcaster")

    
#     tf_broadcaster = TFBroadcaster(MARKER_POSITIONS[LOCATION])

#     tf_broadcaster.execute()

# if __name__ == '__main__':
#     main()


# marker = {"position": (0.327, 1.770, 0.633), "orientation": (-0.711, -0.004, 0.008, 0.703)}