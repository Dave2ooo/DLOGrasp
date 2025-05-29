import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import open3d as o3d


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