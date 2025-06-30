import os
import cv2, numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d
import rospy
import copy

from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, quaternion_slerp, quaternion_matrix, quaternion_from_matrix, quaternion_multiply
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from scipy.interpolate import BSpline

import networkx as nx

from skimage.morphology import skeletonize

from skimage.draw import line as skline

from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------------

def create_stamped_transform_from_trans_and_rot(translation, rotation):
    transform_stamped = TransformStamped()
    transform_stamped.transform.translation.x = translation[0]
    transform_stamped.transform.translation.y = translation[1]
    transform_stamped.transform.translation.z = translation[2]

    transform_stamped.transform.rotation.x = rotation[0]
    transform_stamped.transform.rotation.y = rotation[1]
    transform_stamped.transform.rotation.z = rotation[2]
    transform_stamped.transform.rotation.w = rotation[3]
    return transform_stamped

#region Pointcloud
# get_pointcloud
def convert_depth_map_to_pointcloud(depth: NDArray[np.floating] | Image.Image | NDArray[np.uint8], camera_parameters, stride: int | None = None) -> 'o3d.geometry.PointCloud':
    """Convert *depth map* → Open3D :class:`~open3d.geometry.PointCloud`.

    Parameters
    ----------
    depth
        ``HxW`` depth image in **metres** (``float32``). RGB depth encodings
        will be collapsed to a single channel.
    intrinsics
        Tuple ``(fx, fy, cx, cy)`` in pixels.  *If ``None``*, defaults to a
        simple pin‑hole model with ``fx = fy = max(H, W)``, ``cx = W/2``,
        ``cy = H/2``.
    stride
        Optional pixel stride for pre‑down‑sampling (e.g. ``stride=2`` keeps
        ¼ of the points; often faster than calling ``random_down_sample``
        afterwards).

    Returns
    -------
    o3d.geometry.PointCloud
        3‑D points in the **camera frame**.  Invalid/zero‑depth pixels are
        dropped automatically.
    """
    # --- Prepare depth ----------------------------------------------------
    if isinstance(depth, Image.Image):
        depth = np.asarray(depth).astype(np.float32)
    elif isinstance(depth, np.ndarray):
        depth = depth.astype(np.float32)
    else:
        raise TypeError("depth must be PIL.Image.Image or numpy.ndarray")

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    # Optional sub‑sampling for speed --------------------------------------
    if stride and stride > 1:
        depth = depth[::stride, ::stride]

    H, W = depth.shape
    fx, fy, cx, cy = camera_parameters

    # --- Convert to 3‑D ----------------------------------------------------
    i_coords, j_coords = np.indices((H, W), dtype=np.float32)
    z = depth
    x = (j_coords - cx) * z / fx
    y = (i_coords - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = pts[:, 2] > 0
    pts = pts[valid].astype(np.float64)

    # --- Pack into Open3D --------------------------------------------------
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def mask_pointcloud(pointcloud: 'o3d.geometry.PointCloud',
                    mask: np.ndarray,
                    camera_parameters) -> 'o3d.geometry.PointCloud':
    """
    Filter a point cloud using a 2D mask: retains only points whose projection in the image
    falls within the True region of the mask.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The point cloud returned by get_pointcloud().
    mask : np.ndarray
        2D boolean or binary mask of shape (H, W), where True (or 1) indicates pixels to keep.

    Returns
    -------
    o3d.geometry.PointCloud
        A new point cloud containing only the masked points.
    """
    # sanity checks
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("mask_pointcloud expects an open3d.geometry.PointCloud")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")

    # ensure boolean mask
    mask_bool = mask.astype(bool)
    H, W = mask_bool.shape

    # pull out the Nx3 array of points
    pts = np.asarray(pointcloud.points)
    if pts.size == 0:
        return o3d.geometry.PointCloud()  # nothing to keep

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # project back to pixel coords
    j = np.round(x * camera_parameters[0] / z + camera_parameters[2]).astype(int)
    i = np.round(y * camera_parameters[1] / z + camera_parameters[3]).astype(int)

    # build mask of points whose projection is in-bounds AND True in mask
    keep = (
        (i >= 0) & (i < H) &
        (j >= 0) & (j < W) &
        mask_bool[i, j]
    )

    # filter and rebuild point cloud
    filtered = pts[keep]
    pc_new = o3d.geometry.PointCloud()
    pc_new.points = o3d.utility.Vector3dVector(filtered)
    return pc_new

def transform_pointcloud_to_world(pointcloud: o3d.geometry.PointCloud, camera_pose: TransformStamped) -> o3d.geometry.PointCloud:
    """
    Transform a point cloud from camera coordinates into world coordinates
    using the given camera_pose (a tf2_ros TransformStamped).

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The point cloud in the camera frame.
    camera_pose : tf2_ros.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    o3d.geometry.PointCloud
        A new point cloud expressed in world coordinates.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # Extract translation and quaternion
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

    # Build 4x4 homogeneous transformation matrix
    T = quaternion_matrix(quat)        # rotation part
    T[:3, 3] = trans                  # translation part

    # Convert point cloud to Nx4 homogeneous coordinates
    pts = np.asarray(pointcloud.points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_hom = np.hstack((pts, ones))   # shape (N,4)

    # Apply transform
    pts_world_hom = (T @ pts_hom.T).T  # shape (N,4)
    pts_world = pts_world_hom[:, :3]

    # Build new point cloud in world frame
    pc_world = o3d.geometry.PointCloud()
    pc_world.points = o3d.utility.Vector3dVector(pts_world)
    return pc_world

def transform_points_to_world(points: np.ndarray, camera_pose: TransformStamped) -> np.ndarray:
    """
    Transform an array of 3D points from the camera frame into world coordinates.

    Parameters
    ----------
    points : np.ndarray, shape (N,3)
        Array of points in the camera's coordinate system.
    camera_pose : geometry_msgs.msg.TransformStamped
        Transform from camera frame to world frame.

    Returns
    -------
    np.ndarray, shape (N,3)
        The points expressed in the world coordinate system.
    """
    # Validate inputs
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N,3) array")

    # Build camera→world homogeneous matrix
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    Tcw = quaternion_matrix(quat)    # 4×4 rotation
    Tcw[:3, 3] = trans               # insert translation

    # Convert to homogeneous coordinates and apply
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_hom = np.hstack((pts, ones))  # (N,4)
    world_hom = (Tcw @ pts_hom.T).T   # (N,4)

    return world_hom[:, :3]


def transform_pointcloud_from_world(pointcloud: o3d.geometry.PointCloud, camera_pose: TransformStamped) -> o3d.geometry.PointCloud:
    """
    Transform a point cloud from world coordinates into the camera frame
    using the given camera_pose (camera→world).

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The point cloud in world coordinates.
    camera_pose : tf2_ros.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    o3d.geometry.PointCloud
        A new point cloud expressed in the camera coordinate frame.
    """

    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # Extract translation and rotation
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

    # Build camera→world matrix then invert for world→camera
    T_cam2world = quaternion_matrix(quat)
    T_cam2world[:3, 3] = trans
    T_world2cam = np.linalg.inv(T_cam2world)

    # Convert point cloud to homogeneous coords
    pts_world = np.asarray(pointcloud.points, dtype=np.float64)
    if pts_world.size == 0:
        return o3d.geometry.PointCloud()
    ones = np.ones((pts_world.shape[0], 1), dtype=np.float64)
    pts_hom = np.hstack((pts_world, ones))  # (N,4)

    # Apply world→camera transform
    pts_cam_hom = (T_world2cam @ pts_hom.T).T  # (N,4)
    pts_cam = pts_cam_hom[:, :3]

    # Build and return new point cloud
    pc_cam = o3d.geometry.PointCloud()
    pc_cam.points = o3d.utility.Vector3dVector(pts_cam)
    return pc_cam

def transform_point_to_world(point, camera_pose: TransformStamped) -> np.ndarray:
    """
    Transform a single 3D point from the camera frame into world coordinates.

    Parameters
    ----------
    point : sequence of three floats
        The [x, y, z] coordinates in the camera frame.
    camera_pose : geometry_msgs.msg.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    np.ndarray
        A length-3 array of the point in world coordinates.
    """
    # Validate inputs
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("point must be a sequence of three numbers [x, y, z]")
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # Build the 4×4 camera→world transform
    q = camera_pose.transform.rotation
    t = camera_pose.transform.translation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)

    T = quaternion_matrix(quat)    # 4×4 rotation+1s
    T[:3, 3] = trans               # insert translation

    # Apply to homogeneous point
    hom_pt = np.append(p, 1.0)      # [x, y, z, 1]
    world_hom = T @ hom_pt          # matrix multiplication

    return world_hom[:3]

def transform_point_from_world(point, camera_pose: TransformStamped) -> np.ndarray:
    """
    Transform a single 3D point from world coordinates into the camera frame.

    Parameters
    ----------
    point : sequence of three floats
        The [x, y, z] coordinates in the world frame.
    camera_pose : geometry_msgs.msg.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    np.ndarray
        A length-3 array of the point in camera coordinates.
    """
    # Validate inputs
    p_world = np.asarray(point, dtype=np.float64)
    if p_world.shape != (3,):
        raise ValueError("point must be a sequence of three numbers [x, y, z]")
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # Build the 4×4 camera→world transform
    q = camera_pose.transform.rotation
    t = camera_pose.transform.translation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    T_cam2world = quaternion_matrix(quat)  # 4×4 rotation+1s
    T_cam2world[:3, 3] = trans              # insert translation

    # Invert to get world→camera
    T_world2cam = np.linalg.inv(T_cam2world)

    # Apply to homogeneous point
    hom_pt = np.append(p_world, 1.0)        # [x, y, z, 1]
    cam_hom = T_world2cam @ hom_pt          # matrix multiplication

    return cam_hom[:3]

def transform_points_from_world(points: np.ndarray,
                                camera_pose: TransformStamped) -> np.ndarray:
    """
    Transform an array of 3D points from the world frame into the camera frame.

    Parameters
    ----------
    points : np.ndarray, shape (N,3)
        Points expressed in the world coordinate system.
    camera_pose : geometry_msgs.msg.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    np.ndarray, shape (N,3)
        The same points expressed in the camera coordinate system.
    """
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a TransformStamped")

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")

    # Extract translation and rotation (camera→world)
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

    # Build 4×4 camera→world matrix, then invert to world→camera
    T_cam2world = quaternion_matrix(quat)
    T_cam2world[:3, 3] = trans
    T_world2cam = np.linalg.inv(T_cam2world)

    # Apply to all points
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom_world = np.hstack([pts, ones])             # (N,4)
    hom_cam   = (T_world2cam @ hom_world.T).T      # (N,4)
    return hom_cam[:, :3]


# get_closest_points
def get_closest_points_in_pointclouds(pointcloud1: o3d.geometry.PointCloud, pointcloud2: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Find closest‐point correspondences from pointcloud1 to pointcloud2.

    Parameters
    ----------
    pointcloud1 : o3d.geometry.PointCloud
        Source cloud (N₁ points).
    pointcloud2 : o3d.geometry.PointCloud
        Target cloud (N₂ points).

    Returns
    -------
    corres : np.ndarray of shape (M,2), dtype=int
        Each row [i, j] means point i in pointcloud1 corresponds to
        its nearest neighbor j in pointcloud2.
    """
    # build a KD‐tree on the second cloud
    kdtree = o3d.geometry.KDTreeFlann(pointcloud2)

    pts1 = np.asarray(pointcloud1.points)
    corres = []
    for i, p in enumerate(pts1):
        # search for the single nearest neighbor in cloud2
        [_, idx, _] = kdtree.search_knn_vector_3d(p, 1)
        corres.append([i, idx[0]])

    return np.array(corres, dtype=np.int64)

def triangulate(point1_3d, pose1: TransformStamped, point2_2d, pose2: TransformStamped, camera_parameters) -> np.ndarray:
    """
    Given a 3D point on ray1 (from camera1) and a 2D pixel on ray2 (from camera2),
    compute the 3D point on ray1 that is closest to ray2.

    Parameters
    ----------
    point1_3d : sequence of three floats
        A point [x, y, z] on the first ray, in world coordinates.
    pose1 : TransformStamped
        Transform from camera1 frame to world frame.
    point2_2d : sequence of two floats
        A pixel coordinate [u, v] in image2.
    pose2 : TransformStamped
        Transform from camera2 frame to world frame.

    Returns
    -------
    np.ndarray, shape (3,)
        The world‐coordinate point on ray1 nearest to ray2.
    """
    if not isinstance(pose1, TransformStamped) or not isinstance(pose2, TransformStamped):
        raise TypeError("pose1 and pose2 must be geometry_msgs.msg.TransformStamped")

    # Convert inputs
    p1 = np.asarray(point1_3d, dtype=np.float64)
    if p1.shape != (3,):
        raise ValueError("point1_3d must be [x, y, z]")

    # Camera1 center and ray direction d1
    t1 = pose1.transform.translation
    C1 = np.array([t1.x, t1.y, t1.z], dtype=np.float64)
    v1 = p1 - C1
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 == 0:
        raise ValueError("point1_3d coincides with camera1 center")
    d1 = v1 / norm_v1

    # Camera2 center
    t2 = pose2.transform.translation
    C2 = np.array([t2.x, t2.y, t2.z], dtype=np.float64)

    # Build ray2 direction in world frame
    fx, fy, cx, cy = camera_parameters
    u, v = point2_2d
    # back‐project pixel into camera‐frame ray
    d_cam = np.array([(u - cx) / fx,
                        (v - cy) / fy,
                        1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam)
    # rotate into world
    q2 = pose2.transform.rotation
    M2 = quaternion_matrix([q2.x, q2.y, q2.z, q2.w])
    R2 = M2[:3, :3]
    d2 = R2.dot(d_cam)

    # Solve for t on ray1 and s on ray2 minimizing ||(C1 + t d1) - (C2 + s d2)||²
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    w0 = C1 - C2
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b

    if abs(denom) < 1e-6:
        # near‐parallel: project C2–C1 onto d1
        t = -d / a
    else:
        t = (b * e - c * d) / denom

    # Closest point on ray1
    return C1 + t * d1

#endregion

# ----------------------------------------------------------------------------

#region 2D
def scale_depth_map(depth: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """
    Scale and shift a depth map, preserving any existing mask:
    - If `depth` is a numpy MaskedArray, apply scale/shift to the data
        and retain the mask.
    - Otherwise, treat zeros as masked pixels and ensure they remain zero.

    Parameters
    ----------
    depth : np.ndarray or np.ma.MaskedArray
        Input depth map (H×W), possibly masked.
    scale : float
        Multiplicative factor to apply to the depth values.
    shift : float
        Additive offset to apply after scaling.

    Returns
    -------
    np.ndarray or np.ma.MaskedArray
        The transformed depth map, masked in the same way as input.
    """
    # Numeric checks
    if not np.isscalar(scale) or not np.isscalar(shift):
        raise TypeError("scale and shift must be numeric scalars")

    # Handle masked arrays
    if isinstance(depth, np.ma.MaskedArray):
        # apply scale/shift to data, keep mask
        data = depth.data.astype(np.float32, copy=False)
        scaled = data * scale + shift
        return np.ma.MaskedArray(scaled, mask=depth.mask, copy=False)

    # Otherwise treat zeros as mask
    if not isinstance(depth, np.ndarray):
        raise TypeError("depth must be a numpy.ndarray or MaskedArray")
    depth_f = depth.astype(np.float32, copy=False)
    mask_zero = (depth_f == 0)

    # scale & shift
    scaled = depth_f * scale + shift

    # re‐apply mask: zero out previously masked pixels
    scaled[mask_zero] = 0.0
    return scaled

def mask_depth_map(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a 2D mask to a depth map, zeroing out all pixels outside the mask.

    Parameters
    ----------
    depth : np.ndarray
        Input depth map of shape (H, W), dtype float or int.
    mask : np.ndarray
        Binary or boolean mask of shape (H, W). Non-zero/True means keep.

    Returns
    -------
    np.ndarray
        A new depth map where depth[i, j] is preserved if mask[i, j] else 0.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError("depth must be a numpy.ndarray")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")
    if depth.shape != mask.shape:
        raise ValueError("depth and mask must have the same shape")

    # make boolean mask
    mask_bool = mask.astype(bool)

    # apply mask
    result = depth.copy()
    result[~mask_bool] = 0
    return result

# project_pointcloud
def project_pointcloud_from_world(pointcloud: o3d.geometry.PointCloud, camera_pose: TransformStamped, camera_parameters):
    """
    Project a 3D point cloud (in world coordinates) onto the camera image plane,
    but always output a 480×640 depth map and mask for easy comparison.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The cloud in world coordinates.
    camera_pose : TransformStamped
        Transform from camera frame into world frame.

    Returns
    -------
    depth_map : np.ndarray, shape (480, 640)
        Depth in metres; zero where no points project.
    mask_bw : np.ndarray, shape (480, 640), uint8
        255 where depth_map > 0, else 0.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
    
    # build camera→world, invert to world→camera
    if isinstance(camera_pose, PoseStamped):
        t = camera_pose.pose.position
        q = camera_pose.pose.orientation
    elif isinstance(camera_pose, TransformStamped):
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
    else:
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")
    

    fx, fy, cx, cy = camera_parameters
    # force output resolution
    H_out, W_out = 480, 640

    trans = np.array([t.x, t.y, t.z], np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], np.float64)
    Tcw = quaternion_matrix(quat)
    Tcw[:3,3] = trans
    Twc = np.linalg.inv(Tcw)

    # transform points
    pts = np.asarray(pointcloud.points, np.float64)
    ones = np.ones((pts.shape[0],1), np.float64)
    pts_h = np.hstack((pts, ones))
    cam_pts = (Twc @ pts_h.T).T[:, :3]
    x_cam, y_cam, z_cam = cam_pts.T

    # keep points in front
    valid = z_cam > 0
    x_cam, y_cam, z_cam = x_cam[valid], y_cam[valid], z_cam[valid]

    # project
    u = x_cam * fx / z_cam + cx
    v = y_cam * fy / z_cam + cy
    j = np.round(u).astype(int)
    i = np.round(v).astype(int)

    # mask to fixed bounds
    bounds = (i >= 0) & (i < H_out) & (j >= 0) & (j < W_out)
    i, j, z_cam = i[bounds], j[bounds], z_cam[bounds]

    # build depth
    depth_map = np.full((H_out, W_out), np.inf, dtype=np.float32)
    for ii, jj, zz in zip(i, j, z_cam):
        if zz < depth_map[ii, jj]:
            depth_map[ii, jj] = zz
    depth_map[depth_map == np.inf] = 0.0

    # build mask
    mask_bw = (depth_map > 0).astype(np.uint8) * 255

    return depth_map, mask_bw

def project_point(point, camera_parameters):
    """
    Project a single 3D point in the camera coordinate system onto the image plane.

    Parameters
    ----------
    point : sequence of float or np.ndarray, shape (3,)
        A 3D point [x, y, z] in camera coordinates.

    Returns
    -------
    (u, v) : tuple of float
        The projected pixel coordinates. Raises if z <= 0.
    """
    import numpy as np
    from typing import Tuple

    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("point must be a 3-element sequence or array")
    x, y, z = p
    if z <= 0:
        raise ValueError("Point is behind the camera (z <= 0)")

    fx, fy, cx, cy = camera_parameters
    u = x * fx / z + cx
    v = y * fy / z + cy
    return [u, v]

# project_point_world
def project_point_from_world(point, camera_pose: TransformStamped, camera_parameters):
    """
    Project a single 3D point in world coordinates onto the image plane
    of the camera defined by camera_pose.

    Parameters
    ----------
    point : sequence of three floats
        The [x, y, z] coordinates in world frame.
    camera_pose : geometry_msgs.msg.TransformStamped
        The transform from camera frame to world frame.

    Returns
    -------
    (u, v) : tuple of float
        The projected pixel coordinates.
    """
    # build camera→world, invert to world→camera
    if isinstance(camera_pose, PoseStamped):
        t = camera_pose.pose.position
        q = camera_pose.pose.orientation
    elif isinstance(camera_pose, TransformStamped):
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
    else:
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    T_cam2world = quaternion_matrix(quat)
    T_cam2world[:3, 3] = trans
    T_world2cam = np.linalg.inv(T_cam2world)

    # transform the point
    p_world = np.asarray(point, dtype=np.float64)
    if p_world.shape != (3,):
        raise ValueError("point must be a sequence of three numbers [x, y, z]")
    hom = np.append(p_world, 1.0)
    x_cam, y_cam, z_cam = (T_world2cam @ hom)[:3]

    print(f'x_cam: {x_cam}, y_cam: {y_cam}')

    if z_cam <= 0:
        rospy.logwarn("Point is behind the camera (z <= 0)")

    # project using intrinsics
    fx, fy, cx, cy = camera_parameters
    u = x_cam * fx / z_cam + cx
    v = y_cam * fy / z_cam + cy
    return u, v

def get_closest_points_from_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Find closest‐point correspondences between two binary masks.

    Parameters
    ----------
    mask1, mask2 : np.ndarray, shape (H, W)
        Binary or boolean masks (non‐zero pixels are “on”). Must be same shape.

    Returns
    -------
    np.ndarray, shape (N, 4), dtype=int
        Each row is [r1, c1, r2, c2], where (r1,c1) is a pixel from mask1
        and (r2,c2) is its nearest neighbor in mask2.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    # Extract nonzero pixel coordinates (row, col)
    coords1 = np.column_stack(np.nonzero(mask1.astype(bool)))
    coords2 = np.column_stack(np.nonzero(mask2.astype(bool)))

    # No correspondences if either mask is empty
    if coords1.size == 0 or coords2.size == 0:
        return np.zeros((0, 4), dtype=int)

    # Build a 2D KD‐tree on coords2 (embed in 3D with z=0)
    pts2 = np.zeros((coords2.shape[0], 3), dtype=float)
    pts2[:, 0] = coords2[:, 1]  # x = col
    pts2[:, 1] = coords2[:, 0]  # y = row
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(pts2)
    kdtree = o3d.geometry.KDTreeFlann(pc2)

    # For each point in mask1, find nearest in mask2
    corres = []
    for r1, c1 in coords1:
        query = [float(c1), float(r1), 0.0]
        _, idx, _ = kdtree.search_knn_vector_3d(query, 1)
        r2, c2 = coords2[idx[0]]
        corres.append([r1, c1, r2, c2])

    return np.array(corres, dtype=int)

# get_closest_point
def get_closest_point_2D(mask: np.ndarray, point):
    """
    Find the nearest non-zero pixel in a binary mask to a given point.

    Parameters
    ----------
    mask : np.ndarray
        2D array, dtype float or uint8, where non-zero values are valid mask points.
    point : tuple of two floats
        The (x, y) coordinate to search from.

    Returns
    -------
    (x_closest, y_closest) : tuple of int
        The pixel coordinate in the mask closest to `point`.
    """
    import numpy as np

    # Build boolean mask of valid points
    mask_bool = mask.astype(bool)
    # Extract (row, col) indices of all true pixels
    coords = np.column_stack(np.nonzero(mask_bool))  # shape (N,2): (y, x)
    if coords.size == 0:
        raise ValueError("Mask contains no foreground pixels")

    # Round input point and split into ints
    x0, y0 = point
    x0, y0 = float(x0), float(y0)
    # We'll compare to (row=y, col=x)
    p = np.array([y0, x0], dtype=np.float64)

    # Compute squared distances and find the index of the minimum
    deltas = coords.astype(np.float64) - p
    d2 = np.einsum('ij,ij->i', deltas, deltas)
    idx = np.argmin(d2)

    y_closest, x_closest = coords[idx]
    return int(x_closest), int(y_closest)

def fill_mask_holes(mask_bw: np.ndarray, closing_radius: int = 2, kernel_shape: str = 'disk') -> np.ndarray:
    """
    Pure morphological closing: dilation then erosion with a small SE.

    Parameters
    ----------
    mask_bw : np.ndarray
        Input binary mask (0/255 or bool).
    closing_radius : int
        Radius of the closing structuring element.
    kernel_shape : str
        'disk' for circular SE, 'square' for square SE.

    Returns
    -------
    np.ndarray
        Output mask (uint8 0/255) with holes filled (and borders smoothed).
    """
    # normalize
    m = mask_bw.copy()
    if m.dtype != np.uint8:
        m = (m.astype(bool).astype(np.uint8) * 255)

    # choose SE
    ksz = 2*closing_radius + 1
    if kernel_shape == 'disk':
        # approximate disk with ellipse
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    else:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))

    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    return closed

def reduce_mask(mask: np.ndarray, pixel: int) -> np.ndarray:
    """
    Shrink (erode) a binary mask by removing `pixel` layers from its border.

    Parameters
    ----------
    mask : np.ndarray
        Input mask (2D), dtype bool or uint8 (0/255).
    pixel : int
        Number of pixels to remove from the outer boundary.

    Returns
    -------
    np.ndarray
        The eroded mask, dtype uint8 (0/255).
    """
    if not isinstance(pixel, int) or pixel < 1:
        raise ValueError("pixel must be a positive integer")

    # Normalize to uint8 0/255
    m = mask.copy()
    if m.dtype != np.uint8:
        m = (m.astype(bool).astype(np.uint8) * 255)

    # Create a (2*pixel+1)x(2*pixel+1) rectangular structuring element
    ksize = 2 * pixel + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    # Erode to remove `pixel` layers around the mask
    eroded = cv2.erode(m, kernel, iterations=1)

    return eroded

def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """
    Let the user interactively remove portions of a binary mask by drawing
    one or more rectangles. Inside each drawn rectangle, mask pixels are set to zero.
    """
    import cv2
    import numpy as np

    # Normalize to uint8 0/255
    working = mask.copy()
    if working.dtype != np.uint8:
        working = (working.astype(bool).astype(np.uint8) * 255)

    drawing = False
    ix = iy = cur_x = cur_y = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, ix, iy, cur_x, cur_y, working
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            cur_x, cur_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cur_x, cur_y = x, y
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1, x2 = sorted((ix, cur_x))
            y1, y2 = sorted((iy, cur_y))
            working[y1:y2, x1:x2] = 0

    window = "Cleanup Mask - draw rectangles, Esc to finish"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display_img = working.copy()
        if drawing:
            cv2.rectangle(display_img,
                            (ix, iy),
                            (cur_x, cur_y),
                            color=128, thickness=2)
        cv2.imshow(window, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            break

    cv2.destroyWindow(window)
    return working



#endregion

# ----------------------------------------------------------------------------

#region Calculations
def calculate_angle_from_mask_and_point(mask, point, area=40):
    """
    Calculate angle from mask and point.

    Parameters
    ----------
    mask : NDArray
        A binary mask.
    point : tuple
        A point in the mask.
    area : int, optional
        The size of the patch to check, by default 20.

    Returns
    -------
    angle : float
        The angle of the line that best fits the points in the patch.
    """
    px, py = point
    px, py = int(px), int(py)

    print(f'point: {point}, px: {px}, py: {py}')
    patch = mask[py-area:py+area+1, px-area:px+area+1]
    show_masks([mask], title="Pointcloud projected to horizontal plane")
    show_masks([patch], title="Patch")
    points = cv2.findNonZero(patch)    # get foreground points in the patch
    # fit a line (least squares) through these points
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx)[0]
    print(f'Angle: {angle/np.pi*180:.0f}°')
    return(angle)

def calculate_angle_from_pointcloud(pointcloud: o3d.geometry.PointCloud, point3d, radius: float = 0.05) -> float:
    """
    Compute the orientation of the local structure in a point cloud patch,
    as viewed from above (projected onto the XY plane), and visualize it.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The full 3D point cloud.
    point3d : sequence of three floats
        The [x, y, z] reference point in world coordinates.
    radius : float
        Horizontal search radius around `point3d`.

    Returns
    -------
    float
        The angle (in radians) of the best‐fitting line through the patch,
        measured CCW from the X axis in the XY‐plane.
    """

    # Validate inputs
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an Open3D PointCloud")
    p_ref = np.asarray(point3d, dtype=np.float64)
    if p_ref.shape != (3,):
        raise ValueError("point3d must be a sequence of three floats")

    # Extract points and select neighbors
    pts = np.asarray(pointcloud.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Point cloud is empty")
    dx = pts[:, 0] - p_ref[0]
    dy = pts[:, 1] - p_ref[1]
    mask = (dx**2 + dy**2) <= radius**2
    neighbors = pts[mask]
    if neighbors.shape[0] < 2:
        raise ValueError("Not enough neighbors within radius")

    # Project onto XY plane and center on mean
    xy = neighbors[:, :2]
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy

    # PCA to get principal direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # unit vector [ux, uy]

    # Compute angle CCW from X axis
    angle = float(np.arctan(direction[1] / direction[0]))
    print(f'Angle: {angle/np.pi*180}')

    # --- Visualization with OpenCV ---
    # Map the patch to a square image
    canvas_size = 400
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    scale = (canvas_size * 0.4) / radius  # leave 10% border
    origin = canvas_size // 2

    # Draw neighbor points
    for x, y in centered:
        px = int(origin + x * scale)
        py = int(origin - y * scale)
        if 0 <= px < canvas_size and 0 <= py < canvas_size:
            cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)

    # Draw principal axis line
    length = int(radius * scale)
    ux, uy = direction
    x1 = int(origin - ux * length)
    y1 = int(origin + uy * length)
    x2 = int(origin + ux * length)
    y2 = int(origin - uy * length)
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Annotate angle
    deg = angle * 180.0 / np.pi
    cv2.putText(canvas,
                f"Angle: {deg:.1f} deg",
                (10, canvas_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA)

    cv2.imshow("XY Patch and Principal Direction", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # --- end visualization ---

    return -angle

def count_inliers(depth1: np.ndarray,
                    depth2: np.ndarray) -> int:
    """
    Count the number of pixels/points that are “valid” in both inputs.
    Treats any boolean or integer array as a mask (non-zero == valid),
    and any float array as a depth map (value > 0 == valid).

    Parameters
    ----------
    depth1, depth2 : np.ndarray
        Either depth maps (float) or masks (bool or int). Must have the same shape.

    Returns
    -------
    int
        The count of positions where both inputs are valid.
    """
    if depth1.shape != depth2.shape:
        raise ValueError(f"Inputs must have the same shape\ndepth1.shape: {depth1.shape}\ndepth2.shape: {depth2.shape}")

    def make_mask(x):
        if x.dtype == bool:
            return x
        if np.issubdtype(x.dtype, np.integer):
            return x != 0
        if np.issubdtype(x.dtype, np.floating):
            return x > 0
        raise TypeError(f"Unsupported dtype for count_inliers: {x.dtype}")

    m1 = make_mask(depth1)
    m2 = make_mask(depth2)

    inlier_mask = m1 & m2
    union_mask  = m1 | m2

    return int(np.count_nonzero(inlier_mask)), int(np.count_nonzero(union_mask))

def score_mask_match(P, M, beta: float = 0.05, thresh: float = 0.5) -> float:
    """
    P : projected mask, arbitrary numeric or bool array
    M : reference mask, same shape as P
    thresh : values > thresh are treated as 1
    """

    # ---- sanity checks ------------------------------------------------------
    if not isinstance(P, np.ndarray) or not isinstance(M, np.ndarray):
        raise TypeError("P and M must be numpy.ndarray")

    if P.shape != M.shape:
        raise ValueError(f"shape mismatch {P.shape} vs {M.shape}")

    if P.ndim != 2:
        raise ValueError("masks must be 2-D (H×W)")

    # ---- force binary bool --------------------------------------------------
    if P.dtype != np.bool_:
        P = P > thresh
    if M.dtype != np.bool_:
        M = M > thresh

    # ---- overlap (IoU) ------------------------------------------------------
    inter = np.logical_and(P, M).sum(dtype=np.float32)
    union = np.logical_or (P, M).sum(dtype=np.float32)
    iou   = inter / (union + 1e-8)

    # ---- distance term (ASSD) ----------------------------------------------
    # OpenCV wants uint8 with background=1, object=0
    dt_M = cv2.distanceTransform(np.logical_not(M).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    dt_P = cv2.distanceTransform(np.logical_not(P).astype(np.uint8),
                                 cv2.DIST_L2, 3)

    fp = np.logical_and(P, ~M)  # false positives
    fn = np.logical_and(M, ~P)  # false negatives

    assd = 0.0
    if fp.any():
        assd += dt_M[fp].mean()
    if fn.any():
        assd += dt_P[fn].mean()
    assd *= 0.5

    return np.exp(-beta * assd)

    return float(iou * np.exp(-beta * assd))

#endregion

# ----------------------------------------------------------------------------

#region Show
def show_depth_map(depth, title="Depth Map", wait=True):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cv2.imshow(title, depth)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_masks(masks, title="Masks", wait=True) -> None:
    """
    Display multiple binary or float masks in one window, each colored differently.
    Blocks until any key is pressed.

    Parameters
    ----------
    masks : np.ndarray or sequence of np.ndarray
        If a single 2D mask (H×W), it will be wrapped in a list.
        If an array of shape (N, H, W), each slice along axis 0 is treated as one mask.
        Or you can pass a list/tuple of 2D masks.
    """
    import cv2
    import numpy as np

    # Normalize input to a list of 2D masks
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            raise ValueError("masks array must have 2 or 3 dimensions")
    elif isinstance(masks, (list, tuple)):
        mask_list = list(masks)
    else:
        raise TypeError("masks must be a numpy array or a list/tuple of arrays")

    # All masks must have the same shape
    H, W = mask_list[0].shape

    # Pre‐defined distinct BGR colors
    colors = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (128, 0, 128),  # purple
        (0, 128, 128),  # teal
        (128, 128, 0),  # olive
    ]

    # Create an empty color canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Overlay each mask in its color
    for idx, mask in enumerate(mask_list):
        # If mask has extra dims (e.g. (1, H, W)), take the first channel
        if mask.ndim == 3:
            mask = mask[0]
        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized.astype(bool)
        color = colors[idx % len(colors)]
        canvas[mask_bool] = color

    # Display the combined masks
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 550)
    cv2.imshow(title, canvas)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# show_mask_union
def show_masks_union(mask1: np.ndarray, mask2: np.ndarray, title="Masks Union") -> None:
    """
    Display two masks overlayed in one window, coloring:
        - mask1-only regions in red,
        - mask2-only regions in green,
        - overlapping regions in yellow.

    Parameters
    ----------
    mask1, mask2 : np.ndarray
        2D arrays of the same shape. Can be float in [0,1] or uint8 in [0,255].
    """
    if not isinstance(mask1, np.ndarray) or not isinstance(mask2, np.ndarray):
        raise TypeError("Both mask1 and mask2 must be numpy arrays")
    if mask1.shape != mask2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")
    if mask1.ndim != 2:
        raise ValueError("Masks must be 2D arrays")

    # Normalize masks to uint8 0/255
    def to_uint8(m):
        if m.dtype in (np.float32, np.float64):
            return (m * 255).astype(np.uint8)
        else:
            return m.astype(np.uint8)

    m1 = to_uint8(mask1)
    m2 = to_uint8(mask2)

    H, W = m1.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Boolean versions
    b1 = m1 > 0
    b2 = m2 > 0

    # mask1-only: red
    canvas[b1 & ~b2] = (0, 0, 255)
    # mask2-only: green
    canvas[~b1 & b2] = (0, 255, 0)
    # overlap: purple
    canvas[b1 & b2] = (240, 32, 160)

    # show
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 550)
    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def show_mask_and_points(mask: np.ndarray, points, title = "Mask with Points") -> None:
    """
    Display a single mask with overlaid points using OpenCV.

    Parameters
    ----------
    mask : np.ndarray
        2D mask array (H×W), dtype float in [0,1] or uint8 in [0,255].
    points : sequence of (x, y)
        List or array of pixel coordinates to draw on the mask.
    """
    import cv2
    import numpy as np

    # Normalize mask to uint8 [0,255]
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_img = (mask * 255).astype(np.uint8)
    else:
        mask_img = mask.copy().astype(np.uint8)

    # Convert to BGR so we can draw colored points
    canvas = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # Draw each point as a small circle (red)
    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    # Show result
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 550)
    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_pointclouds(pointclouds: list,  frames: list[TransformStamped] = None, title: str = 'Point Clouds') -> None:
    """
    Display multiple point clouds (each colored differently), optionally with frames.
    Accepts either Open3D PointClouds or numpy arrays of shape (N,3).

    Parameters
    ----------
    pointclouds : list of o3d.geometry.PointCloud or np.ndarray
        The point clouds to display. If an entry is an (N×3) array, it will be
        converted into an Open3D PointCloud.
    frames : list of TransformStamped, optional
        Coordinate frames to draw as triads.
    title : str
        Window title for the visualizer.
    """
    import copy
    import numpy as np
    import open3d as o3d
    from tf.transformations import quaternion_matrix
    from geometry_msgs.msg import TransformStamped

    geometries = []

    # draw origin triad
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=(0,0,0)))

    # draw any provided frames
    if frames:
        for tf in frames:
            if not isinstance(tf, TransformStamped):
                raise TypeError("Each frame must be a TransformStamped")
            t = tf.transform.translation; q = tf.transform.rotation
            T = quaternion_matrix((q.x, q.y, q.z, q.w))
            T[:3,3] = (t.x, t.y, t.z)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            mesh.transform(T)
            geometries.append(mesh)

    # color palette for point clouds
    palette = [
        [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [1,0,1], [0,1,1],
    ]

    # process each cloud
    for idx, pc in enumerate(pointclouds):
        # convert numpy array to PointCloud if needed
        if isinstance(pc, np.ndarray):
            if pc.ndim != 2 or pc.shape[1] != 3:
                raise ValueError(f"NumPy array at index {idx} must be shape (N,3)")
            pc_obj = o3d.geometry.PointCloud()
            pc_obj.points = o3d.utility.Vector3dVector(pc)
        elif isinstance(pc, o3d.geometry.PointCloud):
            pc_obj = copy.deepcopy(pc)
        else:
            raise TypeError("Each entry must be an Open3D PointCloud or an (N×3) ndarray")

        # assign a solid color
        color = palette[idx % len(palette)]
        colors = np.tile(color, (len(pc_obj.points), 1))
        pc_obj.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pc_obj)

    # launch the visualizer
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280, height=720,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

def show_pointclouds_with_frames(pointclouds: list[o3d.geometry.PointCloud],
                                    frames: list[TransformStamped],
                                    title: str = 'Point Clouds with Frames') -> None:
    """
    Display multiple point clouds (each colored differently) together with coordinate frames.

    Parameters
    ----------
    pointclouds : list of o3d.geometry.PointCloud
        The point clouds to display.
    frames : list of TransformStamped
        A list of camera/world frames to draw as coordinate triads.
    title : str
        Window title for the visualizer.
    """
    # origin frame
    geometries: list = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    ]

    # draw each frame
    for tf in frames:
        if not isinstance(tf, TransformStamped):
            raise TypeError("Each frame must be a geometry_msgs.msg.TransformStamped")
        t = tf.transform.translation
        q = tf.transform.rotation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3,3] = [t.x, t.y, t.z]
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame_mesh.transform(T)
        geometries.append(frame_mesh)

    # pick distinct colors (RGB) for each cloud
    palette = [
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
    ]

    # color and add each pointcloud
    for idx, pc in enumerate(pointclouds):
        if not isinstance(pc, o3d.geometry.PointCloud):
            raise TypeError("Each entry in pointclouds must be an Open3D PointCloud")
        # clone so we don't overwrite the original
        pc_colored = copy.deepcopy(pc)
        color = palette[idx % len(palette)]
        # assign a uniform color per point
        colors = np.tile(color, (len(pc_colored.points), 1))
        pc_colored.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pc_colored)

    # launch viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280, height=720,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

def show_pointclouds_with_frames_and_grid(pointclouds: list[o3d.geometry.PointCloud],
                                            frames: list[TransformStamped] = None,
                                            grid_size: float = 5.0,
                                            grid_step: float = 0.5,
                                            title: str = 'Point Clouds (Z-up)'):
    """
    Display point clouds + frames with Z always up, and a ground‐plane grid at z=0.
    Locks Z as vertical by resetting the up‐vector each frame.
    """
    # --- build all geometries ---
    geometries = []

    # origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    # camera frames
    if frames is not None:
        for tf in frames:
            t = tf.transform.translation; q = tf.transform.rotation
            T = quaternion_matrix((q.x, q.y, q.z, q.w))
            T[:3,3] = (t.x, t.y, t.z)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            mesh.transform(T)
            geometries.append(mesh)

    # colored pointclouds
    palette = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
    for idx, pc in enumerate(pointclouds):
        if not isinstance(pc, o3d.geometry.PointCloud):
            raise TypeError("Expected PointCloud")
        pc2 = copy.deepcopy(pc)
        color = palette[idx % len(palette)]
        pc2.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(pc2.points),1)))
        geometries.append(pc2)

    # ground-plane grid
    pts, lines = [], []
    n = 0
    for x in np.arange(-grid_size, grid_size+1e-6, grid_step):
        pts += [[x,-grid_size,0],[x,grid_size,0]]
        lines += [[n,n+1]]; n+=2
    for y in np.arange(-grid_size, grid_size+1e-6, grid_step):
        pts += [[-grid_size,y,0],[grid_size,y,0]]
        lines += [[n,n+1]]; n+=2
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines))
    grid.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5]]*len(lines))
    geometries.append(grid)

    # --- manual visualizer loop with up-vector reset ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    for geo in geometries:
        vis.add_geometry(geo)

    ctr = vis.get_view_control()
    ctr.set_front((0, -0.5, -0.5))
    ctr.set_lookat((0, 0, 0))
    # initial up
    ctr.set_up((0, 0, 1))

    while True:
        if not vis.poll_events():
            break
        # enforce Z-up every frame
        ctr.set_up((0, 0, 1))
        vis.update_renderer()

    vis.destroy_window()

def interactive_scale_shift(depth1: np.ndarray,
                            mask2: np.ndarray,
                            pose1: TransformStamped,
                            pose2: TransformStamped,
                            scale_limits: tuple[float, float] = (0.005, 0.4),
                            shift_limits: tuple[float, float] = (-1.0, 1.0)
                            ) -> tuple[float, float, int]:
    """
    Open an OpenCV window with two trackbars (“scale” and “shift”) that lets you
    interactively adjust the scaling and offset of `depth1`, regenerate the point
    cloud from camera1, project it into camera2, compare against `mask2`, and
    display the count of inliers as well as the current scale and shift.

    Parameters
    ----------
    depth1 : np.ndarray
        Original depth map from camera1 (H×W).
    mask2 : np.ndarray
        Binary mask from camera2 (H×W).
    pose1 : TransformStamped
        camera1→world transform.
    pose2 : TransformStamped
        camera2→world transform.
    scale_limits : tuple(float, float)
        (min_scale, max_scale) for the scale slider.
    shift_limits : tuple(float, float)
        (min_shift, max_shift) for the shift slider.

    Returns
    -------
    (scale, shift, inliers) : Tuple[float, float, int]
        The final slider values and number of inliers when the window is closed.
    """
    H, W = depth1.shape
    window = "Scale/Shift Tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # unpack limits and compute slider resolution
    smin, smax = scale_limits
    scale_res = 0.001
    s_steps = max(1, int(round((smax - smin) / scale_res)))
    scale_res = (smax - smin) / s_steps

    tmin, tmax = shift_limits
    t_steps = 200
    shift_res = (tmax - tmin) / t_steps

    result = {'scale': None, 'shift': None, 'inliers': None}

    def update(_=0):
        v_s = cv2.getTrackbarPos("scale", window)
        v_t = cv2.getTrackbarPos("shift", window)

        scale = smin + v_s * scale_res
        shift = tmin + v_t * shift_res
        # 1) scale & shift depth1
        d1 = scale_depth_map(depth1, scale, shift)
        # 2) to world pointcloud
        pc1 = get_pointcloud(d1)
        pc1_world = transform_pointcloud_to_world(pc1, pose1)
        # 3) project into cam2
        _, reproj_mask = project_pointcloud(pc1_world, pose2)
        # 4) overlay masks
        m2 = (mask2.astype(bool)).astype(np.uint8) * 255
        rm = (reproj_mask.astype(bool)).astype(np.uint8) * 255
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[m2 > 0]               = (0,   0, 255)
        overlay[rm > 0]               = (0, 255,   0)
        overlay[(m2 > 0) & (rm > 0)]  = (0, 255, 255)
        # 5) count inliers
        inliers = count_inliers(reproj_mask, mask2)

        # 6) annotate scale, shift, inliers
        cv2.putText(overlay, f"Scale: {scale:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f"Shift: {shift:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f"Inliers: {inliers}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        result['scale'], result['shift'], result['inliers'] = scale, shift, inliers
        cv2.imshow(window, overlay)

    # create trackbars
    cv2.createTrackbar("scale", window, int(s_steps//2), s_steps, update)
    cv2.createTrackbar("shift", window, int(t_steps//2), t_steps, update)

    update()  # initial draw

    # loop until Esc or window closed
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window)
    return result['scale'], result['shift'], result['inliers']

def show_spline_gradient(mask: np.ndarray, centerline: np.ndarray, title: str = "Centerline Points"):
    """
    Overlay the 2D centerline points on the mask with a blue→red gradient.

    Args:
        mask:        H×W binary mask.
        centerline:  (N×2) array of (row, col) coordinates.
        title:       OpenCV window name.
    """
    # Prepare the background image
    img = (mask.astype(np.uint8) * 255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    N = len(centerline)
    if N == 0:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        return

    # Draw each point with its own gradient color
    for i, (r, c) in enumerate(centerline):
        t = i / (N - 1) if N > 1 else 0
        # Compute B→R gradient: blue at start, red at end
        color = (int(255 * (1 - t)), 0, int(255 * t))
        # Draw a filled circle at each centerline point
        cv2.circle(img, (int(c), int(r)), radius=1, color=color, thickness=-1)

    # Show the result
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

#endregion

#region Save
def save_depth_map(depth_map: np.ndarray,
                    folder: str,
                    filename: str,
                    vmin: float = None,
                    vmax: float = None,
                    colormap: int = None) -> None:
    """
    Save a depth map to a PNG in the specified folder, using `filename` + ".png".

    Parameters
    ----------
    depth_map : np.ndarray
        2D array of depth values (float or int).
    folder : str
        Directory in which to save the image (will be created if needed).
    filename : str
        Base name (without extension) for the PNG file.
    vmin : float, optional
        Minimum depth for normalization (default = min of depth_map).
    vmax : float, optional
        Maximum depth for normalization (default = max of depth_map).
    colormap : int, optional
        OpenCV colormap (e.g. cv2.COLORMAP_JET). If None, grayscale is used.
    """
    import os
    import numpy as np
    import cv2

    # ensure folder exists
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # build path
    save_path = os.path.join(folder, f"{filename}.png")

    # normalize to [0,255]
    dm = depth_map.astype(np.float32)
    mn = vmin if vmin is not None else np.nanmin(dm)
    mx = vmax if vmax is not None else np.nanmax(dm)
    if mx <= mn:
        img8 = np.zeros(dm.shape, dtype=np.uint8)
    else:
        norm = (dm - mn) / (mx - mn)
        norm = np.clip(norm, 0.0, 1.0)
        img8 = (norm * 255).astype(np.uint8)

    # apply colormap if requested
    if colormap is not None:
        img_color = cv2.applyColorMap(img8, colormap)
    else:
        img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    # write out PNG
    cv2.imwrite(save_path, img_color)

def save_masks(masks: list[np.ndarray],
               folder: str,
               filename: str,
               colors: list[tuple[int,int,int]] = None) -> None:
    """
    Overlay and save multiple binary masks as a single color PNG.

    Parameters
    ----------
    masks : list of np.ndarray or np.ndarray (2D or 3D)
        Binary or uint8 masks to overlay. Can be a single 2D mask, a 3D array
        of shape (N, H, W), or a list of 2D arrays.
    folder : str
        Directory to write the output PNG into (will be created if needed).
    filename : str
        Base name (without extension) for the saved file.
    colors : list of (B, G, R) tuples, optional
        Colors for each mask in 0–255. If omitted, defaults to:
        white, red, green, blue, yellow, magenta, cyan, …
    """
    import os
    import numpy as np
    import cv2

    # Normalize input into a list of 2D masks
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            raise ValueError("If masks is an ndarray it must be 2D or 3D (N,H,W)")
    elif isinstance(masks, (list, tuple)):
        mask_list = list(masks)
    else:
        raise TypeError("masks must be an ndarray or a list/tuple of ndarrays")

    if not mask_list:
        raise ValueError("No masks provided")

    # Verify all masks have the same shape
    H, W = mask_list[0].shape
    for m in mask_list:
        if m.shape != (H, W):
            raise ValueError("All masks must have the same shape")

    # Default color palette (BGR)
    default_palette = [
        (255, 255, 255),  # white
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (128, 128, 128),  # gray
    ]
    palette = colors if colors is not None else default_palette

    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    # Prepare blank color image
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, m in enumerate(mask_list):
        # Binarize mask
        mb = (m > 0)
        color = palette[idx % len(palette)]
        out[mb] = color

    # Save image
    path = os.path.join(folder, f"{filename}.png")
    cv2.imwrite(path, out)

def save_mask_spline(mask: np.ndarray,
                        ctrl_points: np.ndarray,
                        order: int,
                        camera_parameters,
                        camera_pose: TransformStamped,
                        folder: str,
                        filename: str,
                        num_samples: int = 200,
                        line_color = (0, 0, 255),
                        line_thickness: int = 2) -> None:
    """
    Overlay a projected 3D B‐spline on top of a binary mask and save as PNG.
    Projects the spline (in world coords) into the image plane defined by camera_pose.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        Binary mask (0/255 or bool) to serve as the background.
    ctrl_points : np.ndarray, shape (N,3)
        Control points of the 3D spline in world coordinates.
    order : int
        Degree of the B‐spline.
    camera_parameters : (fx, fy, cx, cy)
        Intrinsic parameters.
    camera_pose : TransformStamped
        The ROS transform from camera frame to world frame.
    folder : str
        Directory to save the output (will be created if needed).
    filename : str
        Base name (without extension) for the saved file.
    num_samples : int
        How many points to sample along the spline for a smooth curve.
    line_color : (B, G, R)
        Color to draw the spline.
    line_thickness : int
        Thickness of the polyline segments.
    """
    # 1) ensure folder
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"{filename}.png")

    # 2) prepare background
    H, W = mask.shape[:2]
    bg = (mask.astype(bool).astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # 3) build knot vector
    n_ctrl = ctrl_points.shape[0]
    m = n_ctrl - order - 1
    interior = np.linspace(0,1,m+2)[1:-1] if m>0 else np.array([])
    knots = np.concatenate([np.zeros(order+1), interior, np.ones(order+1)])

    # 4) sample spline in world coords
    spline = BSpline(knots, ctrl_points, order)
    t0, t1 = knots[order], knots[-order-1]
    ts = np.linspace(t0, t1, num_samples)
    world_pts = spline(ts)  # (num_samples,3)

    # 5) build world→camera transform
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a TransformStamped")
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    Tcw = quaternion_matrix(quat)
    Tcw[:3,3] = trans
    Twc = np.linalg.inv(Tcw)

    # 6) project each world point into pixel coords
    fx, fy, cx, cy = camera_parameters
    pts_h = np.hstack([world_pts, np.ones((world_pts.shape[0],1))])
    cam_pts = (Twc @ pts_h.T).T
    x_cam, y_cam, z_cam = cam_pts[:,0], cam_pts[:,1], cam_pts[:,2]
    u = (x_cam * fx / z_cam + cx).round().astype(int)
    v = (y_cam * fy / z_cam + cy).round().astype(int)

    # 7) draw polyline
    for i in range(len(u)-1):
        u0, v0, u1, v1 = u[i], v[i], u[i+1], v[i+1]
        if 0 <= u0 < W and 0 <= v0 < H and 0 <= u1 < W and 0 <= v1 < H:
            cv2.line(img, (u0,v0), (u1,v1), line_color, line_thickness)

    # 8) save
    cv2.imwrite(save_path, img)



#endregion

# ----------------------------------------------------------------------------

#region Poses
def interpolate_poses(pose_start, pose_end, num_steps: int) -> list[PoseStamped]:
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

def convert_pose_to_pose_stamped(pose: Pose, frame: str = "map") -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = frame
    ps.pose = pose
    return ps

# rotate_pose
def rotate_pose_around_z(pose: PoseStamped, angle:float) -> PoseStamped:
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

def convert_trans_and_rot_to_stamped_transform(translation, rotation):
    transform_stamped = TransformStamped()
    transform_stamped.transform.translation.x = translation[0]
    transform_stamped.transform.translation.y = translation[1]
    transform_stamped.transform.translation.z = translation[2]

    transform_stamped.transform.rotation.x = rotation[0]
    transform_stamped.transform.rotation.y = rotation[1]
    transform_stamped.transform.rotation.z = rotation[2]
    transform_stamped.transform.rotation.w = rotation[3]
    return transform_stamped

def create_pose(x: float=0, y: float=0, z: float=0, roll: float=0, pitch: float=0, yaw: float=0, reference_frame: str=None) -> PoseStamped:
    """
    Create a PoseStamped message from XYZ and RPY in a given reference frame.

    Parameters
    ----------
    x, y, z : float
        Position coordinates.
    roll, pitch, yaw : float
        Orientation in radians (ZYX convention).
    reference_frame : str
        The frame_id for the header.

    Returns
    -------
    geometry_msgs.msg.PoseStamped
        The stamped pose.
    """
    if reference_frame is None:
        raise TypeError("reference_frame cannot be None")

    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = reference_frame

    # Set position
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)

    # Convert RPY to quaternion
    q = quaternion_from_euler(roll, pitch, yaw)  # returns [x, y, z, w]
    ps.pose.orientation.x = float(q[0])
    ps.pose.orientation.y = float(q[1])
    ps.pose.orientation.z = float(q[2])
    ps.pose.orientation.w = float(q[3])

    return ps


#endregion

def create_pointcloud_message(pointcloud: o3d.geometry.PointCloud, frame: str) -> PointCloud2:
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
  

def get_desired_pose(position,
                     base_footprint,
                     frame: str = "map") -> PoseStamped:
    """
    Return a PoseStamped at `position` in `frame`, with orientation equal to
    the current base_footprint orientation rotated by -90° about its local Y axis.

    Parameters
    ----------
    position : sequence of three floats
        Desired [x, y, z] in the `frame` coordinate.
    base_footprint : str
        The TF frame name of the robot base.
    frame : str
        The target frame in which to express the pose (default "map").

    Returns
    -------
    geometry_msgs.msg.PoseStamped
        Stamped pose at the given position, with orientation = (base_footprint → frame)
        ∘ RotY(-90°).
    """
    # 1) look up the current base_footprint → frame transform as a PoseStamped
    # base_ps = ros_handler.get_current_pose(base_footprint, frame)
    # q = base_ps.pose.orientation
    rotation = base_footprint.transform.rotation
    # q_base = [q.x, q.y, q.z, q.w]
    q_base = [rotation.x, rotation.y, rotation.z, rotation.w]

    # 2) build a -90° rotation about Y
    q_delta = quaternion_from_euler(np.pi, 0, 0.0)  # [x,y,z,w]
    # q_delta = quaternion_from_euler(0.0, 0, 0.0)  # [x,y,z,w]

    # 3) compose: q_new = q_base ∘ q_delta
    q_new = quaternion_multiply(q_base, q_delta)

    # 4) assemble the PoseStamped
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = frame

    ps.pose.position.x = float(position[0])
    ps.pose.position.y = float(position[1])
    ps.pose.position.z = float(position[2])

    ps.pose.orientation.x = float(q_new[0])
    ps.pose.orientation.y = float(q_new[1])
    ps.pose.orientation.z = float(q_new[2])
    ps.pose.orientation.w = float(q_new[3])

    return ps


def get_highest_point(pointcloud: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Return the 3D point in the point cloud with the largest Z coordinate.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The input point cloud.

    Returns
    -------
    np.ndarray
        A length-3 array [x, y, z] of the point with the maximum z value.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an open3d.geometry.PointCloud")

    pts = np.asarray(pointcloud.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Point cloud is empty")

    # find the index of the maximum Z
    idx = np.argmax(pts[:, 2])
    return pts[idx]



#region B-spline Fitting
def fit_spline(data, num_control_points: int, order: int) -> np.ndarray:
    """
    Fit a clamped, uniform B‐spline of given degree and number of control points
    to either:
        - an Open3D PointCloud (unordered) or
        - an (M×3) numpy array of ordered center-line points.

    Returns the (num_control_points×3) control‐point array.

    Parameters
    ----------
    data : o3d.geometry.PointCloud or np.ndarray
        If PointCloud, will parameterize by chord‐length along its PCA‐axis.
        If np.ndarray of shape (M,3), treated as already‐ordered curve.
    num_control_points : int
    order : int
    """
    import numpy as np
    from scipy.interpolate import make_lsq_spline
    import open3d as o3d

    # --- extract points and parameter t ---
    if isinstance(data, o3d.geometry.PointCloud):
        pts = np.asarray(data.points, dtype=float)
        # PCA‐based axis as before
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        proj = centered.dot(axis)
        t_raw = (proj - proj.min()) / (proj.max() - proj.min())
        order_idx = np.argsort(t_raw)
        t_sorted = t_raw[order_idx]
        pts_sorted = pts[order_idx]
    elif isinstance(data, np.ndarray):
        pts = data.astype(float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("numpy array must be shape (M,3)")
        pts_sorted = pts
        # chord‐length parameterization
        dists = np.linalg.norm(np.diff(pts_sorted, axis=0), axis=1)
        s = np.concatenate(([0.], np.cumsum(dists)))
        t_sorted = s / s[-1]
    else:
        raise TypeError("data must be an Open3D PointCloud or an (M×3) numpy array")

    M = pts_sorted.shape[0]
    if M < order + 1:
        raise ValueError("Not enough points for spline")

    # --- build clamped uniform knot vector ---
    m = num_control_points - order - 1
    if m > 0:
        interior = np.linspace(0, 1, m+2)[1:-1]
    else:
        interior = np.array([])
    knots = np.concatenate([
        np.zeros(order+1),
        interior,
        np.ones(order+1)
    ])

    # --- least‐squares fit per dimension ---
    ctrl_pts = np.zeros((num_control_points, 3), dtype=float)
    for dim in range(3):
        coord = pts_sorted[:, dim]
        spline = make_lsq_spline(t_sorted, coord, t=knots, k=order)
        ctrl_pts[:, dim] = spline.c

    return ctrl_pts


def visualize_spline_with_pc(pointcloud, control_pts, degree, num_samples=200, title="Spline with PointCloud"):
    n_ctrl = len(control_pts)
    # Build a clamped, uniform knot vector of length n_ctrl + degree + 1
    # interior knot count = n_ctrl - degree - 1
    if n_ctrl <= degree:
        raise ValueError("Need more control points than the degree")
    m = n_ctrl - degree - 1
    if m > 0:
        interior = np.linspace(0, 1, m+2)[1:-1]
    else:
        interior = np.array([])
    # clamp start/end
    t = np.concatenate([
        np.zeros(degree+1),
        interior,
        np.ones(degree+1)
    ])
    # create BSpline
    spline = BSpline(t, control_pts, degree)

    # sample
    ts = np.linspace(t[degree], t[-degree-1], num_samples)
    curve_pts = spline(ts)

    # build LineSet
    import open3d as o3d
    lines = [[i, i+1] for i in range(len(curve_pts)-1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(curve_pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])

    o3d.visualization.draw_geometries([pointcloud, ls])

def extract_centerline_mst(pointcloud: o3d.geometry.PointCloud, k: int = 6) -> np.ndarray:
    """
    Extract a 1D center-line from a (possibly sparse) DLO point cloud by:
        1) Building a k-NN graph
        2) Computing its undirected MST
        3) Finding the two farthest-apart leaves
        4) Extracting the unique path between them via Dijkstra predecessors

    Returns an ordered array of 3D points along that path.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
    import open3d as o3d

    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an Open3D PointCloud")

    pts = np.asarray(pointcloud.points, float)
    N = pts.shape[0]
    if N < 3:
        raise ValueError("Need at least three points to get a non-trivial centerline")

    # 1) k-NN graph (undirected)
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k+1)
    rows, cols, data = [], [], []
    for i in range(N):
        for dist, j in zip(dists[i,1:], idxs[i,1:]):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([dist, dist])
    G = coo_matrix((data, (rows, cols)), shape=(N, N))

    # 2) build undirected MST
    mst_directed = minimum_spanning_tree(G)
    mst_und = mst_directed + mst_directed.T  # now symmetric
    mst_und = mst_und.tocsr()

    # 3) compute degrees properly
    # adjacency is nonzero entries
    adj_bool = mst_und.copy()
    adj_bool.data = np.ones_like(adj_bool.data)
    degrees = np.array(adj_bool.sum(axis=1)).flatten()
    leaves = np.where(degrees == 1)[0]
    if len(leaves) < 2:
        raise ValueError(f"Only found {len(leaves)} leaf nodes—try increasing k")

    # 4a) tree diameter endpoints
    dist0, _ = dijkstra(mst_und, indices=leaves[0], return_predecessors=True)
    A = np.nanargmax(dist0)
    distA, preds = dijkstra(mst_und, indices=A, return_predecessors=True)
    B = np.nanargmax(distA)

    # 5) reconstruct unique path A→B
    path_idx = []
    cur = B
    while cur != A and cur != -9999:
        path_idx.append(cur)
        cur = preds[cur]
    path_idx.append(A)
    path_idx = path_idx[::-1]

    # map back to 3D
    centerline = pts[path_idx]
    return centerline


#endregion


#region B-spline fitting - paper

def chord_length_parameterize(X):
    """Returns normalized chord‐length params u_k for point cloud X."""
    diffs = np.linalg.norm(np.diff(X, axis=0), axis=1)
    u = np.zeros(len(X))
    u[1:] = np.cumsum(diffs)
    return u / u[-1]

def make_open_uniform_knots(n_ctrl_pts, degree):
    """Creates an open‐uniform knot vector with clamped ends."""
    m = n_ctrl_pts + degree + 1
    # first and last degree+1 knots are 0 and 1, interior uniformly spaced
    knots = np.empty(m)
    knots[:degree+1] = 0.0
    knots[-(degree+1):] = 1.0
    num_int = m - 2*(degree+1)
    if num_int > 0:
        knots[degree+1:- (degree+1)] = np.linspace(0, 1, num_int+2)[1:-1]
    return knots

def find_footpoint(bspline, bspline_d1, bspline_d2, Xk, t0, tol=1e-6, maxiter=10):
    """Newton‐project data point Xk onto spline, returns t in [0,1]."""
    t = t0
    for _ in range(maxiter):
        C   = bspline(t)
        C1  = bspline_d1(t)
        C2  = bspline_d2(t)
        r   = C - Xk
        g1  = 2.0 * np.dot(C1, r)
        g2  = 2.0 * (np.dot(C1, C1) + np.dot(r, C2))
        if abs(g2) < 1e-8:
            break
        t_new = t - g1/g2
        # clamp into valid range
        t = np.clip(t_new, bspline.t[bspline.k], bspline.t[-(bspline.k+1)])
        if abs(t - t_new) < tol:
            break
    return t

def fit_spline_paper(pointcloud, num_ctrl_points, degree,
                     max_iter=10, tol=1e-5):
    """
    Fit a B-spline curve of given degree to a 3D pointcloud,
    following the iterative squared‐distance minimization in Flöry’s thesis.
    Returns an array of shape (num_ctrl_points, 3) of control points.
    
    Parameters
    ----------
    pointcloud : ndarray, shape (N,3)
    num_ctrl_points : int
    degree : int             # p in B-spline terminology
    max_iter : int           # max Gauss–Newton steps
    tol : float              # convergence tol on control point change
    """
    X = np.asarray(pointcloud)
    N_pts = X.shape[0]
    
    # 1) Parameterize and build initial system
    u = chord_length_parameterize(X)  # initial t_k
    U = make_open_uniform_knots(num_ctrl_points, degree)
    
    # Build initial basis matrix N of shape (N_pts, num_ctrl_points)
    # using SciPy’s BSpline.design_matrix functionality
    bspline = BSpline(U, np.zeros((num_ctrl_points,3)), degree)
    Nmat = np.vstack([bspline.design_matrix(u_k) for u_k in u])
    
    # Least‐squares solve N^T N D = N^T X
    D = solve(Nmat.T @ Nmat, Nmat.T @ X)
    
    for it in range(max_iter):
        # update spline object with current control points
        bspline = BSpline(U, D, degree)
        bspline_d1 = bspline.derivative(1)
        bspline_d2 = bspline.derivative(2)
        
        # 2) Foot‐point projection: find new u's
        u_new = np.array([
            find_footpoint(bspline, bspline_d1, bspline_d2, X[k], u[k])
            for k in range(N_pts)
        ])
        
        # 3) Rebuild N matrix at updated parameters
        Nmat_new = np.vstack([bspline.design_matrix(u_k)
                               for u_k in u_new])
        
        # 4) Solve for new control points
        D_new = solve(Nmat_new.T @ Nmat_new, Nmat_new.T @ X)
        
        # check convergence
        if np.linalg.norm(D_new - D) < tol:
            D = D_new
            break
        
        D = D_new
        u = u_new
    
    return D
#endregion

#region B-spline fitting - new
def extract_centerline(pcd: o3d.geometry.PointCloud, k: int = 80):
    """
    Extracts the centerline of a deformable linear object (DLO) from an unordered pointcloud.

    Steps:
      1. Builds a k‐nearest‐neighbor graph on the points.
      2. Computes the minimum spanning tree (MST) of that graph.
      3. Finds the MST’s diameter (longest path between two leaves).
      4. Returns the sequence of 3D points along that path.

    Args:
        pcd: Open3D PointCloud containing only the DLO’s points.
        k:  Number of nearest neighbors to connect in the graph (default: 6).

    Returns:
        centerline_pts: (M×3) numpy array of points along the centerline.
        path_indices:  list of length M giving the original point indices.
    """
    # 1. Prepare data & KD‐tree
    pts = np.asarray(pcd.points)
    N = len(pts)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 2. Build k‐NN graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        _, idx, dist2 = kdtree.search_knn_vector_3d(pts[i], k+1)
        # skip idx[0] == i itself
        for j, d2 in zip(idx[1:], dist2[1:]):
            w = np.sqrt(d2)
            # add edge (i, j) with weight = Euclidean distance
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=w)

    # prune long edges
    lengths = [d['weight'] for _,_,d in G.edges(data=True)]
    thresh = np.mean(lengths) + 2*np.std(lengths)
    for u,v,d in list(G.edges(data=True)):
        if d['weight'] > thresh:
            G.remove_edge(u,v)

    # 3. Compute MST of the graph
    T = nx.minimum_spanning_tree(G, weight='weight')

    # 4. Find tree diameter via two BFS/Dijkstra passes:
    #    a) from arbitrary node (0) find farthest node A
    lengths, _ = nx.single_source_dijkstra(T, source=0, weight='weight')
    A = max(lengths, key=lengths.get)
    #    b) from A find farthest node B
    lengths, _ = nx.single_source_dijkstra(T, source=A, weight='weight')
    B = max(lengths, key=lengths.get)

    # 5. Retrieve the longest path between A and B
    path_indices = nx.shortest_path(T, source=A, target=B, weight='weight')

    # 6. Gather the 3D coordinates
    centerline_pts = pts[path_indices]

    return centerline_pts, path_indices

def fit_bspline_scipy(centerline_pts: np.ndarray, degree: int = 3, smooth: float = None, nest: int = None) -> np.ndarray:
    """
    Fit a B-spline to a 3D centerline using SciPy's smoothing spline.

    Args:
        centerline_pts: (N×3) array of ordered 3D points along the centerline.
        degree: Spline degree (k). Must be <= 5.
        smooth: Smoothing factor (s). If None, defaults to s=0 (interpolating spline).
        nest: Maximum number of knots. Higher values allow more control points.
              If None, SciPy chooses based on data size.

    Returns:
        ctrl_pts: (M×3) array of the spline's control points.
    """
    # Prepare data for splprep: a list of coordinate arrays
    coords = [centerline_pts[:, i] for i in range(3)]
    
    # Compute the B-spline representation
    tck, u = splprep(coords, k=degree, s=smooth, nest=nest)
    
    # Extract control points: tck[1] is a list of arrays for each dimension
    ctrl_pts = np.vstack(tck[1]).T
    
    return ctrl_pts

def extract_centerline_from_mask(depth_image: np.ndarray, mask: np.ndarray, camera_parameters, depth_scale: float = 1.0, connectivity: int = 8) -> np.ndarray:
    """
    Extracts an ordered 3D centerline from a binary mask and its corresponding depth image
    via 2D skeletonization and back-projection.

    Args:
        depth_image: (H×W) depth values (in same units as 'depth_scale' multiplier).
        mask:         (H×W) binary mask (0/1 or False/True) of the DLO region.
        intrinsics:   Open3D PinholeCameraIntrinsic or dict with keys 'fx','fy','cx','cy'.
        depth_scale:  Factor to divide depth_image values by to get metric depth (default=1.0).
        connectivity: 4 or 8 connectivity for the skeleton graph (default=8).

    Returns:
        centerline_pts: (N×3) array of ordered 3D points along the centerline.
    """
    # 1. Skeletonize the 2D mask
    skeleton = skeletonize(mask > 0)

    # 2. Collect skeleton pixel coordinates
    rows, cols = np.where(skeleton)
    coords = list(zip(rows, cols))
    if len(coords) == 0:
        return np.empty((0, 3))

    # 3. Build a pixel adjacency graph over the skeleton
    index_map = {coord: idx for idx, coord in enumerate(coords)}
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   ( 0, -1),           ( 0, 1),
                   ( 1, -1), ( 1, 0), ( 1, 1)]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    for i, (r, c) in enumerate(coords):
        for dr, dc in offsets:
            neighbor = (r + dr, c + dc)
            j = index_map.get(neighbor)
            if j is not None:
                G.add_edge(i, j)

    # 4. Compute the graph diameter (longest path) to order points
    #    a) Find farthest from arbitrary node 0
    lengths = nx.single_source_shortest_path_length(G, source=0)
    A = max(lengths, key=lengths.get)
    #    b) From A find farthest B
    lengths_A = nx.single_source_shortest_path_length(G, source=A)
    B = max(lengths_A, key=lengths_A.get)
    #    c) Extract the ordered path
    path = nx.shortest_path(G, source=A, target=B)

    # 5. Back-project each skeleton pixel to 3D
    #    Resolve intrinsics
    fx, fy, cx, cy = camera_parameters

    centerline_pts = []
    for idx in path:
        r, c = coords[idx]
        z = depth_image[r, c] / depth_scale
        if z <= 0:
            continue
        x = (c - cx) * z / fx
        y = (r - cy) * z / fy
        centerline_pts.append((x, y, z))

    return np.array(centerline_pts)

def project_bspline(ctrl_points: np.ndarray, camera_pose, camera_parameters: tuple, width: int = 640, height: int = 480, degree: int = 3) -> np.ndarray:
    """
    Projects a 3D B-spline (defined by control points) into the camera image plane,
    rendering a continuous curve into a binary mask.

    Args:
        ctrl_points: (N_control x 3) array of B-spline control points.
        camera_pose: TransformStamped with .transform.translation (x,y,z)
                     and .transform.rotation (x,y,z,w) defining camera pose in world.
        camera_parameters: (fx, fy, cx, cy) intrinsic parameters.
        width:  Output image width in pixels.
        height: Output image height in pixels.
        degree: Spline degree (k). Must satisfy N_control > degree.

    Returns:
        mask: (height x width) uint8 mask with the projected spline drawn in 255 on 0.
    """
    # Unpack intrinsics
    fx, fy, cx, cy = camera_parameters

    # Build open-uniform knot vector
    n_ctrl = ctrl_points.shape[0]
    k = degree
    if n_ctrl <= k:
        raise ValueError("Number of control points must exceed spline degree")
    num_inner = n_ctrl - k - 1
    if num_inner > 0:
        inner = np.linspace(1/(num_inner+1), num_inner/(num_inner+1), num_inner)
        t = np.concatenate(([0]*(k+1), inner, [1]*(k+1)))
    else:
        t = np.concatenate(([0]*(k+1), [1]*(k+1)))

    # Create vector-valued spline
    spline = BSpline(t, ctrl_points, k, axis=0)

    # Sample points along spline
    num_samples = max(width, height)
    u = np.linspace(0, 1, num_samples)
    pts_world = spline(u)  # (num_samples x 3)

    # Parse camera pose (world -> camera)
    tx = camera_pose.transform.translation.x
    ty = camera_pose.transform.translation.y
    tz = camera_pose.transform.translation.z
    qx = camera_pose.transform.rotation.x
    qy = camera_pose.transform.rotation.y
    qz = camera_pose.transform.rotation.z
    qw = camera_pose.transform.rotation.w

    # Build rotation matrix from quaternion (camera -> world)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [  2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)],
        [  2*(xz - wy),   2*(yz + wx), 1-2*(xx+yy)]
    ])

    # Transform points into camera frame: p_cam = R^T * (p_world - t)
    diff = pts_world - np.array([tx, ty, tz])
    pts_cam = diff.dot(R)

    # Perspective projection
    x_cam, y_cam, z_cam = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    valid = z_cam > 0
    u_proj = (fx * x_cam[valid] / z_cam[valid] + cx)
    v_proj = (fy * y_cam[valid] / z_cam[valid] + cy)

    # Integer pixel coords
    u_pix = np.round(u_proj).astype(int)
    v_pix = np.round(v_proj).astype(int)

    # Create mask and draw lines between consecutive samples
    mask = np.zeros((height, width), dtype=np.uint8)
    # Filter pixels inside image bounds
    pts2d = list(zip(u_pix, v_pix))
    pts2d = [(u, v) for u, v in pts2d if 0 <= u < width and 0 <= v < height]
    for (u0, v0), (u1, v1) in zip(pts2d, pts2d[1:]):
        rr, cc = skline(v0, u0, v1, u1)
        valid_line = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        mask[rr[valid_line], cc[valid_line]] = 255

    return mask

def project_bspline_pts(ctrl_points, camera_pose, camera_parameters, degree=3, num_samples=200,  width=640, height=480):
    """
    Same as project_bspline but returns sampled 2D pixel coords (floats) along the spline,
    instead of rasterizing to a mask.
    """
    # Unpack intrinsics
    fx, fy, cx, cy = camera_parameters

    # Build open-uniform knot vector
    n_ctrl = ctrl_points.shape[0]
    k = degree
    if n_ctrl <= k:
        raise ValueError("Number of control points must exceed spline degree")
    num_inner = n_ctrl - k - 1
    if num_inner > 0:
        inner = np.linspace(1/(num_inner+1), num_inner/(num_inner+1), num_inner)
        t = np.concatenate(([0]*(k+1), inner, [1]*(k+1)))
    else:
        t = np.concatenate(([0]*(k+1), [1]*(k+1)))

    # Create vector-valued spline
    spline = BSpline(t, ctrl_points, k, axis=0)

    # Sample points along spline
    u = np.linspace(0, 1, num_samples)
    pts_world = spline(u)  # (num_samples x 3)

    # Parse camera pose (world -> camera)
    tx = camera_pose.transform.translation.x
    ty = camera_pose.transform.translation.y
    tz = camera_pose.transform.translation.z
    qx = camera_pose.transform.rotation.x
    qy = camera_pose.transform.rotation.y
    qz = camera_pose.transform.rotation.z
    qw = camera_pose.transform.rotation.w

    # Build rotation matrix from quaternion (camera -> world)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [  2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)],
        [  2*(xz - wy),   2*(yz + wx), 1-2*(xx+yy)]
    ])

    # Transform points into camera frame: p_cam = R^T * (p_world - t)
    diff = pts_world - np.array([tx, ty, tz])
    pts_cam = diff.dot(R.T)

    # Perspective projection
    x_cam, y_cam, z_cam = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    # compute pts_cam, do projection, but **don’t** round or clip:
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    # keep only those with z>0 and inside a slightly padded window
    valid = (z_cam>0) & (u>=-1) & (u<width+1) & (v>=-1) & (v<height+1)
    return np.stack([u[valid], v[valid]], axis=1)  # shape (M,2)

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Compute the 2D skeleton of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input 2D mask, dtype bool or uint8 (0/255).

    Returns
    -------
    np.ndarray
        Skeletonized mask, dtype uint8 (0/255).
    """
    import numpy as np
    from skimage.morphology import skeletonize

    # ensure boolean
    bool_mask = mask.astype(bool)

    # compute skeleton
    skel = skeletonize(bool_mask)

    # return as uint8 0/255
    return (skel.astype(np.uint8) * 255)

def make_bspline_bounds(ctrl_points: np.ndarray, delta: float = 0.1):
    """
    Given an (N×3) array of B-spline control points, return a 
    bounds list of length 3N for differential_evolution, where
    each coordinate is allowed to vary ±delta around its original value.
    """
    flat = ctrl_points.flatten()
    bounds = [(val - delta, val + delta) for val in flat]
    return bounds

def continuous_score(spline_pts_2d, interp_out, interp_in, beta=0.05):
    """
    mask:    H×W bool
    spline_pts_2d: (N×2) float pixels
    """
    # get distances for points that land outside: false positives
    d_out = interp_out(spline_pts_2d)  # how far from mask when outside
    # get distances for points that land inside: false negatives
    d_in  = interp_in(spline_pts_2d)   # how far from background when inside

    # you can combine them however you like; for example:
    assd = 0.5 * (d_out.mean() + d_in.mean())
    return np.exp(-beta * assd)

def score_function_bspline_new(x, datas, camera_pose, camera_parameters, degree, show=False):
    """
    Score = average distance (in pixels) from spline to mask-skeleton.
    Lower is better.

    Returns:
      assd: float
    """
    # 1) Reshape flat vector into (n_ctrl,3)
    ctrl_pts = np.array(x, dtype=float).reshape(-1, 3)

    # 2) Ground truth mask → skeleton
    _, mask, _, _, _ = datas[-1]           # mask: H×W boolean or 0/1
    skeleton = skeletonize(mask > 0)       # shape (H,W) bool

    # 3) Project spline to subpixel 2D points
    pts2d = project_bspline_pts(ctrl_pts,
                                camera_pose,
                                camera_parameters,
                                degree=degree,
                                num_samples=200)    # → (M,2) floats (u,v)

    # 4) Build a single distance transform of the skeleton’s complement
    #    (distance from every pixel to the nearest skeleton pixel)
    dt_skel = distance_transform_edt(~skeleton)

    # 5) Interpolate DT at your spline points (bilinear)
    H, W = skeleton.shape
    grid_y = np.arange(H)
    grid_x = np.arange(W)
    interp = RegularGridInterpolator(
        (grid_y, grid_x),
        dt_skel,
        bounds_error=False,
        fill_value=dt_skel.max()
    )
    # pts2d is (u,v) = (col,row), but interpolator expects (row, col) → (v,u)
    sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
    dists = interp(sample_pts)

    # 6) Compute average distance
    assd = dists.mean()

    # 7) Optional viz
    if show:
        proj_mask = project_bspline(ctrl_pts, camera_pose, camera_parameters,
                                    degree=degree)
        show_masks([skeleton.astype(np.uint8)*255, proj_mask],
                   title=f"Mean dist: {assd:.2f}px")

    # 8) Return the score to MINIMIZE
    # print(f"score: {assd}")
    return assd

def score_function_bspline_reg(x, datas, camera_pose, camera_parameters, degree, init_x, reg_weight: float = 1000.0, show: bool = False):
    """
    Returns a loss = (average pixel‐distance to skeleton) 
    + reg_weight * (average control‐point displacement).

    Args:
        x:               flat array of length 3*n_ctrl, current ctrl‐pts.
        datas:           your data list, last element’s mask at datas[-1][1].
        camera_pose:     TransformStamped.
        camera_parameters: (fx, fy, cx, cy).
        degree:          spline degree.
        init_x:          flat array same shape as x, the original ctrl‐pts.
        reg_weight:      weight for the drift penalty (in pixels per meter).
        show:            if True, pops up the overlay.

    Returns:
        loss: float to MINIMIZE.
    """
    if rospy.is_shutdown(): exit()
    # ----- 1) Reshape and penalty -----
    ctrl_pts = np.array(x, dtype=float).reshape(-1, 3)
    # compute average L2‐displacement per control‐point (in world units)
    displacement = np.linalg.norm(x - init_x) / ctrl_pts.shape[0]

    # ----- 2) Skeletonize mask -----
    _, mask, _, _, _ = datas[-1]
    skeleton = skeletonize(mask > 0)

    # ----- 3) Project to subpixel pts -----
    pts2d = project_bspline_pts(ctrl_pts,
                                camera_pose,
                                camera_parameters,
                                degree=degree,
                                num_samples=200)  # (M,2) floats

    # ----- 4) Distance transform of skeleton -----
    dt = distance_transform_edt(~skeleton)
    H, W = skeleton.shape
    interp = RegularGridInterpolator(
        (np.arange(H), np.arange(W)),
        dt,
        bounds_error=False,
        fill_value=dt.max()
    )

    # ----- 5) Sample DT at spline pts -----
    sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)  # (row, col)
    dists = interp(sample_pts)
    assd = dists.mean()

    # ----- 6) Optional visualization -----
    if show:
        proj_mask = project_bspline(ctrl_pts,
                                    camera_pose,
                                    camera_parameters,
                                    width=W,
                                    height=H,
                                    degree=degree)
        show_masks([skeleton.astype(np.uint8)*255, proj_mask],
                   title=f"Fit: {assd:.2f}px, Drift: {displacement:.3f}m")

    # ----- 7) Final loss -----
    # print(f"reg_weighot: {reg_weight}, punishment. {reg_weight * displacement}")
    loss = assd + reg_weight * displacement
    return loss

def precompute_skeletons_and_interps(datas):
    """
    One‐time preprocessing: from each mask in datas, build:
      - skeleton (bool H×W)
      - dt = distance transform of the skeleton complement
      - interp DT → fast subpixel lookup
    
    Returns:
      skeletons: list of bool arrays
      interps:   list of RegularGridInterpolator objects
    """
    skeletons = []
    interps   = []
    for (_, mask, _, _, _) in datas:
        sk = skeletonize(mask > 0)
        dt = distance_transform_edt(~sk)
        H, W = sk.shape
        interp = RegularGridInterpolator(
            (np.arange(H), np.arange(W)),
            dt,
            bounds_error=False,
            fill_value=dt.max()
        )
        skeletons.append(sk)
        interps.append(interp)
    return skeletons, interps


def score_function_bspline_reg_multiple_pre(x, camera_poses, camera_parameters, degree, init_x, reg_weight: float, decay: float, curvature_weight: float, skeletons: list, interps: list, num_samples: int):
    """
    Loss = decayed mean ASSD over multiple frames
           + reg_weight * mean control‐point drift
           + curvature_weight * mean squared turn‐angle of control‐points.

    All per‐mask work (skeleton/DT/interp) is precomputed.

    Args:
        x:                 flat array (3*n_ctrl) of current ctrl‐pts.
        camera_poses:      list of TransformStamped, per frame.
        camera_parameters: (fx, fy, cx, cy) intrinsics.
        degree:            spline degree.
        init_x:            flat array same shape as x, original ctrl‐pts.
        reg_weight:        weight for drift penalty.
        decay:             exponential decay ∈(0,1] for older frames.
        curvature_weight:  weight for sharp‐turns penalty.
        skeletons:         list of H×W bool skeletons.
        interps:           list of RegularGridInterpolator for each frame.
        num_samples:       how many points to sample along the spline.

    Returns:
        loss: float to MINIMIZE (never NaN).
    """
    # 1) reshape & drift penalty
    ctrl_pts = x.reshape(-1, 3)
    n_ctrl = ctrl_pts.shape[0]
    drift = np.linalg.norm(x - init_x) / n_ctrl

    # 2) curvature penalty
    if n_ctrl >= 3:
        diffs = ctrl_pts[1:] - ctrl_pts[:-1]
        v1, v2 = diffs[:-1], diffs[1:]
        dot = np.einsum('ij,ij->i', v1, v2)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos_t = np.clip(dot/(n1*n2 + 1e-8), -1, 1)
        angles = np.arccos(cos_t)
        curvature_penalty = np.mean(angles**2)
    else:
        curvature_penalty = 0.0

    # 3) frame weights
    n_frames = len(skeletons)
    weights = np.array([decay**(n_frames - 1 - i) for i in range(n_frames)], dtype=float)
    wsum = weights.sum()

    # 4) accumulate decayed ASSD
    assd_sum = 0.0
    fx, fy, cx, cy = camera_parameters
    for i, cam_pose in enumerate(camera_poses):
        # — instead of project_bspline_pts, do the same transform + intrinsics we tested before —
        # 1) sample spline in world coords
        # rebuild knot vector (same as project_bspline)
        n_ctrl = ctrl_pts.shape[0]
        k = degree
        m = n_ctrl - k - 1
        if m > 0:
            interior = np.linspace(0,1,m+2)[1:-1]
        else:
            interior = np.array([])
        knots = np.concatenate([np.zeros(k+1), interior, np.ones(k+1)])
        spline = BSpline(knots, ctrl_pts, k, axis=0)
        ts = np.linspace(knots[k], knots[-k-1], num_samples)
        pts_world = spline(ts)  # (num_samples, 3)

        # 2) transform into camera frame (using your existing helper)
        pts_cam = transform_points_from_world(pts_world, cam_pose)  # (num_samples,3)

        # 3) apply intrinsics
        fx, fy, cx, cy = camera_parameters
        x_c = pts_cam[:,0];  y_c = pts_cam[:,1];  z_c = pts_cam[:,2]
        valid = (z_c > 0)
        u = (fx * x_c[valid] / z_c[valid] + cx)
        v = (fy * y_c[valid] / z_c[valid] + cy)
        pts2d = np.stack([u, v], axis=1)

        # guard against empty projection
        if pts2d.size == 0:
            # no points visible → huge penalty
            assd_i = np.max(interps[i].values)  # max distance
        else:
            # sample the precomputed interpolator
            sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
            dists = interps[i](sample_pts)
            # guard NaNs
            assd_i = np.nanmean(dists) if np.isfinite(dists).any() else np.max(interps[i].values)

        assd_sum += weights[i] * assd_i

    mean_assd = assd_sum / wsum

    # 5) final loss
    loss = mean_assd + reg_weight * drift + curvature_weight * curvature_penalty
    # print(f"loss: {loss:.3f}")
    return float(loss)

def project_bspline_subpixel(ctrl_points: np.ndarray,
                                camera_pose: TransformStamped,
                                camera_parameters,
                                degree: int,
                                num_samples: int) -> np.ndarray:
    """
    Sample the 3D B-spline in world, transform into the given camera frame,
    and return the floating-point image coordinates (u,v) for each sample.

    Parameters
    ----------
    ctrl_points : (N_ctrl×3) array
        Control-points in WORLD coordinates.
    camera_pose : TransformStamped
        camera→world transform.
    camera_parameters : (fx, fy, cx, cy)
    degree : int
        Spline degree.
    num_samples : int
        How many points to sample along the spline.

    Returns
    -------
    pts2d : (M×2) array of floats
        Each row = (u, v) in pixel coordinates.  Points behind the camera
        or outside the image plane are simply omitted.
    """
    fx, fy, cx, cy = camera_parameters

    # 1) build knot vector
    n_ctrl = ctrl_points.shape[0]
    k = degree
    m = n_ctrl - k - 1
    interior = np.linspace(0, 1, m+2)[1:-1] if m>0 else np.array([])
    knots = np.concatenate([
        np.zeros(k+1),
        interior,
        np.ones(k+1)
    ])

    # 2) sample spline in world
    spline = BSpline(knots, ctrl_points, k, axis=0)
    ts = np.linspace(knots[k], knots[-k-1], num_samples)
    world_pts = spline(ts)      # (num_samples,3)

    # 3) transform to camera frame (use your helper to avoid R vs R^T mistakes)
    #    assumes you have a method `transform_points_from_world`
    #    which does exactly: p_cam = (T_cam2world)^{-1} * p_world
    cam_pts = transform_points_from_world(world_pts, camera_pose)
    x_cam, y_cam, z_cam = cam_pts.T

    # 4) perspective project (float)
    # only keep points in front of camera
    valid = z_cam > 0
    x_cam = x_cam[valid]
    y_cam = y_cam[valid]
    z_cam = z_cam[valid]

    u = (fx * x_cam / z_cam) + cx
    v = (fy * y_cam / z_cam) + cy

    # 5) filter to image bounds if you like, or leave to your interpolator
    pts2d = np.stack([u, v], axis=1)
    in_bounds = (
        (u >= 0) & (u < cx*2) &
        (v >= 0) & (v < cy*2)
    )
    return pts2d[in_bounds]


def score_function_bspline_reg_multiple(x, datas, camera_poses, camera_parameters, degree, init_x, reg_weight: float = 1000.0, decay: float = 1.0, curvature_weight: float = 1.0, show: bool = False):
    """
    Loss = decayed mean ASSD over multiple frames
           + reg_weight * mean control‐point drift
           + curvature_weight * mean squared turn‐angle of control‐points.

    Args:
        x:                 flat array (3*n_ctrl) of current ctrl‐pts.
        datas:             list of data tuples; mask = data_i[1].
        camera_poses:      list of TransformStamped, per frame.
        camera_parameters: (fx, fy, cx, cy) intrinsics.
        degree:            spline degree.
        init_x:            flat array same shape as x, original ctrl‐pts.
        reg_weight:        weight for drift penalty.
        decay:             exponential decay ∈(0,1] for older frames.
        curvature_weight:  weight for sharp‐turns penalty.
        show:              if True, visualize overlays.

    Returns:
        loss: float to MINIMIZE.
    """
    if rospy.is_shutdown(): exit()
    # 1) reshape and drift penalty
    ctrl_pts = np.array(x, dtype=float).reshape(-1, 3)
    n_ctrl = ctrl_pts.shape[0]
    drift = np.linalg.norm(x - init_x) / n_ctrl

    # 2) curvature penalty: mean squared turn‐angle at interior ctrl‐pts
    if n_ctrl >= 3:
        # vectors between successive control points
        diffs = ctrl_pts[1:] - ctrl_pts[:-1]        # shape (n_ctrl-1, 3)
        # compute angles between consecutive diffs
        v1 = diffs[:-1]
        v2 = diffs[1:]
        # dot products and norms
        dot = np.einsum('ij,ij->i', v1, v2)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        # avoid division by zero
        cos_theta = np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0)
        angles = np.arccos(cos_theta)               # radians
        curvature_penalty = np.mean(angles**2)
    else:
        curvature_penalty = 0.0

    # 3) decayed ASSD over frames
    n_frames = len(datas)
    weights = np.array([decay**(n_frames - 1 - i) for i in range(n_frames)], dtype=float)
    wsum = weights.sum()

    assd_sum = 0.0
    for i, (data_i, cam_pose) in enumerate(zip(datas, camera_poses)):
        _, mask, _, _, _ = data_i
        skeleton = skeletonize(mask > 0)

        pts2d = project_bspline_pts(ctrl_pts,
                                    cam_pose,
                                    camera_parameters,
                                    degree=degree,
                                    num_samples=200)

        dt = distance_transform_edt(~skeleton)
        H, W = skeleton.shape
        interp = RegularGridInterpolator(
            (np.arange(H), np.arange(W)),
            dt,
            bounds_error=False,
            fill_value=dt.max()
        )
        sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
        dists = interp(sample_pts)
        assd_sum += weights[i] * dists.mean()

    mean_assd = assd_sum / wsum

    # 4) total loss
    loss = mean_assd + reg_weight * drift + curvature_weight * curvature_penalty

    # debug
    if show:
        print(f"Mean ASSD: {mean_assd:.2f}, Drift: {drift*reg_weight:.2f}, "
              f"Curvature: {curvature_penalty:.2f}")
    return loss

def score_bspline_translation(shift_xy: np.ndarray, data, camera_pose, camera_parameters: tuple, degree: int, ctrl_points: np.ndarray, num_samples: int = 200) -> float:
    """
    Score (mean symmetric distance) of a B-spline after translating it
    parallel to the camera's image plane by shift_xy = [dx, dy] in camera coords.

    Args:
        shift_xy:            length-2 array [dx, dy] in camera-frame meters.
        ctrl_points:         (N_ctrl×3) original control points (world).
        data:                a tuple whose second element is the H×W mask.
        camera_pose:         TransformStamped of camera in world.
        camera_parameters:   (fx, fy, cx, cy) intrinsics.
        degree:              spline degree.
        num_samples:         how many points to sample along the spline.

    Returns:
        assd: mean distance (in pixels) between projected spline and mask skeleton.
    """
    if rospy.is_shutdown(): exit()
    # 1) Apply translation in camera frame to control points
    #    Build rotation matrix R from camera_pose quaternion (cam->world)
    q = camera_pose.transform.rotation
    tx, ty, tz = (camera_pose.transform.translation.x,
                  camera_pose.transform.translation.y,
                  camera_pose.transform.translation.z)
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    # rotation matrix from quaternion
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz),   2*(xy - wz),     2*(xz + wy)    ],
        [2*(xy + wz),       1 - 2*(xx + zz), 2*(yz - wx)    ],
        [2*(xz - wy),       2*(yz + wx),     1 - 2*(xx + yy)]
    ])  # world <- camera

    # translation vector in world coords
    dx, dy = shift_xy
    shift_cam = np.array([dx, dy, 0.0], dtype=float)
    shift_world = R.dot(shift_cam)

    # shifted control points
    ctrl_pts_shifted = np.asarray(ctrl_points, dtype=float) + shift_world

    # 2) get mask and skeleton
    _, mask, _, _, _ = data
    skeleton = skeletonize(mask > 0)

    # 3) project shifted spline to subpixel 2D points
    pts2d = project_bspline_pts(ctrl_pts_shifted,
                                camera_pose,
                                camera_parameters,
                                degree=degree,
                                num_samples=num_samples)  # (M,2) floats (u,v)

    # 4) compute distance transform of skeleton complement
    dt = distance_transform_edt(~skeleton)
    H, W = skeleton.shape
    interp = RegularGridInterpolator(
        (np.arange(H), np.arange(W)),
        dt,
        bounds_error=False,
        fill_value=dt.max()
    )

    # 5) sample distances (interpolator expects (row, col))
    sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
    dists = interp(sample_pts)

    # 6) return mean distance (ASSD)
    return float(dists.mean())

def apply_translation_to_ctrl_points(ctrl_points: np.ndarray, shift_xy: np.ndarray, camera_pose) -> np.ndarray:
    """
    Applies a 2D translation (dx, dy) in camera image plane to all B-spline control points.

    Args:
        ctrl_points: (N×3) array of original control points in world coordinates.
        shift_xy:    length-2 array [dx, dy] representing translation in camera-frame meters.
        camera_pose: TransformStamped of camera in world.

    Returns:
        shifted_ctrl_points: (N×3) array of translated control points in world coordinates.
    """
    # Extract translation and quaternion from camera_pose
    tx, ty, tz = (camera_pose.transform.translation.x,
                  camera_pose.transform.translation.y,
                  camera_pose.transform.translation.z)
    qx = camera_pose.transform.rotation.x
    qy = camera_pose.transform.rotation.y
    qz = camera_pose.transform.rotation.z
    qw = camera_pose.transform.rotation.w

    # Build rotation matrix from quaternion (camera->world)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),   1-2*(xx+yy)]
    ])

    # Create camera-frame shift vector and convert to world frame
    dx, dy = shift_xy
    shift_cam = np.array([dx, dy, 0.0], dtype=float)
    shift_world = R.dot(shift_cam)

    # Apply shift to each control point
    shifted_ctrl_points = np.asarray(ctrl_points, dtype=float) + shift_world[np.newaxis, :]

    return shifted_ctrl_points

def shift_control_points(ctrl_points: np.ndarray, mask: np.ndarray, camera_pose, camera_parameters: tuple, degree: int = 3, num_samples: int = 200) -> np.ndarray:
    """
    Shift the entire B-spline parallel to the camera's image plane so that
    the centroid of its projection aligns with the centroid of the mask.

    Args:
        ctrl_points:        (N_ctrl×3) array of control points (world coords).
        mask:               (H×W) binary mask (0/1 or bool).
        camera_pose:        TransformStamped with .transform.translation & rotation.
        camera_parameters:  (fx, fy, cx, cy) intrinsics.
        degree:             spline degree (default=3).
        num_samples:        how many points to sample along the spline.

    Returns:
        shifted_ctrl_points: (N_ctrl×3) array of translated control points.
    """
    fx, fy, cx, cy = camera_parameters

    # 1) Build the B-spline and sample points in world coords
    pts = np.asarray(ctrl_points, dtype=float)
    n_ctrl = pts.shape[0]
    k = degree
    # knot vector: open-uniform
    n_inner = n_ctrl - k - 1
    if n_inner > 0:
        inner = np.linspace(0, 1, n_inner+2)[1:-1]
        knots = np.concatenate((np.zeros(k+1), inner, np.ones(k+1)))
    else:
        knots = np.concatenate((np.zeros(k+1), np.ones(k+1)))
    spline = BSpline(knots, pts, k, axis=0)
    u = np.linspace(0, 1, num_samples)
    samples_world = spline(u)  # (M,3)

    # 2) Compute camera->world rotation matrix R (world = R @ cam)
    q = camera_pose.transform.rotation
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),   1-2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),   1-2*(xx+yy)]
    ])

    # camera translation in world coords
    t = np.array([camera_pose.transform.translation.x,
                  camera_pose.transform.translation.y,
                  camera_pose.transform.translation.z], dtype=float)

    # 3) Transform samples into camera frame: p_cam = R^T @ (p_world - t)
    diff = samples_world - t
    samples_cam = diff.dot(R.T)
    x_cam, y_cam, z_cam = samples_cam[:,0], samples_cam[:,1], samples_cam[:,2]
    valid = z_cam > 0
    x_cam, y_cam, z_cam = x_cam[valid], y_cam[valid], z_cam[valid]

    # 4) Project to pixel coords (float)
    u_proj = fx * x_cam / z_cam + cx
    v_proj = fy * y_cam / z_cam + cy

    # 5) Centroids
    # mask centroid (u_mask, v_mask)
    rows, cols = np.where(mask > 0)
    u_mask = cols.mean()
    v_mask = rows.mean()
    # spline centroid (u_spline, v_spline)
    u_spline = u_proj.mean()
    v_spline = v_proj.mean()

    # 6) Compute pixel shifts
    du = u_mask - u_spline
    dv = v_mask - v_spline

    # 7) Convert pixel shift to camera-frame meters at mean depth
    z_ref = z_cam.mean()
    dx = du * z_ref / fx
    dy = dv * z_ref / fy
    shift_cam = np.array([dx, dy, 0.0], dtype=float)

    # 8) Convert to world-frame shift: shift_world = R @ shift_cam
    shift_world = R.dot(shift_cam)

    # 9) Apply to all control points
    shifted_ctrl_points = pts + shift_world[np.newaxis, :]

    return shifted_ctrl_points

def interactive_bspline_editor(ctrl_points: np.ndarray,
                               datas,
                               camera_pose,
                               camera_parameters: tuple,
                               degree: int,
                               score_function_bspline,
                               init_ctrl_points: np.ndarray,
                               delta: float = 0.5,
                               slider_scale: int = 1000):
    """
    Launch an OpenCV window with sliders to manually adjust B-spline control points,
    overlays the projected spline on the ground-truth mask, and shows the score.

    Args:
        ctrl_points: (N_control × 3) initial control points.
        datas:       List of data tuples; last element's second item is ground-truth mask (H×W bool/uint8).
        camera_pose: TransformStamped for projection.
        camera_parameters: (fx, fy, cx, cy) intrinsics.
        degree:      Spline degree for projection.
        score_function_bspline: Callable(x_flat, datas, camera_pose, camera_parameters, degree) -> -score.
        delta:       ±range around each original ctrl-point coordinate.
        slider_scale: Integer range for sliders (0...slider_scale).

    Returns:
        final_ctrl_pts: (N_control × 3) array of adjusted control points upon exit.
    """
    # Extract mask from datas
    _, mask, _, _, _ = datas[-1]
    skeleton = skeletonize_mask(mask)

    skeleton_img = (skeleton.astype(np.uint8) * 255) if skeleton.dtype != np.uint8 else skeleton
    h, w = skeleton_img.shape[:2]


    # Prepare slider window
    window_name = "B-Spline Editor (press 'q' to finish)"
    cv2.namedWindow(window_name)

    n_ctrl = ctrl_points.shape[0]
    orig_flat = ctrl_points.flatten()

    # Callback placeholder (required by OpenCV)
    def _cb(val):
        pass

    # Create one slider per control-point coordinate
    for idx in range(len(orig_flat)):
        cv2.createTrackbar(f"pt{idx}", window_name, slider_scale // 2, slider_scale, _cb)

    # Main loop
    final_ctrl_pts = orig_flat.copy()
    while True:
        # Read slider positions and map to control-point values
        positions = np.array([cv2.getTrackbarPos(f"pt{i}", window_name) for i in range(len(orig_flat))])
        final_flat = orig_flat - delta + (positions / slider_scale) * (2 * delta)
        final_ctrl_pts = final_flat.reshape(-1, 3)

        # Project spline to mask
        proj_mask = project_bspline(final_ctrl_pts, camera_pose, camera_parameters, width=w, height=h, degree=degree)

        # Overlay: original mask in gray, spline in red
        disp = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        disp[proj_mask > 0] = (0, 0, 255)

        # Compute and display score (positive value)
        score_neg = score_function_bspline(final_flat, datas, camera_pose, camera_parameters, degree, init_ctrl_points.flatten())
        score = -score_neg
        cv2.putText(disp, f"Score: {score:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show window
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return final_ctrl_pts

def get_highest_point_and_angle_spline(ctrl_points: np.ndarray, degree: int = 3, num_samples: int = 1000):
    """
    Samples the B-spline densely, finds the 3D point of maximum z,
    and computes the tangent angle at that point (projection onto XY-plane),
    normalized to the range [-pi/2, pi/2].

    Args:
        ctrl_points:  (N×3) array of control points.
        degree:       spline degree (default 3).
        num_samples:  number of samples along the spline for search.

    Returns:
        highest_pt:   (x, y, z) numpy array of the highest point on the spline.
        angle:        tangent angle in radians in [-pi/2, pi/2].
    """
    pts = np.asarray(ctrl_points, dtype=float)
    N, dim = pts.shape
    if dim != 3:
        raise ValueError("ctrl_points must be an (N,3) array")
    if N <= degree:
        raise ValueError("Need at least degree+1 control points")

    # build open-uniform knot vector
    k = degree
    n_inner = N - k - 1
    if n_inner > 0:
        inner = np.linspace(0, 1, n_inner+2)[1:-1]
        knots = np.concatenate((np.zeros(k+1), inner, np.ones(k+1)))
    else:
        knots = np.concatenate((np.zeros(k+1), np.ones(k+1)))

    spline = BSpline(knots, pts, k, axis=0)
    u = np.linspace(0, 1, num_samples)
    samples = spline(u)  # (num_samples, 3)

    # find index of max z
    idx_max = np.argmax(samples[:, 2])
    highest_pt = samples[idx_max]

    # compute derivative spline and evaluate at same u
    dspline = spline.derivative()
    d_samples = dspline(u)  # (num_samples, 3)
    dx, dy = d_samples[idx_max, 0], d_samples[idx_max, 1]

    # angle of tangent in XY-plane
    angle = np.arctan(dx/dy)
    angle += np.pi/2
    # normalize to [-pi/2, pi/2]
    while angle > np.pi/2:
        angle -= np.pi
    while angle < -np.pi/2:
        angle += np.pi

    return highest_pt, angle

# Example usage:
# ctrl_pts = np.random.rand(10, 3)
# pt, ang = get_highest_point_and_tangent_angle(ctrl_pts)
# print("Highest:", pt, "Angle (rad):", ang)



def convert_bspline_to_pointcloud(ctrl_points: np.ndarray, samples: int = 150, degree: int = 3) -> o3d.geometry.PointCloud:
    """
    Converts a B-spline defined by control points into an Open3D PointCloud
    by sampling points along the curve.

    Args:
        ctrl_points: (N_control × 3) array of control points.
        samples:     Number of points to sample along the spline (integer ≥ 2).
        degree:      Spline degree (k). Must satisfy N_control > degree.

    Returns:
        pcd: Open3D.geometry.PointCloud containing `samples` points sampled
             along the B-spline.
    """
    pts = np.asarray(ctrl_points, dtype=float)
    n_ctrl = pts.shape[0]
    k = degree
    if n_ctrl <= k:
        raise ValueError("Number of control points must exceed spline degree")

    # Open-uniform knot vector: (k+1) zeros, inner knots, (k+1) ones
    n_inner = n_ctrl - k - 1
    if n_inner > 0:
        inner = np.linspace(0, 1, n_inner + 2)[1:-1]
        knots = np.concatenate((np.zeros(k+1), inner, np.ones(k+1)))
    else:
        knots = np.concatenate((np.zeros(k+1), np.ones(k+1)))

    # Build the spline and sample
    spline = BSpline(knots, pts, k, axis=0)
    u = np.linspace(0, 1, samples)
    samples_3d = spline(u)  # shape: (samples, 3)

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples_3d)

    return pcd

#endregion


#region new spline initialization old
# def extract_centerline_from_mask_overlap(
#     depth_image: np.ndarray,
#     mask: np.ndarray,
#     camera_parameters,
#     depth_scale: float = 1.0,
#     connectivity: int = 8,
#     min_length: int = 20,
#     conn_radius: float = 30.0,
#     w_angle: float = 4.0,
#     w_dist: float = 2.0,
#     w_curv: float = 5.0,
#     show: bool = False
# ) -> np.ndarray:
#     """
#     Extracts the 3D centerline from a binary mask of a cable-like structure,
#     handling overlaps by skeletonizing, excising intersections, removing short
#     branches, and reconnecting endpoints based on curvature and distance.

#     Parameters:
#         depth_image (np.ndarray): 2D array of depth values (same resolution as mask).
#         mask (np.ndarray): 2D binary mask (0 or 255) of the object.
#         camera_parameters: tuple (fx, fy, cx, cy) intrinsic parameters.
#         depth_scale (float): Scale factor to convert depth units.
#         connectivity (int): Pixel connectivity for skeleton and components (4 or 8).
#         min_length (int): Minimum branch length (in pixels) to keep.
#         conn_radius (float): Radius (in px) to consider endpoints for reconnection.
#         w_angle (float): Weight for angle cost in matching.
#         w_dist (float): Weight for distance cost in matching.
#         w_curv (float): Weight for curvature cost in matching.
#         show (bool): If True, display intermediate masks via show_masks().

#     Returns:
#         np.ndarray: Ordered centerline points in 3D, shape (N, 3).
#     """
#     fx, fy, cx, cy = camera_parameters
#     # 1. Binarize mask to {0,1}
#     binary = (mask > 0).astype(np.uint8)
#     if show:
#         show_masks([binary * 255], title="Binary mask")

#     # 2. Skeletonize
#     skeleton = skeletonize(binary.astype(bool)).astype(np.uint8)
#     if show:
#         show_masks([skeleton * 255], title="Skeleton")

#     # 3. Excise intersections
#     kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
#     neighbor_count = cv2.filter2D(skeleton, -1, kernel)
#     ints = np.argwhere((skeleton == 1) & (neighbor_count >= 3))
#     dist_map = distance_transform_edt(binary)
#     to_remove = np.zeros_like(skeleton, dtype=bool)
#     H, W = skeleton.shape
#     for y, x in ints:
#         r = dist_map[y, x]
#         R = int(np.ceil(r))
#         y0, y1 = max(0, y - R), min(H, y + R + 1)
#         x0, x1 = max(0, x - R), min(W, x + R + 1)
#         yy, xx = np.ogrid[y0:y1, x0:x1]
#         circle = (yy - y)**2 + (xx - x)**2 <= r**2
#         to_remove[y0:y1, x0:x1][circle] = True
#     skeleton_excised = (skeleton.astype(bool) & ~to_remove).astype(np.uint8)
#     if show:
#         show_masks([skeleton * 255, skeleton_excised * 255], title="After excising intersections")

#     # 4. Remove short segments
#     num_labels, labels = cv2.connectedComponents(skeleton_excised, connectivity=connectivity)
#     for lbl in range(1, num_labels):
#         comp = (labels == lbl)
#         if comp.sum() < min_length:
#             skeleton_excised[comp] = 0
#     if show:
#         show_masks([skeleton * 255, skeleton_excised * 255], title=f"Pruned < {min_length}px segments")

#     # Debug: display each branch individually after removing crossover
#     if show:
#         num_branches, branch_labels = cv2.connectedComponents(skeleton_excised, connectivity=connectivity)
#         for b in range(1, num_branches):
#             branch_mask = (branch_labels == b).astype(np.uint8) * 255
#             show_masks([branch_mask], title=f"Branch {b}")

#     # 5. Reconnect endpoints with curvature
#     neighbor_count2 = cv2.filter2D(skeleton_excised, -1, kernel)
#     endpoints = np.argwhere((skeleton_excised == 1) & (neighbor_count2 == 1))
#     pts = endpoints
#     dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
#     counts = (dmat < conn_radius).sum(axis=1) - 1
#     to_conn = counts >= 1
#     if to_conn.sum() % 2 == 1:
#         sub = dmat[to_conn][:, to_conn]
#         np.fill_diagonal(sub, np.inf)
#         drop = np.argmax(sub.min(axis=1))
#         global_idx = np.where(to_conn)[0][drop]
#         to_conn[global_idx] = False
#     conn_eps = endpoints[to_conn]
#     # compute tangents & curvatures
#     vectors, curv = [], []
#     for y, x in conn_eps:
#         # tangent
#         for dy in (-1,0,1):
#             for dx in (-1,0,1):
#                 if dy==dx==0: continue
#                 ny, nx = y+dy, x+dx
#                 if 0 <= ny < H and 0 <= nx < W and skeleton_excised[ny, nx]:
#                     v = np.array([y-ny, x-nx], float)
#                     vectors.append(v/np.linalg.norm(v))
#                     break
#             else:
#                 continue
#             break
#         # curvature (identical to previous)
#         nbr1 = [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
#                 if not (dy==dx==0) and 0<=y+dy<H and 0<=x+dx<W and skeleton_excised[y+dy,x+dx]]
#         if nbr1:
#             y1, x1 = nbr1[0]
#             nbr2 = [(y1+dy, x1+dx) for dy in (-1,0,1) for dx in (-1,0,1)
#                     if (dy!=0 or dx!=0) and (y1+dy, x1+dx)!=(y,x)
#                     and 0<=y1+dy<H and 0<=x1+dx<W and skeleton_excised[y1+dy,x1+dx]]
#             if nbr2:
#                 y2, x2 = nbr2[0]
#                 v1 = np.array([y1-y, x1-x], float)
#                 v2 = np.array([y2-y1, x2-x1], float)
#                 angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
#                 curv.append(angle/(np.linalg.norm(v1)+1e-8))
#         else:
#             curv.append(0.0)
#     curv = np.array(curv)
#     curv_norm = curv/(curv.max()+1e-8)
#     # build cost matrix & match
#     n = len(conn_eps)
#     dists = np.linalg.norm(conn_eps[:,None,:]-conn_eps[None,:,:],axis=2)
#     d_max = dists.max() if n>1 else 1
#     A = np.zeros((n,n),float)
#     for i in range(n):
#         for j in range(i+1,n):
#             val = np.dot(vectors[i],vectors[j])
#             val = np.clip(val, -1.0, 1.0)
#             ang_cost = np.pi - np.arccos(val)
#             d_cost = dists[i,j]/(d_max+1e-8)
#             k_cost = abs(curv_norm[i]-curv_norm[j])
#             A[i,j] = A[j,i] = w_angle*ang_cost + w_dist*d_cost + w_curv*k_cost
#     from itertools import combinations
#     def find_min_pairs(A_mat):
#         m=A_mat.shape[0]
#         assert m%2==0
#         best_cost, best_pairs=float('inf'),None
#         def recurse(rem,pairs):
#             nonlocal best_cost,best_pairs
#             if not rem:
#                 c=sum(A_mat[i,j] for i,j in pairs)
#                 if c<best_cost:
#                     best_cost,best_pairs=c,pairs.copy()
#                 return
#             i=rem[0]
#             for j in rem[1:]: recurse([k for k in rem if k not in (i,j)],pairs+[(i,j)])
#         recurse(list(range(m)),[])
#         return best_pairs
#     pairs = find_min_pairs(A) if n>0 else []
#     connected = skeleton_excised.copy()
#     disp = (connected*255).astype(np.uint8)
#     for i,j in pairs:
#         y0,x0 = conn_eps[i]
#         y1,x1 = conn_eps[j]
#         cv2.line(disp,(x0,y0),(x1,y1),255,1)
#     connected = (disp>0).astype(np.uint8)
#     if show:
#         show_masks([skeleton_excised*255, disp], title="Reconnected skeleton")
#     # 6. Simplified linear walk from endpoint A to B
#     # Compute degree-1 endpoints
#     neighbor_count3 = cv2.filter2D(connected, -1, kernel)
#     end_pts = np.argwhere((connected == 1) & (neighbor_count3 == 1))
#     # Determine start and end
#     if len(end_pts) >= 2:
#         start = tuple(end_pts[0])
#         end_pt = tuple(end_pts[1])
#     elif len(end_pts) == 1:
#         start = tuple(end_pts[0])
#         end_pt = None
#     else:
#         # No endpoints: pick arbitrary start
#         pts_all = np.argwhere(connected == 1)
#         start = tuple(pts_all[0])
#         end_pt = None
#     # Walk greedily
#     ordered = [start]
#     prev = None
#     cur = start
#     while True:
#         if cur == end_pt:
#             break
#         y, x = cur
#         # collect neighbors (8-connectivity) excluding prev
#         nexts = []
#         for dy in (-1,0,1):
#             for dx in (-1,0,1):
#                 if dy == 0 and dx == 0: continue
#                 ny, nx = y + dy, x + dx
#                 if 0 <= ny < H and 0 <= nx < W and connected[ny, nx]:
#                     if prev is None or (ny, nx) != prev:
#                         nexts.append((ny, nx))
#         if not nexts:
#             break
#         # pick the first
#         nxt = nexts[0]
#         ordered.append(nxt)
#         prev, cur = cur, nxt
#     # 7. Show ordered spline
#     if show:
#         from my_utils import show_spline_gradient
#         centerline_2d = np.array(ordered)
#         show_spline_gradient(mask, centerline_2d, title="Ordered Centerline")
#     # 8. Back-project to 3D Show ordered spline
#     if show:
#         from my_utils import show_spline_gradient
#         centerline_2d = np.array(ordered)
#         show_spline_gradient(mask, centerline_2d, title="Ordered Centerline")

# # 8. Back-project to 3D Show ordered spline
#     if show:
#         from my_utils import show_spline_gradient
#         centerline_2d=np.array(ordered)
#         show_spline_gradient(mask, centerline_2d, title="Ordered Centerline")
#     # 8. Back-project to 3D
#     pts3d=[]
#     for y,x in ordered:
#         z=float(depth_image[y,x])*depth_scale
#         X=(x-cx)*z/fx;Y=(y-cy)*z/fy
#         pts3d.append([X,Y,z])
#     return np.array(pts3d)


#endregion


#region new spline initialization
def extract_centerline_from_mask_overlap(
    depth_image: np.ndarray,
    mask: np.ndarray,
    camera_parameters: tuple,
    depth_scale: float = 1.0,
    connectivity: int = 8,
    min_length: int = 20,
    conn_radius: float = 30.0,
    w_angle: float = 50.0,
    w_dist: float = 0.5,
    w_curv: float = 50.0,
    show: bool = False
    ) -> np.ndarray:
    """
    Extracts a single ordered 3D centerline spline from a cable mask+depth.

    Returns:
        pts3d: (N,3) array of (X,Y,Z) in camera coords.
    """
    fx, fy, cx, cy = camera_parameters

    # 1. Binarise mask
    _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # 2. Skeletonize
    skeleton_bool = skeletonize(binary.astype(bool))
    skeleton_disp = skeleton_bool.astype(np.uint8) * 255
    if show:
        show_masks([binary*255, skeleton_disp], title="Skeleton")

    # 3. Excise intersections
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bool.astype(np.uint8), -1, kernel)
    ints = np.argwhere((skeleton_bool) & (neighbor_count >= 3))
    dist_map = distance_transform_edt(binary)
    to_remove = np.zeros_like(skeleton_bool, bool)
    H, W = skeleton_bool.shape
    for y, x in ints:
        r = dist_map[y, x]
        R = int(np.ceil(r))
        y0, y1 = max(0, y-R), min(H, y+R+1)
        x0, x1 = max(0, x-R), min(W, x+R+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy-y)**2 + (xx-x)**2 <= r**2
        to_remove[y0:y1, x0:x1][circle] = True
    skeleton_excised = skeleton_bool & ~to_remove
    skeleton_excised_disp = skeleton_excised.astype(np.uint8) * 255
    if show:
        show_masks([skeleton_disp, skeleton_excised_disp],
                            title="Before vs After Excising Intersection Zone")

    # 4. Remove short segments
    num_labels, labels = cv2.connectedComponents(skeleton_excised.astype(np.uint8), connectivity)
    for lbl in range(1, num_labels):
        comp = (labels == lbl)
        if comp.sum() < min_length:
            skeleton_excised[comp] = False
    skeleton_excised_disp = skeleton_excised.astype(np.uint8) * 255
    if show:
        show_masks([skeleton_disp, skeleton_excised_disp],
                            title=f"Removed segments shorter than {min_length}px")

    # 5. Extract paths
    num_labels2, labels2 = cv2.connectedComponents(skeleton_excised.astype(np.uint8), connectivity)
    real_paths = []
    connection_paths = []
    for lbl in range(1, num_labels2):
        comp_mask = (labels2 == lbl)
        pixels = [tuple(pt) for pt in np.argwhere(comp_mask)]
        nbrs = {p: [] for p in pixels}
        for y, x in pixels:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==dx==0: continue
                    q = (y+dy, x+dx)
                    if q in nbrs:
                        nbrs[(y,x)].append(q)
        ends = [p for p,ne in nbrs.items() if len(ne)==1]
        start = ends[0] if ends else pixels[0]
        ordered, prev, curr = [], None, start
        while True:
            ordered.append(curr)
            nxt = [n for n in nbrs[curr] if n!=prev]
            if not nxt: break
            prev, curr = curr, nxt[0]
        if show:
            show_spline_gradient(binary, np.array(ordered),
                                          title=f"Segment {lbl}")
        real_paths.append(np.array(ordered))

    # 6. Pair endpoints
    neighbor_count_exc = cv2.filter2D(skeleton_excised.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere((skeleton_excised) & (neighbor_count_exc==1))
    pts = endpoints
    dmat = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
    neigh_counts = (dmat < conn_radius).sum(axis=1) - 1
    mask_conn = neigh_counts>=1
    if mask_conn.sum()%2==1:
        sub = dmat[mask_conn][:,mask_conn]
        np.fill_diagonal(sub, np.inf)
        drop = np.argmax(sub.min(axis=1))
        idx_global = np.where(mask_conn)[0][drop]
        mask_conn[idx_global] = False
    endpoints = endpoints[mask_conn]
    n = len(endpoints)

    # compute tangents & curvatures
    vectors, curv = [], []
    for y,x in endpoints:
        # tangent
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==dx==0: continue
                ny,nx=y+dy,x+dx
                if 0<=ny<H and 0<=nx<W and skeleton_excised[ny,nx]:
                    v=np.array([y-ny,x-nx],float)
                    vectors.append(v/ (np.linalg.norm(v)+1e-8))
                    break
            else:
                continue
            break
        # curvature
        nbrs1=[(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
               if not(dy==dx==0) and 0<=y+dy<H and 0<=x+dx<W and skeleton_excised[y+dy,x+dx]]
        if not nbrs1:
            curv.append(0.0); continue
        y1,x1=nbrs1[0]
        nbrs2=[(y1+dy,x1+dx) for dy in (-1,0,1) for dx in (-1,0,1)
               if not(dy==dx==0) and (y1+dy,x1+dx)!=(y,x)
               and 0<=y1+dy<H and 0<=x1+dx<W and skeleton_excised[y1+dy,x1+dx]]
        if not nbrs2:
            curv.append(0.0); continue
        y2,x2=nbrs2[0]
        v1=np.array([y1-y,x1-x],float)
        v2=np.array([y2-y1,x2-x1],float)
        angle=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
        curv.append(angle/(np.linalg.norm(v1)+1e-8))
    curv=np.array(curv)
    curv_norm=curv/(curv.max()+1e-8)
    dists=dmat[mask_conn][:,mask_conn]
    d_max=dists.max()

    # build cost matrix with clamp and signed-dot
    A=np.zeros((n,n),float)
    for i in range(n):
        for j in range(i+1,n):
            dij=dists[i,j]
            if dij>conn_radius:
                cost=np.inf
            else:
                dot=np.dot(vectors[i],vectors[j])
                ang=np.arccos(-dot)
                dcost=dij/(d_max+1e-8)
                kcost=abs(curv_norm[i]-curv_norm[j])
                cost=w_angle*ang + w_dist*dcost + w_curv*kcost
            A[i,j]=A[j,i]=cost

    def find_min_pairs(M):
        m=M.shape[0]
        assert m%2==0
        best_cost,best=float('inf'),None
        def recurse(rem,pairs):
            nonlocal best_cost,best
            if not rem:
                c=sum(M[i,j] for i,j in pairs)
                if c<best_cost:
                    best_cost,best=c,pairs.copy()
                return
            i=rem[0]
            for j in rem[1:]:
                recurse([k for k in rem if k not in (i,j)],pairs+[(i,j)])
        recurse(list(range(m)),[])
        return best
    pairs=find_min_pairs(A)

    # draw & store connections
    disp_conn=skeleton_excised_disp.copy()
    for i,j in pairs:
        y0,x0=endpoints[i]
        y1,x1=endpoints[j]
        cv2.line(disp_conn,(x0,y0),(x1,y1),255,1)
    if show:
        show_masks([disp_conn],title="Connected Skeleton w/ Curvature")
    for i,j in pairs:
        y0,x0=endpoints[i]
        y1,x1=endpoints[j]
        rr,cc=skline(y0,x0,y1,x1)
        connection_paths.append(np.stack([rr,cc],axis=1))
        if show:
            show_spline_gradient(binary, np.stack([rr,cc],axis=1),
                                          title="Connection")

    # build connection set
    conn_set = {tuple(pt) for cp in connection_paths for pt in cp}

    # merge all paths into one chain, storing tuples
    def endpoints_of(p): return tuple(p[0]), tuple(p[-1])
    total = []
    for p in real_paths: total.append({'coords': p, 'type': 'real'})
    for p in connection_paths: total.append({'coords': p, 'type': 'conn'})

    ep_map = {}
    for idx, entry in enumerate(total):
        a, b = endpoints_of(entry['coords'])
        for e in (a, b): ep_map.setdefault(e, []).append(idx)

    # find starting endpoint
    start = None
    for idx, entry in enumerate(total):
        if entry['type'] != 'real': continue
        a, b = endpoints_of(entry['coords'])
        for e in (a, b):
            if len(ep_map[e]) == 1:
                start = (idx, e)
                break
        if start: break
    if start is None:
        raise RuntimeError("No unique starting endpoint.")

    merged = []
    used = set()
    curr_idx, curr_pt = start

    # initialize merged with first real segment
    coords = total[curr_idx]['coords']
    seq = [tuple(pt) for pt in coords]
    if seq[0] != curr_pt:
        seq = seq[::-1]
    merged.extend(seq)
    used.add(curr_idx)
    curr_pt = merged[-1]

    # iteratively append connected segments
    while True:
        next_idxs = [i for i in ep_map[curr_pt] if i not in used]
        if not next_idxs:
            break
        idx_next = next_idxs[0]
        coords = total[idx_next]['coords']
        seq = [tuple(pt) for pt in coords]
        if seq[0] != curr_pt:
            seq = seq[::-1]
        merged.extend(seq[1:])
        used.add(idx_next)
        curr_pt = merged[-1]

    merged_arr = np.array(merged)
    if show:
        show_spline_gradient(binary, merged_arr,
                                      title="Full Merged Skeleton")

    # remove connection points
    merged_real = [pt for pt in merged if pt not in conn_set]

    # convert to 3D
    points_3d = []
    for y, x in merged_real:
        z = float(depth_image[y, x]) * depth_scale
        if z <= 0:
            continue
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        points_3d.append((X, Y, z))

    return np.array(points_3d, dtype=float)


def extract_centerline_from_mask_individual(
    depth_image: np.ndarray,
    mask: np.ndarray,
    camera_parameters: tuple,
    depth_scale: float = 1.0,
    connectivity: int = 8,
    min_length: int = 20,
    show: bool = False
    ) -> list:
    """
    Extracts individual 3D centerline segments (no connections).

    Returns:
        centerlines: list of (Ni,3) arrays of segment points in camera coords.
    """
    fx, fy, cx, cy = camera_parameters

    # 1. Binarise and skeletonize + excise + remove short segments
    _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    skeleton_bool = skeletonize(binary.astype(bool))
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bool.astype(np.uint8), -1, kernel)
    ints = np.argwhere((skeleton_bool)&(neighbor_count>=3))
    dist_map = distance_transform_edt(binary)
    to_remove = np.zeros_like(skeleton_bool, bool)
    H, W = skeleton_bool.shape
    for y,x in ints:
        r = dist_map[y, x]
        R = int(np.ceil(r))
        y0,y1 = max(0,y-R), min(H, y+R+1)
        x0,x1 = max(0,x-R), min(W, x+R+1)
        yy,xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy-y)**2 + (xx-x)**2 <= r**2
        to_remove[y0:y1, x0:x1][circle] = True
    skeleton_exc = skeleton_bool & ~to_remove
    num_labels, labels = cv2.connectedComponents(skeleton_exc.astype(np.uint8), connectivity)
    for lbl in range(1, num_labels):
        comp = (labels==lbl)
        if comp.sum() < min_length:
            skeleton_exc[comp] = False

    # 2. Extract each segment
    num_labels2, labels2 = cv2.connectedComponents(skeleton_exc.astype(np.uint8), connectivity)
    segments = []
    for lbl in range(1, num_labels2):
        comp_mask = (labels2 == lbl)
        pixels = [tuple(pt) for pt in np.argwhere(comp_mask)]
        nbrs = {p: [] for p in pixels}
        for y, x in pixels:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==dx==0: continue
                    q = (y+dy, x+dx)
                    if q in nbrs:
                        nbrs[(y,x)].append(q)
        ends = [p for p,n in nbrs.items() if len(n)==1]
        start = ends[0] if ends else pixels[0]
        ordered, prev, curr = [], None, start
        while True:
            ordered.append(curr)
            nxt = [n for n in nbrs[curr] if n!=prev]
            if not nxt: break
            prev, curr = curr, nxt[0]
        coords = np.array(ordered)
        if show:
            show_spline_gradient(binary, coords, title=f"Segment {lbl}")

        # 3. Convert to 3D and drop invalid
        pts3d = []
        for y,x in coords:
            z = float(depth_image[y,x]) * depth_scale
            if z <= 0:
                continue
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            pts3d.append((X, Y, z))
        segments.append(np.array(pts3d, dtype=float))

    return segments
#endregion