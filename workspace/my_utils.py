import cv2, numpy as np
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d
import rospy
import copy

from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, quaternion_slerp, quaternion_matrix, quaternion_from_matrix, quaternion_multiply, quaternion_inverse
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev, make_splprep
from scipy.spatial import cKDTree

from scipy.interpolate import BSpline

import networkx as nx

from skimage.morphology import skeletonize

from skimage.draw import line as skline

from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

import time


from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates


import scipy.optimize as opt

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

def transform_pointcloud_to_world(pointcloud: o3d.geometry.PointCloud, camera_pose: PoseStamped) -> o3d.geometry.PointCloud:
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
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # Extract translation and quaternion
    t = camera_pose.pose.position
    q = camera_pose.pose.orientation
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

def transform_points_to_world(points: np.ndarray, camera_pose: PoseStamped) -> np.ndarray:
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
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N,3) array")

    # Build camera→world homogeneous matrix
    t = camera_pose.pose.position
    q = camera_pose.pose.orientation
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
def project_pointcloud_from_world(pointcloud: o3d.geometry.PointCloud, camera_pose: PoseStamped, camera_parameters):
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
    else:
        raise TypeError("camera_pose must be a geometry_msgs.msg.PoseStamped")
    

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

def show_masks(masks, title="Masks") -> None:
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
        (255, 255, 255), # white
        (0, 0, 255),     # red
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (128, 0, 128),   # purple
        (0, 128, 128),   # teal
        (128, 128, 0),   # olive
    ]

    # Create an empty color canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Overlay each mask in its color
    for idx, mask in enumerate(mask_list):
        # If mask has extra dims (e.g. (1, H, W)), take the first channel
        if mask.ndim == 3:
            mask = mask[0]

        # Make a uint8 copy for resizing (so original mask stays untouched)
        mask_uint8 = mask.astype(np.uint8)

        # Resize the uint8 mask
        mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)

        # Convert back to a boolean mask
        mask_bool = mask_resized.astype(bool)

        color = colors[idx % len(colors)]
        canvas[mask_bool] = color

    # Display the combined masks
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 550)
    cv2.imshow(title, canvas)
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

def show_pointclouds(pointclouds: 'o3d.geometry.PointCloud', max_points: int = 200_000, voxel_size: float | None = None, axis_size: float = 0.2, title: str = 'Point Cloud') -> None:
    """Display a point cloud via *Open3D*'s built‑in viewer, showing axes.

    Parameters
    ----------
    pointcloud
        Cloud to display (camera‑frame coordinates).
    max_points
        Random down‑sample threshold for interactivity.
    voxel_size
        If given, voxel grid down‑sampling size **in metres**.
    axis_size
        Size of the origin coordinate frame (in metres).
    window_name
        Title of the visualiser window.
    """
    if not isinstance(pointclouds[0], o3d.geometry.PointCloud):
        raise TypeError("show_pointcloud expects an open3d.geometry.PointCloud")
    
    # Origin coordinate frame ---------------------------------------------
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])]

    # for pc in pointclouds:
    #     if voxel_size is not None and voxel_size > 0:
    #         pc = pc.voxel_down_sample(voxel_size)
    #     elif len(pc.points) > max_points:
    #         pc = pc.random_down_sample(max_points / len(pc.points))
    #     geometries.append(pc)

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

    # Viewer ---------------------------------------------------------------
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False)

def show_pointclouds_with_frames(pointclouds: list[o3d.geometry.PointCloud],
                                    frames: list[PoseStamped],
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
        if not isinstance(tf, PoseStamped):
            raise TypeError("Each frame must be a geometry_msgs.msg.TransformStamped")
        t = tf.pose.position
        q = tf.pose.orientation
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
                                            frames: list[PoseStamped] = None,
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
            t = tf.pose.position; q = tf.pose.orientation
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





# -----------------------------------------------------------------------------
# Helper geometry builders
# -----------------------------------------------------------------------------

def _cylinder_between(p0: np.ndarray, p1: np.ndarray,
                      radius: float = 0.02, resolution: int = 20) -> o3d.geometry.TriangleMesh | None:
    """Build a cylinder mesh whose axis runs p0 → p1."""
    v = p1 - p0
    h = np.linalg.norm(v)
    if h <= 1e-9:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, h, resolution)
    cyl.compute_vertex_normals()

    # orient +Z of the cylinder to match v
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    angle = np.arccos(np.dot(z, v) / h)
    if np.linalg.norm(axis) > 1e-8:
        R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
        cyl.rotate(R_align, center=np.zeros(3))

    cyl.translate(p0 + 0.5 * v)
    return cyl


def _sphere_at(p: np.ndarray, radius: float = 0.05, resolution: int = 10) -> o3d.geometry.TriangleMesh:
    """Return a sphere mesh centred at *p*."""
    sph = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
    sph.compute_vertex_normals()
    sph.translate(p)
    return sph


# -----------------------------------------------------------------------------
# Main visualiser
# -----------------------------------------------------------------------------

def show_bspline_with_frames_and_grid(
    spline: BSpline,
    frames: list | None = None,
    *,
    num_samples: int = 200,
    curve_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    # Rendering style ----------------------------------------------------------
    tube_radius: float | None = None,
    show_dots: bool = True,
    # dot_radius: float = 0.005,
    line_width_px: int = 3,
    # Scene settings -----------------------------------------------------------
    grid_size: float = 5.0,
    grid_step: float = 0.5,
    show_grid: bool = True,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    # Window -------------------------------------------------------------------
    title: str = "B-spline & Frames (Z-up)"
) -> None:
    """Visualise a 3‑D **BSpline** curve with many display options.

    Parameters
    ----------
    spline : BSpline
        The 3‑D vector spline to render.
    frames : list | None
        Optional camera / robot poses (PoseStamped‑like objects).
    num_samples : int, default 200
        Number of points sampled uniformly along the spline.
    curve_color : (R, G, B)
        RGB colour in [0,1] for curve + dots.

    Rendering style
    ~~~~~~~~~~~~~~~
    tube_radius : float | None, default None
        If set, draw the curve as cylinders of this radius.
    show_dots : bool, default False
        Show spheres at each sampled point. Takes precedence over *tube_radius*.
    dot_radius : float, default 0.05
        Sphere radius when *show_dots* is True.
    line_width_px : int, default 3
        Pixel width for *LineSet* when neither *show_dots* nor *tube_radius*.

    Scene settings
    ~~~~~~~~~~~~~~
    grid_size, grid_step : float
        Size (half‑extent) and spacing of ground grid.
    show_grid : bool, default True
        Toggle the ground‑plane grid.
    background_color : (R, G, B), default black
        OpenGL clear colour.

    Window
    ~~~~~~
    title : str
        Window title.
    """

    # ---- sample the spline ----
    t_min, t_max = spline.t[spline.k], spline.t[-spline.k - 1]
    ts = np.linspace(t_min, t_max, num_samples)
    pts = np.asarray(spline(ts), dtype=np.float64, order="C")
    if pts.shape[0] == 3 and pts.shape[1] != 3:  # transpose if 3×N
        pts = pts.T
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Spline must evaluate to (N,3) points, got {pts.shape}")

    geoms: list[o3d.geometry.Geometry] = []

    # world origin frame
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    # optional frames
    if frames:
        for tf in frames:
            t = tf.pose.position
            q = tf.pose.orientation  # x, y, z, w
            T = np.eye(4)
            T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = (t.x, t.y, t.z)
            fmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            fmesh.transform(T)
            geoms.append(fmesh)

    # curve representation (dots > tube > line)
    if show_dots:
        for p in pts:
            sph = _sphere_at(p, radius=dot_radius)
            sph.paint_uniform_color(curve_color)
            geoms.append(sph)
    elif tube_radius is not None:
        for i in range(len(pts) - 1):
            cyl = _cylinder_between(pts[i], pts[i + 1], radius=tube_radius)
            if cyl is not None:
                cyl.paint_uniform_color(curve_color)
                geoms.append(cyl)
    else:
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        curve_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        curve_ls.colors = o3d.utility.Vector3dVector([curve_color] * len(lines))
        geoms.append(curve_ls)

    # ground grid
    if show_grid:
        g_pts, g_lines = [], []
        idx = 0
        for x in np.arange(-grid_size, grid_size + 1e-6, grid_step):
            g_pts += [[x, -grid_size, 0], [x, grid_size, 0]]
            g_lines += [[idx, idx + 1]]; idx += 2
        for y in np.arange(-grid_size, grid_size + 1e-6, grid_step):
            g_pts += [[-grid_size, y, 0], [grid_size, y, 0]]
            g_lines += [[idx, idx + 1]]; idx += 2
        grid_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(g_pts),
            lines=o3d.utility.Vector2iVector(g_lines),
        )
        grid_ls.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(g_lines))
        geoms.append(grid_ls)

    # ---- Open3D visualiser ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.line_width = float(line_width_px)
    opt.background_color = np.asarray(background_color, dtype=float)

    ctr = vis.get_view_control()
    ctr.set_front((0, -0.5, -0.5))
    ctr.set_lookat((0, 0, 0))
    ctr.set_up((0, 0, 1))

    while vis.poll_events():
        ctr.set_up((0, 0, 1))  # re‑enforce Z‑up
        vis.update_renderer()

    vis.destroy_window()





def _process_single(mask: np.ndarray) -> np.ndarray:
    """Compute distance transform of a single mask's skeleton."""
    if mask.ndim != 2:
        raise ValueError("Each mask must be 2‑D (H×W)")

    # Boolean mask – non‑zero / True = foreground
    sk = skeletonize(mask.astype(bool))
    dt = distance_transform_edt(~sk)  # distance to nearest skeleton pixel
    return dt.astype(np.float32)


def skeleton_distance_transform(masks):
    """Distance‑transform of **mask skeletons**.

    Parameters
    ----------
    masks : ndarray or sequence of ndarray
        * **Single H×W array** → returns one H×W float array.
        * **(N, H, W) array**  → returns list of N H×W arrays.
        * **list/tuple of arrays** → returns list of same length.

    Returns
    -------
    ndarray or list[ndarray]
        The distance‑transform(s) (dtype `float32`).  Shape mirrors the input
        pattern so you can drop the result straight into `show_masks()`.
    """
    # normalise input to list of 2‑D arrays
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return _process_single(masks)
        elif masks.ndim == 3:
            return [_process_single(masks[i]) for i in range(masks.shape[0])]
        else:
            raise ValueError("masks array must have 2 or 3 dimensions")
    elif isinstance(masks, (list, tuple)):
        return [_process_single(m) for m in masks]
    else:
        raise TypeError("masks must be a numpy array or a list/tuple of arrays")

from matplotlib import cm       # ← add this near the other imports

def dt_to_color(dt: np.ndarray,
                *,
                cmap: str = 'viridis',
                normalize: bool = True,
                bgr: bool = True) -> np.ndarray:
    """Map a float **distance-transform** to a false-colour `uint8` image.

    Parameters
    ----------
    dt : ndarray (H×W)
        Input scalar field (float).
    cmap : str, default 'viridis'
        Any Matplotlib colormap name.
    normalize : bool, default True
        If *True*, linearly rescale `dt` so min → 0 and max → 1 before
        applying the colormap.  If *False*, values are assumed already in
        [0, 1].
    bgr : bool, default True
        Return image in BGR order (OpenCV convention).  Set to *False* for RGB.

    Returns
    -------
    ndarray (H×W×3, uint8)
        Colour image ready for `cv2.imshow` / `cv2.imwrite`.
    """
    if dt.ndim != 2:
        raise ValueError('dt must be 2-D')

    data = dt.astype(np.float32, copy=False)
    if normalize:
        vmin, vmax = data.min(), data.max()
        rng = vmax - vmin if vmax > vmin else 1.0
        data = (data - vmin) / rng
        data = np.clip(data, 0.0, 1.0)

    cmap_func = cm.get_cmap(cmap)
    rgba = cmap_func(data, bytes=True)          # H×W×4 uint8 RGBA
    rgb = rgba[..., :3]                        # drop alpha

    if bgr:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return rgb


def interactive_scale_shift(depth1: np.ndarray,
                            mask2: np.ndarray,
                            pose1: TransformStamped,
                            pose2: TransformStamped,
                            camera_parameters: tuple[float, float, float, float],
                            scale_limits: tuple[float, float] = (0.1, 0.8),
                            shift_limits: tuple[float, float] = (-2.0, 2.0)
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
        pc1 = convert_depth_map_to_pointcloud(d1, camera_parameters)
        pc1_world = transform_pointcloud_to_world(pc1, pose1)
        # 3) project into cam2
        _, reproj_mask = project_pointcloud_from_world(pc1_world, pose2, camera_parameters)
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
    new_pose.header.stamp = rospy.Time.now()
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
  

def get_desired_pose(position, base_footprint, frame: str = "map") -> PoseStamped:
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
    # 1) look up the current base_footprint → frame pose as a PoseStamped
    # base_ps = ros_handler.get_current_pose(base_footprint, frame)
    # q = base_ps.pose.orientation
    rotation = base_footprint.pose.orientation
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


# def visualize_spline_with_pc(pointcloud, control_pts, degree, num_samples=200, title="Spline with PointCloud"):
#     n_ctrl = len(control_pts)
#     # Build a clamped, uniform knot vector of length n_ctrl + degree + 1
#     # interior knot count = n_ctrl - degree - 1
#     if n_ctrl <= degree:
#         raise ValueError("Need more control points than the degree")
#     m = n_ctrl - degree - 1
#     if m > 0:
#         interior = np.linspace(0, 1, m+2)[1:-1]
#     else:
#         interior = np.array([])
#     # clamp start/end
#     t = np.concatenate([
#         np.zeros(degree+1),
#         interior,
#         np.ones(degree+1)
#     ])
#     # create BSpline
#     spline = BSpline(t, control_pts, degree)

#     # sample
#     ts = np.linspace(t[degree], t[-degree-1], num_samples)
#     curve_pts = spline(ts)

#     # build LineSet
#     import open3d as o3d
#     lines = [[i, i+1] for i in range(len(curve_pts)-1)]
#     ls = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(curve_pts),
#         lines=o3d.utility.Vector2iVector(lines)
#     )
#     ls.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])

#     o3d.visualization.draw_geometries([pointcloud, ls])


def visualize_spline_with_pc(pointcloud: o3d.geometry.PointCloud, spline: BSpline, num_samples: int = 200, title: str = "Spline with PointCloud"):
    """
    Visualize a 3D BSpline alongside an existing Open3D PointCloud.

    Args:
        pointcloud:   An Open3D PointCloud to display.
        spline:       A SciPy.interpolate.BSpline of degree k with 3D coeffs.
        num_samples:  How many points to sample along the spline.
        title:        (Unused) window title placeholder.
    """
    # Extract degree and knot vector
    k = spline.k
    t = spline.t

    # Determine valid parameter interval [t[k], t[-k-1]]
    u_start, u_end = t[k], t[-k-1]
    us = np.linspace(u_start, u_end, num_samples)

    # Evaluate spline: shape (samples, 3) or (3, samples)
    pts = spline(us)
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] == num_samples:
        pts = pts.T

    # Build LineSet for the curve
    lines = [[i, i+1] for i in range(len(pts)-1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(lines))

    # Display both
    if pointcloud is None:
        o3d.visualization.draw_geometries([ls])
    else:
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

def fit_bspline_scipy(
        centerline_pts: np.ndarray,
        degree: int = 3,
        smooth: float | None = None,
        nest: int | None = None,
        num_ctrl: int | None = None,
    ) -> BSpline:
    """
    Fit a 3-D B-spline; if `num_ctrl` is set, force that many control points.
    Returns the BSpline object (coefficients in .c).
    """
    # chord-length parameterisation
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centerline_pts, axis=0), axis=1))]
    u /= u[-1]

    if num_ctrl is not None:
        k = degree
        # how many internal knots do we need?
        n_internal = num_ctrl - (k + 1)  # because of 2*(k+1) clamped end knots
        if n_internal < 0:
            raise ValueError("num_ctrl must exceed degree+1")
        # uniform internal knot vector
        t_internal = np.linspace(u[k+1], u[-k-2], n_internal, endpoint=True)
        # build full knot vector (clamped)
        t_full = np.r_[np.repeat(u[0], k + 1),
                    t_internal,
                    np.repeat(u[-1], k + 1)]
        # small positive s lets FITPACK adjust within fixed knots
        s = 0.0 if smooth is None else smooth
        spline, _ = make_splprep(centerline_pts.T, k=k, s=s, t=t_full, u=u)
    else:
        s = 0.0 if smooth is None else smooth
        spline, _ = make_splprep(centerline_pts.T, k=degree, s=s, nest=nest)

    return spline        # control points are spline.c.shape == (num_ctrl, 3)

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

def project_bspline(spline: BSpline, camera_pose: PoseStamped, camera_parameters: tuple, width: int = 640, height: int = 480, num_samples: int = 50) -> np.ndarray:
    """
    Projects a 3D BSpline into the camera image plane,
    rendering a continuous curve into a binary mask.

    Args:
        spline:            A SciPy.interpolate.BSpline instance whose
                           coefficients describe a 3D curve.
        camera_pose:       A PoseStamped with .pose.position (x,y,z)
                           and .pose.orientation (x,y,z,w) defining
                           camera pose in world.
        camera_parameters: (fx, fy, cx, cy) intrinsic parameters.
        width:             Image width in pixels.
        height:            Image height in pixels.
        num_samples:       Number of points to sample; if None, uses max(width, height).

    Returns:
        mask: (height × width) uint8 mask with the projected spline drawn in 255 on 0.
    """
    fx, fy, cx, cy = camera_parameters

    # 1) Sample along the spline’s natural domain
    k = spline.k
    t = spline.t
    u0, u1 = t[k], t[-k-1]
    N = num_samples
    u = np.linspace(u0, u1, N)
    pts = spline(u)  # could be (N,3) or (3,N)
    if pts.ndim == 2 and pts.shape[0] == 3:
        pts = pts.T  # -> (N,3)

    # 2) Parse camera pose
    tx = camera_pose.pose.position.x
    ty = camera_pose.pose.position.y
    tz = camera_pose.pose.position.z
    qx = camera_pose.pose.orientation.x
    qy = camera_pose.pose.orientation.y
    qz = camera_pose.pose.orientation.z
    qw = camera_pose.pose.orientation.w

    # 3) Build rotation matrix R (camera → world)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [  2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)],
        [  2*(xz - wy),   2*(yz + wx), 1-2*(xx+yy)]
    ])

    # 4) Transform world pts into camera frame
    diff = pts - np.array([tx, ty, tz])
    pts_cam = diff.dot(R)  # since we want R^T*(p - t), and R here is camera→world

    # 5) Project with pinhole model
    x, y, z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    valid = z > 0
    u_proj = fx * x[valid] / z[valid] + cx
    v_proj = fy * y[valid] / z[valid] + cy
    u_pix = np.round(u_proj).astype(int)
    v_pix = np.round(v_proj).astype(int)

    # 6) Draw into mask
    mask = np.zeros((height, width), dtype=np.uint8)
    pts2d = [
        (u, v) for u, v in zip(u_pix, v_pix)
        if 0 <= u < width and 0 <= v < height
    ]
    for (u0, v0), (u1, v1) in zip(pts2d, pts2d[1:]):
        rr, cc = skline(v0, u0, v1, u1)
        inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        mask[rr[inside], cc[inside]] = 255

    return mask

from skimage.draw import disk          # optional if you want >1-px markers

def project_bspline_points(
    spline: BSpline,
    camera_pose: PoseStamped,
    camera_parameters: tuple,
    width: int = 640,
    height: int = 480,
    num_samples: int = 50,
    radius_px: int = 0,                # 0 → single pixel, >0 → filled disk
) -> np.ndarray:
    """
    Same signature and return type as the original function but
    only the sampled points are drawn – no interpolation between them.
    """
    fx, fy, cx, cy = camera_parameters

    # ------------------------------------------------------------------ #
    # 1) sample the spline
    k = spline.k
    t = spline.t
    u0, u1 = t[k], t[-k - 1]
    u = np.linspace(u0, u1, num_samples)
    pts = spline(u)
    if pts.ndim == 2 and pts.shape[0] == 3:
        pts = pts.T                                             # (N, 3)
    # ------------------------------------------------------------------ #
    # 2) camera pose → rotation R and translation t
    tx, ty, tz = (camera_pose.pose.position.x,
                  camera_pose.pose.position.y,
                  camera_pose.pose.position.z)
    qx, qy, qz, qw = (camera_pose.pose.orientation.x,
                      camera_pose.pose.orientation.y,
                      camera_pose.pose.orientation.z,
                      camera_pose.pose.orientation.w)

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz),   2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),     1 - 2*(xx + yy)]
    ])
    # ------------------------------------------------------------------ #
    # 3) world → camera, then pinhole projection
    pts_cam = (pts - np.array([tx, ty, tz])) @ R            # Rᵀ (p - t)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = z > 0
    u_pix = np.round(fx * x[valid] / z[valid] + cx).astype(int)
    v_pix = np.round(fy * y[valid] / z[valid] + cy).astype(int)

    # ------------------------------------------------------------------ #
    # 4) rasterise *only* the sampled pixels
    mask = np.zeros((height, width), dtype=np.uint8)
    in_bounds = (0 <= u_pix) & (u_pix < width) & (0 <= v_pix) & (v_pix < height)
    u_pix, v_pix = u_pix[in_bounds], v_pix[in_bounds]

    if radius_px == 0:
        mask[v_pix, u_pix] = 255
    else:
        for u_, v_ in zip(u_pix, v_pix):
            rr, cc = disk((v_, u_), radius_px, shape=mask.shape)
            mask[rr, cc] = 255

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
    tx = camera_pose.pose.position.x
    ty = camera_pose.pose.position.y
    tz = camera_pose.pose.position.z
    qx = camera_pose.pose.orientation.x
    qy = camera_pose.pose.orientation.y
    qz = camera_pose.pose.orientation.z
    qw = camera_pose.pose.orientation.w

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
    # pts_cam = diff.dot(R.T)
    pts_cam = diff.dot(R)

    # Perspective projection
    x_cam, y_cam, z_cam = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    # compute pts_cam, do projection, but **don’t** round or clip:
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    # keep only those with z>0 and inside a slightly padded window
    valid = (z_cam>0) & (u>=-1) & (u<width+1) & (v>=-1) & (v<height+1)
    return np.stack([u[valid], v[valid]], axis=1)  # shape (M,2)

def project_bspline_pts_new(spline, camera_pose, camera_parameters, degree=3, num_samples=200,  width=640, height=480):
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
    tx = camera_pose.pose.position.x
    ty = camera_pose.pose.position.y
    tz = camera_pose.pose.position.z
    qx = camera_pose.pose.orientation.x
    qy = camera_pose.pose.orientation.y
    qz = camera_pose.pose.orientation.z
    qw = camera_pose.pose.orientation.w

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
    # pts_cam = diff.dot(R.T)
    pts_cam = diff.dot(R)

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

def make_bspline_bounds(spline: np.ndarray, delta: float = 0.1):
    """
    Given an (N×3) array of B-spline control points, return a 
    bounds list of length 3N for differential_evolution, where
    each coordinate is allowed to vary ±delta around its original value.
    """
    ctrl_points = spline.c
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

def precompute_skeletons_and_interps(masks):
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
    for mask in masks:
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


def score_function_bspline_reg_multiple_pre(x, camera_poses: PoseStamped, camera_parameters, degree, init_x, reg_weight: float, decay: float, curvature_weight: float, skeletons: list, interps: list, num_samples: int):
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
        # project to subpixel 2D
        pts2d = project_bspline_pts(
            ctrl_pts,
            cam_pose,
            camera_parameters,
            degree=degree,
            num_samples=num_samples
        )  # shape (M,2) floats

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

def score_bspline_translation(shift_xy: np.ndarray, mask: np.ndarray, camera_pose: PoseStamped, camera_parameters: tuple, degree: int, spline: BSpline, num_samples: int = 200) -> float:
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
    q = camera_pose.pose.orientation
    tx, ty, tz = (camera_pose.pose.position.x,
                  camera_pose.pose.position.y,
                  camera_pose.pose.position.z)
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
    ctrl_points = spline.c
    ctrl_pts_shifted = np.asarray(ctrl_points, dtype=float) + shift_world

    # 2) get mask and skeleton
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

def apply_translation_to_ctrl_points(spline: BSpline, shift_xy: np.ndarray, camera_pose: PoseStamped) -> np.ndarray:
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
    tx, ty, tz = (camera_pose.pose.position.x,
                  camera_pose.pose.position.y,
                  camera_pose.pose.position.z)
    qx = camera_pose.pose.orientation.x
    qy = camera_pose.pose.orientation.y
    qz = camera_pose.pose.orientation.z
    qw = camera_pose.pose.orientation.w

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
    ctrl_points = spline.c
    shifted_ctrl_points = np.asarray(ctrl_points, dtype=float) + shift_world[np.newaxis, :]

    spline.c = shifted_ctrl_points

    return spline

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

def get_highest_point_and_angle_spline(spline: BSpline,
                                       num_samples: int = 1000):
    """
    Samples the given 3D BSpline densely, finds the 3D point of maximum z,
    and computes the tangent angle at that point (projection onto XY-plane),
    via arctan2(dy,dx). Raises if something unexpected shows up.
    """
    # --- 1) domain of definition
    k = spline.k
    t = spline.t
    u_start = t[k]
    u_end   = t[-(k+1)]
    assert u_end > u_start, f"Invalid param range: [{u_start}, {u_end}]"

    # --- 2) sample
    u = np.linspace(u_start, u_end, num_samples)
    pts = spline(u)
    pts = np.atleast_2d(pts)
    # now expect shape (num_samples, D)
    if pts.shape[0] != num_samples:
        if pts.shape[1] == num_samples:
            pts = pts.T
        else:
            raise ValueError(f"BSpline(u) gave shape {pts.shape}, "
                             "couldn't reorient to (N, D).")
    N, D = pts.shape
    if D < 3:
        raise ValueError(f"Spline output has dimension {D}<3; need 3D points.")

    # --- 3) find highest‐z sample
    idx_max = np.argmax(pts[:, 2])
    highest_pt = pts[idx_max]  # (x,y,z)

    # --- 4) derivative & tangent
    dspline = spline.derivative(nu=1)
    d_pts = dspline(u)
    d_pts = np.atleast_2d(d_pts)
    if d_pts.shape[0] != num_samples:
        if d_pts.shape[1] == num_samples:
            d_pts = d_pts.T
        else:
            raise ValueError(f"Derivative gave shape {d_pts.shape}.")
    dx, dy = d_pts[idx_max, 0], d_pts[idx_max, 1]

    # --- 5) angle via atan2
    angle = np.arctan2(dy, dx)
    angle += np.pi/2
    # normalize to [-pi/2, pi/2]
    while angle > np.pi/2:
        angle -= np.pi
    while angle < -np.pi/2:
        angle += np.pi

    # if you really need to clamp to [-pi/2, pi/2], you can do:
    # angle = np.clip(angle, -np.pi/2, np.pi/2)

    return highest_pt, angle


# Example usage:
# ctrl_pts = np.random.rand(10, 3)
# pt, ang = get_highest_point_and_tangent_angle(ctrl_pts)
# print("Highest:", pt, "Angle (rad):", ang)



# def convert_bspline_to_pointcloud(ctrl_points: np.ndarray, samples: int = 150, degree: int = 3) -> o3d.geometry.PointCloud:
#     """
#     Converts a B-spline defined by control points into an Open3D PointCloud
#     by sampling points along the curve.

#     Args:
#         ctrl_points: (N_control × 3) array of control points.
#         samples:     Number of points to sample along the spline (integer ≥ 2).
#         degree:      Spline degree (k). Must satisfy N_control > degree.

#     Returns:
#         pcd: Open3D.geometry.PointCloud containing `samples` points sampled
#              along the B-spline.
#     """
#     pts = np.asarray(ctrl_points, dtype=float)
#     n_ctrl = pts.shape[0]
#     k = degree
#     if n_ctrl <= k:
#         raise ValueError("Number of control points must exceed spline degree")

#     # Open-uniform knot vector: (k+1) zeros, inner knots, (k+1) ones
#     n_inner = n_ctrl - k - 1
#     if n_inner > 0:
#         inner = np.linspace(0, 1, n_inner + 2)[1:-1]
#         knots = np.concatenate((np.zeros(k+1), inner, np.ones(k+1)))
#     else:
#         knots = np.concatenate((np.zeros(k+1), np.ones(k+1)))

#     # Build the spline and sample
#     spline = BSpline(knots, pts, k, axis=0)
#     u = np.linspace(0, 1, samples)
#     samples_3d = spline(u)  # shape: (samples, 3)

#     # Create Open3D PointCloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(samples_3d)

#     return pcd

def convert_bspline_to_pointcloud(spline: BSpline, samples: int = 1000) -> o3d.geometry.PointCloud:
    """
    Converts a SciPy BSpline into an Open3D PointCloud by sampling points along the curve.

    Args:
        spline:   A SciPy.interpolate.BSpline instance whose coefficients describe
                  a 3D curve (e.g., built via splprep + BSpline).
        samples:  Number of points to sample along the spline (integer ≥ 2).

    Returns:
        pcd:      Open3D.geometry.PointCloud containing `samples` XYZ points.
    """
    # Extract the true domain of the spline:
    k = spline.k
    t = spline.t
    u_start, u_end = t[k], t[-k-1]

    # Uniform parameter values over [u_start, u_end]
    u = np.linspace(u_start, u_end, samples)

    # Evaluate — this yields an array of shape (samples, 3) if your spline.c
    # was shaped appropriately (axis=0 for the spatial dimension).
    pts = spline(u)

    # Some BSpline constructions return (3, samples), so transpose if needed:
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] == samples:
        pts = pts.T

    # Build the Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return pcd


#endregion


#region new spline initialization
def extract_2d_spline(mask: np.ndarray,
                      connectivity: int = 8,
                      alpha: float = 5.0) -> np.ndarray:
    """
    Extract a continuous 2D centerline from a binary DLO mask—even through overlaps—
    by skeletonizing, measuring local mask thickness, and taking the shortest path
    between the two endpoints in a graph where edge‐costs rise in thicker regions.

    Args:
        mask:        H×W binary (0/1 or bool) mask of the DLO.
        connectivity:4 or 8-connected for the skeleton graph.
        alpha:       thickness penalty factor (>0). Larger α avoids overlaps harder.

    Returns:
        centerline:  (N×2) array of (row, col) pixel coordinates along the cable.
    """
    # 1) Skeletonize to 1-px centerline
    sk = skeletonize(mask > 0)
    rows, cols = np.where(sk)
    coords = list(zip(rows, cols))
    if not coords:
        return np.zeros((0,2), int)

    # 2) Precompute mask thickness (distance to background)
    thick = distance_transform_edt(mask > 0)
    max_th = thick.max() if thick.max()>0 else 1.0

    # 3) Build graph over skeleton pixels
    idx = {pt:i for i, pt in enumerate(coords)}
    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neigh = [(-1,0),(0,-1),(0,1),(1,0)]

    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    for i, (r,c) in enumerate(coords):
        for dr,dc in neigh:
            j = idx.get((r+dr, c+dc))
            if j is None: 
                continue
            # base length
            d = np.hypot(dr, dc)
            # thickness penalty: average of the two nodes
            t_avg = (thick[r,c] + thick[coords[j][0], coords[j][1]]) / 2.0
            w = d * (1.0 + alpha * (t_avg / max_th))
            G.add_edge(i, j, weight=w)

    # 4) Identify the two cable endpoints (degree==1)
    deg = dict(G.degree())
    ends = [n for n,d in deg.items() if d==1]
    if len(ends) < 2:
        # fallback: pick the two farthest nodes in unweighted graph
        dists = dict(nx.all_pairs_shortest_path_length(G))
        a,b,maxd = 0,0,0
        for i in deg:
            for j in deg:
                if dists[i][j] > maxd:
                    a, b, maxd = i, j, dists[i][j]
        ends = [a, b]
    start, goal = ends[:2]

    # 5) Compute the weighted shortest path
    path = nx.shortest_path(G, source=start, target=goal, weight='weight')

    # 6) Map back to pixel coords
    centerline = np.array([coords[i] for i in path], dtype=int)
    return centerline


def display_2d_spline_gradient(mask: np.ndarray,
                               centerline: np.ndarray,
                               window: str = "Centerline"):
    """
    Overlay the 2D centerline on the mask with a blue→red gradient.

    Args:
        mask:        H×W binary mask.
        centerline:  (N×2) array of (row, col) coordinates.
        window:      OpenCV window name.
    """
    img = (mask.astype(np.uint8)*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    N = len(centerline)
    for i in range(N-1):
        r0, c0 = centerline[i]
        r1, c1 = centerline[i+1]
        t = i / (N-1)
        color = (int(255*(1-t)), 0, int(255*t))  # B→R gradient
        cv2.line(img, (c0, r0), (c1, r1), color, 2)
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window)



#endregion












#region additional after branching
def create_bspline(ctrl_points: np.ndarray, degree: int = 3) -> o3d.geometry.PointCloud:
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

    return spline


def convert_transform_stamped_to_pose_stamped(transform: 'TransformStamped') -> 'PoseStamped':
    """
    Convert a geometry_msgs/TransformStamped into a geometry_msgs/PoseStamped.

    Args:
        transform (TransformStamped): The incoming transform message.

    Returns:
        PoseStamped: A pose message with the same header, translation, and rotation.
    """
    pose = PoseStamped()

    # Copy the header verbatim (frame_id, stamp, etc.)
    pose.header = transform.header

    # Map translation → position
    pose.pose.position.x = transform.transform.translation.x
    pose.pose.position.y = transform.transform.translation.y
    pose.pose.position.z = transform.transform.translation.z

    # Map rotation (already a quaternion) → orientation
    pose.pose.orientation = transform.transform.rotation

    return pose



def get_midpoint_and_angle_spline(spline: BSpline):
    """
    Evaluates the given 3D BSpline at the midpoint of its parameter domain,
    and computes the tangent angle at that point (projection onto the XY-plane)
    via arctan2(dy, dx). Raises if something unexpected shows up.
    """
    # --- 1) domain of definition
    k = spline.k
    t = spline.t
    u_start = t[k]
    u_end   = t[-(k+1)]
    if u_end <= u_start:
        raise ValueError(f"Invalid parameter range: [{u_start}, {u_end}]")

    # --- 2) midpoint parameter
    u_mid = (u_start + u_end) / 2.0

    # --- 3) evaluate spline at midpoint
    pt = spline(u_mid)
    pt = np.atleast_1d(pt)
    if pt.ndim != 1:
        raise ValueError(f"BSpline({u_mid}) returned shape {pt.shape}; expected 1D array")
    D = pt.shape[0]
    if D < 3:
        raise ValueError(f"Spline output has dimension {D}<3; need 3D point.")

    # --- 4) derivative & tangent
    dspline = spline.derivative(nu=1)
    d_pt = dspline(u_mid)
    d_pt = np.atleast_1d(d_pt)
    if d_pt.ndim != 1 or d_pt.shape[0] < 2:
        raise ValueError(f"Derivative at {u_mid} returned shape {d_pt.shape}; expected >=2D.")
    dx, dy = d_pt[0], d_pt[1]

    # --- 5) angle via atan2
    # angle = np.arctan2(dy, dx)
    angle = np.arctan2(dx, dy)
    # angle += np.pi/2
    # normalize to [-pi/2, pi/2]
    while angle > np.pi/2:
        angle -= np.pi
    while angle < -np.pi/2:
        angle += np.pi

    return pt, angle

def optimize_bspline_custom(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        num_samples=200,
        smoothness_weight=1e-4,
        symmetric=True,
        translate=False,
        verbose=2,
    ):
    # Initial control points
    ctrl_pts_init = initial_spline.c.copy()
    ctrl_pts_init_flat = ctrl_pts_init.reshape(-1)
    degree = initial_spline.k
    knot_vector = initial_spline.t.copy()
    dim = ctrl_pts_init.shape[1]
    num_ctrl = ctrl_pts_init.shape[0]

    def make_bspline(ctrl_points_flat):
        ctrl_points = ctrl_points_flat.reshape(num_ctrl, dim)
        return BSpline(knot_vector, ctrl_points, degree)

    # Precompute skeletons, distance transforms, and coordinates
    dts = []
    skeleton_coords = []
    for mask in masks:
        sk = skeletonize(mask > 0)
        coords = np.vstack(np.nonzero(sk)).T  # (v,u)
        skeleton_coords.append((sk, coords))
        dts.append(distance_transform_edt(~sk))

    # Set initial optimization parameters
    initial_params = np.zeros(dim) if translate else ctrl_pts_init_flat.copy()

    def objective(params):
        # Compute effective control points
        if translate:
            translation = params
            ctrl_flat_eff = ctrl_pts_init_flat + np.tile(translation, num_ctrl)
        else:
            ctrl_flat_eff = params

        # Build spline and sample
        spline = make_bspline(ctrl_flat_eff)
        sampled_pts_3d = sample_bspline(spline, num_samples)

        residuals = []
        M = len(dts)
        for i, ((sk, coords), dt, cam_pose) in enumerate(zip(skeleton_coords, dts, camera_poses)):
            proj = project_3d_points(sampled_pts_3d, cam_pose, camera_parameters)
            u, v = proj[:,0], proj[:,1]
            w = decay ** (M - 1 - i)

            # Forward Chamfer: curve → skeleton
            img_coords = np.vstack((v, u))
            d_to_skel = map_coordinates(dt, img_coords, order=1, mode='nearest')
            residuals.extend(np.sqrt(w) * d_to_skel)

            # Backward Chamfer: skeleton → curve
            if symmetric:
                pts2d = np.vstack((v, u)).T
                tree = cKDTree(pts2d)
                d_to_curve, _ = tree.query(coords)
                residuals.extend(np.sqrt(w) * d_to_curve)

        # Smoothness only if shape can change
        if not translate:
            residuals.extend(curvature_residuals(spline, smoothness_weight, num_samples))

        return residuals

    # Print initial state
    init_res = objective(initial_params)
    print(f"initial_residuals: {init_res}")

    # Setup bounds for non-translate mode
    if not translate:
        bounds = make_bspline_bounds_new(ctrl_pts_init_flat, delta=0.1)
        print(f"initial control points: {ctrl_pts_init_flat}")
        print(f"bounds: {bounds}")

    # Solver configuration
    # solver_opts = dict(verbose=verbose,
    #                    xtol=1e-12,
    #                    ftol=1e-10,
    #                    gtol=1e-10,
    #                    max_nfev=1_000_000,
    #                    x_scale='jac')
    # method = 'dogbox' # 'trf'

    # Run optimization
    start = time.perf_counter()
    if translate:
        # result = least_squares(objective, initial_params,
        #                        method=method,**solver_opts)
        result = least_squares(
            fun              = objective,    # your function
            x0               = initial_params,    # shape (n,)
            method           = "lm", # "dogbox", # "trf",
            jac              = "3-point",            # or supply analytic/complex-step
            # bounds           = bounds,             # e.g. physical limits on control pts
            # loss             = "soft_l1",            # robust to skeleton gaps/outliers
            f_scale          = np.median(init_res),  # soft inlier threshold
            x_scale          = "jac",                # auto-scaling
            ftol             = 1e-6,
            xtol             = 1e-4,
            gtol             = 1e-8,
            diff_step        = None,                 # let SciPy pick
            max_nfev         = 1_000_000,                 # give it room
            verbose          = 2,
        )

    else:
        # result = least_squares(objective, initial_params,
        #                        bounds=bounds,
        #                        method=method, **solver_opts)
        result = least_squares(
            fun              = objective,    # your function
            x0               = initial_params,    # shape (n,)
            method           = "lm", # "dogbox", # "trf",
            jac              = "3-point",            # or supply analytic/complex-step
            # bounds           = bounds,             # e.g. physical limits on control pts
            # loss             = "soft_l1",            # robust to skeleton gaps/outliers
            f_scale          = np.median(init_res),  # soft inlier threshold
            x_scale          = "jac",                # auto-scaling
            ftol             = 1e-6,
            xtol             = 1e-4,
            gtol             = 1e-8,
            diff_step        = None,                 # let SciPy pick
            max_nfev         = 1_000_000,                 # give it room
            verbose          = 2,
        )

    end = time.perf_counter()
    print(f"B-spline optimization took {end - start:.2f} seconds")

    # Report final residuals
    opt_res = objective(result.x)
    print(f"optimal_residuals: {opt_res}")
    print(f"result: {result}")

    # Construct optimal spline
    if translate:
        best_flat = ctrl_pts_init_flat + np.tile(result.x, num_ctrl)
    else:
        best_flat = result.x
    optimal_spline = make_bspline(best_flat)

    # Visualization per view
    for idx, ((sk, _), cam_pose) in enumerate(zip(skeleton_coords, camera_poses)):
        # pts3d = sample_bspline(optimal_spline, num_samples)
        # proj3d = project_3d_points(pts3d, cam_pose, camera_parameters)
        # show_2d_points_with_mask(proj3d, sk, title=f"Optimal Spline on Skeleton {idx}")

        projected_spline = project_bspline(optimal_spline, camera_poses[-1], camera_parameters)
        show_masks([sk, projected_spline], title=f"Optimal Spline on Skeleton {idx}")

    return optimal_spline

def make_bspline_bounds_new(ctrl_points: np.ndarray, delta: float = 0.1):
    """
    Generate lower and upper bounds for B-spline control points.

    Parameters
    ----------
    ctrl_points : np.ndarray
        Initial control points, either shape (n_params,) or (n_points, dims).
    delta : float
        Maximum absolute deviation allowed for each parameter.

    Returns
    -------
    lb : np.ndarray
        Lower‐bound array of shape (n_params,), equal to ctrl_points−delta.
    ub : np.ndarray
        Upper‐bound array of shape (n_params,), equal to ctrl_points+delta.
    """
    flat = ctrl_points.ravel()
    lb = flat - delta
    ub = flat + delta
    return lb, ub

def curvature_residuals(bspline: BSpline, weight: float, num_curv_samples: int):
    # sample in the same parameter domain you use for fitting
    t0, t1 = bspline.t[bspline.k], bspline.t[-bspline.k-1]
    u_curv = np.linspace(t0, t1, num_curv_samples)
    d2 = bspline(u_curv, 2)                      # second derivative vectors
    # L2 norm of each second derivative, scaled by sqrt(weight)
    return np.sqrt(weight) * np.linalg.norm(d2, axis=1)

def sample_bspline(spline: BSpline, n_samples: int):
    """
    Sample a 3-dimensional BSpline so that you get an (n_samples, 3) array back.

    Parameters
    ----------
    spline : BSpline
        A BSpline whose coefficient dimension is 3.
    n_samples : int
        Number of points to sample.

    Returns
    -------
    u : (n_samples,) float array
        Parameter values uniformly spaced on [t[k], t[-k-1]].
    pts : (n_samples, 3) float array
        Spline evaluated at each u, shape guaranteed to be (n_samples,3).
    """
    # domain
    t, c, k = spline.t, spline.c, spline.k
    u_min, u_max = t[k], t[-k-1]
    u = np.linspace(u_min, u_max, n_samples)

    # evaluate; for a 3D spline spline(u) returns shape (n_samples, 3)
    pts = spline(u)

    # sometimes spline(u) returns shape (3, n_samples), so transpose if needed
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] == n_samples:
        pts = pts.T

    # final sanity check
    if pts.shape != (n_samples, 3):
        raise ValueError(f"Expected output shape ({n_samples},3), got {pts.shape}")

    return pts

def project_3d_points(points, camera_pose, camera_parameters):
    """
    Project 3D world points into pixel coordinates, given camera pose & intrinsics.

    Parameters
    ----------
    points : (n,3) array-like of float
        3D points in the world frame.
    camera_pose : geometry_msgs.msg.PoseStamped
        Pose of the camera in the world frame.
    camera_parameters : tuple of 4 floats
        (fx, fy, cx, cy) intrinsic parameters.

    Returns
    -------
    projected : (n,2) ndarray of float
        Pixel coordinates (u, v) for each 3D point.
    """
    # Unpack intrinsics
    fx, fy, cx, cy = camera_parameters

    # Convert input to ndarray
    pts_w = np.asarray(points, dtype=float)  # shape (n,3)

    # Extract translation
    t = camera_pose.pose.position
    cam_t = np.array([t.x, t.y, t.z], dtype=float)

    # Extract quaternion
    q = camera_pose.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w

    # Build rotation matrix R_cam→world from q,
    # then invert to get R_world→cam = R_cam→world.T
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    R_wc = R.T  # world → camera

    # Transform points into the camera frame
    # p_cam = R_wc @ (p_world - cam_t)
    pts_cam = (R_wc @ (pts_w - cam_t).T).T  # still shape (n,3)

    # Perspective projection (no rounding!)
    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]
    if np.any(Z <= 0):
        # points behind the camera will give nonsense—warn if you like
        pass

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    return np.stack([u, v], axis=1)

def show_2d_points_with_mask(projected_points, mask, 
                                    title="Projected Points on Mask",
                                    point_color=(0, 0, 255),  # red in BGR
                                    point_radius=0,
                                    point_thickness=-1):      # filled circle
    """
    Overlay projected 2D points on a mask and display with OpenCV.

    Parameters
    ----------
    projected_points : (n,2) array-like of float
        Pixel coordinates (u, v) to draw.
    mask : 2D ndarray
        Binary or grayscale mask image. Values will be shown in gray.
    title : str
        Window title.
    point_color : tuple of 3 ints
        BGR color for the points.
    point_radius : int
        Radius of each circle in pixels.
    point_thickness : int
        Thickness (-1 for filled).
    """
    if mask.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    
    # Convert mask to uint8 gray if needed
    m = mask
    m = (mask.astype(np.uint8) * 255)

    # Make BGR canvas
    canvas = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    # Draw each point
    for (u, v) in projected_points:
        # Round down to int pixel coords
        pt = (int(np.floor(u)), int(np.floor(v)))
        cv2.circle(canvas, pt, point_radius, point_color, point_thickness)

    # Display
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 550)
    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_centerline_from_mask_individual(depth_image: np.ndarray, mask: np.ndarray, camera_parameters: tuple, save_data_class, depth_scale: float = 1.0, connectivity: int = 8, min_length: int = 20, show: bool = False) -> list:
    """
    Extracts individual 3D centerline segments (no connections).

    Returns:
        centerlines: list of (Ni,3) arrays of segment points in camera coords.
    """
    fx, fy, cx, cy = camera_parameters

    # 1. Ensure mask is in uint8 format (0 or 1)
    # If mask is boolean, convert directly
    if mask.dtype == bool:
        mask_u8 = mask.astype(np.uint8)
    else:
        # If mask has other types (e.g., 0/255), normalize to 0/1
        mask_u8 = np.where(mask > 127, 1, 0).astype(np.uint8)

    # 2. Binarise and skeletonize + excise + remove short segments
    # Use threshold on uint8 mask
    _, binary = cv2.threshold(mask_u8, 0, 1, cv2.THRESH_BINARY)

    skeleton_bool = skeletonize(binary.astype(bool))
    save_data_class.save_skeleton([skeleton_bool], "original_skeleton", invert_color=True)

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


    save_data_class.save_skeleton([to_remove], "to_remove", invert_color=True)
    save_data_class.save_skeleton([skeleton_bool, to_remove], "junction_skeleton", invert_color=True)

    skeleton_exc = skeleton_bool & ~to_remove

    save_data_class.save_skeleton([skeleton_exc], "junction_deleted", invert_color=True)

    longest_skeleton = longest_skeleton_path_mask(skeleton_exc)

    show_masks([longest_skeleton], title="longest_skeleton")
    save_data_class.save_skeleton([longest_skeleton], "final_skeleton", invert_color=True)

    num_labels, labels = cv2.connectedComponents(skeleton_exc.astype(np.uint8), connectivity)
    for lbl in range(1, num_labels):
        comp = (labels==lbl)
        if comp.sum() < min_length:
            skeleton_exc[comp] = False
    
    if show:
        show_masks([skeleton_bool], title="Skeleton")
        show_masks([to_remove], title="To Remove")
        show_masks([skeleton_bool, to_remove], title="Both")
        show_masks([skeleton_exc], title="skeleton_exc")

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

    print(f"Found {len(segments)} segments")
    return segments

import numpy as np
from scipy import ndimage as ndi
import math
import heapq

def longest_skeleton_path_mask(skeleton_exc: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Return a boolean mask (same shape) with only the single longest continuous path on the
    skeleton (after junction removal) kept. Uses 4- or 8-neighborhood.
    """
    assert connectivity in (4, 8), "connectivity must be 4 or 8"
    h, w = skeleton_exc.shape
    if not skeleton_exc.any():
        return np.zeros_like(skeleton_exc, dtype=bool)

    # Connected components on the junction-free skeleton
    structure = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=bool) if connectivity == 8 else \
                np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    labeled, ncc = ndi.label(skeleton_exc, structure=structure)

    # Neighbor offsets
    if connectivity == 8:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
    else:
        nbrs = [(-1,0), (1,0), (0,-1), (0,1)]

    def build_graph(coords):
        index_of = {rc: i for i, rc in enumerate(coords)}
        adj = [[] for _ in coords]
        deg = np.zeros(len(coords), dtype=int)
        for i, (r, c) in enumerate(coords):
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                j = index_of.get((nr, nc), None)
                if j is not None:
                    w = 1.0 if (dr == 0 or dc == 0) else math.sqrt(2.0)
                    adj[i].append((j, w))
                    deg[i] += 1
        return adj, deg

    def dijkstra(adj, start):
        n = len(adj)
        dist = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, parent

    def longest_path_component(coords):
        """
        For one connected component (list of (r,c)), compute the longest shortest path
        and return (node_index_path_list, path_length_float).
        """
        n = len(coords)
        if n == 1:
            return [0], 0.0  # <-- FIX: consistent 2-value return

        adj, deg = build_graph(coords)

        # prefer an endpoint (deg==1); fallback to node 0
        endpoints = np.where(deg == 1)[0]
        start = int(endpoints[0]) if endpoints.size > 0 else 0

        # First pass: from start to farthest u
        dist1, _ = dijkstra(adj, start)
        u = int(np.argmax(dist1))  # all finite in a connected component

        # Second pass: from u to farthest v; track parents
        dist2, parent2 = dijkstra(adj, u)
        v = int(np.argmax(dist2))
        length = float(dist2[v])

        # Reconstruct u -> v
        path = []
        cur = v
        while cur != -1:
            path.append(cur)
            if cur == u:
                break
            cur = parent2[cur]
        path.reverse()
        return path, length

    best_len = -1.0
    best_coords = None
    best_path_nodes = None

    for label in range(1, ncc + 1):
        coords = list(zip(*np.nonzero(labeled == label)))
        if not coords:
            continue
        path_nodes, length = longest_path_component(coords)
        if length > best_len:
            best_len = length
            best_coords = coords
            best_path_nodes = path_nodes

    longest_mask = np.zeros((h, w), dtype=bool)
    if best_coords is not None and best_path_nodes is not None:
        for idx in best_path_nodes:
            r, c = best_coords[idx]
            longest_mask[r, c] = True
    return longest_mask



def show_spline_gradient(mask: np.ndarray, centerline: np.ndarray, title: str = "Gradient"):
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



def control_law(current_pose: PoseStamped,
                      target_pose: PoseStamped,
                      step_size: float) -> PoseStamped:
    """
    Compute the next gripper pose via a proportional control step.

    Parameters
    ----------
    current_pose : PoseStamped
        The gripper's current pose.
    target_pose : PoseStamped
        The desired target pose (with downward-pointing z-axis).
    step_size : float
        Control gain γ ∈ [0,1]. 1 = jump straight to target.

    Returns
    -------
    PoseStamped
        The next pose: pₙₑₓₜ = (1-γ)p₀ + γp₁,
                         qₙₑₓₜ = q₀ ⊗ (q₀⁻¹ ⊗ q₁)^γ.
    """
    # Extract translation and quaternion
    def unpack(ps: PoseStamped):
        t = ps.pose.position
        q = ps.pose.orientation
        trans = np.array([t.x, t.y, t.z], dtype=float)
        quat  = np.array([q.x, q.y, q.z, q.w], dtype=float)
        hdr   = copy.deepcopy(ps.header)
        return trans, quat, hdr

    p0, q0, hdr = unpack(current_pose)
    p1, q1, _   = unpack(target_pose)

    # 1) Linear step in position:
    p_next = (1.0 - step_size) * p0 + step_size * p1

    # 2) Compute delta quaternion: q_delta = q0⁻¹ ⊗ q1
    q_delta = quaternion_multiply(quaternion_inverse(q0), q1)

    # 3) Raise delta to the γ power via slerp from identity [0,0,0,1]
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q_delta_gamma = quaternion_slerp(identity, q_delta, step_size)

    # 4) Apply: q_next = q0 ⊗ q_delta^γ
    q_next = quaternion_multiply(q0, q_delta_gamma)

    # Build the next PoseStamped
    ps = PoseStamped()
    ps.header = hdr
    # Optionally update timestamp:
    # ps.header.stamp = rospy.Time.now()
    ps.pose.position.x = float(p_next[0])
    ps.pose.position.y = float(p_next[1])
    ps.pose.position.z = float(p_next[2])
    ps.pose.orientation.x = float(q_next[0])
    ps.pose.orientation.y = float(q_next[1])
    ps.pose.orientation.z = float(q_next[2])
    ps.pose.orientation.w = float(q_next[3])

    return ps




def optimize_bspline_pre_working(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        reg_weight=0,
        curvature_weight=0,
        num_samples=200,
        symmetric=True,
        translate=False,
        disp=True,
    ):
    # Initial control points
    ctrl_pts_init = initial_spline.c.copy()
    ctrl_pts_init_flat = ctrl_pts_init.reshape(-1)
    degree = initial_spline.k
    knot_vector = initial_spline.t.copy()
    dim = ctrl_pts_init.shape[1]
    num_ctrl = ctrl_pts_init.shape[0]

    n_frames = len(masks)
    weights = np.array([decay**(n_frames - 1 - i) for i in range(n_frames)], dtype=float)
    wsum = weights.sum()

    _, interps = precompute_skeletons_and_interps(masks) 

    def make_bspline(ctrl_points_flat):
        ctrl_points = ctrl_points_flat.reshape(num_ctrl, dim)
        return BSpline(knot_vector, ctrl_points, degree)  
    
    # Precompute skeletons, distance transforms, and coordinates
    dts = []
    skeleton_coords = []
    for mask in masks:
        sk = skeletonize(mask > 0)
        coords = np.vstack(np.nonzero(sk)).T  # (v,u)
        skeleton_coords.append((sk, coords))
        dts.append(distance_transform_edt(~sk))


    def objective(ctrl_points_flat):
        ctrl_pts = ctrl_points_flat.reshape(-1, 3)

        # Build spline and sample
        spline = make_bspline(ctrl_points_flat)
        # spline = create_bspline(ctrl_points_flat.reshape(-1, 3))
        sampled_pts_3d = sample_bspline(spline, num_samples, equal_spacing=False)

        # 1) reshape & drift penalty
        drift = np.linalg.norm(ctrl_points_flat - ctrl_pts_init_flat) / num_ctrl

        # 2) curvature penalty
        if num_ctrl >= 3:
            # diffs = ctrl_pts[1:] - ctrl_pts[:-1]
            # v1, v2 = diffs[:-1], diffs[1:]
            # dot = np.einsum('ij,ij->i', v1, v2)
            # n1 = np.linalg.norm(v1, axis=1)
            # n2 = np.linalg.norm(v2, axis=1)
            # cos_t = np.clip(dot/(n1*n2 + 1e-8), -1, 1)
            # angles = np.arccos(cos_t)
            # curvature_penalty = np.mean(angles**2)
            curvature_penalty = np.mean(curvature_residuals(spline, 1, num_samples*2)**2)
        else:
            curvature_penalty = 0.0

        
        assd_sum = 0.0
        for i, ((sk, coords), dt, cam_pose) in enumerate(zip(skeleton_coords, dts, camera_poses)):
            pts2d = project_3d_points(sampled_pts_3d, cam_pose, camera_parameters)
            print(f"num_samples: {num_samples}, pts2d.shape: {pts2d.shape}")
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
        # print(f"mean_assd: {mean_assd:.2f}, drift: {drift*reg_weight:.2f}, curvature_penalty: {curvature_penalty*curvature_weight:.2f}, loss: {loss:.2f}")
        return float(loss)
    
    bounds = make_bspline_bounds(initial_spline, delta=0.4)

    start = time.perf_counter()
    result = opt.minimize(
        fun=objective,
        x0=ctrl_pts_init_flat,
        method='L-BFGS-B', # 'Powell', # 
        bounds=bounds,
        jac='3-point',
        options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-4, 'disp':True, 'maxfun':1e6}
    )
    end = time.perf_counter()
    # print(f"Optimization took {end - start:.2f} seconds")
    print(f"Optimization cost: {result.fun:.2f}")
    return make_bspline(result.x), result.fun
    # return create_bspline(result.x.reshape(-1, 3))

def optimize_bspline_pre_least_squares_manualy_coded(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        reg_weight=0,
        curvature_weight=0,
        num_samples=200,
        symmetric=True,
        translate=False,
        disp=True,
    ):
    # Initial control points
    ctrl_pts_init = initial_spline.c.copy()
    ctrl_pts_init_flat = ctrl_pts_init.reshape(-1)
    degree = initial_spline.k
    knot_vector = initial_spline.t.copy()
    dim = ctrl_pts_init.shape[1]
    num_ctrl = ctrl_pts_init.shape[0]

    n_frames = len(masks)
    weights = np.array([decay**(n_frames - 1 - i) for i in range(n_frames)], dtype=float)
    wsum = weights.sum()

    _, interps = precompute_skeletons_and_interps(masks) 

    def make_bspline(ctrl_points_flat):
        ctrl_points = ctrl_points_flat.reshape(num_ctrl, dim)
        return BSpline(knot_vector, ctrl_points, degree)  
    
    # Precompute skeletons, distance transforms, and coordinates
    dts = []
    skeleton_coords = []
    for mask in masks:
        sk = skeletonize(mask > 0)
        coords = np.vstack(np.nonzero(sk)).T  # (v,u)
        skeleton_coords.append((sk, coords))
        dts.append(distance_transform_edt(~sk))

    #region old residuals
    # def objective(ctrl_points_flat):
    #     residuals = []
    #     ctrl_pts = ctrl_points_flat.reshape(-1, 3)

    #     # 1) reshape & drift penalty
    #     drift_penalty = (ctrl_points_flat - ctrl_pts_init_flat) / num_ctrl * reg_weight
    #     residuals.extend(drift_penalty)

    #     # 2) curvature penalty
    #     if num_ctrl >= 3:
    #         diffs = ctrl_pts[1:] - ctrl_pts[:-1]
    #         v1, v2 = diffs[:-1], diffs[1:]
    #         dot = np.einsum('ij,ij->i', v1, v2)
    #         n1 = np.linalg.norm(v1, axis=1)
    #         n2 = np.linalg.norm(v2, axis=1)
    #         cos_t = np.clip(dot/(n1*n2 + 1e-8), -1, 1)
    #         angles = np.arccos(cos_t)
    #         # curvature_penalty = np.mean(angles**2)
    #         curvature_penalty = angles**2 * curvature_weight
    #         residuals.extend(curvature_penalty)
    #     else:
    #         # curvature_penalty = 0.0
    #         pass


    #     # Build spline and sample
    #     spline = make_bspline(ctrl_points_flat)
    #     sampled_pts_3d = sample_bspline(spline, num_samples)
        
    #     assd_sum = 0.0
    #     for i, ((sk, coords), dt, cam_pose) in enumerate(zip(skeleton_coords, dts, camera_poses)):
    #         pts2d = project_3d_points(sampled_pts_3d, cam_pose, camera_parameters)
    #         # guard against empty projection
    #         if pts2d.size == 0:
    #             # no points visible → huge penalty
    #             assd_i = np.max(interps[i].values)  # max distance
    #         else:
    #             # sample the precomputed interpolator
    #             sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
    #             dists = interps[i](sample_pts)
    #             # guard NaNs
    #             assd_i = np.nanmean(dists) if np.isfinite(dists).any() else np.max(interps[i].values)

    #         # assd_sum += weights[i] * assd_i
    #         assd_i *= weights[i]
    #         residuals.append(assd_i)

    #     # mean_assd = assd_sum / wsum

    #     # 5) final loss
    #     # loss = mean_assd + reg_weight * drift + curvature_weight * curvature_penalty
    #     # print(f"loss: {loss:.3f}")
    #     print(f"residuals: {residuals}")
    #     return residuals
    #endregion old residuals

    def objective(ctrl_points_flat):
        """
        Residual vector for scipy.optimize.least_squares.
        0.5 * sum(residual**2)  reproduces  *exactly*:
            mean_assd + reg_weight*drift + curvature_weight*curv
        """
        ctrl_pts = ctrl_points_flat.reshape(-1, 3)
        res = []

        # --- drift term: reg_weight * drift  ----------------------------------
        drift = np.linalg.norm(ctrl_points_flat - ctrl_pts_init_flat) / num_ctrl
        # choose r so that 0.5*r**2 = reg_weight * drift
        res.append(np.sqrt(2 * reg_weight * max(drift, 1e-12)))

        # --- curvature term: curvature_weight * mean(angles**2) ---------------
        if num_ctrl >= 3 and curvature_weight:
            v = ctrl_pts[1:] - ctrl_pts[:-1]
            cos_t = np.einsum('ij,ij->i', v[:-1], v[1:]) / (
                    np.linalg.norm(v[:-1], axis=1) *
                    np.linalg.norm(v[1:],  axis=1) + 1e-12)
            angles2 = np.arccos(np.clip(cos_t, -1, 1)) ** 2
            curv = angles2.mean()
            res.append(np.sqrt(2 * curvature_weight * max(curv, 1e-12)))

        # --- data term:   (weights/wsum) * assd_i  ----------------------------
        spline = make_bspline(ctrl_points_flat)
        sampled_pts_3d = sample_bspline(spline, num_samples)

        for i, (cam_pose, interp) in enumerate(zip(camera_poses, interps)):
            pts2d = project_3d_points(sampled_pts_3d, cam_pose, camera_parameters)
            if pts2d.size == 0:
                assd_i = np.max(interp.values)
            else:
                dists = interp(np.stack([pts2d[:, 1], pts2d[:, 0]], axis=1))
                assd_i = np.nanmean(dists) if np.isfinite(dists).any() else np.max(interp.values)

            weight_i = weights[i] / wsum                      # same weight as before
            res.append(np.sqrt(2 * weight_i * max(assd_i, 1e-12)))

        return np.asarray(res, dtype=float)
    
    bounds = make_bspline_bounds(initial_spline, delta=0.4) # wring format for least squares
    lo, hi = np.array(bounds, dtype=float).T      # shape both (n,)
    init_res = objective(ctrl_pts_init_flat)

    start = time.perf_counter()
    # result = opt.minimize(
    #     fun=objective,
    #     x0=ctrl_pts_init_flat,
    #     method='L-BFGS-B', # 'Powell', # 
    #     bounds=bounds,
    #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-6, 'xtol':1e-4, 'disp':True, 'maxfun':1e6}
    # )

    result = least_squares(
        fun              = objective,    # your function
        x0               = ctrl_pts_init_flat,    # shape (n,)
        method           = "trf", # "lm", # "dogbox",
        jac              = "2-point",            # or supply analytic/complex-step
        bounds           = (lo, hi),             # e.g. physical limits on control pts
        # loss             = "soft_l1",            # robust to skeleton gaps/outliers
        f_scale          = np.median(init_res),  # soft inlier threshold
        x_scale          = "jac",                # auto-scaling
        ftol             = 1e-9,
        xtol             = 1e-9,
        gtol             = 1e-9,
        diff_step        = None,                 # let SciPy pick
        max_nfev         = 2000,                 # give it room
        verbose          = disp,
    )

    end = time.perf_counter()
    print(f"Optimization took {end - start:.2f} seconds")
    return make_bspline(result.x), result.cost

def optimize_bspline_pre_least_squares_one_residual(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        reg_weight=0,
        curvature_weight=0,
        num_samples=200,
        symmetric=True,
        translate=False,
        disp=True,
    ):
    # ---------- unchanged bookkeeping ----------
    ctrl_pts_init   = initial_spline.c.copy()
    ctrl_pts_init_f = ctrl_pts_init.reshape(-1)
    degree          = initial_spline.k
    knot_vector     = initial_spline.t.copy()
    dim, num_ctrl   = ctrl_pts_init.shape[1], ctrl_pts_init.shape[0]

    n_frames  = len(masks)
    weights   = np.array([decay**(n_frames-1-i) for i in range(n_frames)], float)
    wsum      = weights.sum()

    _, interps = precompute_skeletons_and_interps(masks)

    def make_bspline(flat):
        return BSpline(knot_vector, flat.reshape(num_ctrl, dim), degree)

    # ---------- per-frame pre-computations ----------
    skel_coords, dts = [], []
    for m in masks:
        sk = skeletonize(m > 0)
        skel_coords.append(np.vstack(np.nonzero(sk)).T)  # (v,u)
        dts.append(distance_transform_edt(~sk))

    # ---------- residual function ----------
    def residuals(flat):
        ctrl_pts = flat.reshape(-1, 3)

        # drift
        drift = np.linalg.norm(flat - ctrl_pts_init_f) / num_ctrl

        # curvature
        if num_ctrl >= 3:
            diffs = ctrl_pts[1:] - ctrl_pts[:-1]
            v1, v2  = diffs[:-1], diffs[1:]
            cos_t   = np.einsum('ij,ij->i', v1, v2) / (
                        np.linalg.norm(v1, 2, 1)*np.linalg.norm(v2, 2, 1) + 1e-8)
            curvature = np.mean(np.arccos(np.clip(cos_t, -1, 1))**2)
        else:
            curvature = 0.0

        spline        = make_bspline(flat)
        pts3d_sampled = sample_bspline(spline, num_samples)

        res = []

        # data term – one residual per visible sample point
        for i, (interp, cam_pose, wt) in enumerate(zip(interps, camera_poses, weights)):
            pts2d = project_3d_points(pts3d_sampled, cam_pose, camera_parameters)
            if pts2d.size == 0:
                # if nothing is visible, emit one huge residual so the solver moves
                res.append(np.sqrt(2.0*wt/wsum) * np.max(interp.values))
                continue

            # interp expects (v,u) = (row, col)
            samples = np.stack([pts2d[:, 1], pts2d[:, 0]], axis=1)
            dists   = interp(samples)
            # replace NaNs with max distance
            dists[~np.isfinite(dists)] = np.max(interp.values)

            # scale each point: sqrt(2*wt/wsum) matches original weighting
            res.extend(np.sqrt(2.0*wt/wsum) * dists)

        # regularisation residuals
        if reg_weight:
            res.append(np.sqrt(2.0*reg_weight) * drift)
        if curvature_weight:
            res.append(np.sqrt(2.0*curvature_weight) * curvature)

        return np.asarray(res, float)

    # ---------- bounds ----------
    b_low, b_up = map(np.asarray, zip(*make_bspline_bounds(initial_spline, delta=0.4)))

    # ---------- solve ----------
    t0 = time.perf_counter()
    sol = opt.least_squares(
        fun       = residuals,
        x0        = ctrl_pts_init_f,
        jac       = '3-point',
        bounds    = (b_low, b_up),
        method    = 'trf',      # bounds-aware, robust to rank-deficiency
        loss      = 'linear',   # same as your original scalar loss
        ftol=1e-6, xtol=1e-6, gtol=1e-6,
        verbose   = 2 if disp else 0
    )
    if disp:
        elapsed = time.perf_counter() - t0
        final_loss = 0.5*np.sum(residuals(sol.x)**2)
        print(f'Optimisation finished in {elapsed:.2f}s — loss={final_loss:.6f}')

    return create_bspline(sol.x.reshape(-1, 3))

def optimize_bspline_pre_least_squares_many_residuals(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        reg_weight=0.0,
        curvature_weight=0.0,
        num_samples=200,
        symmetric=True,
        translate=False,
        disp=True,
    ):
    # --------------- bookkeeping -------------------------------------
    ctrl_init     = initial_spline.c.copy()
    ctrl_init_f   = ctrl_init.reshape(-1)
    k, t          = initial_spline.k, initial_spline.t.copy()
    dim, n_ctrl   = ctrl_init.shape[1], ctrl_init.shape[0]

    n_frames      = len(masks)
    w_frame       = decay ** np.arange(n_frames-1, -1, -1, dtype=float)
    wsum          = w_frame.sum()

    _, interps    = precompute_skeletons_and_interps(masks)

    def make_bspline(flat):
        return BSpline(t, flat.reshape(n_ctrl, dim), k)

    # --------------- frame-level pre-computations ---------------------
    dts, skel_coords = [], []
    for m in masks:
        skel = skeletonize(m > 0)
        skel_coords.append(np.vstack(np.nonzero(skel)).T)      # (v,u)
        dts.append(distance_transform_edt(~skel))

    # --------------- residual function -------------------------------
    def residuals(flat):
        ctrl   = flat.reshape(-1, 3)

        # drift regulariser
        drift  = np.linalg.norm(flat - ctrl_init_f) / n_ctrl
        r_drift = np.sqrt(2*reg_weight) * drift if reg_weight else None

        # curvature regulariser
        if n_ctrl >= 3:
            d01     = ctrl[1:] - ctrl[:-1]
            v1, v2  = d01[:-1], d01[1:]
            cosang  = np.einsum('ij,ij->i', v1, v2) / (
                       np.linalg.norm(v1,2,1)*np.linalg.norm(v2,2,1) + 1e-8)
            curvature = np.mean(np.arccos(np.clip(cosang, -1, 1))**2)
        else:
            curvature = 0.0
        r_curv = np.sqrt(2*curvature_weight) * curvature if curvature_weight else None

        # sample B-spline in 3-D
        pts3d = sample_bspline(make_bspline(flat), num_samples)

        # data residuals
        res = []
        for w_i, interp, cam_pose in zip(w_frame, interps, camera_poses):
            pts2d = project_3d_points(pts3d, cam_pose, camera_parameters)
            if pts2d.size == 0:
                res.append(np.sqrt(2*w_i/wsum) * np.max(interp.values))
                continue

            samples = np.stack([pts2d[:,1], pts2d[:,0]], 1)   # (v,u)
            dists   = interp(samples)
            dists[~np.isfinite(dists)] = np.max(interp.values)
            # scale distances the same way the frame weight was scaled in mean-assd
            res.extend(np.sqrt(2*w_i / wsum) * dists)

        if r_drift is not None:
            res.append(r_drift)
        if r_curv is not None:
            res.append(r_curv)

        return np.asarray(res, float)

    # --------------- bounds ------------------------------------------
    lb, ub = map(np.asarray, zip(*make_bspline_bounds(initial_spline, 0.4)))

    # --------------- solve -------------------------------------------
    t0 = time.perf_counter()
    sol = opt.least_squares(
        fun     = residuals,
        x0      = ctrl_init_f,
        jac     = '3-point',
        bounds  = (lb, ub),
        method  = 'trf',
        loss    = 'linear', # 'soft_l1',      # robust, closer to L1 than pure LS
        ftol=1e-4, xtol=1e-4, gtol=1e-4,
        verbose = 2 if disp else 0
    )
    if disp:
        secs = time.perf_counter() - t0
        print(f'least_squares finished in {secs:.2f}s; ½‖r‖² = {0.5*np.sum(sol.fun**2):.6f}')

    return make_bspline(sol.x.reshape(-1, 3))

def sample_bspline(
        spline: BSpline,
        n_samples: int,
        *,
        equal_spacing: bool = False
    ):
    """
    Sample a 3-dimensional BSpline so that you get an (n_samples, 3) array back.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
        A BSpline whose coefficient dimension is 3.
    n_samples : int
        Number of points to sample.
    equal_spacing : bool, default False
        If False (default) sample at uniformly spaced *parameter* values.
        If True, return points that are (approximately) equally spaced
        in Euclidean distance along the curve.

    Returns
    -------
    pts : (n_samples, 3) ndarray
        Sampled points on the spline.
    """
    # Domain in parameter space
    t, k = spline.t, spline.k
    u_min, u_max = t[k], t[-k-1]

    if not equal_spacing:
        u = np.linspace(u_min, u_max, n_samples)
        pts = spline(u)

    else:
        # --- 1. Oversample to build an arclength lookup --------------------
        oversample = max(200, n_samples * 20)
        u_dense = np.linspace(u_min, u_max, oversample)
        pts_dense = spline(u_dense)

        # shape-fix
        if pts_dense.ndim == 2 and pts_dense.shape[0] == 3:
            pts_dense = pts_dense.T

        # --- 2. Cumulative arclength along the dense set -------------------
        deltas = np.diff(pts_dense, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        s_cum = np.concatenate(([0.0], np.cumsum(seg_len)))          # length = oversample
        total_len = s_cum[-1]

        # --- 3. Target arclengths and corresponding parameter values -------
        s_target = np.linspace(0.0, total_len, n_samples)
        u = np.interp(s_target, s_cum, u_dense)                      # length = n_samples

        # --- 4. Evaluate spline at those parameter values -----------------
        pts = spline(u)

    # Ensure shape is (n_samples, 3)
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] == n_samples:
        pts = pts.T
    if pts.shape != (n_samples, 3):
        raise ValueError(f"Expected output shape ({n_samples}, 3), got {pts.shape}")

    return pts


def optimize_depth_map(depths, masks, camera_poses, camera_parameters, show=False):
    index_mask = 0
    mask = masks[index_mask]
    camera_pose_mask = camera_poses[index_mask]

    index_depth = 1
    depth = depths[index_depth]
    camera_pose_depth = camera_poses[index_depth]

    _, interps = precompute_skeletons_and_interps(masks=[mask])

    def objective(scale_shift):
        scale, shift = scale_shift

        scaled_shifted_depth = scale_depth_map(depth=depth, scale=scale, shift=shift)
        depth_pointcloud = convert_depth_map_to_pointcloud(depth=scaled_shifted_depth, camera_parameters=camera_parameters)
        depth_pointcloud_world = transform_pointcloud_to_world(pointcloud=depth_pointcloud, camera_pose=camera_pose_depth)

        pts2d = project_pointcloud_exact(pointcloud=depth_pointcloud_world, camera_pose=camera_pose_mask, camera_parameters=camera_parameters)

        # print(f"pts2d: {pts2d.shape}")


        # guard against empty projection
        if pts2d.size == 0:
            # no points visible → huge penalty
            assd_i = np.max(interps[0].values)  # max distance
        else:
            # sample the precomputed interpolator
            sample_pts = np.stack([pts2d[:,1], pts2d[:,0]], axis=1)
            dists = interps[0](sample_pts)
            # guard NaNs
            assd_i = np.nanmean(dists) if np.isfinite(dists).any() else np.max(interps[0].values)

        # print(f"pts2d shape: {pts2d.shape}, ASSD: {assd_i}")
        return assd_i
        

    # bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]
    bounds = [(0.05, 1), (-1, 1)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    start = time.perf_counter()
    result = opt.minimize(
        objective,
        bounds=bounds,
        x0 = [0.28, 0.03],
        method= 'Powell', #'L-BFGS-B', 
            options={'maxiter':1e8, 'ftol':1e-2, 'eps':1e-6, 'disp':True, 'maxfun':1e8}
    )
    end = time.perf_counter()
    print(f"Estimating scale & shift took {end - start:.2f} seconds")

    scale, shift = result.x
    print(f"Scale: {scale:.2f}, Shift: {shift:.2f}")

    optimal_depth = scale_depth_map(depth=depth, scale=scale, shift=shift)

    scaled_shifted_depth = scale_depth_map(depth=depth, scale=scale, shift=shift)
    depth_pointcloud = convert_depth_map_to_pointcloud(depth=scaled_shifted_depth, camera_parameters=camera_parameters)
    depth_pointcloud_world = transform_pointcloud_to_world(pointcloud=depth_pointcloud, camera_pose=camera_pose_depth)
    _, projection = project_pointcloud_from_world(depth_pointcloud_world, camera_pose_mask, camera_parameters)


    if show:
        show_masks_union(mask, projection)

    return scale, shift, depth_pointcloud_world, None


def project_pointcloud_exact(pointcloud: 'o3d.geometry.PointCloud',
                       camera_pose: 'PoseStamped',
                       camera_parameters: tuple[float,float,float,float] | None = None
                      ) -> np.ndarray:
    """
    Transform a pointcloud into camera frame and project.
    
    Parameters
    ----------
    pointcloud
        Open3D PointCloud in the same frame as camera_pose.
    camera_pose
        geometry_msgs.msg.PoseStamped describing camera→world.
    intrinsics
        (fx, fy, cx, cy). If provided, returns pixel coords
        [u_px, v_px]. Otherwise returns normalized [x/z, y/z].

    Returns
    -------
    pts2d : (M,2) ndarray
        2D points either normalized (if intrinsics=None) or in
        pixel coordinates.
    """
    # 1) Build world→camera transform
    q = camera_pose.pose.orientation
    t = camera_pose.pose.position
    T_cam_w = quaternion_matrix([q.x, q.y, q.z, q.w])
    T_cam_w[0:3, 3] = [t.x, t.y, t.z]
    T_w_cam = np.linalg.inv(T_cam_w)

    # 2) Load points and make homogeneous
    pts_w = np.asarray(pointcloud.points)            # (N,3)
    pts_h = np.hstack([pts_w, np.ones((pts_w.shape[0],1))])  # (N,4)

    # 3) Transform to camera frame
    pts_cam = (T_w_cam @ pts_h.T).T[:, :3]           # (N,3)

    # 4) Keep only in‐front points
    mask = pts_cam[:,2] > 0
    pts_cam = pts_cam[mask]

    # 5) Project
    x, y, z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    u_norm = x / z
    v_norm = y / z
    pts2d = np.vstack([u_norm, v_norm]).T            # (M,2)

    if camera_parameters is not None:
        fx, fy, cx, cy = camera_parameters
        u_px = fx * u_norm + cx
        v_px = fy * v_norm + cy
        pts2d = np.vstack([u_px, v_px]).T            # pixel coords

    return pts2d


#endregion additional after branching


#region bspline otpimization - many residuals new
# --------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------
def bspline_basis_matrix(knot_vector, degree, u, n_ctrl):
    """
    Return the (len(u) , n_ctrl) matrix whose (s,j) entry is N_{j,k}(u_s).
    """
    B = np.empty((len(u), n_ctrl), dtype=float)
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=float)
        coeff[j] = 1.0
        B[:, j] = BSpline(knot_vector, coeff, degree, extrapolate=False)(u)
    return B


# --------------------------------------------------------------------------
# MAIN OPTIMISER
# --------------------------------------------------------------------------
def optimize_bspline_pre_ls_fast(
        initial_spline,
        camera_parameters,
        masks,
        camera_poses,
        decay=0.95,
        reg_weight=0.0,
        curvature_weight=0.0,
        coarse_levels=(50, 100, 200),          # sample counts for the cascade
        bounds_delta=0.4,
        disp=True,
    ):
    """
    Fast least-squares B-spline fitting with coarse-to-fine sampling and
    a pre-computed spline basis matrix.

    Parameters
    ----------
    * initial_spline      : scipy.interpolate.BSpline
    * camera_parameters   : whatever your `project_3d_points` expects
    * masks               : list/array of binary 2-D numpy arrays
    * camera_poses        : one 4×4 pose (world→camera) per frame
    * decay               : float, temporal weight decay
    * reg_weight          : λ_drift  (0 = off)
    * curvature_weight    : λ_curv   (0 = off)
    * coarse_levels       : tuple of `num_samples` for the cascade
    * bounds_delta        : ±bound range in world units
    * disp                : bool, print progress
    """
    # ------------------------------------------------------------------
    # basic spline data
    ctrl_init = initial_spline.c.copy()               # (n_ctrl , 3)
    ctrl_init_flat = ctrl_init.reshape(-1)            # 1-D view
    n_ctrl, dim = ctrl_init.shape
    degree      = initial_spline.k
    knot_vec    = initial_spline.t.copy()

    # ------------------------------------------------------------------
    # frame weights (same as your original)
    n_frames = len(masks)
    w_frame  = np.array([decay ** (n_frames - 1 - i) for i in range(n_frames)],
                        dtype=float)
    w_sum    = w_frame.sum()

    # ------------------------------------------------------------------
    # helpers pre-computed once per frame
    _, interps = precompute_skeletons_and_interps(masks)      # your util
    dts  = []
    for m in masks:
        skel = skeletonize(m > 0)
        dts.append(distance_transform_edt(~skel))

    # ------------------------------------------------------------------
    # bounds in least_squares format
    lb, ub = map(np.asarray,
                 zip(*make_bspline_bounds(initial_spline, delta=bounds_delta)))

    # ------------------------------------------------------------------
    # start the coarse-to-fine cascade
    x = ctrl_init_flat.copy()
    total_start = time.perf_counter()

    for level_idx, S in enumerate(coarse_levels, 1):
        if disp:
            print(f'\n=== Level {level_idx}/{len(coarse_levels)} '
                  f'— {S} samples ===')

        # --------------------------------------------------------------
        # 1) pre-compute basis matrix for this sample count
        u_samples = np.linspace(0.0, 1.0, S, endpoint=True)
        B = bspline_basis_matrix(knot_vec, degree, u_samples, n_ctrl)  # (S,n)

        # function that maps ctrl-points → sampled 3-D points
        def spline_points(flat):
            return B @ flat.reshape(n_ctrl, 3)         # (S,3)

        # --------------------------------------------------------------
        # 2) residual vector (Option B: one per visible sample + regs)
        def residual_vec(flat):
            P      = flat.reshape(n_ctrl, 3)
            pts3d  = B @ P                              # (S,3)
            res    = []

            # data term
            for w_i, interp, cam_pose in zip(w_frame, interps, camera_poses):
                pts2d = project_3d_points(pts3d, cam_pose, camera_parameters)
                if pts2d.size == 0:
                    res.append(np.sqrt(2*w_i/w_sum) * np.max(interp.values))
                    continue

                sample_uv = np.stack([pts2d[:, 1], pts2d[:, 0]], axis=1)
                dists = interp(sample_uv)
                dists[~np.isfinite(dists)] = np.max(interp.values)
                res.extend(np.sqrt(2*w_i / w_sum) * dists)

            # drift regulariser
            if reg_weight:
                drift = np.linalg.norm(flat - ctrl_init_flat) / n_ctrl
                res.append(np.sqrt(2*reg_weight) * drift)

            # curvature regulariser
            if curvature_weight and n_ctrl >= 3:
                diffs = P[1:] - P[:-1]
                v1, v2 = diffs[:-1], diffs[1:]
                cosang = np.einsum('ij,ij->i', v1, v2) / (
                           np.linalg.norm(v1,2,1) * np.linalg.norm(v2,2,1) + 1e-8)
                curv = np.mean(np.arccos(np.clip(cosang, -1, 1))**2)
                res.append(np.sqrt(2*curvature_weight) * curv)

            return np.asarray(res, float)

        # --------------------------------------------------------------
        # 3) (optional) analytic Jacobian would go *here*
        #    For now keep finite-difference but use '2-point' to halve calls.
        # --------------------------------------------------------------
        start = time.perf_counter()
        sol = opt.least_squares(
            fun     = residual_vec,
            x0      = x,
            jac     = '2-point',        # much cheaper than '3-point'
            bounds  = (lb, ub),
            method  = 'trf',
            loss    = 'soft_l1',        # robust, close to your L1 objective
            ftol=1e-6, xtol=1e-6, gtol=1e-6,
            verbose = 2 if disp else 0
        )
        if disp:
            print(f'  ↳ level finished in {time.perf_counter()-start:.2f}s  '
                  f'(½‖r‖² = {0.5*np.sum(sol.fun**2):.6f})')
        x = sol.x                       # refine next level from here

    if disp:
        print(f'\nTOTAL optimisation time: '
              f'{time.perf_counter()-total_start:.2f}s')

    # ------------------------------------------------------------------
    return create_bspline(x.reshape(n_ctrl, 3))
#endregion bspline otpimization - many residuals new

















# ────────────────────────────────────────────────────────────────────────────────
# 1.  PRECOMPUTE   — add the pixel coordinates of the mask’s skeleton
# ────────────────────────────────────────────────────────────────────────────────
def precompute_skeletons_and_interps_new(masks):
    """
    From each mask build
      • skeleton                   (bool  H×W)
      • dt of skeleton complement  (float H×W)
      • RegularGridInterpolator    (sub-pixel lookup into dt)
      • sk_coords                  (N×2  int)  ← NEW: (row, col) of skeleton pix
    """
    skeletons, interps, sk_coords_list = [], [], []
    for mask in masks:
        sk = skeletonize(mask > 0)
        dt = distance_transform_edt(~sk)
        H, W = sk.shape

        interps.append(
            RegularGridInterpolator(
                (np.arange(H), np.arange(W)),
                dt,
                bounds_error=False,
                fill_value=dt.max(),
            )
        )
        skeletons.append(sk)
        # save the integer coords once; they never change during optimisation
        sk_coords_list.append(np.column_stack(np.nonzero(sk)))

    return skeletons, interps, sk_coords_list


# ────────────────────────────────────────────────────────────────────────────────
# 2.  OPTIMISATION   — symmetric distance inside the objective
# ────────────────────────────────────────────────────────────────────────────────
def optimize_depth_map_new(depths, masks, camera_poses, camera_parameters, show=False):

    # fixed reference (= mask) and moving (= depth) views ---------------
    idx_mask, idx_depth = 0, 1
    mask, depth = masks[idx_mask], depths[idx_depth]
    pose_mask, pose_depth = camera_poses[idx_mask], camera_poses[idx_depth]

    # pre-computation that never changes across objective calls ---------
    _, interps, sk_coords_list = precompute_skeletons_and_interps_new([mask])
    interp_mask_dt = interps[0]           # distance-transform of the mask skeleton
    sk_coords = sk_coords_list[0]         # integer pixel coords of that skeleton
    H, W = mask.shape                     # image size for quick reuse
    max_dt = interp_mask_dt.values.max()  # used as a large penalty

    # -------------------------------------------------------------------
    def objective(scale_shift):
        scale, shift = scale_shift

        # 1.  transform the depth map as before -------------------------
        d_world = transform_pointcloud_to_world(
            convert_depth_map_to_pointcloud(
                scale_depth_map(depth, scale, shift), camera_parameters
            ),
            pose_depth,
        )

        pts2d = project_pointcloud_exact(
            d_world, camera_pose=pose_mask, camera_parameters=camera_parameters
        )

        # ----------------------------------------------------------------
        # Ⓐ distance mask ← pts2d  (old direction, sub-pixel)
        # ----------------------------------------------------------------
        if pts2d.size == 0:
            d12 = max_dt                          # nothing is visible → huge error
        else:
            sample_pts = np.stack([pts2d[:, 1], pts2d[:, 0]], axis=1)  # (row, col)
            d12 = interp_mask_dt(sample_pts)      # mask DT sampled at projected pts
            d12 = d12[np.isfinite(d12)]           # guard against NaNs
            if d12.size == 0:
                d12 = np.array([max_dt])          # all NaN → huge error

        # ----------------------------------------------------------------
        # Ⓑ distance pts2d ← mask  (new direction)
        #     1. build a binary image of projected pts
        #     2. DT of its complement
        #     3. sample that DT at every skeleton pixel of the mask
        # ----------------------------------------------------------------
        if pts2d.size == 0:
            d21 = np.array([max_dt])              # nothing to project
        else:
            # integer pixel positions, clipped to image bounds
            pix = np.round(pts2d).astype(int)
            inside = (0 <= pix[:, 0]) & (pix[:, 0] < W) & (0 <= pix[:, 1]) & (pix[:, 1] < H)
            pix = pix[inside]

            img_pts = np.zeros((H, W), dtype=bool)
            if pix.size:                          # protect distance_transform_edt
                img_pts[pix[:, 1], pix[:, 0]] = True

            dt_pts = distance_transform_edt(~img_pts)
            d21 = dt_pts[sk_coords[:, 0], sk_coords[:, 1]]

        # ----------------------------------------------------------------
        # ASSD   (average symmetric surface distance) --------------------
        # note: “surface” = mask skeleton, projected points
        # ----------------------------------------------------------------
        assd = 0.5 * (np.mean(d12) + np.mean(d21))
        return assd
    # -------------------------------------------------------------------

    bounds = [(0.05, 1.0), (-1.0, 1.0)]
    t0 = time.perf_counter()
    result = opt.minimize(
        objective,
        x0=[0.28, 0.03],
        bounds=bounds,
        method="Powell",
        options=dict(maxiter=1e8, maxfun=1e8, ftol=1e-2, eps=1e-6, disp=True),
    )
    print(f"Estimating scale & shift took {time.perf_counter() - t0:.2f} s")

    scale, shift = result.x
    print(f"Scale: {scale:.3f}   Shift: {shift:.3f}")

    # optional visualisation -------------------------------------------
    if show:
        _, proj = project_pointcloud_from_world(
            transform_pointcloud_to_world(
                convert_depth_map_to_pointcloud(scale_depth_map(depth, scale, shift),
                                                camera_parameters),
                pose_depth),
            pose_mask, camera_parameters)
        show_masks_union(mask, proj)

    # return world-space point cloud in the depth view so caller can reuse it
    depth_world = transform_pointcloud_to_world(
        convert_depth_map_to_pointcloud(
            scale_depth_map(depth, scale, shift), camera_parameters
        ),
        pose_depth,
    )
    return scale, shift, depth_world, None



from scipy.spatial.transform import Rotation as R

def transform_pose_intrinsic_xy(
    pose: PoseStamped, 
    alpha: float, 
    beta: float, 
    x: float, 
    y: float, 
    z: float
) -> PoseStamped:
    """
    1. Translate by (x, y, z) in the original local frame.
    2. Rotate intrinsically: alpha about local X, then beta about *new* local Y.
    
    Angles in radians, translation in meters.
    """
    q = pose.pose.orientation
    p = np.array([pose.pose.position.x,
                  pose.pose.position.y,
                  pose.pose.position.z])

    # Original orientation
    R0 = R.from_quat([q.x, q.y, q.z, q.w])

    # --- Step 1: translation in the original local frame ---
    translation_world = R0.apply([x, y, z])  # convert local offset to world coords
    p_new = p + translation_world

    # --- Step 2: intrinsic rotations ---
    Rx = R.from_rotvec(alpha * np.array([1, 0, 0]))   # local X
    Ry_local = R.from_rotvec(beta * np.array([0, 1, 0]))  # new local Y after Rx
    R_final = R0 * Rx * Ry_local

    # --- Output ---
    qn = R_final.as_quat()
    out = PoseStamped()
    out.header = pose.header
    out.header.stamp = rospy.Time.now()
    out.pose.position.x = p_new[0]
    out.pose.position.y = p_new[1]
    out.pose.position.z = p_new[2]
    out.pose.orientation.x = qn[0]
    out.pose.orientation.y = qn[1]
    out.pose.orientation.z = qn[2]
    out.pose.orientation.w = qn[3]
    return out
