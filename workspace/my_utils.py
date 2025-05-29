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
from tf.transformations import quaternion_from_euler, quaternion_slerp, quaternion_matrix, quaternion_from_matrix, quaternion_multiply
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

# ----------------------------------------------------------------------------

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
    if not isinstance(camera_pose, TransformStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

    # world→camera: invert camera→world
    t = camera_pose.transform.translation
    q = camera_pose.transform.rotation
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

    if z_cam <= 0:
        raise ValueError("Point is behind the camera (z <= 0)")

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

#endregion

# ----------------------------------------------------------------------------

#region Calculations
def calculate_angle_from_mask_and_point(mask, point, area=20):
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

    print(f'point: {point}')
    print(f'px: {px}, py: {py}')
    patch = mask[py-area:py+area+1, px-area:px+area+1]
    points = cv2.findNonZero(patch)    # get foreground points in the patch
    # fit a line (least squares) through these points
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx)[0]
    print(f'Angle: {angle}')
    return(angle)

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
    cv2.imshow(title, canvas)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    for pc in pointclouds:
        if voxel_size is not None and voxel_size > 0:
            pc = pc.voxel_down_sample(voxel_size)
        elif len(pc.points) > max_points:
            pc = pc.random_down_sample(max_points / len(pc.points))
        geometries.append(pc)

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
    import cv2
    import numpy as np

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
  

def get_desired_pose(position, frame="map"):
    # build Pose
    p = Pose()
    p.position.x = float(position[0])
    p.position.y = float(position[1])
    p.position.z = float(position[2])
    p.orientation.x = float(1)
    p.orientation.y = float(0)
    p.orientation.z = float(0)
    p.orientation.w = float(0)
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = frame
    ps.pose = p
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
