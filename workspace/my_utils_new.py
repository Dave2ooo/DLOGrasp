#region Imports
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

import time

import scipy.optimize as opt

import torch
#endregion

# TransformStamped -> PoseStamped 
# t = camera_pose.pose.position
# q = camera_pose.pose.orientation

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
                        camera_pose: PoseStamped,
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
    camera_pose : PoseStamped
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
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a PoseStamped")
    t = camera_pose.pose.position
    q = camera_pose.pose.orientation
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

def convert_depth_map_to_pointcloud(depth: NDArray[np.floating] | Image.Image | NDArray[np.uint8], camera_parameters, stride: int | None = None) -> o3d.geometry.PointCloud:
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

def mask_pointcloud(pointcloud: o3d.geometry.PointCloud, mask: np.ndarray, camera_parameters) -> o3d.geometry.PointCloud:
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
    using the given camera_pose (a tf2_ros PoseStamped).

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The point cloud in the camera frame.
    camera_pose : tf2_ros.PoseStamped
        The transform from camera frame to world frame.

    Returns
    -------
    o3d.geometry.PointCloud
        A new point cloud expressed in world coordinates.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise TypeError("pointcloud must be an open3d.geometry.PointCloud")

    if isinstance(camera_pose, PoseStamped):
        t = camera_pose.pose.position
        q = camera_pose.pose.orientation
    else:
        raise TypeError("camera_pose must be a geometry_msgs.msg.PoseStamped")

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
    camera_pose : geometry_msgs.msg.PoseStamped
        Transform from camera frame to world frame.

    Returns
    -------
    np.ndarray, shape (N,3)
        The points expressed in the world coordinate system.
    """
    # Validate inputs
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.PoseStamped")
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

def transform_points_from_world(points: np.ndarray, camera_pose: PoseStamped) -> np.ndarray:
    """
    Transform an array of 3D points from the world frame into the camera frame.

    Parameters
    ----------
    points : np.ndarray, shape (N,3)
        Points expressed in the world coordinate system.
    camera_pose : geometry_msgs.msg.PoseStamped
        The transform from camera frame to world frame.

    Returns
    -------
    np.ndarray, shape (N,3)
        The same points expressed in the camera coordinate system.
    """
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a PoseStamped")

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")

    # Extract translation and rotation (camera→world)
    t = camera_pose.pose.position
    q = camera_pose.pose.orientation
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

def project_pointcloud_from_world(pointcloud: o3d.geometry.PointCloud, camera_pose: PoseStamped, camera_parameters):
    """
    Project a 3D point cloud (in world coordinates) onto the camera image plane,
    but always output a 480×640 depth map and mask for easy comparison.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
        The cloud in world coordinates.
    camera_pose : PoseStamped
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

def show_depth_map(depth, title="Depth Map"):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cv2.imshow(title, depth)
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

def show_pointclouds(pointclouds: list[o3d.geometry.PointCloud],  frames: list[PoseStamped] = None, title: str = 'Point Clouds') -> None:
    """
    Display multiple point clouds (each colored differently), optionally with frames.
    Accepts either Open3D PointClouds or numpy arrays of shape (N,3).

    Parameters
    ----------
    pointclouds : list of o3d.geometry.PointCloud or np.ndarray
        The point clouds to display. If an entry is an (N×3) array, it will be
        converted into an Open3D PointCloud.
    frames : list of PoseStamped, optional
        Coordinate frames to draw as triads.
    title : str
        Window title for the visualizer.
    """

    geometries = []

    # draw origin triad
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=(0,0,0)))

    # draw any provided frames
    if frames:
        for tf in frames:
            if not isinstance(tf, PoseStamped):
                raise TypeError("Each frame must be a PoseStamped")
            t = camera_pose.pose.position
            q = camera_pose.pose.orientation
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





def interpolate_poses(pose_start: PoseStamped, pose_end: PoseStamped, num_steps: int) -> list[PoseStamped]:
    """
    Generate a smooth path of PoseStamped messages interpolating between two poses.
    Inputs can be either PoseStamped or PoseStamped. Outputs are PoseStamped.

    Parameters
    ----------
    pose_start : PoseStamped
        Starting pose.
    pose_end : PoseStamped
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
        if isinstance(t, PoseStamped):
            p = t.pose.position
            q = t.pose.orientation
            hdr = copy.deepcopy(t.header)
            return (np.array([p.x, p.y, p.z], dtype=float),
                    np.array([q.x, q.y, q.z, q.w], dtype=float),
                    hdr)
        else:
            raise TypeError("pose_start and pose_end must be PoseStamped or PoseStamped")

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

def interpolate_poses(pose_start: PoseStamped, pose_end: PoseStamped, num_steps: int) -> list[PoseStamped]:
    """
    Generate a smooth path of PoseStamped messages interpolating between two poses.
    Inputs can be either PoseStamped or PoseStamped. Outputs are PoseStamped.

    Parameters
    ----------
    pose_start : PoseStamped
        Starting pose.
    pose_end : PoseStamped
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
        if isinstance(t, PoseStamped):
            pos = t.pose.position
            ori = t.pose.orientation
            hdr = copy.deepcopy(t.header)
            return (np.array([pos.x, pos.y, pos.z], dtype=float),
                    np.array([ori.x, ori.y, ori.z, ori.w], dtype=float),
                    hdr)
        else:
            raise TypeError("pose_start and pose_end must be PoseStamped")

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

def extract_centerline_from_mask_individual(depth_image: np.ndarray, mask: np.ndarray, camera_parameters: tuple, depth_scale: float = 1.0, connectivity: int = 8, min_length: int = 20, show: bool = False) -> list:
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

def project_bspline(ctrl_points: np.ndarray, camera_pose, camera_parameters: tuple, width: int = 640, height: int = 480, degree: int = 3) -> np.ndarray:
    """
    Projects a 3D B-spline (defined by control points) into the camera image plane,
    rendering a continuous curve into a binary mask.

    Args:
        ctrl_points: (N_control x 3) array of B-spline control points.
        camera_pose: PoseStamped with .pose.position (x,y,z)
                     and .pose.orientation (x,y,z,w) defining camera pose in world.
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
    # ensure boolean
    bool_mask = mask.astype(bool)

    # compute skeleton
    skel = skeletonize(bool_mask)

    # return as uint8 0/255
    return (skel.astype(np.uint8) * 255)


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

def score_function_bspline_point_ray(x, camera_poses, camera_parameters, degree, skeletons: list, num_samples: int, decay: float = 0.9):
    """
    Loss = decayed mean bidirectional point-ray loss over multiple frames.

    Args:
        x:                 flat np.array of shape (3*n_ctrl,)
        camera_poses:      list of PoseStamped, per frame
        camera_parameters: (fx, fy, cx, cy)
        degree:            spline degree (k)
        skeletons:         list of H×W bool masks (the 2D skeletons)
        num_samples:       how many points to sample along spline
        decay:             exponential decay factor for older frames (default 0.9)

    Returns:
        loss: torch scalar
    """
    # reshape control points
    ctrl_pts = x.reshape(-1, 3)
    n_ctrl = ctrl_pts.shape[0]

    fx, fy, cx, cy = camera_parameters
    frame_losses = []
    n_frames = len(skeletons)

    # compute per-frame decayed losses
    for i, (skel_mask, cam_pose) in enumerate(zip(skeletons, camera_poses)):
        # build B-spline in world coords
        k = degree
        m = n_ctrl - k - 1
        if m > 0:
            interior = np.linspace(0, 1, m + 2)[1:-1]
        else:
            interior = np.zeros((0,))
        knots = np.concatenate([np.zeros(k+1), interior, np.ones(k+1)])
        spline = BSpline(knots, ctrl_pts, k, axis=0)
        ts = np.linspace(knots[k], knots[-k-1], num_samples)
        pts_world = spline(ts)

        # transform to camera frame
        pts_cam = transform_points_from_world(pts_world, cam_pose)

        # build ray directions from skeleton pixels
        ys, xs = np.nonzero(skel_mask)
        x_c = (xs - cx) / fx
        y_c = (ys - cy) / fy
        dirs = np.stack([x_c, y_c, np.ones_like(x_c)], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # compute point-ray loss
        rays_t = torch.from_numpy(dirs).float()
        pts_t = torch.from_numpy(pts_cam).float()
        loss_i = point_ray_loss(rays_t, pts_t)

        # apply decay weight
        weight = decay ** (n_frames - 1 - i)
        frame_losses.append(weight * loss_i)

    # normalized weighted mean
    weights = torch.tensor([decay ** (n_frames - 1 - i) for i in range(n_frames)], dtype=torch.float32)
    mean_loss = torch.stack(frame_losses).sum() / weights.sum()

    return mean_loss

def point_ray_loss(rays: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    # rays: (R,3); pts: (P,3)
    # 1) dot‐products via mat‐mul = (P,R)
    t = pts @ rays.transpose(0,1)          # shape (P,R)

    # 2) same unpacking:
    diff = t.unsqueeze(-1) * rays.unsqueeze(0) - pts.unsqueeze(1)
    dists = diff.norm(dim=-1)

    pt_to_ray = dists.min(dim=1).values.mean()
    ray_to_pt = dists.min(dim=0).values.mean()

    return 0.5 * (pt_to_ray + ray_to_pt)

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

def make_bspline_bounds(ctrl_points: np.ndarray, delta: float = 0.1):
    """
    Given an (N×3) array of B-spline control points, return a 
    bounds list of length 3N for differential_evolution, where
    each coordinate is allowed to vary ±delta around its original value.
    """
    flat = ctrl_points.flatten()
    bounds = [(val - delta, val + delta) for val in flat]
    return bounds

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

def score_bspline_translation(shift_xy: np.ndarray, mask, camera_pose: PoseStamped, camera_parameters: tuple, degree: int, ctrl_points: np.ndarray, num_samples: int = 200) -> float:
    """
    Score (mean symmetric distance) of a B-spline after translating it
    parallel to the camera's image plane by shift_xy = [dx, dy] in camera coords.

    Args:
        shift_xy:            length-2 array [dx, dy] in camera-frame meters.
        ctrl_points:         (N_ctrl×3) original control points (world).
        data:                a tuple whose second element is the H×W mask.
        camera_pose:         PoseStamped of camera in world.
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

def apply_translation_to_ctrl_points(ctrl_points: np.ndarray, shift_xy: np.ndarray, camera_pose: PoseStamped) -> np.ndarray:
    """
    Applies a 2D translation (dx, dy) in camera image plane to all B-spline control points.

    Args:
        ctrl_points: (N×3) array of original control points in world coordinates.
        shift_xy:    length-2 array [dx, dy] representing translation in camera-frame meters.
        camera_pose: PoseStamped of camera in world.

    Returns:
        shifted_ctrl_points: (N×3) array of translated control points in world coordinates.
    """
    if not isinstance(camera_pose, PoseStamped):
        raise TypeError("camera_pose must be a geometry_msgs.msg.PoseStamped")


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
    shifted_ctrl_points = np.asarray(ctrl_points, dtype=float) + shift_world[np.newaxis, :]

    return shifted_ctrl_points

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
    pts_cam = diff.dot(R.T)

    # Perspective projection
    x_cam, y_cam, z_cam = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    # compute pts_cam, do projection, but **don’t** round or clip:
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    # keep only those with z>0 and inside a slightly padded window
    valid = (z_cam>0) & (u>=-1) & (u<width+1) & (v>=-1) & (v<height+1)
    return np.stack([u[valid], v[valid]], axis=1)  # shape (M,2)



def interactive_bspline_editor(ctrl_points: np.ndarray,
                               mask,
                               camera_pose,
                               camera_parameters: tuple,
                               degree: int,
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
        # score_neg = score_function_bspline_point_ray(final_flat, mask, camera_pose, camera_parameters, degree, init_ctrl_points.flatten())

        skeletons, interps = precompute_skeletons_and_interps([mask])  
        score = score_function_bspline_point_ray(final_flat, camera_pose, camera_parameters, degree, skeletons, 100)

        cv2.putText(disp, f"Score: {score:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show window
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return final_ctrl_pts