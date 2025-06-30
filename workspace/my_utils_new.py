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
#endregion

# TransformStamped -> PoseStamped 
# t = camera_pose.pose.position
# q = camera_pose.pose.orientation

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

    if isinstance(camera_pose, PoseStamped):
        t = camera_pose.pose.position
        q = camera_pose.pose.orientation
    else:
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
        (255, 255, 255) # white
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


def interpolate_poses(pose_start: PoseStamped, pose_end: PoseStamped, num_steps: int) -> list[PoseStamped]:
    """
    Generate a smooth path of PoseStamped messages interpolating between two poses.
    Inputs can be either PoseStamped or TransformStamped. Outputs are PoseStamped.

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

def estimate_scale_shift_new_distance(self, depth1, mask2, camera_pose1, camera_pose2, show=False):
    """
    Estimates the optimal scale and shift values of data1 to fit data2.
    
    Parameters
    ----------
    data1 : tuple
        A tuple containing the first dataset, including the mask and depth map.
    data2 : tuple
        A tuple containing the second dataset, including the mask.
    transform1 : TransformStamped
        The pose from the first camera in world coordinates.
    transform2 : TransformStamped
        The pose from the second camera world coordinates.
    show : bool, optional
        If True, display visualizations of the masks and point clouds.
    
    Returns
    -------
    best_alpha : float
        The estimated optimal scale factor.
    best_beta : float
        The estimated optimal shift value.
    best_pointcloud_world : np.ndarray
        The transformed point cloud in world coordinates after applying the scale and shift.
    """
    print('Estimating scale and shift...')

        if not isinstance(camera_pose1, PoseStamped):
            raise TypeError("camera_pose1 must be a geometry_msgs.msg.TransformStamped")
        if not isinstance(camera_pose2, PoseStamped):
            raise TypeError("camera_pose2 must be a geometry_msgs.msg.TransformStamped")

    bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    start = time.perf_counter()
    result = opt.differential_evolution(
        score_function,
        bounds,
        args=(depth1, camera_pose1, mask2, camera_pose2),
        strategy='best1bin',
        maxiter=20,
        popsize=15,
        tol=1e-2,
        disp=False
    )
    end = time.perf_counter()
    print(f"Optimization took {end - start:.2f} seconds")

    alpha_opt, beta_opt = result.x
    # start = time.perf_counter()
    y_opt = -self.score_function([alpha_opt, beta_opt], depth1, transform1, mask2, transform2)
    # end = time.perf_counter()
    # print(f"One function call took {end - start:.4f} seconds")

    depth_opt = scale_depth_map(depth1, scale=alpha_opt, shift=beta_opt)
    pc_cam1_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
    pc_cam0_opt = transform_pointcloud_to_world(pc_cam1_opt, transform1)
    projection_pc_depth_cam2_opt, projection_pc_mask_cam2_opt = project_pointcloud_from_world(pc_cam0_opt, transform2, camera_parameters)

    fixed_projection_pc_depth_cam2_opt = fill_mask_holes(projection_pc_mask_cam2_opt)

    # self.depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_cam0_opt], [transform1])

    score = score_mask_match(fixed_projection_pc_depth_cam2_opt, mask2)

    print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Score: {y_opt}')


    # Show mask and pointcloud
    if show:
        show_masks([projection_pc_mask_cam2_opt], title='Optimal Projection')
        show_masks([fixed_projection_pc_depth_cam2_opt], title='Projection holes fixed')
        show_masks_union(projection_pc_mask_cam2_opt, fixed_projection_pc_depth_cam2_opt, title="Optimal Projection orig & holes fixed Projection")
        show_masks_union(fixed_projection_pc_depth_cam2_opt, mask2, title='Optimal Projection with holes fixed vs Mask to fit')

    return alpha_opt, beta_opt, pc_cam0_opt, score

def score_function(self, x, depth1, camera_pose1, mask2, camera_pose2):
    if rospy.is_shutdown(): exit()
    alpha, beta = x
    # Scale and shift
    depth_new = scale_depth_map(depth1, scale=alpha, shift=beta)
    # Get scaled/shifted pointcloud
    new_pc1 = convert_depth_map_to_pointcloud(depth_new, camera_parameters)
    # Transform scaled pointcloud into world coordinates
    new_pc0 = transform_pointcloud_to_world(new_pc1, camera_pose1)
    # Get projection of scaled pointcloud into camera 2
    projection_new_pc2_depth, projection_new_pc2_mask = project_pointcloud_from_world(new_pc0, camera_pose2, camera_parameters)
    # Fill holes
    fixed_projection_new_pc2_mask = fill_mask_holes(projection_new_pc2_mask)
    # Count the number of inliers between mask 2 and projection of scaled pointcloud
    score = score_mask_match(fixed_projection_new_pc2_mask, mask)
    # print(f'num_inliers: {num_inliers}')
    return -score

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

def interpolate_poses(self, pose_start: PoseStamped, pose_end: PoseStamped, num_steps: int) -> list[PoseStamped]:
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




