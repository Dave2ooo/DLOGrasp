#!/usr/bin/env python3.11

import os
import cv2, numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d
import rospy
import copy

from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from tf.transformations import quaternion_from_euler, quaternion_slerp, quaternion_matrix, quaternion_from_matrix, quaternion_multiply, translation_matrix, quaternion_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

from scipy.interpolate import BSpline

from scipy.spatial.transform import Rotation

from datetime import datetime

from save_load_numpy import *

from my_utils import *

import pickle

import json

camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)


class save_data:
    def __init__(self) -> None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M")
        self.folder_name = f'/root/workspace/images/thesis_images/{timestamp}'
        self.counter = 0


    def save_all(self, image, mask, depth_orig, depth_masked, camera_pose, palm_pose, spline_coarse, spline_fine, aruco_pose):
        folder_name_image = f'{self.folder_name}/image'
        folder_name_mask_cv2 = f'{self.folder_name}/mask_cv2'
        folder_name_mask_cv2_inverted = f'{self.folder_name}/mask_cv2_inverted'
        folder_name_mask_numpy = f'{self.folder_name}/mask_numpy'
        folder_name_depth_orig = f'{self.folder_name}/depth_orig'
        folder_name_depth_masked = f'{self.folder_name}/depth_masked'
        folder_name_camera_pose = f'{self.folder_name}/camera_pose'
        folder_name_palm_pose = f'{self.folder_name}/palm_pose'
        folder_name_spline_coarse = f'{self.folder_name}/spline_coarse'
        folder_name_spline_fine = f'{self.folder_name}/spline_fine'
        folder_name_aruco_pose = f'{self.folder_name}/aruco_pose'

        # OpenCV images
        if image is not None:
            save_image(image, folder_name_image, f'{self.counter}.png')
        if mask is not None:
            save_masks([mask], folder_name_mask_cv2, f'{self.counter}')
            save_masks([mask], folder_name_mask_cv2_inverted, f'{self.counter}', invert_color=True)
            save_numpy_to_file(mask, folder_name_mask_numpy, f'{self.counter}')
        if depth_orig is not None:
            save_depth_map(depth_orig, folder_name_depth_orig, f'{self.counter}')
            save_numpy_to_file(depth_orig, folder_name_depth_orig, f'{self.counter}')
        if depth_masked is not None:
            save_depth_map(depth_masked, folder_name_depth_masked, f'{self.counter}')
            save_numpy_to_file(depth_masked, folder_name_depth_masked, f'{self.counter}')

        if camera_pose is not None:
            save_pose_stamped(camera_pose, folder_name_camera_pose, f'{self.counter}')
        if palm_pose is not None:
            save_pose_stamped(palm_pose, folder_name_palm_pose, f'{self.counter}')

        if spline_coarse is not None:
            save_bspline(spline_coarse, folder_name_spline_coarse, f'{self.counter}')
        if spline_fine is not None:
            save_bspline(spline_fine, folder_name_spline_fine, f'{self.counter}')

        if aruco_pose is not None:
            save_pose_stamped(aruco_pose, folder_name_aruco_pose, f'{self.counter}')

        self.counter += 1

    def save_initial_spline(self, spline):
        save_bspline(spline, self.folder_name, 'initial_spline')

    def save_misc_params(self, scale, shift, optimization_time_translate, optimization_time_coarse, optimization_time_fine, optimization_cost_translate, optimization_cost_coarse, optimization_cost_fine, grasp_success):
        save_misc_params(scale, shift, optimization_time_translate, optimization_time_coarse, optimization_time_fine, optimization_cost_translate, optimization_cost_coarse, optimization_cost_fine, grasp_success, self.folder_name, 'misc_params')

    def save_skeleton(self, skeleton, name: str, invert_color: bool = False):
        folder_name_skeleton = f'{self.folder_name}/skeleton'

        save_masks(skeleton, folder_name_skeleton, name, invert_color=invert_color)

    def save_pointcloud_and_spline(self, pointcloud, spline,  name: str):
        folder_name_skeleton = f'{self.folder_name}/pointcloud'

        save_pointcloud_snapshots(pointcloud, folder_name_skeleton, name, spline=spline)

def save_image(image, folder: str, filename: str) -> bool:
    """
    Saves an OpenCV image to the specified folder and filename.
    Creates the folder if it does not exist.

    Parameters:
        image (numpy.ndarray): The image to save.
        folder (str): Destination folder path.
        filename (str): Name of the file including extension (e.g., 'snapshot.png').

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    # Ensure the output directory exists
    os.makedirs(folder, exist_ok=True)

    # Build full path and write
    path = os.path.join(folder, filename)
    success = cv2.imwrite(path, image)
    return success
        
def save_depth_map(depth_map: np.ndarray,
                   folder: str,
                   filename: str,
                   gamma: float = 0.5,
                   colormap: int = cv2.COLORMAP_PLASMA) -> None:
    """
    Save a depth map to PNG for visualization, stretching only the non-zero depths
    and painting the background white.

    Parameters
    ----------
    depth_map : np.ndarray
        2D array of depth values (float or int), with 0 indicating background.
    folder : str
        Directory to save into (will be created if needed).
    filename : str
        Base name (no extension) for the saved PNG.
    gamma : float
        Gamma correction exponent (<1 brightens close depths).
    colormap : int
        OpenCV colormap (default = COLORMAP_TURBO).
    """

    # ensure output folder exists
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"{filename}.png")

    # convert to float32
    dm = depth_map.astype(np.float32)

    # mask of valid (non-background) pixels
    valid = dm > 0

    # if no valid depths, just write a white image
    H, W = dm.shape
    if not valid.any():
        black = 255 * np.ones((H, W, 3), np.uint8)
        cv2.imwrite(save_path, black)
        return

    # compute vmin/vmax over non-zero depths
    vmin = float(dm[valid].min())
    vmax = float(dm[valid].max())

    # normalize and gamma-correct only valid pixels
    norm = np.zeros_like(dm, dtype=np.float32)
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    norm_val = (dm[valid] - vmin) / denom
    norm_val = np.clip(norm_val, 0.0, 1.0) ** gamma
    norm[valid] = norm_val

    # scale to 8-bit
    img8 = (norm * 255).astype(np.uint8)

    # apply Turbo colormap
    img_color = cv2.applyColorMap(img8, colormap)

    # paint background (zero-depth) white
    img_color[~valid] = (255, 255, 255)

    # save PNG
    cv2.imwrite(save_path, img_color)

def save_masks(masks,
                folder: str,
                filename: str,
                colors: list[tuple[int,int,int]] = None,
                invert_color: bool = False) -> None:
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
        white, red, green, blue, yellow, magenta, cyan, gray.
    invert_color : bool
        If True, use a white background and draw each mask in a
        contrasting color (first mask drawn in black).
        Otherwise background is black and masks drawn in the default palette.
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

    # If invert_color, swap to white background and ensure first mask is black
    if invert_color:
        background_color = (255, 255, 255)
        if colors is None:
            # first mask black, then use standard palette for rest
            palette = [(0, 0, 0)] + default_palette[1:]
        else:
            # user-provided: force first entry to be black
            palette = [(0,0,0)] + colors[1:]
    else:
        background_color = (0, 0, 0)
        palette = colors if colors is not None else default_palette

    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    # Prepare background
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:] = background_color

    # Overlay each mask
    for idx, m in enumerate(mask_list):
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

def save_bspline(spline: BSpline, folder: str, filename: str) -> str:
    """
    Saves a scipy.interpolate.BSpline to a .npz file.

    Parameters:
        spline    (BSpline):  The spline object to save.
        folder    (str)     :  Target folder path.
        filename  (str)     :  Base filename (with or without '.npz').

    Returns:
        str: Full path of the written .npz file.
    """
    # Ensure folder exists
    os.makedirs(folder, exist_ok=True)

    # Guarantee .npz extension
    if not filename.endswith('.npz'):
        filename = filename + '.npz'

    path = os.path.join(folder, filename)
    # Save knot vector (t), coefficients (c), and degree (k)
    np.savez(path, t=spline.t, c=spline.c, k=spline.k)
    return path

def save_misc_params(scale: float,
                     shift: float,
                     optimization_time_translate: list,
                     optimization_time_coarse: list,
                     optimization_time_fine: list,
                     optimization_cost_translate, optimization_cost_coarse, optimization_cost_fine, 
                     grasp_success: bool,
                     folder: str,
                     filename: str) -> str:
    """
    Save run parameters/timings to a JSON file.

    Returns
    -------
    str
        Full path to the saved JSON file.
    """
    # Ensure folder exists
    os.makedirs(folder, exist_ok=True)

    # Normalize filename to end with .json
    out_name = filename if filename.lower().endswith(".json") else f"{filename}.json"
    out_path = os.path.join(folder, out_name)

    # Convert lists (possibly numpy types) to plain Python floats
    def to_float_list(x):
        return np.asarray(x, dtype=float).tolist()

    payload = {
        "scale": float(scale),
        "shift": float(shift),
        "optimization_time": {
            "translate": to_float_list(optimization_time_translate),
            "coarse":    to_float_list(optimization_time_coarse),
            "fine":      to_float_list(optimization_time_fine),
        },
        "optimization_cost": {
            "translate": to_float_list(optimization_cost_translate),
            "coarse":    to_float_list(optimization_cost_coarse),
            "fine":      to_float_list(optimization_cost_fine),
        },
        "grasp_success": bool(grasp_success),
    }

    # Write JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path

def load_bspline(folder: str, filename: str) -> BSpline:
    """
    Loads a scipy.interpolate.BSpline from a .npz file.

    Parameters:
        folder   (str): Folder where the .npz is stored.
        filename (str): Filename (with or without '.npz').

    Returns:
        BSpline: Reconstructed spline object.
    """
    # Guarantee .npz extension
    if not filename.endswith('.npz'):
        filename = filename + '.npz'

    path = os.path.join(folder, filename)
    data = np.load(path)
    # t and c are arrays, k is stored as a scalar
    t = data['t']
    c = data['c']
    k = int(data['k'])
    return BSpline(t, c, k)

def save_pose_stamped(pose: PoseStamped, folder: str, filename: str) -> str:
    """
    Serializes a geometry_msgs/PoseStamped to disk using pickle.

    Parameters:
        pose     (PoseStamped): The PoseStamped message to save.
        folder    (str)       : Destination folder path.
        filename  (str)       : Base filename (with or without '.pkl').

    Returns:
        str: Full path of the written .pkl file.
    """
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Guarantee .pkl extension
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'

    path = os.path.join(folder, filename)
    # Serialize with pickle
    with open(path, 'wb') as f:
        pickle.dump(pose, f)
    return path

def load_pose_stamped(folder: str, filename: str) -> PoseStamped:
    """
    Loads a geometry_msgs/PoseStamped from a pickle file.

    Parameters:
        folder   (str): Folder where the .pkl is stored.
        filename (str): Filename (with or without '.pkl').

    Returns:
        PoseStamped: The deserialized PoseStamped message.
    """
    # Guarantee .pkl extension
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'

    path = os.path.join(folder, filename)
    with open(path, 'rb') as f:
        pose = pickle.load(f)
    return pose

import open3d as o3d
import numpy as np
import os
from scipy.interpolate import BSpline

import open3d as o3d
import numpy as np
import os
from scipy.interpolate import BSpline

def save_pointcloud_snapshots(pcd: o3d.geometry.PointCloud, folder: str, filename: str,
                               width: int = 800, height: int = 600,
                               point_size: float = 3.0,
                               spline: BSpline = None,
                               spline_color: tuple = (1.0, 0.0, 0.0),
                               spline_line_width: float = 10.0):
    """
    Saves snapshots of a 3D point cloud (and optional B-spline) from multiple views as PNG images.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to render.
        folder (str): Directory where images will be saved.
        filename (str): Base filename (without extension).
        width (int): Image width.
        height (int): Image height.
        point_size (float): Point size used for rendering.
        spline (BSpline, optional): Optional 3D BSpline (scipy).
        spline_color (tuple): RGB color for the spline.
        spline_line_width (float): Width of the rendered spline line.
    """
    os.makedirs(folder, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # white background

    # Point cloud material
    pc_mat = o3d.visualization.rendering.MaterialRecord()
    pc_mat.shader = "defaultUnlit"
    pc_mat.point_size = point_size

    scene.clear_geometry()
    scene.add_geometry("pcd", pcd, pc_mat)

    # Add spline if provided
    if spline is not None:
        u_samples = np.linspace(spline.t[spline.k], spline.t[-spline.k-1], 200)

        points = spline(u_samples)

        # Ensure shape is (N, 3) for Open3D
        if points.shape[0] == 3:
            points = points.T  # convert from (3, N) to (N, 3)

        points = np.ascontiguousarray(points, dtype=np.float64)

        spline_line = o3d.geometry.LineSet()
        spline_line.points = o3d.utility.Vector3dVector(points)
        lines = [[i, i+1] for i in range(len(points)-1)]
        spline_line.lines = o3d.utility.Vector2iVector(lines)

        # Color all segments the same
        spline_line.colors = o3d.utility.Vector3dVector([spline_color] * len(lines))

        spline_mat = o3d.visualization.rendering.MaterialRecord()
        spline_mat.shader = "unlitLine"
        spline_mat.line_width = spline_line_width
        scene.add_geometry("spline", spline_line, spline_mat)

    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    diameter = np.linalg.norm(bbox.get_extent())
    distance = 1.5 * diameter

    camera_views = {
        "front": [0, 0, 1],
        "top": [0, 1, 0],
        "side": [1, 0, 0],
        "iso": [1, 1, 1],
    }

    for view_name, eye in camera_views.items():
        eye_vector = np.array(eye, dtype=float)
        eye_vector = eye_vector / np.linalg.norm(eye_vector) * distance
        eye_position = center + eye_vector

        renderer.setup_camera(60.0, center, eye_position, [0, 1, 0])
        image = renderer.render_to_image()
        save_path = os.path.join(folder, f"{filename}_{view_name}.png")
        o3d.io.write_image(save_path, image)
        print(f"Saved: {save_path}")








if __name__ == "__main__":
    experiment_timestamp_str = '2025_08_11_15-44'

    experiment_folder = '/root/workspace/images/thesis_images/'
    b_spline_folder = experiment_folder + experiment_timestamp_str + '/spline_fine'
    pose_folder = experiment_folder + experiment_timestamp_str + '/camera_pose'
    depth_folder = experiment_folder + experiment_timestamp_str + '/depth_orig'

    # b_spline = load_bspline(b_spline_folder, '4')
    b_spline = load_bspline(experiment_folder + experiment_timestamp_str, 'initial_spline')
    camera_pose = load_pose_stamped(pose_folder, '4')
    
    projected_spline_cam2_fine = project_bspline(b_spline, camera_pose, camera_parameters)
    # show_masks([projected_spline_cam2_fine])

    depth_orig = load_numpy_from_file(depth_folder, '0')
    
    show_depth_map(depth_orig)






