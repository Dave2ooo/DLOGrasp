import cv2, numpy as np
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d
import rospy
import copy

from geometry_msgs.msg import Pose, PoseStamped
from save_data import *
from my_utils import show_masks
from bspline_from_voxel import *

import pickle


import json

from pathlib import Path

from voxel_carving import *
from scipy.interpolate import BSpline, interp1d
from scipy.spatial import cKDTree
import ast

from scipy.interpolate import make_lsq_spline

from guided_bspline_from_voxels import (
    pick_guides_open3d, fit_bspline_guided, GuidedParams, GuideWeights
)

from skimage.morphology import skeletonize
from tf.transformations import euler_from_quaternion

camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)

def _dict_from_string(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not found. Install opencv-contrib-python.")
    try:
        return getattr(cv2.aruco, dict_name)
    except AttributeError:
        raise ValueError(f"Unknown ArUco dictionary '{dict_name}'. Example: 'DICT_5X5_1000'.")

def _quat_from_R(R: np.ndarray):
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q  # x, y, z, w

def get_camera_pose_from_aruco_marker(
    image: np.ndarray,
    camera_parameters: tuple,
    marker_length_m: float,
    dict: str = "DICT_5X5_1000",
    dist_coeffs: np.ndarray | None = None,
    frame_id: str = "marker",
    show: bool = False,
    window_title: str = "ArUco Pose",
) -> PoseStamped:
    """
    Returns the CAMERA pose in the MARKER frame as PoseStamped.
    If show=True, displays the image with the chosen marker and axes overlaid.
    """
    fx, fy, cx, cy = camera_parameters
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

    if image is None or image.size == 0:
        raise ValueError("Empty image passed to get_camera_pose_from_aruco_marker().")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not found. Install opencv-contrib-python.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dict_id = _dict_from_string(dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") \
             else cv2.aruco.DetectorParameters_create()

    # Detect (prefer new class API)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _rej = detector.detectMarkers(gray)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    else:
        raise RuntimeError("This OpenCV build lacks both ArUco detection APIs.")

    if ids is None or len(ids) == 0:
        # raise RuntimeError("No ArUco markers detected in the image.")
        return None

    # Pick the largest marker by image area
    areas = [cv2.contourArea(c.reshape(-1, 1, 2)) for c in corners]
    i_best = int(np.argmax(areas))
    c_best = corners[i_best]
    id_best = int(ids[i_best])

    # Pose estimation
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([c_best], marker_length_m, K, dist)
        rvec = rvecs[0]
        tvec = tvecs[0]
    else:
        L = marker_length_m
        objp = np.array([
            [-L/2,  L/2, 0.0],
            [ L/2,  L/2, 0.0],
            [ L/2, -L/2, 0.0],
            [-L/2, -L/2, 0.0],
        ], dtype=np.float64)
        imgp = c_best.reshape(-1, 2).astype(np.float64)
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, np.zeros((5,1)) if dist is None else dist, flags=flag)
        if not ok:
            raise RuntimeError("solvePnP failed for the detected marker.")

    # Marker -> Camera
    R_mc, _ = cv2.Rodrigues(rvec)
    t_mc = tvec.reshape(3, 1)

    # Camera -> Marker (pose we return)
    R_cm = R_mc.T
    t_cm = -R_cm @ t_mc
    qx, qy, qz, qw = _quat_from_R(R_cm)

    # Visualize if requested
    if show:
        img_show = image.copy()
        # Draw marker border/ID
        cv2.aruco.drawDetectedMarkers(img_show, [c_best], np.array([[id_best]], dtype=np.int32))
        # Draw axes (use zeros if no dist)
        dist_for_axes = np.zeros((5, 1)) if dist is None else dist
        cv2.drawFrameAxes(img_show, K, dist_for_axes, rvec, tvec, marker_length_m * 0.5)
        # Display (as requested)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 700, 550)
        cv2.imshow(window_title, img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Build PoseStamped
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    # (Leave header.stamp default unless you're in a ROS node)
    pose.pose.position.x = float(t_cm[0])
    pose.pose.position.y = float(t_cm[1])
    pose.pose.position.z = float(t_cm[2])
    pose.pose.orientation.x = float(qx)
    pose.pose.orientation.y = float(qy)
    pose.pose.orientation.z = float(qz)
    pose.pose.orientation.w = float(qw)
    return pose

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q if n == 0.0 else q / n

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    # q = [x, y, z, w]
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + y1*z2 - z1*y2 + x1*w2,
        -x1*z2 + w1*y2 + z1*x2 + y1*w2,
        x1*y2 - y1*x2 + w1*z2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)

def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v by unit quaternion q."""
    q = _quat_normalize(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    q_conj = _quat_conjugate(q)
    r = _quat_multiply(_quat_multiply(q, vq), q_conj)
    return r[:3]

def invert_pose(p: Pose) -> Pose:
    """
    Invert a geometry_msgs/Pose.
    If p encodes T_A_B (frame B pose in frame A),
    returns T_B_A (frame A pose in frame B).
    """
    # Original orientation and position
    q = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], dtype=np.float64)
    t = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float64)

    q = _quat_normalize(q)
    q_inv = _quat_conjugate(q)          # R^{-1} = R^T corresponds to q^{-1} = conjugate for unit quats
    t_inv = -_quat_rotate(q_inv, t)     # t' = -R^{-1} t

    p_out = Pose()
    p_out.position.x, p_out.position.y, p_out.position.z = map(float, t_inv)
    p_out.orientation.x, p_out.orientation.y, p_out.orientation.z, p_out.orientation.w = map(float, q_inv)
    return p_out

def invert_pose_stamped(pose_st: PoseStamped, new_frame_id: str | None = None) -> PoseStamped:
    """
    Invert a geometry_msgs/PoseStamped.

    Example: if pose_st is the CAMERA pose in MARKER frame (T_marker_camera),
    the result is the MARKER pose in CAMERA frame (T_camera_marker).
    """
    out = PoseStamped()
    out.header.stamp = pose_st.header.stamp
    out.header.frame_id = new_frame_id if new_frame_id is not None else (pose_st.header.frame_id or "") + "_inv"
    out.pose = invert_pose(pose_st.pose)
    return out

def compose_pose_stamped(
    handcam_in_map: PoseStamped,
    marker_in_handcam: PoseStamped,
    out_frame_id: str = "map",
    child_frame_id: str = "marker",
) -> PoseStamped:
    """
    Compute marker pose in map frame:
      T_map_marker = T_map_handcam ∘ T_handcam_marker

    Args:
        handcam_in_map: PoseStamped for the hand camera in the map frame (T_map_handcam)
        marker_in_handcam: PoseStamped for the marker in the hand camera frame (T_handcam_marker)
        out_frame_id: frame_id to assign to the output PoseStamped (default 'map')
        child_frame_id: not set in PoseStamped (PoseStamped has only header.frame_id), but kept for clarity

    Returns:
        PoseStamped for the marker in the map frame.
    """
    # Extract quaternions as [x,y,z,w]
    q_map_cam = np.array([
        handcam_in_map.pose.orientation.x,
        handcam_in_map.pose.orientation.y,
        handcam_in_map.pose.orientation.z,
        handcam_in_map.pose.orientation.w,
    ], dtype=np.float64)
    q_map_cam = _quat_normalize(q_map_cam)

    q_cam_marker = np.array([
        marker_in_handcam.pose.orientation.x,
        marker_in_handcam.pose.orientation.y,
        marker_in_handcam.pose.orientation.z,
        marker_in_handcam.pose.orientation.w,
    ], dtype=np.float64)
    q_cam_marker = _quat_normalize(q_cam_marker)

    # Compose rotations: R_map_marker = R_map_cam * R_cam_marker  →  q_map_marker = q_map_cam ⊗ q_cam_marker
    q_map_marker = _quat_multiply(q_map_cam, q_cam_marker)
    q_map_marker = _quat_normalize(q_map_marker)

    # Compose translations: t_map_marker = t_map_cam + R_map_cam * t_cam_marker
    t_map_cam = np.array([
        handcam_in_map.pose.position.x,
        handcam_in_map.pose.position.y,
        handcam_in_map.pose.position.z,
    ], dtype=np.float64)

    t_cam_marker = np.array([
        marker_in_handcam.pose.position.x,
        marker_in_handcam.pose.position.y,
        marker_in_handcam.pose.position.z,
    ], dtype=np.float64)

    t_map_marker = t_map_cam + _quat_rotate(q_map_cam, t_cam_marker)

    # Build output PoseStamped
    out = PoseStamped()
    out.header.stamp = handcam_in_map.header.stamp  # keep upstream timestamp; change if you need something else
    out.header.frame_id = out_frame_id

    out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, t_map_marker)
    out.pose.orientation.x = float(q_map_marker[0])
    out.pose.orientation.y = float(q_map_marker[1])
    out.pose.orientation.z = float(q_map_marker[2])
    out.pose.orientation.w = float(q_map_marker[3])
    return out

def load_mask(folder: str, filename: str) -> list[np.ndarray]:
    """
    Load a mask PNG and return it as a list containing one binary ndarray (0/1).

    Parameters
    ----------
    folder : str
        Directory containing the PNG.
    filename : str
        File name (with or without .png extension).

    Returns
    -------
    list[np.ndarray]
        [mask] where mask is a 2D array of dtype uint8 with values {0,1}.
    """

    # Build full path (append .png if missing)
    fname = filename if filename.lower().endswith(".png") else f"{filename}.png"
    path = os.path.join(folder, fname)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load mask at: {path}")

    # Convert to grayscale if needed (handles 3- or 4-channel images too)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img

    # Binarize: any nonzero becomes 1
    mask_bin = (gray > 0).astype(np.uint8)

    return mask_bin

def distances_to_reference(rec_spline: BSpline,
                           ref_spline: BSpline,
                           n_samples: int = 200,
                           ref_grid_size: int = 5000) -> np.ndarray:
    """
    Sample the *reconstructed* spline at `n_samples` points that are
    (approximately) equidistant in **arc-length**, then return the
    Euclidean distance from each sampled point to the *closest* point
    on the *reference* spline.

    Parameters
    ----------
    rec_spline : scipy.interpolate.BSpline
        The shorter, reconstructed spline.
    ref_spline : scipy.interpolate.BSpline
        The correct/reference spline.
    n_samples : int, optional
        How many arc-length-equidistant samples to take on the
        reconstructed spline.  Default is 200.
    ref_grid_size : int, optional
        How finely to pre-sample the reference spline for the
        nearest-neighbour search (KD-tree).  Larger → more accurate
        but slower. Default is 5000.

    Returns
    -------
    np.ndarray
        A 1-D array of length `n_samples` containing the distances
        (in the same units as the spline coordinates).
    """
    # ---------- helper: cumulative arc length for a spline ----------
    def _arc_length_parameterisation(spl: BSpline, n: int = 4000):
        # sample the parameter uniformly over its valid range
        t_min = spl.t[spl.k]
        t_max = spl.t[-spl.k-1]
        t_dense = np.linspace(t_min, t_max, n)
        pts = spl(t_dense)
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(dists)))   # cumulative arc-length
        return s, interp1d(s, t_dense, kind='linear')

    # ---------- 1. build arc-length ↔ parameter map for rec_spline ----------
    s_rec,  s_to_u = _arc_length_parameterisation(rec_spline)
    total_len_rec = s_rec[-1]

    # equidistant arc-length positions along the *reconstructed* curve
    s_target = np.linspace(0.0, total_len_rec, n_samples)
    u_samples = s_to_u(s_target)          # parameter values on rec_spline
    rec_pts   = rec_spline(u_samples)     # (n_samples, dim)

    # ---------- 2. pre-sample the reference spline & build KD-tree ----------
    t_min_ref = ref_spline.t[ref_spline.k]
    t_max_ref = ref_spline.t[-ref_spline.k-1]
    t_ref_grid = np.linspace(t_min_ref, t_max_ref, ref_grid_size)
    ref_pts = ref_spline(t_ref_grid)      # (ref_grid_size, dim)

    tree = cKDTree(ref_pts)

    # ---------- 3. query distances ----------
    dists, _ = tree.query(rec_pts, k=1)   # nearest neighbour distances

    return dists

def distances_to_reference_step(rec_spline: BSpline,
                                ref_spline: BSpline,
                                step: float = 0.001,
                                ref_grid_size: int = 10000,
                                min_samples: int = 2):
    """
    Sample the *reconstructed* spline every `step` millimetres (arc-length)
    and return the distance from each sample point to the nearest point on the
    *reference* spline.

    Parameters
    ----------
    rec_spline : BSpline
        Shorter, reconstructed spline.
    ref_spline : BSpline
        Correct/reference spline.
    step : float
        Desired spacing between consecutive samples, in the same units as the
        spline coordinates (e.g. millimetres).
    ref_grid_size : int, optional
        Number of points used to pre-sample the reference spline for the KD-tree.
    min_samples : int, optional
        Guarantee at least this many samples (useful if `step` > total length).

    Returns
    -------
    np.ndarray
        1-D array of distances (len ≥ `min_samples`).
    """

    # ---------- helper: build arc-length ↔ parameter map ----------
    def _arc_length_param(spl: BSpline, n: int = 4000):
        t_min, t_max = spl.t[spl.k], spl.t[-spl.k-1]
        t_dense = np.linspace(t_min, t_max, n)
        pts = spl(t_dense)
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(dists)))      # cumulative length
        return s, interp1d(s, t_dense, kind="linear")

    s_rec, s2u = _arc_length_param(rec_spline)
    L_rec = s_rec[-1]
    # print(f"Reconstructed length: {L_rec:.3f} (same units as control-points)")


    # ---------- arc-length grid: 0, step, 2*step, ... ----------
    if step <= 0:
        raise ValueError("step must be positive")
    n_steps = max(int(np.floor(L_rec / step)) + 1, min_samples)
    s_target = np.linspace(0.0, min(n_steps - 1, L_rec / step) * step, n_steps)

    # parameter values & sample points on reconstructed spline
    u_target = s2u(s_target)
    # print(f"u_target: {u_target}")
    rec_pts = rec_spline(u_target)                         # (n_steps, dim)

    # ---------- KD-tree on densely sampled reference spline ----------
    t_min, t_max = ref_spline.t[ref_spline.k], ref_spline.t[-ref_spline.k-1]
    t_ref = np.linspace(t_min, t_max, ref_grid_size)
    ref_pts = ref_spline(t_ref)
    tree = cKDTree(ref_pts)

    # nearest-neighbour distances
    dists, _ = tree.query(rec_pts, k=1)
    return dists


def save_array(array, folder, save_name, delimiter=","):
    """
    Save a 1-D or 2-D NumPy array as <folder>/<save_name>.csv.

    Parameters
    ----------
    array : array-like
        Data to save (e.g. the distances returned by `distances_to_reference`).
    folder : str or Path
        Target directory. It is created if it does not already exist.
    save_name : str
        Desired base-name *without* extension. “distances” → distances.csv
    delimiter : str, optional
        Field separator written to the file. Default is ','.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    file_path = folder / f"{save_name}.csv"
    # savetxt writes plain text, one row per line
    np.savetxt(file_path, np.asarray(array), delimiter=delimiter)
    # (Optional) return the full path for convenience
    return file_path


def load_array(folder, file_name, delimiter=","):
    """
    Load the array saved by `save_array`.

    Parameters
    ----------
    folder : str or Path
        Directory that contains the file.
    file_name : str
        Base-name *without* extension (same string you passed as `save_name`).
    delimiter : str, optional
        Field separator used when the file was written. Default is ','.

    Returns
    -------
    np.ndarray
        The array stored in <folder>/<file_name>.csv.
    """
    folder = Path(folder)
    file_path = folder / f"{file_name}.csv"
    return np.loadtxt(file_path, delimiter=delimiter)

def save_grasp_error(diff, dist, folder, filename="grasp_error"):
    """
    Save the grasp error vector and distance as a JSON file.

    Args:
        folder (str): Directory to save the file.
        filename (str): Filename without .json extension.
        diff (np.ndarray or list): (3,) vector (x, y, z error)
        dist (float): Euclidean error
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename + ".json")
    data = {
        "diff": [float(x) for x in diff],
        "dist": float(dist)
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def calculate_statistics(distances, ignore_nan=True):
    """
    Return the five-number summary (Q0–Q4) of a 1-D array.

    Parameters
    ----------
    distances : array-like
        The sample of distances.
    ignore_nan : bool, default True
        If True, NaNs are ignored.  If False and NaNs are present,
        the result will contain NaN.

    Returns
    -------
    q0, q1, q2, q3, q4 : float
        Minimum, 25th-percentile (Q1), median (Q2),
        75th-percentile (Q3), maximum.
    """
    dist = np.asarray(distances).ravel()
    if ignore_nan:
        dist = dist[~np.isnan(dist)]

    q0 = np.min(dist)
    q1 = np.percentile(dist, 25)
    q2 = np.median(dist)
    q3 = np.percentile(dist, 75)
    q4 = np.max(dist)
    return q0, q1, q2, q3, q4

def calculate_grasp_distance(bs_world, grasp_point, num_samples=10000):
    """
    Returns (xyz_diff, euclidean_dist) between grasp_point and the closest point on the sampled B-spline.

    Args:
        bs_world: scipy.interpolate.BSpline object (3D)
        grasp_point: np.ndarray, shape (3,)
        num_samples: Number of points to sample on the spline

    Returns:
        diff: np.ndarray (3,) = grasp_point - closest_spline_point
        dist: float = Euclidean distance
    """
    t0, t1 = bs_world.t[bs_world.k], bs_world.t[-bs_world.k-1]
    t_vals = np.linspace(t0, t1, num_samples)
    points = bs_world(t_vals)  # shape (num_samples, 3)
    # print(f"points: {points}")
    dists = np.linalg.norm(points - grasp_point, axis=1)
    # print(f"dists: {dists}")
    idx = np.argmin(dists)
    closest_pt = points[idx]
    diff = grasp_point - closest_pt
    dist = np.linalg.norm(diff)
    return diff, dist

def fit_guided_bspline_to_voxels(
    occupancy: np.ndarray,
    guides_world: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    num_ctrl_pts: int = 10,
    degree: int = 3,
    smoothing: float = 1e-2,
    guide_weight: float = 10.0,
    sample_frac: float = 0.1,
    random_seed: int = 42
) -> BSpline:
    """
    Fit a cubic B-spline to a set of occupied voxels, guided by user-selected world points.
    Endpoints are clamped to first and last guide.
    """
    # --- 1. Sample points from occupancy ---
    occ_idx = np.argwhere(occupancy)
    n_vox = occ_idx.shape[0]
    # Subsample voxels for efficiency
    n_samples = max(int(n_vox * sample_frac), 1)
    rng = np.random.default_rng(random_seed)
    sampled_idx = occ_idx[rng.choice(n_vox, size=n_samples, replace=n_samples>n_vox)]
    sampled_world = sampled_idx * voxel_size + origin[None,:]  # (n_samples, 3)
    
    # --- 2. Combine with guides ---
    X = np.vstack([sampled_world, guides_world])
    N = X.shape[0]
    # Guides go at the end (for weighting, easier logic)

    # --- 3. Parametrize all points with t in [0,1] (arc-length) ---
    # We'll sort by closest-point assignment to the polyline defined by guides.
    from scipy.spatial import cKDTree

    # Assign each point in X to the nearest segment on the guides polyline
    def project_onto_polyline(points, polyline):
        # For each point, find closest point on piecewise-linear path
        segs = np.stack([polyline[:-1], polyline[1:]], axis=1)  # (M-1,2,3)
        seg_vec = segs[:,1] - segs[:,0]  # (M-1,3)
        seg_len = np.linalg.norm(seg_vec, axis=1)
        seg_vec_norm = seg_vec / (seg_len[:,None] + 1e-10)
        # Compute projections
        closest_t = np.zeros(len(points))
        for i, p in enumerate(points):
            proj = []
            for j, (a,b) in enumerate(segs):
                v = b-a
                t = np.dot(p-a, v) / (np.dot(v,v) + 1e-12)
                t_clamped = np.clip(t, 0, 1)
                closest = a + t_clamped*v
                d = np.linalg.norm(p-closest)
                proj.append((d, j, t_clamped))
            dmin, jmin, tmin = min(proj, key=lambda tup: tup[0])
            # Compute normalized u along [0,1]
            seg_start = np.sum(seg_len[:jmin])
            u = (seg_start + tmin*seg_len[jmin]) / np.sum(seg_len)
            closest_t[i] = u
        return closest_t

    t_X = project_onto_polyline(X, guides_world)

    # --- 4. Prepare knot vector ---
    # Clamped at ends, evenly spaced in [0,1]
    k = degree
    n = num_ctrl_pts
    # knots: (n + k + 1,)
    knots = np.concatenate((
        np.zeros(k), 
        np.linspace(0, 1, n - k + 1),
        np.ones(k)
    ))

    # --- 5. Weights ---
    weights = np.ones(N)
    weights[-len(guides_world):] = guide_weight
    # Clamp endpoints with huge weight
    weights[-len(guides_world)] = 1e8  # First guide = start
    weights[-1] = 1e8                  # Last guide = end

    # --- 6. Least-squares fit ---
    # Add smoothing by augmenting weights (not exact, but common hack)
    weights = weights / (1 + smoothing)

    # Must sort t_X, X, weights by t_X for make_lsq_spline!
    idx_sort = np.argsort(t_X)
    t_X_sorted = t_X[idx_sort]
    X_sorted = X[idx_sort]
    weights_sorted = weights[idx_sort]

    # Fit separately for x,y,z (vector-valued spline)
    spl = make_lsq_spline(
        t_X_sorted, X_sorted, t=knots, k=k, w=weights_sorted
    )
    return spl

def carve_bspline(image_folder,
                mask_folder,
                pose_folder,
                voxel_folder,
                correct_pose_index,
                index_array = [0, 1, 2, 3, 4, 5, 6],
                center = (10.54, 1.4, 0.4),      # meters, in 'map'
                side_lengths = (1.2, 1, 1),        # meters
                voxel_size = 0.002,               # 2 mm voxels
                tolerance_px = 0
                ):
    #region Correct Camera Poses
    camera_poses_in_map_frame_from_experiment = []
    camera_poses_in_marker_frame = []
    marker_poses_in_camera_frame = []

    camera_poses_in_map_frame_corrected = []

    masks = []


    for i in index_array:
        print(f"i: {i}")
        camera_pose = load_pose_stamped(pose_folder, str(i))
        image = cv2.imread(image_folder + str(i) + '.png')

        camera_pose_in_marker_frame = get_camera_pose_from_aruco_marker(image, camera_parameters, marker_length_m=0.150, show=False, dict="DICT_4X4_1000")
        if camera_pose_in_marker_frame is None:
            print(f"No marker detected in image {i}, skipping...")
            continue
        # camera_pose_in_marker_frame = get_camera_pose_from_aruco_marker(image, camera_parameters, marker_length_m=0.199, show=False, dict="DICT_5X5_1000")
        camera_poses_in_marker_frame.append(camera_pose_in_marker_frame)

        marker_pose_in_camera_frame = invert_pose_stamped(camera_pose_in_marker_frame, 'hand_camera_frame')
        marker_poses_in_camera_frame.append(marker_pose_in_camera_frame)
        
        mask = load_mask(mask_folder, str(i))
        masks.append(mask)
        # show_masks(mask, title=f"Mask {i}")

        camera_poses_in_map_frame_from_experiment.append(camera_pose)

    
    marker_pose_in_map_frame = compose_pose_stamped(camera_poses_in_map_frame_from_experiment[correct_pose_index], marker_poses_in_camera_frame[correct_pose_index])
    # marker_pose_in_map_frame = compose_pose_stamped(camera_poses_in_map_frame_from_experiment[0], marker_poses_in_camera_frame[0])
    print(f"marker_pose_in_map_frame: {marker_pose_in_map_frame}")

    for i, camera_pose_in_marker_frame in enumerate(camera_poses_in_marker_frame):
        camera_pose_in_map_frame_corrected = compose_pose_stamped(marker_pose_in_map_frame, camera_pose_in_marker_frame)
        camera_poses_in_map_frame_corrected.append(camera_pose_in_map_frame_corrected)

        print("-------------------------------------------------------------")
        compare_poses(camera_poses_in_map_frame_from_experiment[i], camera_poses_in_map_frame_corrected[i])
        # print(f"camera_pose_in_map_frame_from_experiment: {camera_poses_in_map_frame_from_experiment[i]}")
        # print(f"camera_pose_in_map_frame_corrected: {camera_poses_in_map_frame_corrected[i]}")



    #region Voxel Carving
    vg = carve_voxels(masks, camera_poses_in_map_frame_corrected, camera_parameters, center, side_lengths, voxel_size, tolerance_px=tolerance_px)
    # vg = carve_voxels(masks, camera_poses_in_map_frame_from_experiment, camera_parameters, center, side_lengths, voxel_size, tolerance_px=tolerance_px)

    save_voxel_grid(vg, voxel_folder, 'vox_ref_orig')
    print(vg.occupancy.shape, vg.origin, vg.voxel_size)

    # experiment_spline = load_bspline(bspline_folder, '4')
    # show_voxel_grid_with_bspline(vg, experiment_spline, num_samples=400, line_radius=0.007)
    #endregion Voxel Carving

def fit_bspline_wrapper(voxel_folder, bspline_folder):
    vg = load_voxel_grid(voxel_folder, 'vox_ref_mod', 'vox_ref_orig')
    experiment_spline = load_bspline(bspline_folder, '4')

    #region Fit B-spline new
    guides_idx, guides_world = pick_guides_open3d(vg)

    bs_world = fit_guided_bspline_to_voxels(
        occupancy=vg.occupancy,
        guides_world=guides_world,
        voxel_size=vg.voxel_size,
        origin=vg.origin,
        num_ctrl_pts=10,
        degree=3,
        smoothing=1e-2,
        guide_weight=10.0,
    )

    save_bspline(bs_world, experiment_folder, "spline_ref_final")
    #endregion Fit B-spline new

    show_voxel_grid_with_bspline(vg, bs_world, num_samples=400, line_radius=0.007)
    show_bsplines([experiment_spline, bs_world], num_samples=400, line_radius=0.007, show_axes=True)

def calculate_distances(experiment_folder, bspline_folder):
    experiment_spline = load_bspline(bspline_folder, '4')
    bs_world = load_bspline(experiment_folder, 'spline_ref_final')  # bs_world

    # distances = distances_to_reference(experiment_spline, bs_world, n_samples=50)
    distances = distances_to_reference_step(experiment_spline, bs_world)
    save_array(distances, experiment_folder, "distances")
    print(f"Num distances: {len(distances)}")

    statistics = calculate_statistics(distances)
    print(f"min: {statistics[0]:.3f}, q1: {statistics[1]:.3f}, median: {statistics[2]:.3f}, q3: {statistics[3]:.3f}, max: {statistics[4]:.3f}")


    grasp_point = load_json(experiment_folder, key="grasp_point")
    grasp_point = np.array(ast.literal_eval(grasp_point))  # convert string to numpy array
    grasp_point[2] -= 0.083 # Compensate for gripper offset
    diffs, dists = calculate_grasp_distance(bs_world, grasp_point)
    print(f"Grasp Point error: diffs: {diffs}, dists: {dists}")
    save_grasp_error(diffs, dists, experiment_folder)


def show_carved_bspline(voxel_folder, bspline_folder):
    vg = load_voxel_grid(voxel_folder, 'vox_ref_orig', 'vox_ref_orig')
    # vg = load_voxel_grid(voxel_folder, 'vox_ref_mod', 'vox_ref_orig')
    experiment_spline = load_bspline(bspline_folder, '4')  
    # bs_world = load_bspline(experiment_folder, 'spline_ref_final')  # bs_world 

    # show_voxel_grid_with_bspline(vg, bs_world, curve_color=(0, 0, 1), num_samples=400, line_radius=0.007)
    show_voxel_grid_with_bspline(vg, experiment_spline, num_samples=400, line_radius=0.007)

def show_both_splines(experiment_folder, bspline_folder):
    experiment_spline = load_bspline(bspline_folder, '4')
    bs_world = load_bspline(experiment_folder, 'spline_ref_final')  # bs_world 
    show_bsplines([experiment_spline, bs_world], num_samples=400, line_radius=0.007, show_axes=True)

def show_depth_with_skeleton(
    depth_map: np.ndarray,
    skeleton: np.ndarray,
    *,
    skeleton_color: tuple[int, int, int] = (255, 255, 255),  # BGR for red
    skeleton_alpha: float = 1.0,
    win_name: str = "Depth + Skeleton",
    wait_ms: int = 0,
    save_path: str | None = None,
) -> None:
    """
    Display a depth map with an overlaid skeleton using OpenCV.

    Zero-depth pixels are shown in white.

    Parameters
    ----------
    depth_map : np.ndarray
        2-D depth array.
    skeleton : np.ndarray
        Boolean/binary array (same shape) from skimage.morphology.skeletonize.
    skeleton_color : tuple, optional
        BGR color for the skeleton. Default is red.
    skeleton_alpha : float, optional
        Overlay opacity [0–1]. Default 1 = fully opaque.
    win_name : str, optional
        Window title.
    wait_ms : int, optional
        cv2.waitKey delay; 0 waits indefinitely.
    save_path : str | None, optional
        If given, saves the result to this path.
    """
    if depth_map.shape != skeleton.shape:
        raise ValueError("`depth_map` and `skeleton` must have identical shapes.")

    # --- normalise depth (ignoring zeros) ---
    depth_f32 = depth_map.astype(np.float32)
    non_zero_mask = depth_f32 > 0

    depth_norm = np.zeros_like(depth_f32)
    if np.any(non_zero_mask):
        d_min, d_max = depth_f32[non_zero_mask].min(), depth_f32[non_zero_mask].max()
        if d_max > d_min:
            depth_norm[non_zero_mask] = 255 * (depth_f32[non_zero_mask] - d_min) / (d_max - d_min)

    depth_u8 = depth_norm.astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)

    # Set zero-depth pixels to white
    depth_bgr[~non_zero_mask] = (255, 255, 255)

    # --- build skeleton overlay ---
    overlay = np.zeros_like(depth_bgr, dtype=np.uint8)
    overlay[skeleton.astype(bool)] = skeleton_color

    blended = cv2.addWeighted(overlay, skeleton_alpha, depth_bgr, 1.0, 0)

    # --- display & optional save ---
    cv2.imshow(win_name, blended)
    if save_path is not None:
        cv2.imwrite(save_path, blended)
    cv2.waitKey(wait_ms)
    cv2.destroyWindow(win_name)

def compare_poses(pose1, pose2):
    """
    Compare two geometry_msgs/PoseStamped poses and print the difference in position and orientation (Euler angles).
    """
    # Extract positions
    p1 = pose1.pose.position
    p2 = pose2.pose.position
    pos1 = np.array([p1.x, p1.y, p1.z])
    pos2 = np.array([p2.x, p2.y, p2.z])

    diff_pos = np.abs(pos1 - pos2)
    dist_pos = np.linalg.norm(pos1 - pos2)

    print("Position differences (abs):")
    print(f"  x: {diff_pos[0]:.6f}")
    print(f"  y: {diff_pos[1]:.6f}")
    print(f"  z: {diff_pos[2]:.6f}")
    print(f"  Euclidean distance: {dist_pos:.6f} m")

    # Extract orientations (quaternions)
    q1 = pose1.pose.orientation
    q2 = pose2.pose.orientation
    quat1 = [q1.x, q1.y, q1.z, q1.w]
    quat2 = [q2.x, q2.y, q2.z, q2.w]

    # Convert to Euler angles (in radians)
    euler1 = np.array(euler_from_quaternion(quat1))  # roll, pitch, yaw
    euler2 = np.array(euler_from_quaternion(quat2))

    # Convert to degrees
    euler1_deg = np.degrees(euler1)
    euler2_deg = np.degrees(euler2)
    diff_euler_deg = np.abs(euler1_deg - euler2_deg)

    print("Orientation differences (Euler angles, abs):")
    print(f"  Roll : {diff_euler_deg[0]:.3f}°")
    print(f"  Pitch: {diff_euler_deg[1]:.3f}°")
    print(f"  Yaw  : {diff_euler_deg[2]:.3f}°")

# Example usage:
# compare_poses(pose1, pose2)



if __name__ == "__main__":
    # rospy.init_node("experiment_results", anonymous=True)

    # Cable Black
    # experiment_timestamp_str = '2025_08_27_11-39'
    # experiment_timestamp_str = '2025_08_27_12-29'
    # experiment_timestamp_str = '2025_08_27_13-03'
    # experiment_timestamp_str = '2025_08_27_13-10'
    experiment_timestamp_str = '2025_08_27_13-35'
    # experiment_timestamp_str = '2025_08_27_13-40'
    # experiment_timestamp_str = '2025_08_27_13-47'
    # experiment_timestamp_str = '2025_08_27_13-53'
    # experiment_timestamp_str = '2025_08_27_14-05'
    experiment_folder = '/root/workspace/images/experiment_images/cable/black/' + experiment_timestamp_str

    # Cable Tablecloth
    # experiment_folder = '/root/workspace/images/experiment_images/cable/tablecloth/' + experiment_timestamp_str
    # experiment_timestamp_str = '2025_08_29_09-55'
    # experiment_timestamp_str = '2025_08_29_10-57'
    # experiment_timestamp_str = '2025_08_29_11-02'
    # experiment_timestamp_str = '2025_08_29_11-10'
    # experiment_timestamp_str = '2025_09_03_08-36'
    # experiment_timestamp_str = '2025_09_03_08-39'

    # Cable Wood
    # experiment_folder = '/root/workspace/images/experiment_images/cable/wood/' + experiment_timestamp_str
    # experiment_timestamp_str = '2025_08_27_14-26'
    # experiment_timestamp_str = '2025_08_27_14-44'
    # experiment_timestamp_str = '2025_08_27_14-49'
    # experiment_timestamp_str = '2025_08_29_09-33'
    # experiment_timestamp_str = '2025_09_03_08-31'
    # experiment_timestamp_str = '2025_09_03_08-31'

    # Single Tube Black
    # experiment_folder = '/root/workspace/images/experiment_images/tube_single/black/' + experiment_timestamp_str
    # experiment_timestamp_str = '2025_09_01_10-59'
    # experiment_timestamp_str = '2025_09_01_11-04'
    # experiment_timestamp_str = '2025_09_01_11-12'
    # experiment_timestamp_str = '2025_09_01_11-18'
    # experiment_timestamp_str = '2025_09_01_11-23'
    # experiment_timestamp_str = '2025_09_01_11-27'

    # Single Tube Wood
    # experiment_folder = '/root/workspace/images/experiment_images/tube_single/wood/' + experiment_timestamp_str
    # experiment_timestamp_str = '2025_09_01_09-27'
    # experiment_timestamp_str = '2025_09_01_10-09'
    # experiment_timestamp_str = '2025_09_01_10-27'
    # experiment_timestamp_str = '2025_09_01_10-34'
    # experiment_timestamp_str = '2025_09_01_10-39'
    # experiment_timestamp_str = '2025_09_01_10-45'
    # experiment_timestamp_str = '2025_09_01_10-52'


    # Double Tube Black
    # experiment_timestamp_str = '2025_09_01_11-36'
    # experiment_timestamp_str = '2025_09_01_11-40'
    # experiment_timestamp_str = '2025_09_01_11-45'
    # # experiment_timestamp_str = '2025_09_01_12-03'
    # experiment_timestamp_str = '2025_09_01_12-10'
    # experiment_folder = '/root/workspace/images/experiment_images/tube_double/black/' + experiment_timestamp_str

    # Double Tube Wood
    # experiment_timestamp_str = '2025_09_01_12-17'
    # experiment_timestamp_str = '2025_09_01_12-26'
    # experiment_timestamp_str = '2025_09_01_12-30'
    # experiment_timestamp_str = '2025_09_01_12-34'
    # experiment_timestamp_str = '2025_09_01_12-39'
    # experiment_timestamp_str = '2025_09_01_12-43'
    # experiment_folder = '/root/workspace/images/experiment_images/tube_double/wood/' + experiment_timestamp_str

    # experiment_timestamp_str = '2025_08_04_11-17'
    # experiment_folder = '/root/workspace/images/thesis_images/' + experiment_timestamp_str

    pose_folder = experiment_folder + '/camera_pose'
    image_folder = experiment_folder + '/image/'

    mask_folder = experiment_folder + '/mask_cv2'
    # mask_folder = experiment_folder + '/mask_correct'

    bspline_folder = experiment_folder + '/spline_fine'

    voxel_folder = experiment_folder



    index_array=[4,1,5,6] 
    voxel_size = 0.002
    # voxel_size = 0.005
    carve_bspline(image_folder, mask_folder, pose_folder, voxel_folder, 0, index_array=index_array, tolerance_px=0)
    show_carved_bspline(voxel_folder, bspline_folder)


    # fit_bspline_wrapper(voxel_folder, bspline_folder)
    # calculate_distances(experiment_folder, bspline_folder)

    # show_both_splines(experiment_folder, bspline_folder)






    # depth_map_folder = experiment_folder + '/depth_orig'
    # depth_map = load_numpy_from_file(depth_map_folder, '1')
    # mask = load_mask(mask_folder, '1')
    # skeleton = skeletonize(mask.astype(bool))

    # # show_masks([mask, skeleton])
    # show_depth_map(depth_map[100:400, 200:300])

    # show_depth_with_skeleton(depth_map, skeleton)

    # width = 640
    # height = 480
    # left = 260
    # right = 130
    # top = 150
    # bottom = 280

    # save_depth_map(depth_map[top: width-bottom, left: height-right], depth_map_folder, 'cropped')