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

from guided_bspline_from_voxels import (
    pick_guides_open3d, fit_bspline_guided, GuidedParams, GuideWeights
)

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
        raise RuntimeError("No ArUco markers detected in the image.")

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


import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

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



if __name__ == "__main__":
    # rospy.init_node("experiment_results", anonymous=True)

    # experiment_timestamp_str = '2025_08_11_11-00'
    # experiment_timestamp_str = '2025_08_11_15-27'
    # experiment_timestamp_str = '2025_08_11_15-44'
    # experiment_timestamp_str = '2025_08_16_08-35'
    # experiment_timestamp_str = '2025_08_19_13-24'
    # experiment_timestamp_str = '2025_08_21_10-50'
    # experiment_timestamp_str = '2025_08_25_09-56'
    # experiment_timestamp_str = '2025_08_27_11-39'
    # experiment_timestamp_str = '2025_08_27_12-29'
    # experiment_timestamp_str = '2025_08_27_13-03'
    # experiment_timestamp_str = '2025_08_27_13-10'
    # experiment_timestamp_str = '2025_08_27_13-35'
    # experiment_timestamp_str = '2025_08_27_13-40'
    # experiment_timestamp_str = '2025_08_27_13-47'
    # experiment_timestamp_str = '2025_08_27_14-05'
    # experiment_timestamp_str = '2025_08_27_14-37'
    experiment_timestamp_str = '2025_08_27_14-44'

    
    
    # items = (0, 4)
    items = [0, 1, 2, 3, 4, 5, 6]
    # items = (6)
    correct_pose_index = 4


    experiment_folder = '/root/workspace/images/thesis_images/' + experiment_timestamp_str
    pose_folder = experiment_folder + '/camera_pose'
    image_folder = experiment_folder + '/image/'

    mask_folder = experiment_folder + '/mask_cv2'
    # mask_folder = experiment_folder + '/mask_correct'

    bspline_folder = experiment_folder + '/spline_fine'

    #region Correct Camera Poses
    camera_poses_in_map_frame_from_experiment = []
    camera_poses_in_marker_frame = []
    marker_poses_in_camera_frame = []

    camera_poses_in_map_frame_corrected = []

    for i in items:
        print(f"i: {i}")
        camera_pose = load_pose_stamped(pose_folder, str(i))
        camera_poses_in_map_frame_from_experiment.append(camera_pose)
        image = cv2.imread(image_folder + str(i) + '.png')

        camera_pose_in_marker_frame = get_camera_pose_from_aruco_marker(image, camera_parameters, marker_length_m=0.150, show=True, dict="DICT_4X4_1000")
        # camera_pose_in_marker_frame = get_camera_pose_from_aruco_marker(image, camera_parameters, marker_length_m=0.199, show=False, dict="DICT_5X5_1000")
        camera_poses_in_marker_frame.append(camera_pose_in_marker_frame)

        marker_pose_in_camera_frame = invert_pose_stamped(camera_pose_in_marker_frame, 'hand_camera_frame')
        marker_poses_in_camera_frame.append(marker_pose_in_camera_frame)

    
    marker_pose_in_map_frame = compose_pose_stamped(camera_poses_in_map_frame_from_experiment[correct_pose_index], marker_poses_in_camera_frame[correct_pose_index])
    # marker_pose_in_map_frame = compose_pose_stamped(camera_poses_in_map_frame_from_experiment[0], marker_poses_in_camera_frame[0])
    print(f"marker_pose_in_map_frame: {marker_pose_in_map_frame}")

    for i, camera_pose_in_marker_frame in enumerate(camera_poses_in_marker_frame):
        camera_pose_in_map_frame_corrected = compose_pose_stamped(marker_pose_in_map_frame, camera_pose_in_marker_frame)
        camera_poses_in_map_frame_corrected.append(camera_pose_in_map_frame_corrected)

        # print("-------------------------------------------------------------")
        # print(f"camera_pose_in_map_frame_from_experiment: {camera_poses_in_map_frame_from_experiment[i]}")
        # print(f"camera_pose_in_map_frame_corrected: {camera_pose_in_map_frame_corrected}")
    #endregion Correct Camera Poses

    # All camera poses are corrected: camera_poses_in_map_frame_corrected


    #region Load Masks
    masks = []

    for i in items:
        mask = load_mask(mask_folder, str(i))
        masks.append(mask)
        # show_masks(mask, title=f"Mask {i}")
    #endregion Load Masks

    #region Voxel Carving
    # center = (0.54, 1.4, 0.4)           # meters, in 'map'
    center = (10.54, 1.4, 0.4)           # meters, in 'map'
    side_lengths = (1, 1, 1)        # meters
    voxel_size = 0.002                 # 5 mm voxels

    vg = carve_voxels(masks, camera_poses_in_map_frame_corrected, camera_parameters, center, side_lengths, voxel_size, tolerance_px=0)
    # vg = carve_voxels(masks, camera_poses_in_map_frame_from_experiment, camera_parameters, center, side_lengths, voxel_size, tolerance_px=0)

    print(vg.occupancy.shape, vg.origin, vg.voxel_size)





    # show_voxel_grid(vg, backend="open3d", render="voxels")   # interactive cubes (fast enough up to a few 100k)
    # show_voxel_grid_solid_voxels(vg)
    #endregion Voxel Carving


    #region B-spline
    spline = load_bspline(bspline_folder, '4')
    show_voxel_grid_with_bspline(vg, spline, num_samples=400, line_radius=0.007)
    # show_voxel_grid_with_bspline(vg, spline, num_samples=400, line_radius=0.007)
    #endregion B-spline

    #region Fti B-spline
    # L_target_m   = 1 # meter
    # L_target_vox = L_target_m / voxel_size  # 1500 for 2 mm voxels

    # params = FitParams(
    #     # Field & A*
    #     sigma_vox=1.8, dilate_iter=1, use_dt=True,
    #     a_star_eps=0.02, a_star_p_floor=0.08,
    #     a_star_goal_radius_vox=2.0, a_star_margin_vox=28,
    #     a_star_pow=3.0, a_star_w_occ=3.0,
    #     # Waypoints
    #     n_waypoints=5, waypoint_min_sep_vox=12,
    #     # Smoother input polyline
    #     resample_step_vox=2.0,          # ← coarser
    #     spline_smooth=5e-1,             # ← stronger pre-fit smoothing
    #     # Refinement
    #     refine=True, refine_u_samples=600, refine_tau_inside=0.06,
    #     fix_endpoints=True,
    #     refine_weights=RefinementWeights(
    #         alpha=0.5,   # rely a bit less on P
    #         beta=2.0,    # keep mostly in corridor
    #         gamma=5e-3,  # ← stronger smoothness
    #         delta=1e-3,  # keep length reasonable
    #         eta=300.0,   # keep endpoints pinned
    #         zeta=5.0     # moderate voxel pull to avoid zig-zag
    #     ),
    #     length_prior=L_target_vox
    # )
    # result = fit_with_manual_endpoints(vg, params)  # or fit_bspline_from_numpy_voxel_grid(vg, params, endpoints_zyx=...)



    # # params = FitParams(
    # #     endpoint_strategy="pca",
    # #     sigma_vox=3.0,          # more blur to bridge bigger gaps
    # #     pick_tau=0.05,          # include weaker field for endpoints
    # #     a_star_eps=0.02,        # cheaper to traverse low-P zones
    # #     resample_step_vox=1.0,
    # #     refine=True,
    # #     refine_tau_inside=0.08, # don't over-penalize low P during bridging
    # #     refine_u_samples=500,
    # #     fix_endpoints=False,    # allow curve to grow to meet length
    # #     refine_weights=RefinementWeights(
    # #         alpha=1.0, beta=4.0,   # softer outside penalty
    # #         gamma=5e-3,            # keep it smooth
    # #         delta=1e-3,            # <-- enable length prior
    # #         eta=15.0               # soft pins only (since not fixing endpoints)
    # #     ),
    # #     length_prior=L_target_vox
    # # )

    # # result = fit_bspline_from_numpy_voxel_grid(vg, params=params)

    # # vg is your NumpyVoxelGrid-like object
    # # result = fit_bspline_from_numpy_voxel_grid(vg, params=FitParams(
    # #     sigma_vox=1.5,         # Gaussian (in voxels), ~ tube radius
    # #     use_dt=True,           # multiply by distance transform (centers the field)
    # #     resample_step_vox=1.5, # path resampling in voxels
    # #     spline_degree=3,
    # #     spline_smooth=1e-3,
    # #     refine=True,           # turn off if you just want the initial fit
    # #     refine_u_samples=300,
    # #     refine_tau_inside=0.15 # treat P<tau as “outside”
    # # ))
    # bs_world = result.bs_world  # SciPy BSpline in (x,y,z)
    # bs_vox = result.bs_vox  # SciPy BSpline in (x,y,z)
    # u0, u1 = result.u_domain
    # pts_xyz = bs_world(np.linspace(u0, u1, 200))  # sample for visualization

    # debug_report(vg, result)
    #endregion Fit B-spline

    #region Fit B-spline new
    # 1) Click an ordered set of guide points (Shift+LMB, then Q)
    guides_idx, guides_world = pick_guides_open3d(vg)

    # 2) Fit a guided spline
    # Work sometimes and very shooth
    # params = GuidedParams(
    #     sigma_vox=2.0, dilate_iter=1, use_dt=True,
    #     a_star_eps=0.02, a_star_p_floor=0.08, a_star_goal_radius_vox=2.0,
    #     a_star_margin_vox=28, a_star_pow=3.0, a_star_w_occ=4.0,
    #     resample_step_vox=2, spline_degree=3, spline_smooth=1e1, # 1e1,
    #     refine_u_samples=900, refine_tau_inside=0.1,
    #     weights=GuideWeights(alpha=0.4, beta=2.0, gamma=5e-1, zeta=5.0, kappa=30.0, delta=1e-3, eta=200.0),
    #     fix_endpoints=True,
    #     length_prior=None  # or set to expected length in voxels
    # )






    # params = GuidedParams(
    #     sigma_vox=2.0,
    #     dilate_iter=1, 
    #     use_dt=True,
    #     a_star_eps=0.02, 
    #     a_star_p_floor=0.08, 
    #     a_star_goal_radius_vox=2.0, 
    #     a_star_margin_vox=28, 
    #     a_star_pow=3.0, 
    #     a_star_w_occ=4.0, 
    #     resample_step_vox=1.2, 
    #     spline_degree=3, 
    #     spline_smooth=1e0, # 1e1, 
    #     refine_u_samples=600, 
    #     refine_tau_inside=0.05, 
    #     weights=GuideWeights(alpha=0.6, beta=2.0, gamma=5e-3, zeta=5.0, kappa=20.0, delta=1e-3, eta=200.0), 
    #     fix_endpoints=True, 
    #     length_prior=None # or set to expected length in voxels
    # )

    # res = fit_bspline_guided(vg, guides_idx, params)
    # bs_world = res.bs_world




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


    save_bspline(bs_world, experiment_folder, "spline_reference")
    #endregion Fit B-spline new

    distances = distances_to_reference(spline, bs_world, n_samples=50)
    print(f"distances: {distances}")

    show_voxel_grid_with_bspline(vg, bs_world, num_samples=400, line_radius=0.007)

    show_bsplines([spline, bs_world], num_samples=400, line_radius=0.007, show_axes=True)


    save_array(distances, experiment_folder, "distances")

    statistics = calculate_statistics(distances)
    print(f"min: {statistics[0]:.3f}, q1: {statistics[1]:.3f}, median: {statistics[2]:.3f}, q3: {statistics[3]:.3f}, max: {statistics[4]:.3f}")