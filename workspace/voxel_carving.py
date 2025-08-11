import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    # For optional mask dilation/erosion
    import cv2
except ImportError:
    cv2 = None

try:
    from geometry_msgs.msg import PoseStamped
except Exception:
    # Type hints only; code works as long as objects expose .pose.position/.pose.orientation
    PoseStamped = object


@dataclass
class NumpyVoxelGrid:
    """
    Axis-aligned voxel grid in the 'map' frame.
    - occupancy[i, j, k] corresponds to the voxel centered at:
        origin + [ (i+0.5)*voxel_size, (j+0.5)*voxel_size, (k+0.5)*voxel_size ]
      with axes aligned to map X,Y,Z respectively.
    """
    occupancy: np.ndarray          # shape (Nx, Ny, Nz), dtype=bool
    origin: np.ndarray             # (3,), min corner of grid in map frame
    voxel_size: float              # meters

    def indices_to_points(self, idx: np.ndarray) -> np.ndarray:
        """idx: (...,3) integer indices (i,j,k) -> (...,3) points (x,y,z) in map."""
        idx = np.asarray(idx, dtype=np.int64)
        return self.origin + (idx + 0.5) * self.voxel_size

    def points_to_indices(self, pts: np.ndarray) -> np.ndarray:
        """pts: (...,3) points (x,y,z) in map -> (...,3) integer indices (i,j,k)."""
        rel = (np.asarray(pts) - self.origin) / self.voxel_size - 0.5
        return np.floor(rel + 1e-9).astype(np.int64)

    def save(self, path: str) -> None:
        np.savez_compressed(path, occupancy=self.occupancy, origin=self.origin, voxel_size=self.voxel_size)

    @staticmethod
    def load(path: str) -> "NumpyVoxelGrid":
        data = np.load(path, allow_pickle=False)
        return NumpyVoxelGrid(
            occupancy=data["occupancy"].astype(bool),
            origin=data["origin"].astype(float),
            voxel_size=float(data["voxel_size"])
        )

    def to_open3d(self):
        """Optional: convert to Open3D VoxelGrid (requires open3d)."""
        import open3d as o3d
        vg = o3d.geometry.VoxelGrid()
        vg.voxel_size = float(self.voxel_size)
        occ = np.argwhere(self.occupancy)  # (M,3) indices
        # Build voxel centers in map frame
        centers = self.indices_to_points(occ).astype(np.float64)
        voxels = [o3d.geometry.Voxel(o) for o in occ.astype(np.int64)]
        vg.voxels = voxels
        vg.origin = self.origin.astype(np.float64)
        # Colors are optional; skip for speed
        return vg


def _quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix. Returns R that maps camera->map if given q_m_c."""
    # Normalize to be safe
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.dot(q, q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q *= 1.0 / np.sqrt(n)
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R


def _pose_to_Rcm_tmap(pose: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PoseStamped (camera pose in map) to:
      R_c_m: rotation mapping map->camera
      t_m:   camera origin in map coords
    """
    p = pose.pose.position
    o = pose.pose.orientation
    R_m_c = _quat_to_rotmat(o.x, o.y, o.z, o.w)  # camera->map
    t_m = np.array([p.x, p.y, p.z], dtype=np.float64)
    R_c_m = R_m_c.T  # inverse
    return R_c_m, t_m


def _prepare_masks(masks: List[np.ndarray], tolerance_px: int) -> List[np.ndarray]:
    """
    Ensure binary uint8 masks; apply optional dilation/erosion by |tolerance_px| pixels.
    Positive -> dilate; negative -> erode; zero -> unchanged.
    """
    prepped = []
    use_cv = (cv2 is not None) and (tolerance_px != 0)
    ksz = abs(int(tolerance_px))
    kernel = None
    if use_cv and ksz > 0:
        kernel = np.ones((2*ksz+1, 2*ksz+1), np.uint8)

    for m in masks:
        m_bool = (m.astype(np.uint8) > 0).astype(np.uint8)
        if tolerance_px > 0 and use_cv:
            m_bool = cv2.dilate(m_bool, kernel, iterations=1)
        elif tolerance_px < 0 and use_cv:
            m_bool = cv2.erode(m_bool, kernel, iterations=1)
        prepped.append(m_bool)
    return prepped


def carve_voxels(
    masks: List[np.ndarray],
    poses: List[PoseStamped],
    intrinsics: Tuple[float, float, float, float],
    center_xyz: Tuple[float, float, float],
    side_lengths_xyz: Tuple[float, float, float],
    voxel_size: float,
    tolerance_px: int = 0,
    chunk_points: int = 2_000_000,
) -> NumpyVoxelGrid:
    """
    Shape-from-silhouette voxel carving.

    Args
    ----
    masks : list of (H,W) binary arrays; 1=object, 0=background. All same size.
    poses : list of PoseStamped, each is the camera pose in 'map' (i.e., pose of camera frame expressed in map).
    intrinsics : (fx, fy, cx, cy) for all views.
    center_xyz : (cx, cy, cz) center of the AABB in map.
    side_lengths_xyz : (sx, sy, sz) side lengths (meters) of the AABB in map.
    voxel_size : edge length of a voxel (meters).
    tolerance_px : mask growth/shrink in pixels (>=0 dilates, <=0 erodes, 0 disabled).
    chunk_points : process this many voxel centers per chunk to limit peak RAM.

    Returns
    -------
    NumpyVoxelGrid with occupancy (Nx,Ny,Nz), origin (min corner), and voxel_size.
    """
    assert len(masks) == len(poses) and len(masks) > 0, "masks and poses must be same non-zero length"
    H, W = masks[0].shape
    for m in masks:
        assert m.shape == (H, W), "All masks must have identical shape"

    fx, fy, cx, cy = map(float, intrinsics)
    center = np.asarray(center_xyz, dtype=np.float64)
    sides = np.asarray(side_lengths_xyz, dtype=np.float64)
    assert np.all(sides > 0), "Side lengths must be positive"
    assert voxel_size > 0, "voxel_size must be positive"

    # Grid setup
    mins = center - 0.5 * sides
    Nx, Ny, Nz = np.maximum(1, np.ceil(sides / voxel_size).astype(int))
    # Ensure exact coverage
    xs = mins[0] + (np.arange(Nx) + 0.5) * voxel_size
    ys = mins[1] + (np.arange(Ny) + 0.5) * voxel_size
    zs = mins[2] + (np.arange(Nz) + 0.5) * voxel_size

    # Precompute per-view transforms map->camera
    Rcm_list = []
    tmap_list = []
    for pose in poses:
        R_c_m, t_m = _pose_to_Rcm_tmap(pose)
        Rcm_list.append(R_c_m)
        tmap_list.append(t_m)
    Rcm_list = [np.asarray(R) for R in Rcm_list]
    tmap_list = [np.asarray(t) for t in tmap_list]

    # Prep masks (apply optional tolerance)
    masks_u8 = _prepare_masks(masks, tolerance_px)

    # Occupancy buffer
    occ_flat = np.zeros(Nx * Ny * Nz, dtype=bool)

    # Iterate voxel centers in chunks
    # Create a flat list of voxel centers in map frame
    # Use np.meshgrid with 'ij' so index mapping is (i->x, j->y, k->z)
    Xi, Yj, Zk = np.meshgrid(xs, ys, zs, indexing='ij')  # each shape (Nx,Ny,Nz)
    P_all = np.stack([Xi.ravel(), Yj.ravel(), Zk.ravel()], axis=1)  # (N,3)
    N_total = P_all.shape[0]

    start = 0
    while start < N_total:
        end = min(N_total, start + int(chunk_points))
        P = P_all[start:end]  # (M,3)
        keep = np.ones(P.shape[0], dtype=bool)

        for (R_c_m, t_m, m_u8) in zip(Rcm_list, tmap_list, masks_u8):
            if not np.any(keep):
                break  # early exit for this chunk

            # Transform map->camera: X_cam = R_c_m @ (X_map - t_m)
            X_rel = P - t_m[None, :]                 # (M,3)
            X_cam = (R_c_m @ X_rel.T).T              # (M,3)

            z = X_cam[:, 2]
            valid = z > 0.0
            if not np.any(valid):
                keep[:] = False
                break

            x = X_cam[valid, 0] / z[valid]
            y = X_cam[valid, 1] / z[valid]
            u = fx * x + cx
            v = fy * y + cy

            # Nearest-neighbor lookup into mask
            u_round = np.rint(u).astype(np.int32)
            v_round = np.rint(v).astype(np.int32)

            inside = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)
            n_valid = int(np.count_nonzero(valid))
            ok = np.zeros(n_valid, dtype=bool)

            ok[inside] = (m_u8[v_round[inside], u_round[inside]] > 0)

            # Merge back into full-length boolean
            valid_full = np.zeros_like(keep, dtype=bool)
            valid_full[valid] = ok

            # Carve: must be inside silhouette in this view
            keep &= valid_full

        occ_flat[start:end] = keep
        start = end

    occupancy = occ_flat.reshape(Nx, Ny, Nz)
    return NumpyVoxelGrid(occupancy=occupancy, origin=mins, voxel_size=float(voxel_size))

def show_voxel_grid(vg,
                    backend: str = "auto",     # "open3d", "matplotlib", or "auto"
                    render: str = "voxels",    # "voxels" or "points"
                    max_voxels: int = 300_000, # subsample if more than this are occupied
                    point_size: int = 2):
    """
    Display a NumpyVoxelGrid-like object with attributes:
      - occupancy: (Nx,Ny,Nz) bool
      - origin: (3,) float (min corner in world/map frame)
      - voxel_size: float meters
      - indices_to_points(idx) -> (N,3) world coords (optional; used if present)

    backend="open3d" gives an interactive viewer (preferred).
    If Open3D is missing or construction fails, falls back to "matplotlib".
    """
    occ = np.asarray(vg.occupancy, dtype=bool)
    inds = np.argwhere(occ)  # (M,3) integer indices (i,j,k)

    if inds.size == 0:
        print("Voxel grid is empty (no occupied voxels). Nothing to show.")
        return

    # Subsample if gigantic
    M = inds.shape[0]
    if M > max_voxels:
        sel = np.random.choice(M, size=max_voxels, replace=False)
        inds = inds[sel]
        M = inds.shape[0]

    # Try Open3D unless user forces matplotlib
    if backend in ("auto", "open3d"):
        try:
            import open3d as o3d
            if render == "voxels":
                # Try to build a VoxelGrid directly. If it fails, fall back to points.
                try:
                    vg_o3d = o3d.geometry.VoxelGrid()
                    vg_o3d.voxel_size = float(vg.voxel_size)
                    vg_o3d.origin = np.asarray(vg.origin, dtype=np.float64)
                    vg_o3d.voxels = [o3d.geometry.Voxel(idx.astype(np.int64)) for idx in inds]
                    o3d.visualization.draw_geometries([vg_o3d])
                    return
                except Exception:
                    # Fall through to point rendering
                    render = "points"

            # Points mode (robust)
            if hasattr(vg, "indices_to_points"):
                centers = vg.indices_to_points(inds)
            else:
                centers = np.asarray(vg.origin, float) + (inds + 0.5) * float(vg.voxel_size)

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
            # Optionally estimate normals or set a uniform color
            pcd.paint_uniform_color([0.2, 0.6, 1.0])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            if opt is not None:
                opt.point_size = float(point_size)
                opt.background_color = np.array([1, 1, 1])
            vis.run()
            vis.destroy_window()
            return
        except Exception as e:
            if backend == "open3d":
                raise
            # else fall back to matplotlib

    # Matplotlib fallback (static)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if hasattr(vg, "indices_to_points"):
        centers = vg.indices_to_points(inds)
    else:
        centers = np.asarray(vg.origin, float) + (inds + 0.5) * float(vg.voxel_size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=point_size)
    # Set equal aspect
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f"Occupied voxels: {M}  |  voxel_size={vg.voxel_size:.4f} m")
    plt.show()
