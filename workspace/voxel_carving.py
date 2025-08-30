import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


import os
import json
import struct

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

def show_voxel_grid_solid_voxels(vg):
    """
    Render a solid (cubic) voxel grid in Open3D, compatible with CUDA builds.
    vg must have: occupancy (Nx,Ny,Nz bool), origin (3,), voxel_size (float)
    """
    import open3d as o3d

    occ = np.asarray(vg.occupancy, dtype=bool)
    occ_idx = np.argwhere(occ)  # (M,3) integer indices (i,j,k)
    if occ_idx.size == 0:
        print("Empty grid.")
        return

    # Voxel centers in map frame
    centers = np.asarray(vg.origin, float) + (occ_idx + 0.5) * float(vg.voxel_size)

    # Build a point cloud at voxel centers
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    # Exact bounds of your grid (so alignment matches your occupancy)
    min_bound = np.asarray(vg.origin, float)
    max_bound = min_bound + np.array(occ.shape, float) * float(vg.voxel_size)

    # Create a voxel grid from those points within your bounds
    grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=float(vg.voxel_size),
        min_bound=min_bound,
        max_bound=max_bound,
    )

    o3d.visualization.draw_geometries([grid])

def show_voxel_grid_with_bspline(
    vg,
    spline,                           # scipy.interpolate.BSpline, vector-valued -> R^3
    num_samples: int = 300,           # samples along the curve
    line_radius: float = None,        # None -> thin polyline; float (meters) -> tube of this radius
    voxel_color=(0.75, 0.75, 0.75),   # RGB in [0,1]
    curve_color=(1, 0, 0)       # RGB in [0,1]
):
    """
    Render a solid voxel grid and a 3D B-spline curve in the SAME 'map' frame.

    Requirements:
    - Open3D installed
    - 'spline' is a vector-valued BSpline returning (N,3) for vector inputs.
      (i.e., coef has last dim 3; evaluating on a 1D array yields shape (N,3))

    Notes:
    - The curve is sampled uniformly in the spline parameter domain [t[k], t[-k-1]].
      If you need arc-length uniformity, say so and we can add it.
    """
    import open3d as o3d
    from numpy.linalg import norm

    # --- Build solid voxel grid from occupied centers (CUDA-safe path) ---
    occ = np.asarray(vg.occupancy, dtype=bool)
    occ_idx = np.argwhere(occ)
    if occ_idx.size == 0:
        print("Empty grid.")
        return

    centers = np.asarray(vg.origin, float) + (occ_idx + 0.5) * float(vg.voxel_size)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    pcd.paint_uniform_color(voxel_color)

    min_bound = np.asarray(vg.origin, float)
    max_bound = min_bound + np.array(occ.shape, float) * float(vg.voxel_size)

    grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size=float(vg.voxel_size), min_bound=min_bound, max_bound=max_bound
    )

    # --- Sample the spline in its valid domain ---
    # SciPy BSpline valid domain = [t[k], t[-k-1]]
    t = np.asarray(spline.t, dtype=float)
    k = int(spline.k)
    u0, u1 = t[k], t[-k-1]
    u = np.linspace(u0, u1, int(num_samples))
    C = spline(u)  # expect shape (num_samples, 3)
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("BSpline must evaluate to Nx3 points. Got shape: %r" % (C.shape,))

    # --- Make curve geometry ---
    geoms = [grid]

    if line_radius is None or line_radius <= 0.0:
        # Fast polyline via LineSet
        lines = np.column_stack([np.arange(len(C)-1), np.arange(1, len(C))]).astype(np.int32)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(C),
            lines=o3d.utility.Vector2iVector(lines)
        )
        colors = np.tile(curve_color, (lines.shape[0], 1))
        ls.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(ls)
    else:
        # Pretty tube: chain of short cylinders
        def _cylinder_between(p0, p1, radius, radial=24):
            v = p1 - p0
            L = float(norm(v))
            if L < 1e-12:
                return None
            # Cylinder is along +Z by default
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=radial)
            cyl.compute_vertex_normals()
            # Rotate +Z to direction v
            z = np.array([0.0, 0.0, 1.0], dtype=float)
            d = v / L
            # Handle parallel/anti-parallel robustly
            c = float(np.dot(z, d))
            if c > 0.999999:
                R = np.eye(3)
            elif c < -0.999999:
                # 180° around any axis orthogonal to z; pick x-axis
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
            else:
                axis = np.cross(z, d)
                axis /= norm(axis)
                angle = float(np.arccos(c))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cyl.rotate(R, center=(0, 0, 0))
            # Move center to midpoint
            mid = (p0 + p1) * 0.5
            cyl.translate(mid)
            return cyl

        tubes = []
        for i in range(len(C) - 1):
            seg = _cylinder_between(C[i], C[i+1], line_radius)
            if seg is not None:
                tubes.append(seg)
        if not tubes:
            print("Curve collapsed to a point; nothing to draw.")
        else:
            mesh = tubes[0]
            for tmesh in tubes[1:]:
                mesh += tmesh
            mesh.paint_uniform_color(curve_color)
            geoms.append(mesh)

    # --- Show ---
    o3d.visualization.draw_geometries(geoms)


def show_bsplines(
    splines,
    num_samples: int = 300,
    line_radius: float | None = None,
    colors=None,
    show_axes: bool = True,
    tube_resolution: int = 24
):
    import numpy as np
    import open3d as o3d
    from numpy.linalg import norm

    def _ensure_color_list(n, colors):
        default_palette = [ # (0.10, 0.55, 0.85)
            (0.90, 0.20, 0.10), (0.0, 0.0, 1.0), (0.10, 0.75, 0.35),
            (0.95, 0.70, 0.10), (0.60, 0.30, 0.90), (0.20, 0.80, 0.80),
            (0.80, 0.40, 0.40), (0.40, 0.80, 0.40), (0.40, 0.40, 0.80),
            (0.70, 0.70, 0.70),
        ]
        if colors is None:
            return [default_palette[i % len(default_palette)] for i in range(n)]
        colors = list(colors)
        if len(colors) == 3 and all(isinstance(c, (int, float)) for c in colors):
            return [tuple(float(c) for c in colors)] * n
        return [tuple(float(c) for c in colors[i % len(colors)]) for i in range(n)]

    def _sample_points_3d(bs, N):
        t = np.asarray(bs.t, float); k = int(bs.k)
        u0, u1 = t[k], t[-k-1]
        u = np.linspace(u0, u1, int(N))
        P = np.asarray(bs(u), float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"BSpline must evaluate to (N,3). Got {P.shape=}")
        return P

    def _tube_segment(p0, p1, radius, radial):
        v = p1 - p0; L = float(norm(v))
        if L < 1e-12: return None
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=radial)
        mesh.compute_vertex_normals()
        z = np.array([0.0, 0.0, 1.0], float); d = v / L
        c = float(np.dot(z, d))
        if c > 0.999999:
            R = np.eye(3)
        elif c < -0.999999:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        else:
            axis = np.cross(z, d); axis /= norm(axis); angle = float(np.arccos(c))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh.rotate(R, center=(0, 0, 0))
        mesh.translate((p0 + p1) * 0.5)
        return mesh

    splines = list(splines)
    if not splines:
        print("show_bsplines: nothing to draw.")
        return

    curve_colors = _ensure_color_list(len(splines), colors)
    geoms = []
    all_sampled_points = []

    if show_axes:
        # Scale the axes later once we know the scene size
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geoms.append(axes)

    # Build geometry + collect points for centering
    for bs, col in zip(splines, curve_colors):
        C = _sample_points_3d(bs, num_samples)
        all_sampled_points.append(C)

        if not line_radius or line_radius <= 0.0:
            if C.shape[0] >= 2:
                lines = np.column_stack([np.arange(len(C)-1), np.arange(1, len(C))]).astype(np.int32)
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(C),
                    lines=o3d.utility.Vector2iVector(lines)
                )
                ls.colors = o3d.utility.Vector3dVector(np.tile(col, (lines.shape[0], 1)))
                geoms.append(ls)
        else:
            segs = []
            for i in range(C.shape[0] - 1):
                m = _tube_segment(C[i], C[i+1], float(line_radius), tube_resolution)
                if m is not None: segs.append(m)
            if segs:
                mesh = segs[0]
                for m in segs[1:]: mesh += m
                mesh.paint_uniform_color(col)
                geoms.append(mesh)

    if not geoms:
        print("show_bsplines: nothing valid to draw.")
        return

    # ---- Auto-center (and heuristic auto-zoom) ------------------------------
    import numpy as np
    pts = np.vstack(all_sampled_points) if all_sampled_points else np.zeros((1, 3))
    min_xyz = pts.min(axis=0); max_xyz = pts.max(axis=0)
    center = 0.5 * (min_xyz + max_xyz)
    extent = max(max_xyz - min_xyz)  # scalar “scene size”
    extent = float(extent if extent > 1e-9 else 1.0)

    # Scale the axes to ~10% of the scene size (if present)
    if show_axes:
        geoms[0].scale(extent * 0.1 / 0.1, center=[0, 0, 0])  # original size=0.1

    # Choose a stable view: look from a diagonal above, Z-up
    up = [0.0, 0.0, 1.0]
    front = np.array([0.6, -0.5, -1.0], float)
    front = (front / np.linalg.norm(front)).tolist()

    # Heuristic zoom based on scene size (works well across scales)
    if extent > 10.0:
        zoom = 0.15
    elif extent > 5.0:
        zoom = 0.22
    elif extent > 2.0:
        zoom = 0.30
    elif extent > 1.0:
        zoom = 0.45
    else:
        zoom = 0.70

    # ---- Render with explicit camera parameters -----------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="BSplines", width=1280, height=800, visible=True)
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer()

    ctr = vis.get_view_control()
    # If your Open3D version supports it, set a field of view (optional):
    # try: ctr.change_field_of_view(step=-5.0) except: pass
    ctr.set_lookat(center.tolist())
    ctr.set_up(up)
    ctr.set_front(front)
    ctr.set_zoom(zoom)

    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------

# import os, json, numpy as np, struct
# from dataclasses import dataclass
# from typing import Tuple

# # ---------------------------------------------------------------------
# # Low-level Binvox RLE helpers
# # ---------------------------------------------------------------------

# def _write_binvox(occ: np.ndarray,
#                   path: str,
#                   translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
#                   scale: float = 1.0):
#     """
#     Save a boolean (Nx,Ny,Nz) array to `path` in Binvox 1 format.
#     translate, scale follow the Binvox spec (they’re optional for most viewers).
#     """
#     Nx, Ny, Nz = occ.shape
#     if max(Nx, Ny, Nz) > 1024:
#         raise ValueError("Binvox spec limits each dimension to 1024.")

#     header = (
#         "#binvox 1\n"
#         f"dim {Nx} {Ny} {Nz}\n"
#         f"translate {translate[0]} {translate[1]} {translate[2]}\n"
#         f"scale {scale}\n"
#         "data\n"
#     ).encode()

#     flat = occ.flatten(order="C").astype(np.uint8)  # x-fastest, then y, then z
#     # Run-length encode (value,count) pairs, each a single byte
#     with open(path, "wb") as f:
#         f.write(header)
#         val = flat[0]
#         count = 0
#         for v in flat:
#             if v == val and count < 255:
#                 count += 1
#             else:
#                 f.write(bytes([val, count]))
#                 val, count = v, 1
#         f.write(bytes([val, count]))  # final run


# def _read_binvox(path: str):
#     """Return (occ ndarray bool, translate tuple3, scale float)."""
#     with open(path, "rb") as f:
#         # --- header ---
#         if f.readline().strip() != b"#binvox 1":
#             raise ValueError("Not a binvox v1 file")
#         dims = list(map(int, f.readline().split()[1:]))
#         Nx, Ny, Nz = dims
#         translate = tuple(map(float, f.readline().split()[1:]))
#         scale = float(f.readline().split()[1])
#         assert f.readline().strip() == b"data"

#         # --- data ---
#         data = np.frombuffer(f.read(), dtype=np.uint8)
#         assert data.size % 2 == 0
#         values = data[0::2]
#         counts = data[1::2]

#         flat = np.repeat(values, counts).astype(bool)
#         if flat.size != Nx * Ny * Nz:
#             raise ValueError("Size mismatch in binvox data")
#         occ = flat.reshape((Nx, Ny, Nz), order="C")
#         return occ, translate, scale


# # ---------------------------------------------------------------------
# # Public helpers (match your carve_voxels() grid type)
# # ---------------------------------------------------------------------

# def save_voxel_grid(vg, folder: str, name: str):
#     """
#     Save `vg` (from carve_voxels) as  `folder/name.binvox`
#     plus `folder/name_meta.json` (origin, voxel_size).
#     """
#     os.makedirs(folder, exist_ok=True)
#     base = os.path.join(folder, name)

#     _write_binvox(
#         np.asarray(vg.occupancy, bool),
#         base + ".binvox",
#         translate=(0, 0, 0),            # Binvox viewers ignore these for editing
#         scale=1.0
#     )
#     meta = {
#         "origin": list(map(float, vg.origin)),
#         "voxel_size": float(vg.voxel_size)
#     }
#     with open(base + "_meta.json", "w") as jf:
#         json.dump(meta, jf, indent=2)

#     print(f"Saved {base+'.binvox'}   dims={vg.occupancy.shape}  occ={int(vg.occupancy.sum())}")


# def load_voxel_grid(folder: str, name: str):
#     """
#     Load `folder/name.binvox` (and optional _meta.json) → NumpyVoxelGrid
#     identical in spirit to carve_voxels() output.
#     """
#     @dataclass(eq=False, slots=True)
#     class NumpyVoxelGrid:
#         occupancy: np.ndarray
#         origin: np.ndarray
#         voxel_size: float
#         def indices_to_points(self, idx):
#             return self.origin + (np.asarray(idx)+0.5)*self.voxel_size

#     base = os.path.join(folder, name)
#     occ, _, _ = _read_binvox(base + ".binvox")

#     meta_path = base + "_meta.json"
#     if os.path.exists(meta_path):
#         with open(meta_path) as jf:
#             meta = json.load(jf)
#         origin = np.asarray(meta["origin"], float)
#         voxel_size = float(meta["voxel_size"])
#     else:
#         origin = np.zeros(3, float)
#         voxel_size = 1.0
#         print("Meta file missing – using origin=0, voxel_size=1 m")

#     return NumpyVoxelGrid(occ, origin, voxel_size)

# -----------------------------------------------------


# import os
# import json
# import struct
# import numpy as np


# def _write_vox_150(occupancy: np.ndarray, path: str, palette_index: int = 1) -> None:
#     """
#     Minimal VOX 150 writer: 1 model, default palette, uniform colour.
#     occupancy : (Nx,Ny,Nz) bool grid (X,Y,Z)  — each axis must be <=255.
#     palette_index : 1-255 (0 is reserved for "empty").
#     """
#     if occupancy.ndim != 3:
#         raise ValueError("occupancy must be 3-D")

#     Nx, Ny, Nz = occupancy.shape
#     if max(Nx, Ny, Nz) > 255:
#         raise ValueError("Each grid dimension must be ≤255 for MagicaVoxel.")

#     # --- Build chunks ---
#     vox_idx = np.argwhere(occupancy)  # [[ix,iy,iz], ...]  ints
#     num_vox = vox_idx.shape[0]

#     # SIZE chunk -------------------------------------------------
#     size_chunk = (
#         b"SIZE" +
#         struct.pack("<II", 12, 0) +            # content bytes, children bytes
#         struct.pack("<III", Nx, Ny, Nz)
#     )

#     # XYZI chunk -------------------------------------------------
#     xyzi_content = struct.pack("<I", num_vox)  # N
#     for ix, iy, iz in vox_idx:
#         xyzi_content += struct.pack("BBBB", ix, iy, iz, palette_index)
#     xyzi_chunk = (
#         b"XYZI" +
#         struct.pack("<II", len(xyzi_content), 0) +
#         xyzi_content
#     )

#     # MAIN chunk -------------------------------------------------
#     main_children = size_chunk + xyzi_chunk
#     main_chunk = (
#         b"MAIN" +
#         struct.pack("<II", 0, len(main_children)) +
#         main_children
#     )

#     # --- Write file ---
#     with open(path, "wb") as f:
#         f.write(b"VOX ")
#         f.write(struct.pack("<I", 150))   # version
#         f.write(main_chunk)


# def _read_vox_150(path: str):
#     """
#     Very small VOX 150 reader that returns (Nx,Ny,Nz), occupied indices ndarray (M,3).
#     No palette/material parsing — only SIZE and XYZI are handled.
#     """
#     with open(path, "rb") as f:
#         if f.read(4) != b"VOX ":
#             raise ValueError("Not a VOX file")
#         version = struct.unpack("<I", f.read(4))[0]
#         if version != 150:
#             raise ValueError(f"Unsupported VOX version {version}")

#         # Consume chunks until we hit SIZE then XYZI
#         Nx = Ny = Nz = None
#         voxels = None

#         while True:
#             chunk_id = f.read(4)
#             if not chunk_id:
#                 break  # EOF
#             content_size, child_size = struct.unpack("<II", f.read(8))
#             content = f.read(content_size)
#             _ = f.read(child_size)  # skip children if any (we don't expect nested in SIZE/XYZI)

#             if chunk_id == b"SIZE":
#                 Nx, Ny, Nz = struct.unpack("<III", content[:12])
#             elif chunk_id == b"XYZI":
#                 N = struct.unpack("<I", content[:4])[0]
#                 data = content[4:]
#                 voxels = np.frombuffer(data, dtype=np.uint8).reshape(N, 4)[:, :3]  # (N,3) x,y,z

#         if None in (Nx, Ny, Nz) or voxels is None:
#             raise ValueError("SIZE / XYZI chunks missing")

#         return (Nx, Ny, Nz), voxels


# # ----------------------------------------------------------------------
# # Public helpers -------------------------------------------------------
# # ----------------------------------------------------------------------

# def save_voxel_grid(vg, folder: str, name: str) -> None:
#     """
#     Save a `NumpyVoxelGrid` to `folder/name.vox` (MagicaVoxel) and
#     `folder/name_meta.json` (origin + voxel_size).  Creates the folder.
#     """
#     os.makedirs(folder, exist_ok=True)
#     base = os.path.join(folder, name)

#     # 1) .vox file
#     _write_vox_150(np.asarray(vg.occupancy, bool), base + ".vox")

#     # 2) side-car metadata
#     meta = {"origin": list(map(float, vg.origin)), "voxel_size": float(vg.voxel_size)}
#     with open(base + "_meta.json", "w") as jf:
#         json.dump(meta, jf, indent=2)

#     print(f"Saved {base+'.vox'}  (occupied={int(vg.occupancy.sum())})")


# def load_voxel_grid(folder: str, name: str):
#     """
#     Load `folder/name.vox` (+ optional _meta.json) and return a fresh
#     `NumpyVoxelGrid` identical in layout to `carve_voxels` output.
#     If the meta file is missing, origin=(0,0,0) and voxel_size=1 m.
#     """
#     from dataclasses import dataclass

#     @dataclass(eq=False, slots=True)
#     class NumpyVoxelGrid:
#         occupancy: np.ndarray   # (Nx,Ny,Nz) bool
#         origin: np.ndarray      # (3,) float
#         voxel_size: float

#         def indices_to_points(self, idx):
#             idx = np.asarray(idx, dtype=np.int64)
#             return self.origin + (idx + 0.5) * self.voxel_size

#     base = os.path.join(folder, name)

#     # 1) VOX data
#     (Nx, Ny, Nz), voxels = _read_vox_150(base + ".vox")
#     occ = np.zeros((Nx, Ny, Nz), dtype=bool)
#     occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True

#     # 2) metadata
#     meta_path = base + "_meta.json"
#     if os.path.exists(meta_path):
#         with open(meta_path) as jf:
#             meta = json.load(jf)
#         origin = np.asarray(meta.get("origin", [0.0, 0.0, 0.0]), dtype=float)
#         voxel_size = float(meta.get("voxel_size", 1.0))
#     else:
#         origin = np.zeros(3, dtype=float)
#         voxel_size = 1.0
#         print("Meta file missing – origin defaulted to (0,0,0), voxel_size=1 m.")

#     return NumpyVoxelGrid(occ, origin, voxel_size)



import os, json, numpy as np
from dataclasses import dataclass

# ───────────────────────────────────────────────────────────
# Save  →  folder/name.txt   (Goxel text) + side-car metadata
# Load  ←  same              → NumpyVoxelGrid
# ───────────────────────────────────────────────────────────

def save_voxel_grid(vg, folder: str, name: str,
                    default_hex: str = "ffffff",
                    add_header: bool = True) -> None:
    """
    • vg            : the NumpyVoxelGrid returned by carve_voxels()
    • folder/name   : target path (folder is auto-created)
    • default_hex   : colour for every voxel (RRGGBB).  Goxel ignores alpha.
    • add_header    : writes '# Goxel …' comment line for clarity (optional)
    """

    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, name)

    coords = np.argwhere(vg.occupancy)             # (N,3) integer indices
    hexcol = default_hex.lower()

    with open(base + ".txt", "w") as f:
        if add_header:
            f.write("# Goxel text format — one line per voxel: X Y Z RRGGBB\n")
        for i, j, k in coords:
            f.write(f"{i} {j} {k} {hexcol}\n")

    # keep real-world placement in a tiny JSON:
    meta = {"origin": list(map(float, vg.origin)),
            "voxel_size": float(vg.voxel_size)}
    with open(base + "_meta.json", "w") as jf:
        json.dump(meta, jf)

    print(f"Saved {base+'.txt'}   voxels={len(coords)}")

# ───────────────────────────────────────────────────────────

def load_voxel_grid(folder: str, name: str, name_orig: str):
    """
    Read folder/name.txt  (+ optional _meta.json) → fresh NumpyVoxelGrid.
    """

    @dataclass(eq=False, slots=True)
    class NumpyVoxelGrid:
        occupancy: np.ndarray
        origin: np.ndarray
        voxel_size: float
        # helper for map-space coordinates
        def indices_to_points(self, idx):
            idx = np.asarray(idx, int)
            return self.origin + (idx + 0.5) * self.voxel_size

    base = os.path.join(folder, name)
    base_orig = os.path.join(folder, name_orig)

    # read only the first 4 columns; comments (‘# …’) are ignored by np.loadtxt
    coords = np.loadtxt(base + ".txt",
                        comments="#",
                        usecols=(0, 1, 2),
                        dtype=np.int64)
    coords = coords.reshape(-1, 3)                 # works for single voxel too

    Nx, Ny, Nz = coords.max(axis=0) + 1
    occ = np.zeros((Nx, Ny, Nz), bool)
    occ[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    # spatial metadata (if present)
    meta_path = base_orig + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as jf:
            meta = json.load(jf)
        origin     = np.asarray(meta["origin"], float)
        voxel_size = float(meta["voxel_size"])
    else:
        origin = np.zeros(3)
        voxel_size = 1.0

    return NumpyVoxelGrid(occ, origin, voxel_size)
