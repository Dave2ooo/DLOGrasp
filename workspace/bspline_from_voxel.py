"""
Fit a 3D B-spline centerline to a (possibly disconnected) voxelized DLO.

Pipeline (gap-robust):
  A) Build a smooth tubularity field P from the binary occupancy.
  B) Find a gap-bridging initial path with A* on a 26-neighbor grid using
     edge costs inversely proportional to P.
  C) Fit a cubic B-spline to the resampled path.
  D) Optionally refine control points by minimizing a continuous loss on P
     with curvature regularization and optional endpoint/length priors.

Deps: numpy, scipy (ndimage, interpolate, optimize)

Expected input: a "NumpyVoxelGrid"-like object with these attributes:
  - occupancy / grid / data : 3D boolean or {0,1} ndarray shaped (Z, Y, X)
  - voxel_size / resolution : float or (sz, sy, sx)
  - origin / world_origin   : (x0, y0, z0) of voxel (0,0,0) in world units
If your object uses different names, edit `extract_voxel_grid()` below.

All internal computations for path and refinement are done in voxel index
coordinates (z,y,x) to simplify interpolation on arrays. The final BSpline
is returned both in voxel coordinates and in world coordinates.
"""
from __future__ import annotations
import math
import heapq
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.interpolate import splprep, BSpline
from scipy.optimize import minimize

# ==== Coordinate convention toggles (set for your NumpyVoxelGrid) ====
# Your grid uses (X, Y, Z) array order and you visualize voxel CENTERS via
# origin + (idx + 0.5) * voxel_size. Keep these True to match that.
ARRAY_ORDER = "xyz"            # "xyz" for (X,Y,Z) arrays, "zyx" for (Z,Y,X)
VOXEL_COORD_IS_CENTER = True    # indices represent voxel centers, not corners


# ------------------------------------------------------------
# Utilities: voxel grid extraction and coordinate transforms
# ------------------------------------------------------------

def extract_voxel_grid(voxel_grid: Any) -> Tuple[np.ndarray, Tuple[float,float,float], Tuple[float,float,float]]:
    """Extract (occupancy_zyx, voxel_size_zyx, origin_xyz) from an input object.

    occupancy_zyx: bool ndarray (Z, Y, X)
    voxel_size_zyx: (sz, sy, sx) in world units per voxel index step
    origin_xyz: (x0, y0, z0) world coords of voxel (0,0,0)
    """
    # Occupancy
    occ = None
    for name in ("occupancy", "grid", "data", "voxels"):
        if hasattr(voxel_grid, name):
            occ = getattr(voxel_grid, name)
            break
    if occ is None:
        raise AttributeError("voxel_grid must have one of: occupancy/grid/data/voxels")
    occ = np.asarray(occ)
    if occ.ndim != 3:
        raise ValueError("Occupancy array must be 3D (Z,Y,X)")
    occ_bool = occ.astype(bool)

    # Voxel size
    vs = None
    for name in ("voxel_size", "resolution", "scale", "voxel_size_zyx"):
        if hasattr(voxel_grid, name):
            vs = getattr(voxel_grid, name)
            break
    if vs is None:
        raise AttributeError("voxel_grid must have voxel_size/resolution/scale")
    vs = np.array(vs, dtype=float).reshape(-1)
    if vs.size == 1:
        sz = sy = sx = float(vs[0])
    elif vs.size == 3:
        # Heuristic: many libs store as (sx, sy, sz). We need (sz, sy, sx).
        # If your class already stores (sz, sy, sx), set attribute name to "voxel_size_zyx"
        sx, sy, sz = float(vs[0]), float(vs[1]), float(vs[2])
    else:
        raise ValueError("voxel_size-like attribute must be scalar or length-3")
    voxel_size_zyx = (sz, sy, sx)

    # Origin
    ori = None
    for name in ("origin", "world_origin", "origin_xyz"):
        if hasattr(voxel_grid, name):
            ori = getattr(voxel_grid, name)
            break
    if ori is None:
        ori = (0.0, 0.0, 0.0)
    origin_xyz = tuple(float(v) for v in ori)

    return occ_bool, voxel_size_zyx, origin_xyz


def vox2world(points_idx: np.ndarray,
              origin_xyz: Tuple[float,float,float],
              voxel_size_zyx: Tuple[float,float,float]) -> np.ndarray:
    """Map array indices -> world.
    Supports array orders:
      - "xyz": points_idx[:,0]->x, [:,1]->y, [:,2]->z  (your NumpyVoxelGrid)
      - "zyx": points_idx[:,0]->z, [:,1]->y, [:,2]->x  (legacy assumption)
    Also supports center vs corner indexing via VOXEL_COORD_IS_CENTER.
    """
    sz, sy, sx = voxel_size_zyx  # note: internal tuple is (sz, sy, sx)
    off = 0.5 if VOXEL_COORD_IS_CENTER else 0.0
    if ARRAY_ORDER == "xyz":
        x = (points_idx[:,0] + off) * sx
        y = (points_idx[:,1] + off) * sy
        z = (points_idx[:,2] + off) * sz
    else:  # "zyx"
        z = (points_idx[:,0] + off) * sz
        y = (points_idx[:,1] + off) * sy
        x = (points_idx[:,2] + off) * sx
    X = np.stack([x + origin_xyz[0], y + origin_xyz[1], z + origin_xyz[2]], axis=1)
    return X


def world2vox(points_xyz: np.ndarray,
              origin_xyz: Tuple[float,float,float],
              voxel_size_zyx: Tuple[float,float,float]) -> np.ndarray:
    """Inverse of vox2world with the same ARRAY_ORDER and VOXEL_COORD_IS_CENTER."""
    sz, sy, sx = voxel_size_zyx
    off = 0.5 if VOXEL_COORD_IS_CENTER else 0.0
    x = (points_xyz[:,0] - origin_xyz[0]) / sx - off
    y = (points_xyz[:,1] - origin_xyz[1]) / sy - off
    z = (points_xyz[:,2] - origin_xyz[2]) / sz - off
    if ARRAY_ORDER == "xyz":
        return np.stack([x, y, z], axis=1)
    else:  # "zyx"
        return np.stack([z, y, x], axis=1)

# ------------------------------------------------------------
# Field building (A)
# ------------------------------------------------------------

def build_tubularity_field(occ_zyx: np.ndarray,
                           sigma_vox: float = 1.5,
                           use_dt: bool = True,
                           dilate_iter: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a smooth attraction/tubularity field P in [0,1] from occupancy.

    - Optionally dilate occupancy to thicken sparse/gappy shapes.
    - Smooth with a Gaussian (sigma ~ tube radius in voxels).
    - Optionally multiply by interior distance transform of the *thickened* mask to emphasize medial axis.

    Returns:
        P: float field in [0,1]
        M_thick: bool mask after dilation (used for distance-to-occupancy)
    """
    M = occ_zyx.astype(bool)
    if dilate_iter and dilate_iter > 0:
        st = generate_binary_structure(3, 2)  # 18-connectivity (close to tubular)
        M = binary_dilation(M, structure=st, iterations=int(dilate_iter))
    P = gaussian_filter(M.astype(float), sigma=float(sigma_vox))
    P /= (P.max() + 1e-12)
    if use_dt:
        DT_in = distance_transform_edt(M)  # distance to boundary *inside* object
        DT_in /= (DT_in.max() + 1e-12)
        P = P * DT_in
        P /= (P.max() + 1e-12)
    return P, M

# ------------------------------------------------------------
# A* path on a 26-neighbor grid (B)
# ------------------------------------------------------------

NBR_26 = np.array([(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)], dtype=int)
NBR_26_STEP = np.linalg.norm(NBR_26.astype(float), axis=1)  # 1, sqrt(2), sqrt(3)


def pick_endpoints_from_field(P: np.ndarray, tau: float = 0.15) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Pick two far-apart endpoints from high-P voxels (fast heuristic).
    Returns (z,y,x) integer tuples.
    """
    mask = P > tau
    if not np.any(mask):
        # fallback: global max and farthest away by Euclidean
        z0,y0,x0 = np.unravel_index(np.argmax(P), P.shape)
        coords = np.stack(np.nonzero(P > 0), axis=1)
    else:
        z0,y0,x0 = np.unravel_index(np.argmax(P * mask), P.shape)
        coords = np.stack(np.nonzero(mask), axis=1)
    if coords.size == 0:
        raise RuntimeError("No candidate voxels found to choose endpoints.")
    # farthest point from (z0,y0,x0)
    dif = coords - np.array([z0,y0,x0])[None,:]
    d2 = np.einsum('ij,ij->i', dif, dif)
    idx = int(np.argmax(d2))
    z1,y1,x1 = coords[idx]
    return (int(z0),int(y0),int(x0)), (int(z1),int(y1),int(x1))


def pick_endpoints_by_pca(occ_zyx: np.ndarray) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Pick endpoints as extremal projections along the first PCA axis of occupied voxels.
    Robust to gaps / disconnected components; works directly on occupancy indices.
    Returns integer (z,y,x) tuples matching the array's index order.
    """
    pts = np.stack(np.nonzero(occ_zyx), axis=1).astype(float)
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough occupied voxels for PCA endpoint selection.")
    mu = pts.mean(axis=0)
    A = pts - mu
    # SVD for principal direction
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[0]
    proj = A @ v
    i0 = int(np.argmin(proj)); i1 = int(np.argmax(proj))
    p0 = pts[i0]; p1 = pts[i1]
    return (int(round(p0[0])), int(round(p0[1])), int(round(p0[2]))), \
           (int(round(p1[0])), int(round(p1[1])), int(round(p1[2])))

# --- Waypoint selection to prevent "straight chord" skipping ---

def _pca_axis_and_proj(occ_zyx: np.ndarray):
    pts = np.stack(np.nonzero(occ_zyx), axis=1).astype(float)
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough occupied voxels for PCA.")
    mu = pts.mean(axis=0)
    A = pts - mu
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[0]
    proj = A @ v
    return pts, v, proj, mu


def waypoints_by_pca_quantiles(occ_zyx: np.ndarray,
                               start_zyx: Tuple[int,int,int],
                               goal_zyx: Tuple[int,int,int],
                               n_waypoints: int = 6,
                               min_sep_vox: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Pick intermediate anchors along the first PCA axis.
    - Project occupied voxels onto PCA axis.
    - Keep anchors at fixed quantiles between the projections of start and goal.
    Returns (anchors_idx (K,3) int), (anchors_proj (K,) float) sorted from start->goal.
    """
    if n_waypoints <= 0:
        return np.empty((0,3), dtype=int), np.empty((0,), dtype=float)
    pts, v, proj, mu = _pca_axis_and_proj(occ_zyx)
    start_proj = (np.asarray(start_zyx, float) - mu) @ v
    goal_proj  = (np.asarray(goal_zyx,  float) - mu) @ v
    a, b = (start_proj, goal_proj) if start_proj <= goal_proj else (goal_proj, start_proj)
    # sort pts by proj
    order = np.argsort(proj)
    pts_sorted = pts[order]
    proj_sorted = proj[order]
    # indices within [a,b]
    mask = (proj_sorted >= a) & (proj_sorted <= b)
    if not np.any(mask):
        return np.empty((0,3), dtype=int), np.empty((0,), dtype=float)
    pts_seg = pts_sorted[mask]
    proj_seg = proj_sorted[mask]
    # target quantiles excluding endpoints
    qs = np.linspace(0.0, 1.0, n_waypoints + 2)[1:-1]
    anchors = []
    anchors_proj = []
    last_pick = -1e9
    for q in qs:
        t = proj_seg[0] + q * (proj_seg[-1] - proj_seg[0])
        idx = int(np.searchsorted(proj_seg, t))
        idx = np.clip(idx, 0, len(proj_seg)-1)
        # enforce separation in index space (approx arc-length along PCA)
        if anchors and abs(idx - last_pick) < min_sep_vox:
            # nudge to meet separation
            idx = min(len(proj_seg)-1, max(0, last_pick + min_sep_vox))
        anchors.append(tuple(int(x) for x in np.round(pts_seg[idx])))
        anchors_proj.append(float(proj_seg[idx]))
        last_pick = idx
    anchors = np.array(anchors, dtype=int)
    anchors_proj = np.array(anchors_proj, dtype=float)
    # ensure order matches start->goal direction
    if start_proj > goal_proj:
        anchors = anchors[::-1]
        anchors_proj = anchors_proj[::-1]
    return anchors, anchors_proj


def a_star_path(P: np.ndarray,
                start_zyx: Tuple[int,int,int],
                goal_zyx: Tuple[int,int,int],
                eps: float = 1e-3,
                p_floor: float = 0.0,
                goal_radius_vox: float = 0.0,
                max_expansions: int = 2_000_000,
                D_occ: Optional[np.ndarray] = None,
                w_occ: float = 0.0,
                p_pow: float = 2.0) -> np.ndarray:
    """A* on 26-neighbor 3D grid with edge cost ~ step / (eps + max(p_floor, avg P)).
    Supports a goal tolerance `goal_radius_vox` (accepts reaching within radius).
    Returns an ordered path of voxel indices (N,3) from start to goal.
    """
    Z,Y,X = P.shape
    start = tuple(int(v) for v in start_zyx)
    goal  = tuple(int(v) for v in goal_zyx)

    def in_bounds(z,y,x):
        return 0 <= z < Z and 0 <= y < Y and 0 <= x < X

    def heuristic(z,y,x):
        dz = goal[0]-z; dy = goal[1]-y; dx = goal[2]-x
        return math.sqrt(dz*dz + dy*dy + dx*dx)

    open_heap = []  # (f, g, (z,y,x))
    came_from: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    gscore = {start: 0.0}
    heapq.heappush(open_heap, (heuristic(*start), 0.0, start))

    visited = set()
    expansions = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        # Goal test with tolerance
        if current == goal:
            pass_goal = True
        else:
            dz = goal[0]-current[0]; dy = goal[1]-current[1]; dx = goal[2]-current[2]
            pass_goal = (goal_radius_vox > 0.0 and (dz*dz + dy*dy + dx*dx) <= goal_radius_vox*goal_radius_vox)
        if pass_goal:
            # reconstruct
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return np.array(path, dtype=float)

        z,y,x = current
        pcur = P[z,y,x]
        # neighbors
        for (ddz,ddy,ddx), step_len in zip(NBR_26, NBR_26_STEP):
            nz,ny,nx = z+int(ddz), y+int(ddy), x+int(ddx)
            if not in_bounds(nz,ny,nx):
                continue
            pnb = P[nz,ny,nx]
            pav = 0.5*(pcur + pnb)
            pav = max(pav, p_floor)
            cost_p = 1.0 / ((eps + pav) ** p_pow)
            cost_occ = 0.0
            if D_occ is not None and w_occ > 0.0:
                cost_occ = w_occ * 0.5 * (D_occ[z,y,x] + D_occ[nz,ny,nx])
            cost = step_len * (cost_p + cost_occ)
            ng = g + cost
            nnode = (nz,ny,nx)
            if nnode in gscore and ng >= gscore[nnode]:
                continue
            gscore[nnode] = ng
            came_from[nnode] = current
            heapq.heappush(open_heap, (ng + heuristic(nz,ny,nx), ng, nnode))
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("A* aborted: too many expansions; consider cropping or increasing eps/p_floor/goal_radius or margin.")

    raise RuntimeError("A*: open set exhausted without reaching goal.")


def a_star_path_cropped(P: np.ndarray,
                        start_zyx: Tuple[int,int,int],
                        goal_zyx: Tuple[int,int,int],
                        margin: int = 24,
                        eps: float = 1e-3,
                        p_floor: float = 0.02,
                        goal_radius_vox: float = 1.5,
                        max_expansions: int = 5_000_000,
                        D_occ: Optional[np.ndarray] = None,
                        w_occ: float = 0.0,
                        p_pow: float = 2.0) -> np.ndarray:
    """Run A* in a cropped ROI around the endpoints to limit search.
    Returns path in the *global* index space.
    """
    Z,Y,X = P.shape
    z0 = max(0, min(start_zyx[0], goal_zyx[0]) - margin)
    y0 = max(0, min(start_zyx[1], goal_zyx[1]) - margin)
    x0 = max(0, min(start_zyx[2], goal_zyx[2]) - margin)
    z1 = min(Z, max(start_zyx[0], goal_zyx[0]) + margin + 1)
    y1 = min(Y, max(start_zyx[1], goal_zyx[1]) + margin + 1)
    x1 = min(X, max(start_zyx[2], goal_zyx[2]) + margin + 1)

    P_sub = P[z0:z1, y0:y1, x0:x1]
    s_sub = (start_zyx[0]-z0, start_zyx[1]-y0, start_zyx[2]-x0)
    g_sub = (goal_zyx[0]-z0,  goal_zyx[1]-y0,  goal_zyx[2]-x0)

    path_sub = a_star_path(P_sub, s_sub, g_sub,
                           eps=eps, p_floor=p_floor,
                           goal_radius_vox=goal_radius_vox, max_expansions=max_expansions,
                           D_occ=(D_occ[z0:z1, y0:y1, x0:x1] if D_occ is not None else None),
                           w_occ=w_occ, p_pow=p_pow)
    # map back to global
    path_global = path_sub.copy()
    path_global[:,0] += z0
    path_global[:,1] += y0
    path_global[:,2] += x0
    return path_global

# ------------------------------------------------------------
# Path processing & spline fit (C)
# ------------------------------------------------------------

def resample_polyline(points_zyx: np.ndarray, step_vox: float = 1.5) -> np.ndarray:
    """Resample ordered points (N,3) to approximately uniform arc-length spacing."""
    if len(points_zyx) < 2:
        return points_zyx.copy()
    seg = np.linalg.norm(np.diff(points_zyx, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    m = max(2, int(np.ceil(L/step_vox)) + 1)
    s_new = np.linspace(0.0, L, m)
    out = np.empty((m,3), dtype=float)
    for d in range(3):
        out[:,d] = np.interp(s_new, s, points_zyx[:,d])
    return out


def fit_bspline_to_polyline(points_zyx: np.ndarray,
                            degree: int = 3,
                            smooth: float = 1e-3) -> BSpline:
    """Fit a smoothing spline to 3D polyline in voxel coords."""
    coords = [points_zyx[:,0], points_zyx[:,1], points_zyx[:,2]]
    (t, c, k), u = splprep(coords, k=degree, s=smooth)
    ctrl = np.asarray(c).T  # (n_ctrl, 3)
    return BSpline(t, ctrl, k)

# ------------------------------------------------------------
# Continuous refinement on field P (D)
# ------------------------------------------------------------

def trilinear_interp(volume: np.ndarray, pts_zyx: np.ndarray) -> np.ndarray:
    """Trilinear interpolate volume at fractional voxel coords (z,y,x)."""
    z = pts_zyx[:,0]; y = pts_zyx[:,1]; x = pts_zyx[:,2]
    Z,Y,X = volume.shape
    z0 = np.clip(np.floor(z).astype(int), 0, Z-2)
    y0 = np.clip(np.floor(y).astype(int), 0, Y-2)
    x0 = np.clip(np.floor(x).astype(int), 0, X-2)
    dz = z - z0; dy = y - y0; dx = x - x0
    c000 = volume[z0,   y0,   x0  ]
    c001 = volume[z0,   y0,   x0+1]
    c010 = volume[z0,   y0+1, x0  ]
    c011 = volume[z0,   y0+1, x0+1]
    c100 = volume[z0+1, y0,   x0  ]
    c101 = volume[z0+1, y0,   x0+1]
    c110 = volume[z0+1, y0+1, x0  ]
    c111 = volume[z0+1, y0+1, x0+1]
    c00 = c000*(1-dx) + c001*dx
    c01 = c010*(1-dx) + c011*dx
    c10 = c100*(1-dx) + c101*dx
    c11 = c110*(1-dx) + c111*dx
    c0 = c00*(1-dy) + c01*dy
    c1 = c10*(1-dy) + c11*dy
    c = c0*(1-dz) + c1*dz
    return c


def spline_sample(bs: BSpline, u: np.ndarray, der: int = 0) -> np.ndarray:
    """Evaluate vector-valued BSpline at parameters u (der=0,1,2). Returns (len(u),3)."""
    vals = bs(u, der)
    vals = np.atleast_2d(vals)
    if vals.shape[0] != len(u):
        vals = vals.T
    return vals


def chord_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


@dataclass
class RefinementWeights:
    alpha: float = 1.0    # attraction to high P (maximize P)
    beta:  float = 2.0    # penalty for low P (soft outside)
    gamma: float = 1e-3   # curvature regularization
    delta: float = 0.0    # length prior weight (0 to disable)
    eta:   float = 100.0  # endpoint pin weight
    zeta:  float = 0.0    # attraction to occupied voxels via D_occ (mean distance in voxels)


def refine_bspline_on_field(P: np.ndarray,
                            bs_vox: BSpline,
                            u_samples: int = 300,
                            tau_inside: float = 0.15,
                            weights: RefinementWeights = RefinementWeights(),
                            fix_endpoints_to: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            length_prior: Optional[float] = None,
                            D_occ: Optional[np.ndarray] = None,
                            max_iter: int = 200) -> BSpline:
    """Optimize control points of bs_vox to minimize continuous loss on P.

    NOTE: We keep knots and degree fixed; only control points move. We rely on
    numerical gradients (L-BFGS-B without user-supplied jac) for robustness.
    """
    t = bs_vox.t.copy(); k = bs_vox.k
    C0 = bs_vox.c.copy()  # (n_ctrl,3) in (z,y,x)

    # Parameter domain and samples
    u0 = t[k]; u1 = t[-k-1]
    u = np.linspace(u0, u1, u_samples)

    # Endpoint pin targets (in curve space)
    s_start = spline_sample(BSpline(t, C0, k), np.array([u0]))[0]
    s_end   = spline_sample(BSpline(t, C0, k), np.array([u1]))[0]
    if fix_endpoints_to is None:
        fix_endpoints_to = (s_start, s_end)

    if (length_prior is not None) and (weights.delta <= 0.0):
        print("[WARN] length_prior provided but refinement weight delta==0; length term is disabled.")

    def loss_and_spline(C_flat: np.ndarray) -> Tuple[float, BSpline, Dict[str,float]]:
        C = C_flat.reshape(C0.shape)
        bs = BSpline(t, C, k)
        S = spline_sample(bs, u)            # (n,3)
        Pval = trilinear_interp(P, S)
        # Attraction: maximize P -> minimize -alpha * mean(P)
        L_attr = -weights.alpha * float(np.mean(Pval))
        # Soft outside: penalize P below tau
        short = np.maximum(0.0, tau_inside - Pval)
        L_out = weights.beta * float(np.mean(short*short))
        # Curvature: mean squared norm of second derivative
        S2 = spline_sample(bs, u, der=2)
        L_curv = weights.gamma * float(np.mean(np.sum(S2*S2, axis=1)))
        # Length prior (approx via chord length on samples, in VOXELS)
        L_len = 0.0
        if (weights.delta > 0.0) and (length_prior is not None):
            L = chord_length(S)
            L_len = weights.delta * (L - float(length_prior))**2
        # Occupancy attraction: mean distance-to-occupied (0 on occupied voxels)
        L_occ = 0.0
        if (D_occ is not None) and (weights.zeta > 0.0):
            Dval = trilinear_interp(D_occ, S)
            L_occ = weights.zeta * float(np.mean(Dval))
        # Endpoint pins in curve space
        S0 = spline_sample(bs, np.array([u0]))[0]
        S1 = spline_sample(bs, np.array([u1]))[0]
        L_pin = weights.eta * (float(np.sum((S0 - fix_endpoints_to[0])**2)) +
                               float(np.sum((S1 - fix_endpoints_to[1])**2)))
        total = L_attr + L_out + L_curv + L_len + L_occ + L_pin
        parts = {"attr":L_attr, "out":L_out, "curv":L_curv, "len":L_len, "occ":L_occ, "pin":L_pin}
        return total, bs, parts

    def fun(C_flat: np.ndarray) -> float:
        val, _, _ = loss_and_spline(C_flat)
        return val

    res = minimize(fun, C0.ravel(), method="L-BFGS-B", options={"maxiter": max_iter, "ftol":1e-9})
    C_opt = res.x.reshape(C0.shape)
    return BSpline(t, C_opt, k)

# ------------------------------------------------------------
# High-level API
# ------------------------------------------------------------

@dataclass
class FitParams:
    sigma_vox: float = 1.5
    dilate_iter: int = 0
    use_dt: bool = True
    pick_tau: float = 0.15
    a_star_eps: float = 1e-3
    a_star_p_floor: float = 0.02
    a_star_goal_radius_vox: float = 1.5
    a_star_margin_vox: int = 24
    a_star_max_expansions: int = 5_000_000
    a_star_pow: float = 2.0
    a_star_w_occ: float = 0.0
    resample_step_vox: float = 1.5
    spline_degree: int = 3
    spline_smooth: float = 1e-3
    refine: bool = True
    refine_u_samples: int = 300
    refine_tau_inside: float = 0.15
    refine_weights: RefinementWeights = RefinementWeights()
    refine_max_iter: int = 200
    fix_endpoints: bool = True
    length_prior: Optional[float] = None  # in voxel units; leave None to disable
    endpoint_strategy: str = "field_farthest"  # "field_farthest" | "pca"
    # Waypoints
    n_waypoints: int = 0
    waypoint_min_sep_vox: int = 8
    waypoint_strategy: str = "pca_quantiles"  # currently only PCA quantiles



@dataclass
class FitResult:
    P: np.ndarray                    # tubularity field
    path_vox: np.ndarray             # initial path (N,3) in (z,y,x)
    path_world: np.ndarray           # path in (x,y,z)
    bs_vox: BSpline                  # final spline in voxel coords
    bs_world: BSpline                # final spline in world coords
    u_domain: Tuple[float,float]     # (u0,u1)
    voxel_size_zyx: Tuple[float,float,float]
    origin_xyz: Tuple[float,float,float]



def bspline_vox_to_world(bs_vox: BSpline,
                         origin_xyz: Tuple[float,float,float],
                         voxel_size_zyx: Tuple[float,float,float]) -> BSpline:
    """Convert a BSpline in voxel coords (z,y,x) to world coords (x,y,z)."""
    C_zyx = bs_vox.c
    # Map control points to world and reorder axes to (x,y,z)
    C_world_xyz = vox2world(C_zyx, origin_xyz, voxel_size_zyx)
    # BSpline expects control points as (n_ctrl, dim); already so.
    return BSpline(bs_vox.t.copy(), C_world_xyz, bs_vox.k)


def fit_bspline_from_numpy_voxel_grid(voxel_grid: Any,
                                      params: FitParams = FitParams(),
                                      endpoints_zyx: Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = None) -> FitResult:
    """Full pipeline A–D on a NumpyVoxelGrid-like input.

    Args:
        voxel_grid: object with occupancy + metadata (see `extract_voxel_grid`).
        params: hyperparameters (see FitParams).
        endpoints_zyx: optional ((z0,y0,x0),(z1,y1,x1)) integer endpoints. If None,
                       they are auto-picked from the field.

    Returns:
        FitResult with field, path, and BSplines (voxel/world).
    """
    occ, voxel_size_zyx, origin_xyz = extract_voxel_grid(voxel_grid)
    P, M_thick = build_tubularity_field(occ, sigma_vox=params.sigma_vox, use_dt=params.use_dt, dilate_iter=params.dilate_iter)
    # Distance-to-occupied (in voxels), promotes hugging actual voxels in refine
    D_occ = distance_transform_edt(~M_thick)

    if endpoints_zyx is None:
        if params.endpoint_strategy == "pca":
            start_zyx, goal_zyx = pick_endpoints_by_pca(occ)
        else:
            start_zyx, goal_zyx = pick_endpoints_from_field(P, tau=params.pick_tau)
    else:
        start_zyx, goal_zyx = endpoints_zyx

        # Build path, possibly with intermediate waypoints to avoid straight-chord skipping
    def run_astar_pair(s, g, margin, eps, p_floor, goal_r, w_occ, p_pow):
        return a_star_path_cropped(P, s, g, margin=margin, eps=eps, p_floor=p_floor,
                                   goal_radius_vox=goal_r, max_expansions=params.a_star_max_expansions,
                                   D_occ=D_occ, w_occ=w_occ, p_pow=p_pow)

    nodes = [start_zyx]
    if params.n_waypoints > 0 and params.waypoint_strategy == "pca_quantiles":
        anchors, _ = waypoints_by_pca_quantiles(occ, start_zyx, goal_zyx,
                                                n_waypoints=params.n_waypoints,
                                                min_sep_vox=params.waypoint_min_sep_vox)
        for a in anchors:
            nodes.append(tuple(int(v) for v in a))
    nodes.append(goal_zyx)

    paths = []
    for s, g in zip(nodes[:-1], nodes[1:]):
        try:
            seg = run_astar_pair(s, g,
                                 margin=int(params.a_star_margin_vox),
                                 eps=float(params.a_star_eps),
                                 p_floor=float(params.a_star_p_floor),
                                 goal_r=float(params.a_star_goal_radius_vox),
                                 w_occ=float(params.a_star_w_occ),
                                 p_pow=float(params.a_star_pow))
        except RuntimeError as e:
            # relaxed retry
            seg = run_astar_pair(s, g,
                                 margin=int(max(48, params.a_star_margin_vox*2)),
                                 eps=float(max(params.a_star_eps*5.0, 5e-3)),
                                 p_floor=float(max(params.a_star_p_floor*2.0, 0.05)),
                                 goal_r=float(max(params.a_star_goal_radius_vox*2.0, 3.0)),
                                 w_occ=float(max(params.a_star_w_occ, 1.0)),
                                 p_pow=float(max(params.a_star_pow, 2.0)))
        paths.append(seg)
    # stitch segments (avoid duplicate boundary points)
    path_vox_raw = paths[0]
    for seg in paths[1:]:
        if len(seg) > 1 and len(path_vox_raw) > 0 and np.allclose(seg[0], path_vox_raw[-1]):
            path_vox_raw = np.vstack([path_vox_raw, seg[1:]])
        else:
            path_vox_raw = np.vstack([path_vox_raw, seg])

    path_vox = resample_polyline(path_vox_raw, step_vox=params.resample_step_vox)

    # Initial spline fit
    bs0 = fit_bspline_to_polyline(path_vox, degree=params.spline_degree, smooth=params.spline_smooth)

    # Length prior (if requested) measured in voxel units along current path
    length_prior = params.length_prior
    if length_prior is None and params.refine and params.refine_weights.delta > 0.0:
        length_prior = chord_length(path_vox)

    # Refinement
    bs_final = bs0
    if params.refine:
        fix_to = None
        if params.fix_endpoints:
            # pin curve endpoints to the initial spline's endpoints
            u0_b, u1_b = bs0.t[bs0.k], bs0.t[-bs0.k-1]
            s0 = spline_sample(bs0, np.array([u0_b]))[0]
            s1 = spline_sample(bs0, np.array([u1_b]))[0]
            fix_to = (s0, s1)
        bs_final = refine_bspline_on_field(
            P=P,
            bs_vox=bs0,
            u_samples=params.refine_u_samples,
            tau_inside=params.refine_tau_inside,
            weights=params.refine_weights,
            fix_endpoints_to=fix_to,
            length_prior=length_prior,
            D_occ=D_occ,
            max_iter=params.refine_max_iter,
        )

    # Convert outputs to world coordinates
    path_world = vox2world(path_vox, origin_xyz, voxel_size_zyx)
    bs_world = bspline_vox_to_world(bs_final, origin_xyz, voxel_size_zyx)

    return FitResult(
        P=P,
        path_vox=path_vox,
        path_world=path_world,
        bs_vox=bs_final,
        bs_world=bs_world,
        u_domain=(bs_final.t[bs_final.k], bs_final.t[-bs_final.k-1]),
        voxel_size_zyx=voxel_size_zyx,
        origin_xyz=origin_xyz,
    )

# ------------------------------------------------------------
# Debugging utilities
# ------------------------------------------------------------
from scipy.spatial import cKDTree


def debug_report(voxel_grid: Any, result: FitResult) -> None:
    """Print diagnostics to help locate typical failure modes (offsets, axis/order issues)."""
    occ, vs_zyx, ori_xyz = extract_voxel_grid(voxel_grid)
    P = result.P
    bs_vox = result.bs_vox

    # Sample spline in voxel coords
    u0, u1 = result.u_domain
    u = np.linspace(u0, u1, 300)
    S_vox = spline_sample(bs_vox, u)

    # 1) Distance from spline to occupied voxels (in voxel units)
    occ_pts = np.stack(np.nonzero(occ), axis=1).astype(float)
    if len(occ_pts) == 0:
        print("[DEBUG] Occupancy is empty.")
        return
    tree = cKDTree(occ_pts)
    d, _ = tree.query(S_vox, k=1)
    print(f"[DEBUG] Dist spline->occ (vox): min={d.min():.3f}, mean={d.mean():.3f}, 95%={np.quantile(d,0.95):.3f}")

    # 2) Field values along spline (should be fairly high if well-aligned)
    P_along = trilinear_interp(P, S_vox)
    print(f"[DEBUG] P along spline: min={P_along.min():.3f}, mean={P_along.mean():.3f}, max={P_along.max():.3f}")

    # 3) Compare path vs occ (should be near zero distances)
    path_vox = result.path_vox
    dp, _ = tree.query(path_vox, k=1)
    print(f"[DEBUG] Dist path->occ (vox): min={dp.min():.3f}, mean={dp.mean():.3f}, 95%={np.quantile(dp,0.95):.3f}")

    # 3b) Path length vs PCA span (voxels)
    def _arc_len(a):
        return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))
    L_path = _arc_len(path_vox)
    mu = occ_pts.mean(axis=0); A = occ_pts - mu
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[0]; proj = A @ v
    span = float(proj.max() - proj.min())
    print(f"[DEBUG] Path length (vox) ~ {L_path:.2f}; PCA span (vox) ~ {span:.2f}")

    # 4) World mapping sanity: centroid of occ vs centroid of spline in world
    occ_centroid_vox = occ_pts.mean(axis=0)
    occ_centroid_world = vox2world(occ_centroid_vox.reshape(1,3), ori_xyz, vs_zyx)[0]
    S_world = result.bs_world(np.linspace(u0, u1, 300))
    spline_centroid_world = S_world.mean(axis=0)
    delta = spline_centroid_world - occ_centroid_world
    print(f"[DEBUG] World centroid delta (spline - occ): {delta} [world units]; |delta| = {np.linalg.norm(delta):.4f}")
    # Length in vox & meters (approx via samples)
    L_vox = float(np.sum(np.linalg.norm(np.diff(S_vox, axis=0), axis=1)))
    L_m = L_vox * float(vs_zyx[2])  # if isotropic, sx==sy==sz; using sx here
    print(f"[DEBUG] Approx spline length: {L_vox:.1f} vox ~ {L_m:.3f} m")
# ------------------------------------------------------------
# Interactive endpoint picking (Open3D)
# ------------------------------------------------------------

def _voxel_centers_from_grid(vg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (occ_idx, centers_xyz) where occ_idx are integer array indices
    of occupied voxels in the grid's native axis order, and centers_xyz are
    their world coordinates. Works with scalar or 3-tuple voxel_size.
    """
    occ = np.asarray(getattr(vg, "occupancy"))
    occ_idx = np.argwhere(occ)  # shape (M,3) in the grid's axis order
    vs = getattr(vg, "voxel_size")
    if np.isscalar(vs):
        vs_tuple = (float(vs), float(vs), float(vs))
    else:
        vs = np.asarray(vs, float).reshape(3)
        # vg.voxel_size is expected (sx, sy, sz) in world coords
        vs_tuple = (float(vs[0]), float(vs[1]), float(vs[2]))
    origin = np.asarray(getattr(vg, "origin"), float).reshape(3)

    # Build centers as origin + (idx + 0.5) * voxel_size (viewer convention)
    centers = origin + (occ_idx + 0.5) * np.asarray(vs_tuple)
    return occ_idx, centers


def pick_voxel_endpoints_open3d(vg: Any, window_name: str = "Pick endpoints (Shift+LMB; Q to finish)") -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Open an Open3D window to pick two occupied voxels as endpoints.

    Usage:
      - Shift + Left Mouse Button to pick a point (occupied voxel center)
      - Press 'Q' or 'Esc' to finish. The first two picks are used.

    Returns:
      ((i0,j0,k0), (i1,j1,k1)) in the grid's axis order (same as vg.occupancy).
      You can pass this directly as `endpoints_zyx` to the fitter; axis *names*
      don't matter as long as they're consistent with the array.
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError("Open3D is required for picking. pip install open3d") from e

    occ_idx, centers = _voxel_centers_from_grid(vg)
    if centers.shape[0] == 0:
        raise RuntimeError("Voxel grid is empty; nothing to pick.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    pcd.paint_uniform_color([0.75, 0.75, 0.75])

    print("[Pick] Shift + Left Click to select points; press Q when done.")
    print("[Pick] Please choose TWO endpoints along the DLO.")

    # Use VisualizerWithEditing to capture picked point indices
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = 3.0
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()

    if not picked:
        raise RuntimeError("No points picked.")
    if len(picked) < 2:
        print(f"[Pick] Only {len(picked)} point(s) picked; duplicating the last.")
        picked = list(picked) + [picked[-1]]

    # Take the first two picks
    p0 = int(picked[0]); p1 = int(picked[1])
    if p0 == p1 and len(picked) >= 3:
        p1 = int(picked[2])

    idx0 = tuple(int(v) for v in occ_idx[p0])
    idx1 = tuple(int(v) for v in occ_idx[p1])

    print(f"[Pick] Endpoint 0 (array idx): {idx0}")
    print(f"[Pick] Endpoint 1 (array idx): {idx1}")
    return idx0, idx1


def fit_with_manual_endpoints(vg: Any, params: FitParams) -> FitResult:
    """Convenience wrapper: pick endpoints interactively and run the fit."""
    end0, end1 = pick_voxel_endpoints_open3d(vg)
    return fit_bspline_from_numpy_voxel_grid(vg, params=params, endpoints_zyx=(end0, end1))

