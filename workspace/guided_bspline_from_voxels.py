"""
Guided B-spline fitting on a voxelized deformable linear object (DLO).

You can pick ANY number of ordered guide points on the voxel grid; the spline
will *roughly* follow them (softly) while also being attracted to the voxelized
object and staying smooth. It does not have to interpolate guides exactly.

Assumptions (customize near the top if needed):
- Voxel array shape is (X, Y, Z) and origin is the world min-corner.
- Voxel centers are at: origin + (idx + 0.5) * voxel_size.
- `voxel_size` can be scalar (isotropic) or a 3-tuple (sx, sy, sz).

Deps: numpy, scipy (ndimage, interpolate, optimize), open3d (for picking)

Entry points:
- pick_guides_open3d(vg): returns ordered voxel indices and world coords.
- fit_bspline_guided(vg, guides_idx, params): does field build, path init
  (via segmented A* between consecutive guides), spline fit, and refinement
  with losses: attraction to field P, attraction to occupied voxels (D_occ),
  soft curvature, soft anchor to guides, and optional length prior.

Tip: If your array is (Z,Y,X) instead of (X,Y,Z), set ARRAY_ORDER = "zyx".
"""
from __future__ import annotations
import math
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter, distance_transform_edt,
    binary_dilation, generate_binary_structure
)
from scipy.interpolate import splprep, BSpline
from scipy.optimize import minimize

# =====================
# Conventions & mapping
# =====================
ARRAY_ORDER = "xyz"           # "xyz" for (X,Y,Z) arrays, "zyx" for (Z,Y,X)
VOXEL_COORD_IS_CENTER = True  # indices represent voxel centers


def _as_float_tuple3(v) -> Tuple[float,float,float]:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1:
        return (float(a[0]), float(a[0]), float(a[0]))
    assert a.size == 3, "voxel_size/origin must be scalar or 3-tuple"
    return (float(a[0]), float(a[1]), float(a[2]))


def vox2world(points_idx: np.ndarray,
              origin_xyz: Tuple[float,float,float],
              voxel_size_zyx: Tuple[float,float,float]) -> np.ndarray:
    sz, sy, sx = voxel_size_zyx
    off = 0.5 if VOXEL_COORD_IS_CENTER else 0.0
    if ARRAY_ORDER == "xyz":
        x = (points_idx[:,0] + off) * sx
        y = (points_idx[:,1] + off) * sy
        z = (points_idx[:,2] + off) * sz
    else:
        z = (points_idx[:,0] + off) * sz
        y = (points_idx[:,1] + off) * sy
        x = (points_idx[:,2] + off) * sx
    return np.stack([origin_xyz[0] + x, origin_xyz[1] + y, origin_xyz[2] + z], 1)


def world2vox(points_xyz: np.ndarray,
              origin_xyz: Tuple[float,float,float],
              voxel_size_zyx: Tuple[float,float,float]) -> np.ndarray:
    sz, sy, sx = voxel_size_zyx
    off = 0.5 if VOXEL_COORD_IS_CENTER else 0.0
    x = (points_xyz[:,0] - origin_xyz[0]) / sx - off
    y = (points_xyz[:,1] - origin_xyz[1]) / sy - off
    z = (points_xyz[:,2] - origin_xyz[2]) / sz - off
    if ARRAY_ORDER == "xyz":
        return np.stack([x, y, z], 1)
    else:
        return np.stack([z, y, x], 1)


# ============
# Voxel access
# ============

def extract_voxel_grid(vg: Any) -> Tuple[np.ndarray, Tuple[float,float,float], Tuple[float,float,float]]:
    occ = np.asarray(getattr(vg, "occupancy"))
    if occ.ndim != 3:
        raise ValueError("occupancy must be 3D")
    vs = _as_float_tuple3(getattr(vg, "voxel_size"))
    origin = _as_float_tuple3(getattr(vg, "origin"))
    # reorder to (sz,sy,sx) expected internally
    if ARRAY_ORDER == "xyz":
        sx, sy, sz = vs
        voxel_size_zyx = (sz, sy, sx)
    else:
        voxel_size_zyx = vs  # already sz,sy,sx
    return occ.astype(bool), voxel_size_zyx, origin


# ==================
# Field & distances
# ==================

def build_field_and_masks(occ: np.ndarray, sigma_vox: float, use_dt: bool, dilate_iter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = occ.astype(bool)
    if dilate_iter > 0:
        st = generate_binary_structure(3, 2)
        M = binary_dilation(M, structure=st, iterations=int(dilate_iter))
    P = gaussian_filter(M.astype(float), sigma=float(sigma_vox))
    P /= (P.max() + 1e-12)
    if use_dt:
        DT_in = distance_transform_edt(M)
        DT_in /= (DT_in.max() + 1e-12)
        P = P * DT_in
        P /= (P.max() + 1e-12)
    D_occ = distance_transform_edt(~M)  # distance in voxels to nearest occupied
    return P, M, D_occ


# =========
# A* search
# =========
NBR_26 = np.array([(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)], dtype=int)
NBR_STEP = np.linalg.norm(NBR_26.astype(float), axis=1)


def a_star_path_cropped(P: np.ndarray,
                        D_occ: Optional[np.ndarray],
                        start_zyx: Tuple[int,int,int],
                        goal_zyx: Tuple[int,int,int],
                        margin: int = 24,
                        eps: float = 0.02,
                        p_floor: float = 0.08,
                        goal_radius_vox: float = 2.0,
                        max_expansions: int = 5_000_000,
                        p_pow: float = 3.0,
                        w_occ: float = 3.0) -> np.ndarray:
    Z,Y,X = P.shape
    z0 = max(0, min(start_zyx[0], goal_zyx[0]) - margin)
    y0 = max(0, min(start_zyx[1], goal_zyx[1]) - margin)
    x0 = max(0, min(start_zyx[2], goal_zyx[2]) - margin)
    z1 = min(Z, max(start_zyx[0], goal_zyx[0]) + margin + 1)
    y1 = min(Y, max(start_zyx[1], goal_zyx[1]) + margin + 1)
    x1 = min(X, max(start_zyx[2], goal_zyx[2]) + margin + 1)

    Psub = P[z0:z1, y0:y1, x0:x1]
    Dsub = None if D_occ is None else D_occ[z0:z1, y0:y1, x0:x1]
    s = (start_zyx[0]-z0, start_zyx[1]-y0, start_zyx[2]-x0)
    g = (goal_zyx[0]-z0,  goal_zyx[1]-y0,  goal_zyx[2]-x0)

    def inb(z,y,x):
        return 0 <= z < Psub.shape[0] and 0 <= y < Psub.shape[1] and 0 <= x < Psub.shape[2]

    def h(z,y,x):
        dz = g[0]-z; dy = g[1]-y; dx = g[2]-x
        return math.sqrt(dz*dz + dy*dy + dx*dx)

    open_heap = [(h(*s), 0.0, s)]
    came = {}
    gscore = {s: 0.0}
    visited = set()
    expansions = 0

    while open_heap:
        f, gc, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == g:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            path = np.array(path, float)
            path[:,0]+=z0; path[:,1]+=y0; path[:,2]+=x0
            return path
        z,y,x = cur
        pcur = Psub[z,y,x]
        for (ddz,ddy,ddx), step in zip(NBR_26, NBR_STEP):
            nz,ny,nx = z+int(ddz), y+int(ddy), x+int(ddx)
            if not inb(nz,ny,nx):
                continue
            pnb = Psub[nz,ny,nx]
            pav = max(0.5*(pcur + pnb), p_floor)
            cost_p = 1.0 / ((eps + pav) ** p_pow)
            cost_occ = 0.0
            if Dsub is not None and w_occ > 0.0:
                cost_occ = w_occ * 0.5 * (Dsub[z,y,x] + Dsub[nz,ny,nx])
            ng = gc + step * (cost_p + cost_occ)
            nxt = (nz,ny,nx)
            if nxt in gscore and ng >= gscore[nxt]:
                continue
            gscore[nxt] = ng
            came[nxt] = cur
            heapq.heappush(open_heap, (ng + h(nz,ny,nx), ng, nxt))
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("A* too many expansions; adjust margin/eps/p_floor/w_occ")
    raise RuntimeError("A* failed in cropped region")


# ======================
# Spline fit & refinement
# ======================

def resample_polyline(points_zyx: np.ndarray, step: float) -> np.ndarray:
    """Arc-length resample a 3D polyline. Robust to duplicates/degeneracies.
    Returns a new (M,3) array. If total length≈0, returns the unique point.
    """
    P = np.asarray(points_zyx, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points_zyx must be (N,3)")
    if len(P) == 0:
        return P.copy()
    # Drop exact duplicate consecutive points
    keep = [0]
    for i in range(1, len(P)):
        if not np.allclose(P[i], P[i-1]):
            keep.append(i)
    P = P[keep]
    if len(P) == 1:
        return P.copy()
    # Segment lengths and cumulative arclength
    dif = np.diff(P, axis=0)
    seg = np.linalg.norm(dif, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L < 1e-9:
        return P[[0]].copy()
    m = max(2, int(np.ceil(L/float(step))) + 1)
    sn = np.linspace(0.0, L, m)
    # Interp per coordinate
    out = np.column_stack([np.interp(sn, s, P[:, d]) for d in range(3)])
    return out


def fit_spline(points_zyx: np.ndarray, degree: int = 3, smooth: float = 1e-3) -> BSpline:
    coords = [points_zyx[:,0], points_zyx[:,1], points_zyx[:,2]]
    (t, c, k), _ = splprep(coords, k=degree, s=smooth)
    return BSpline(t, np.asarray(c).T, k)


def trilinear_interp(volume: np.ndarray, pts_zyx: np.ndarray) -> np.ndarray:
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
    return c0*(1-dz) + c1*dz


def eval_bs(bs: BSpline, u: np.ndarray, der: int = 0) -> np.ndarray:
    V = bs(u, der)
    V = np.atleast_2d(V)
    return V if V.shape[0]==len(u) else V.T


def chord_len(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts,0),1)))


@dataclass
class GuideWeights:
    alpha: float = 0.6   # attraction to P
    beta:  float = 2.0   # penalty if P below tau
    gamma: float = 5e-3  # curvature
    delta: float = 0.0   # length prior
    eta:   float = 200.0 # endpoint pins (if used)
    zeta:  float = 5.0   # attraction to occupied voxels (distance field)
    kappa: float = 2.0   # anchor-to-guide soft term


@dataclass
class GuidedParams:
    # Field
    sigma_vox: float = 2.0
    dilate_iter: int = 1
    use_dt: bool = True
    # A*
    a_star_eps: float = 0.02
    a_star_p_floor: float = 0.08
    a_star_goal_radius_vox: float = 2.0
    a_star_margin_vox: int = 28
    a_star_pow: float = 3.0
    a_star_w_occ: float = 4.0
    # Spline fit
    resample_step_vox: float = 1.2
    spline_degree: int = 3
    spline_smooth: float = 5e-3
    # Refinement
    refine_u_samples: int = 600
    refine_tau_inside: float = 0.05
    weights: GuideWeights = GuideWeights()
    fix_endpoints: bool = True
    length_prior: Optional[float] = None  # in voxels


def refine_guided(P: np.ndarray,
                  D_occ: np.ndarray,
                  bs_vox: BSpline,
                  guide_pts_vox: np.ndarray,
                  weights: GuideWeights,
                  tau_inside: float,
                  length_prior: Optional[float],
                  u_samples: int,
                  fix_endpoints: bool) -> BSpline:
    t = bs_vox.t.copy(); k = bs_vox.k
    C0 = bs_vox.c.copy()

    u0 = t[k]; u1 = t[-k-1]
    u = np.linspace(u0, u1, u_samples)

    # Assign guide anchors to parameter values by uniform quantiles
    m = max(2, len(guide_pts_vox))
    u_guides = np.linspace(u0, u1, m)

    # Endpoint pins target
    bs_init = BSpline(t, C0, k)
    g0 = eval_bs(bs_init, np.array([u0]))[0]
    g1 = eval_bs(bs_init, np.array([u1]))[0]
    pin_targets = (g0, g1)

    def loss(Cflat: np.ndarray) -> float:
        C = Cflat.reshape(C0.shape)
        bs = BSpline(t, C, k)
        S = eval_bs(bs, u)
        Pval = trilinear_interp(P, S)
        # Field terms
        L_attr = -weights.alpha * float(np.mean(Pval))
        short = np.maximum(0.0, tau_inside - Pval)
        L_out = weights.beta * float(np.mean(short*short))
        # Curvature
        S2 = eval_bs(bs, u, der=2)
        L_curv = weights.gamma * float(np.mean(np.sum(S2*S2,1)))
        # Length
        L_len = 0.0
        if (weights.delta>0.0) and (length_prior is not None):
            L = chord_len(S)
            L_len = weights.delta * (L - float(length_prior))**2
        # Occupancy attraction
        Dval = trilinear_interp(D_occ, S)
        L_occ = weights.zeta * float(np.mean(Dval))
        # Guide anchors (soft): match spline to guides at pre-assigned params
        if len(guide_pts_vox) > 0:
            Sg = eval_bs(bs, u_guides)
            dif = Sg - guide_pts_vox
            L_anchor = weights.kappa * float(np.mean(np.sum(dif*dif,1)))
        else:
            L_anchor = 0.0
        # Endpoint pins
        S0 = eval_bs(bs, np.array([u0]))[0]; S1 = eval_bs(bs, np.array([u1]))[0]
        L_pin = (weights.eta if fix_endpoints else 0.0) * (
            float(np.sum((S0-pin_targets[0])**2)) + float(np.sum((S1-pin_targets[1])**2))
        )
        return L_attr + L_out + L_curv + L_len + L_occ + L_anchor + L_pin

    res = minimize(loss, C0.ravel(), method="L-BFGS-B", options={"maxiter":300, "ftol":1e-9})
    Copt = res.x.reshape(C0.shape)
    return BSpline(t, Copt, k)


# ======================
# Picking & high-level
# ======================

def _voxel_centers_from_grid(vg: Any) -> Tuple[np.ndarray, np.ndarray]:
    occ = np.asarray(getattr(vg, "occupancy"))
    idx = np.argwhere(occ)
    vs = _as_float_tuple3(getattr(vg, "voxel_size"))
    origin = _as_float_tuple3(getattr(vg, "origin"))
    if ARRAY_ORDER == "xyz":
        sx, sy, sz = vs
    else:
        sz, sy, sx = vs
    centers = np.array(origin) + (idx + 0.5) * np.array([sx, sy, sz])
    return idx, centers


def pick_guides_open3d(vg: Any, min_points: int = 2, window_name: str = "Pick guides (Shift+LMB; Q to finish)") -> Tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError("Open3D is required for picking. pip install open3d") from e
    occ_idx, centers = _voxel_centers_from_grid(vg)
    if centers.shape[0]==0:
        raise RuntimeError("Empty grid; nothing to pick.")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    pcd.paint_uniform_color([0.75,0.75,0.75])
    print("\n[Pick] Shift+Left-Click to select an ordered set of points; press Q when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = 3.0
    vis.run(); vis.destroy_window()
    picked = list(vis.get_picked_points())
    if len(picked) < min_points:
        raise RuntimeError(f"Need at least {min_points} points; got {len(picked)}.")
    guides_idx = occ_idx[np.array(picked, dtype=int)]
    guides_xyz = centers[np.array(picked, dtype=int)]
    print(f"[Pick] {len(picked)} guides selected.")
    return guides_idx, guides_xyz


@dataclass
class GuidedFitResult:
    bs_vox: BSpline
    bs_world: BSpline
    u_domain: Tuple[float,float]
    guides_idx: np.ndarray
    guides_world: np.ndarray
    P: np.ndarray
    D_occ: np.ndarray
    voxel_size_zyx: Tuple[float,float,float]
    origin_xyz: Tuple[float,float,float]


def fit_bspline_guided(vg: Any,
                       guides_idx: Sequence[Sequence[int]],
                       params: GuidedParams = GuidedParams()) -> GuidedFitResult:
    occ, voxel_size_zyx, origin = extract_voxel_grid(vg)
    P, M, D_occ = build_field_and_masks(occ, params.sigma_vox, params.use_dt, params.dilate_iter)

    # A* segmented path through ordered guides (in voxel index space)
    nodes = [tuple(int(v) for v in guides_idx[0])]
    for g in guides_idx[1:]:
        nodes.append(tuple(int(v) for v in g))

    segs = []
    for s, g in zip(nodes[:-1], nodes[1:]):
        seg = a_star_path_cropped(
            P, D_occ, s, g,
            margin=params.a_star_margin_vox,
            eps=params.a_star_eps,
            p_floor=params.a_star_p_floor,
            goal_radius_vox=params.a_star_goal_radius_vox,
            p_pow=params.a_star_pow,
            w_occ=params.a_star_w_occ,
        )
        # Guard: if degenerate, fall back to straight voxel line between s and g
        if seg.shape[0] < 2 or np.allclose(seg[-1], seg[0]):
            # simple linear interpolation in index space
            nlin = max(2, int(np.linalg.norm(np.asarray(g, float) - np.asarray(s, float)) + 1))
            t = np.linspace(0.0, 1.0, nlin)
            sg = np.asarray(s, float)[None, :] * (1 - t[:, None]) + np.asarray(g, float)[None, :] * t[:, None]
            seg = sg
        segs.append(seg)
    path = segs[0]
    for seg in segs[1:]:
        if len(path) > 0 and len(seg) > 0 and np.allclose(path[-1], seg[0]):
            seg = seg[1:]
        path = np.vstack([path, seg])

    # Resample and initial smoothing spline
    path = resample_polyline(path, step=params.resample_step_vox)
    bs0 = fit_spline(path, degree=params.spline_degree, smooth=params.spline_smooth)

    # Prepare guide points in voxel coords (ordered)
    guides_idx_arr = np.asarray(guides_idx, float)
    guides_vox = guides_idx_arr.copy()  # already in voxel index space

    # Refinement with soft guide anchors + occupancy attraction
    L_prior = params.length_prior
    if L_prior is None:
        L_prior = chord_len(path)
    bs_final = refine_guided(P, D_occ, bs0, guides_vox, params.weights,
                             tau_inside=params.refine_tau_inside,
                             length_prior=L_prior,
                             u_samples=params.refine_u_samples,
                             fix_endpoints=params.fix_endpoints)

    # World-space spline by mapping control points
    C_zyx = bs_final.c
    C_world = vox2world(C_zyx, origin, voxel_size_zyx)
    bs_world = BSpline(bs_final.t.copy(), C_world, bs_final.k)

    return GuidedFitResult(
        bs_vox=bs_final,
        bs_world=bs_world,
        u_domain=(bs_final.t[bs_final.k], bs_final.t[-bs_final.k-1]),
        guides_idx=np.asarray(guides_idx, int),
        guides_world=vox2world(np.asarray(guides_idx, float), origin, voxel_size_zyx),
        P=P, D_occ=D_occ,
        voxel_size_zyx=voxel_size_zyx,
        origin_xyz=origin,
    )


# =====================
# Tiny usage example
# =====================
if __name__ == "__main__":
    # This block expects you to provide your own NumpyVoxelGrid `vg`.
    # 1) Pick guides interactively
    # guides_idx, guides_xyz = pick_guides_open3d(vg)
    # 2) Fit guided spline
    # res = fit_bspline_guided(vg, guides_idx)
    # print("u-domain:", res.u_domain, "#ctrl:", res.bs_vox.c.shape[0])
    pass
