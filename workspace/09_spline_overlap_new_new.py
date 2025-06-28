import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from itertools import combinations
from collections import deque
from skimage.draw import line as skline
from my_utils import show_masks, show_spline_gradient

# containers for partial paths
real_paths = []
connection_paths = []

# 1. Load your mask in grayscale
mask = cv2.imread('/root/workspace/images/cable_loop_mask.png', cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread('/root/workspace/images/tube_mask_holes.png', cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("Couldn’t load mask — check your path")

# 2. Binarise (0 or 1)
_, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

# 3. Skeletonise
skeleton_bool = skeletonize(binary.astype(bool))
skeleton_disp  = (skeleton_bool.astype(np.uint8) * 255)
show_masks([binary * 255, skeleton_disp], title="Skeleton")

# ─── Step 3: Excise the intersection zone ────────────────────────────────
kernel = np.array([[1,1,1], [1,0,1], [1,1,1]], dtype=np.uint8)
neighbor_count = cv2.filter2D(skeleton_bool.astype(np.uint8), -1, kernel)
ints = np.argwhere((skeleton_bool) & (neighbor_count >= 3))
dist_map = distance_transform_edt(binary)

to_remove = np.zeros_like(skeleton_bool, dtype=bool)
H, W = skeleton_bool.shape
for y, x in ints:
    r = dist_map[y, x]
    R = int(np.ceil(r))
    y0, y1 = max(0, y - R), min(H, y + R + 1)
    x0, x1 = max(0, x - R), min(W, x + R + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (yy - y)**2 + (xx - x)**2 <= r**2
    to_remove[y0:y1, x0:x1][circle] = True

skeleton_excised      = skeleton_bool & ~to_remove
skeleton_excised_disp = (skeleton_excised.astype(np.uint8) * 255)
show_masks([skeleton_disp, skeleton_excised_disp], title="Before vs After Excising Intersection Zone")

# ─── REMOVE SHORT SEGMENTS ────────────────────────────────────────────────
min_length = 20
num_labels, labels = cv2.connectedComponents(skeleton_excised.astype(np.uint8), connectivity=8)
for lbl in range(1, num_labels):
    comp_mask = (labels == lbl)
    if comp_mask.sum() < min_length:
        skeleton_excised[comp_mask] = False
skeleton_excised_disp = (skeleton_excised.astype(np.uint8) * 255)
show_masks([skeleton_disp, skeleton_excised_disp], title=f"Removed segments shorter than {min_length}px")

# ─── SHOW INDIVIDUAL PATH SEGMENTS ───────────────────────────────────────
num_labels2, labels2 = cv2.connectedComponents(skeleton_excised.astype(np.uint8), connectivity=8)
for lbl in range(1, num_labels2):
    comp_mask = (labels2 == lbl)
    pixels = [tuple(pt) for pt in np.argwhere(comp_mask)]  # (y,x)
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
    ordered_rc = []
    prev = None
    curr = start
    while True:
        ordered_rc.append(curr)
        nbr = [n for n in nbrs[curr] if n != prev]
        if not nbr: break
        prev, curr = curr, nbr[0]
    show_spline_gradient(binary, np.array(ordered_rc), title=f"Segment {lbl}")
    # store the segment path
    real_paths.append(np.array(ordered_rc))

# ─── Step 5: Pair endpoints and generate explicit connection lines ──────── Pair endpoints and generate explicit connection lines ────────
neighbor_count_excised = cv2.filter2D(skeleton_excised.astype(np.uint8), -1, kernel)
endpoints = np.argwhere((skeleton_excised) & (neighbor_count_excised == 1))
conn_radius = 30
pts = endpoints
dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
neighbor_counts = (dmat < conn_radius).sum(axis=1) - 1

to_connect_mask = neighbor_counts >= 1
conn_eps = endpoints[to_connect_mask]
leave_eps = endpoints[~to_connect_mask]
n_conn = len(conn_eps)
if n_conn % 2 == 1:
    sub_dmat = dmat[to_connect_mask][:, to_connect_mask]
    np.fill_diagonal(sub_dmat, np.inf)
    min_dist = sub_dmat.min(axis=1)
    drop = np.argmax(min_dist)
    global_idx = np.where(to_connect_mask)[0][drop]
    to_connect_mask[global_idx] = False
    conn_eps = endpoints[to_connect_mask]
    leave_eps = endpoints[~to_connect_mask]

endpoints = conn_eps
n = len(endpoints)

vectors = []
for y, x in endpoints:
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W and skeleton_excised[ny,nx]:
                v = np.array([y-ny, x-nx], float)
                vectors.append(v/np.linalg.norm(v))
                break
        else: continue
        break

pts_arr = np.array([[y,x] for y,x in endpoints])
dists = np.linalg.norm(pts_arr[:, None, :] - pts_arr[None, :, :], axis=2)
d_max = dists.max()
curvatures = []
for y, x in endpoints:
    nbrs1 = [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
             if not(dy==dx==0) and 0<=y+dy<H and 0<=x+dx<W and skeleton_excised[y+dy,x+dx]]
    if not nbrs1:
        curvatures.append(0.0); continue
    y1, x1 = nbrs1[0]
    nbrs2 = [(y1+dy, x1+dx) for dy in (-1,0,1) for dx in (-1,0,1)
             if not(dy==dx==0) and (y1+dy, x1+dx)!=(y, x) and 0<=y1+dy<H and 0<=x1+dx<W and skeleton_excised[y1+dy,x1+dx]]
    if not nbrs2:
        curvatures.append(0.0); continue
    y2, x2 = nbrs2[0]
    v1 = np.array([y1-y, x1-x], float)
    v2 = np.array([y2-y1, x2-x1], float)
    angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
    curvatures.append(angle/(np.linalg.norm(v1)+1e-8))
curv_arr = np.array(curvatures)
curv_norm = curv_arr / (curv_arr.max()+1e-8)

w_angle, w_dist, w_curv = 1.0, 2.0, 5.0
A = np.zeros((n,n), float)
for i in range(n):
    for j in range(i+1, n):
        ang = np.pi - np.arccos(abs(np.dot(vectors[i], vectors[j])))
        dcost = dists[i,j]/(d_max+1e-8)
        kcost = abs(curv_norm[i]-curv_norm[j])
        A[i,j] = A[j,i] = w_angle*ang + w_dist*dcost + w_curv*kcost

def find_min_pairs(A):
    m = A.shape[0]
    assert m%2==0, "Need even number of endpoints"
    best_cost, best = float('inf'), None
    def recurse(rem, pairs):
        nonlocal best_cost, best
        if not rem:
            total = sum(A[i,j] for i,j in pairs)
            if total < best_cost:
                best_cost, best = total, pairs.copy()
            return
        i = rem[0]
        for j in rem[1:]:
            recurse([k for k in rem if k not in (i,j)], pairs+[(i,j)])
    recurse(list(range(m)), [])
    return best

pairs = find_min_pairs(A)

# Draw combined lines for visualization
combined_disp = skeleton_excised_disp.copy()
for i,j in pairs:
    y0,x0 = endpoints[i]
    y1,x1 = endpoints[j]
    cv2.line(combined_disp, (x0,y0), (x1,y1), 255, 1)
show_masks([combined_disp], title="Connected Skeleton w/ Curvature")

# ─── SHOW EACH CONNECTION LINE INDIVIDUALLY ──────────────────────────────
for idx, (i,j) in enumerate(pairs, 1):
    y0,x0 = endpoints[i]
    y1,x1 = endpoints[j]
    rr, cc = skline(y0, x0, y1, x1)
    coords_line = np.stack([rr, cc], axis=1)  # (row, col)
    show_spline_gradient(binary, coords_line, title=f"Connection {idx}")
    connection_paths.append(coords_line)

# ─── MERGE ALL PATHS INTO ONE CONTINUOUS SKELETON ────────────────────────
# Build lookup of path endpoints
def endpoints_of(path):
    return (tuple(path[0]), tuple(path[-1]))

# label paths: even indices real_paths, odd connection_paths
total_paths = []
for rp in real_paths:
    total_paths.append({'coords': rp, 'type': 'real'})
for cp in connection_paths:
    total_paths.append({'coords': cp, 'type': 'conn'})

# build mapping from endpoint to list of path indices
ep_map = {}
for idx, p in enumerate(total_paths):
    a, b = endpoints_of(p['coords'])
    for e in (a, b):
        ep_map.setdefault(e, []).append(idx)

# find starting real endpoint (occurs only once and belongs to a real path)
start = None
for idx, p in enumerate(total_paths):
    if p['type']!='real': continue
    a, b = endpoints_of(p['coords'])
    for e in (a,b):
        if len(ep_map[e])==1:
            start = (idx, e)
            break
    if start: break
if start is None:
    raise RuntimeError("Could not find a unique starting endpoint to merge.")

merged = []
used = set()
curr_idx, curr_end = start
# orientation: if start matches path end, reverse
coords = total_paths[curr_idx]['coords']
if tuple(coords[0])==curr_end:
    seq = coords
else:
    seq = coords[::-1]
for pt in seq:
    merged.append(tuple(pt))
used.add(curr_idx)
curr_end = merged[-1]

# iterate until no unused connected path
while True:
    # get next paths sharing this endpoint
    next_idxs = [i for i in ep_map[curr_end] if i not in used]
    if not next_idxs:
        break
    # pick the unique other path
    next_idx = next_idxs[0]
    coords = total_paths[next_idx]['coords']
    # orient to match current start
    if tuple(coords[0])==curr_end:
        seq = coords
    else:
        seq = coords[::-1]
    # append without duplicating first point
    for pt in seq[1:]: merged.append(tuple(pt))
    used.add(next_idx)
    curr_end = merged[-1]

# show final merged skeleton
display_coords = np.array(merged)
show_spline_gradient(binary, display_coords, title="Full Merged Skeleton")
