import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from my_utils import show_masks

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
kernel = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]], dtype=np.uint8)

# 3a. Count 8‐neighbours
neighbor_count = cv2.filter2D(skeleton_bool.astype(np.uint8), -1, kernel)
# 3b. Find intersection pixels
ints = np.argwhere((skeleton_bool) & (neighbor_count >= 3))
# 3c. Distance transform
dist_map = distance_transform_edt(binary)

# 3d. Remove a disk around each intersection
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

# 3e. Excise
skeleton_excised      = skeleton_bool & ~to_remove
skeleton_excised_disp = (skeleton_excised.astype(np.uint8) * 255)
show_masks([skeleton_disp, skeleton_excised_disp],
           title="Before vs After Excising Intersection Zone")

# ─── REMOVE SHORT SEGMENTS ────────────────────────────────────────────────
min_length = 20
num_labels, labels = cv2.connectedComponents(skeleton_excised.astype(np.uint8),
                                            connectivity=8)
for lbl in range(1, num_labels):
    comp_mask = (labels == lbl)
    if comp_mask.sum() < min_length:
        skeleton_excised[comp_mask] = False

skeleton_excised_disp = (skeleton_excised.astype(np.uint8) * 255)
show_masks([skeleton_disp, skeleton_excised_disp],
           title=f"Removed segments shorter than {min_length}px")

# ─── Step 5: Pair endpoints and reconnect with curvature ────────────────
neighbor_count_excised = cv2.filter2D(skeleton_excised.astype(np.uint8), -1, kernel)

# 5a) Find all endpoints
endpoints = np.argwhere((skeleton_excised) &
                        (neighbor_count_excised == 1))

# 5b-prep) Only attempt to reconnect clustered endpoints
conn_radius     = 30
pts             = endpoints
dmat            = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
neighbor_counts = (dmat < conn_radius).sum(axis=1) - 1

# endpoints to reconnect vs leave alone
to_connect_mask = neighbor_counts >= 1
conn_eps        = endpoints[to_connect_mask]
leave_eps       = endpoints[~to_connect_mask]

# ——— **NEW**: enforce even count by dropping the most isolated if odd ———
n_conn = len(conn_eps)
if n_conn % 2 == 1:
    # compute for each conn‐endpoint its nearest‐neighbor distance
    sub_dmat = dmat[to_connect_mask][:, to_connect_mask]
    # ignore diagonal by adding large constant
    np.fill_diagonal(sub_dmat, np.inf)
    min_dist_to_any = sub_dmat.min(axis=1)
    # drop the one with largest min-distance
    drop_idx = np.argmax(min_dist_to_any)
    global_idx = np.where(to_connect_mask)[0][drop_idx]
    to_connect_mask[global_idx] = False
    conn_eps = endpoints[to_connect_mask]
    leave_eps = endpoints[~to_connect_mask]

# now work with an even number
endpoints = conn_eps
n         = len(endpoints)

# 5b) Compute tangents
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
        else:
            continue
        break

# 5c) Distances
pts_arr = np.array([[y,x] for y,x in endpoints])
dists   = np.linalg.norm(pts_arr[:,None,:] - pts_arr[None,:,:], axis=2)
d_max   = dists.max()

# 5d) Curvatures
curvatures = []
for y, x in endpoints:
    # first neighbor
    nbrs1 = [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
             if not(dy==dx==0) and 0<=y+dy<H and 0<=x+dx<W
             and skeleton_excised[y+dy,x+dx]]
    if not nbrs1:
        curvatures.append(0.0)
        continue
    y1, x1 = nbrs1[0]
    nbrs2 = [(y1+dy, x1+dx) for dy in (-1,0,1) for dx in (-1,0,1)
             if not(dy==dx==0) and (y1+dy,x1+dx)!=(y,x)
             and 0<=y1+dy<H and 0<=x1+dx<W
             and skeleton_excised[y1+dy,x1+dx]]
    if not nbrs2:
        curvatures.append(0.0)
        continue
    y2, x2 = nbrs2[0]
    v1 = np.array([y1-y, x1-x], float)
    v2 = np.array([y2-y1, x2-x1], float)
    angle = np.arccos(
        np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    )
    curvatures.append(angle/(np.linalg.norm(v1)+1e-8))

curv_arr   = np.array(curvatures)
curv_norm  = curv_arr / (curv_arr.max()+1e-8)

# 5e) Build cost matrix
w_angle = 1.0
w_dist  = 2.0
w_curv  = 5.0

A = np.zeros((n,n), float)
for i in range(n):
    for j in range(i+1, n):
        ang   = np.pi - np.arccos(abs(np.dot(vectors[i],vectors[j])))
        dcost = dists[i,j]/(d_max+1e-8)
        kcost = abs(curv_norm[i]-curv_norm[j])
        A[i,j] = A[j,i] = w_angle*ang + w_dist*dcost + w_curv*kcost

# 5f) Perfect matching
from itertools import combinations
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

# 5g) Draw bridges
connected_disp = skeleton_excised_disp.copy()
for i,j in pairs:
    y0,x0 = endpoints[i]
    y1,x1 = endpoints[j]
    cv2.line(connected_disp, (x0,y0), (x1,y1), 255, 1)

# 6. Show final path
show_masks([connected_disp], title="Connected Skeleton w/ Curvature")

