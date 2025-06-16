import numpy as np
import open3d as o3d
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from my_utils import *
import cv2

print("here")
# --- helper to display a point cloud in‐notebook ---
def display_point_cloud(points, title="Point Cloud", point_size=1):
    """
    points: (N,3) numpy array
    """
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, c=points[:, 2], cmap='viridis', linewidth=0)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# --- 0. Define camera intrinsics, poses, and image paths -------------------
camera_intrinsics = (149.09148, 187.64966, 334.87706, 268.23742)
poses = [
    ([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001]),
    # ([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
]
images = [
    "/root/workspace/images/moves/cable0_mask.jpg",
    # "/root/workspace/images/moves/cable3_mask.jpg",
]
print("here2")

# --- 1. Create dense voxel grid --------------------------------------------
# vox_size = 5e-3   # 0.5 mm voxels
vox_size = 1e-2   # 0.5 mm voxels
grid = o3d.geometry.VoxelGrid.create_dense(
    # origin=[0.8, 0, 0.3],
    origin=[-5, -5, -5],
    color=[1, 0, 0],
    voxel_size=vox_size,
    # width=1, height=1, depth=1
    width=10, height=10, depth=10
)
fx, fy, cx, cy = camera_intrinsics
print("here3")
# --- 2. Carve with each silhouette ----------------------------------------
for (t_vec, q), img_path in zip(poses, images):
    print(f"Carving image {img_path}")
    # load mask, threshold to binary
    img_o3d = o3d.io.read_image(img_path)
    np_img  = np.asarray(img_o3d)
    # if color, take one channel
    if np_img.ndim == 3:
        np_img = np_img[..., 0]
    binary = (np_img > 127).astype(np.uint8) * 255
    mask_o3d = o3d.geometry.Image(binary)

    # plt.imshow(mask_o3d, cmap="gray")
    # plt.title(f"{img_path} — unique values: {np.unique(mask_o3d)}")
    # plt.show()
    # show the mask_o3d with cv2
    img_bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("mask", img_bgr)
    cv2.waitKey(0)

    # debug: show unique values
    print(f"{img_path} → unique mask values:", np.unique(binary))

    # build extrinsic
    rot   = Rotation.from_quat(q)
    extr  = np.eye(4)
    extr[:3, :3] = rot.as_matrix()
    extr[:3, 3] = np.array(t_vec)

    # set camera params using actual mask size
    h, w = binary.shape
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    param.extrinsic = extr

    # carve and reassign
    grid = grid.carve_silhouette(mask_o3d, param)

# --- 3. Extract occupied voxels and skeletonize ----------------------------
vox_list = grid.get_voxels()
if len(vox_list) == 0:
    raise RuntimeError(
        "No voxels remain after carving — verify your masks and camera setup."
    )

coords = np.vstack([v.grid_index for v in vox_list])

# compute volume shape from bounds
min_b = np.array(grid.get_min_bound())
max_b = np.array(grid.get_max_bound())
dims  = np.ceil((max_b - min_b) / vox_size).astype(int) + 1

vol_bin = np.zeros(dims, dtype=bool)
# mark occupied in boolean grid
vol_bin[coords[:,0], coords[:,1], coords[:,2]] = True

# skeletonize to one-voxel-wide centreline
skel = skeletonize(vol_bin)

# --- 4. Extract points & fit cubic B-spline -------------------------------
xyz_idx = np.stack(np.nonzero(skel), axis=1)
xyz_pts = xyz_idx * vox_size + min_b

# optional reorder via nearest neighbours
tree, order = KDTree(xyz_pts), [0]
while len(order) < len(xyz_pts):
    _, nxt = tree.query(xyz_pts[order[-1]], k=2)
    order.append(int(nxt[1]))
xyz_ordered = xyz_pts[order]

# fit C2 cubic spline
tck, _    = splprep(xyz_ordered.T, s=1e-4, k=3)
u_fine    = np.linspace(0, 1, 500)
spline3d  = np.vstack(splev(u_fine, tck)).T


print("Done – reconstructed spline written to spline_points.ply")
display_point_cloud(spline3d, title="Reconstructed Spline", point_size=2)