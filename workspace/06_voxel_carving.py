import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraParameters
from open3d.visualization import gui
from open3d.visualization.rendering import Open3DScene, MaterialRecord
import cv2


def build_extrinsic_from_pose(t, q_xyzw):
    """
    t : (3,) float – camera origin in world coords
    q_xyzw : (4,) float – quaternion as [x, y, z, w]
    returns E : (4,4) world→camera extrinsic
    """
    # reorder into [w, x, y, z] for Open3D
    w, x, y, z = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    R_w_c = o3d.geometry.get_rotation_matrix_from_quaternion([w, x, y, z])
    # invert to get world→camera
    R_c_w = R_w_c.T
    t_c   = -R_c_w @ np.asarray(t, float)
    E = np.eye(4)
    E[:3, :3] = R_c_w
    E[:3,  3] = t_c
    return E

def carve(voxel_grid, mask, camera_pose, camera_intrinsics):
    t, q_xyzw = camera_pose
    fx, fy, cx, cy = camera_intrinsics
    extrinsic = build_extrinsic_from_pose(t, q_xyzw)

    H, W = mask.shape
    survivors = []
    for v in voxel_grid.get_voxels():
        idx    = np.array(v.grid_index, float)
        center = voxel_grid.origin + (idx + 0.5) * voxel_grid.voxel_size
        Xc, Yc, Zc, _ = extrinsic @ np.hstack((center, 1.0))
        if Zc <= 0: 
            continue
        u = int(round(fx * Xc / Zc + cx))
        vv = int(round(fy * Yc / Zc + cy))
        if 0 <= u < W and 0 <= vv < H and mask[vv, u]:
            survivors.append(v)

    # rebuild via pointcloud→voxel as before
    if not survivors:
        return o3d.geometry.VoxelGrid()
    inds    = np.stack([v.grid_index for v in survivors])
    centers = voxel_grid.origin + (inds + 0.5) * voxel_grid.voxel_size
    colors  = np.stack([v.color for v in survivors])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    carved = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_grid.voxel_size,
        voxel_grid.get_min_bound(),
        voxel_grid.get_max_bound()
    )
    return carved

def create_dense_voxel_grid(center, size, voxel_size, color=[1, 0, 0]):
    """
    Create a dense VoxelGrid of the given physical size, centered at `center`.

    Parameters
    ----------
    center : array-like of 3 floats
        (x, y, z) world-coordinates of the grid’s center.
    size : array-like of 3 floats
        (width, height, depth) of the grid in world units.
    voxel_size : float
        Edge length of each cubic voxel.
    color : list of 3 floats, optional
        RGB color for all voxels (each in [0,1]). Default is red [1,0,0].

    Returns
    -------
    open3d.geometry.VoxelGrid
    """
    center = np.asarray(center, dtype=float)
    size   = np.asarray(size,   dtype=float)

    # Validate dimensions
    if center.shape != (3,):
        raise ValueError(f"`center` must have shape (3,), got {center.shape}")
    if size.shape != (3,):
        raise ValueError(f"`size` must have shape (3,), got {size.shape}")

    # Compute origin of the box
    origin = center - size / 2.0

    grid = o3d.geometry.VoxelGrid.create_dense(
        origin=origin,
        color=np.asarray(color, dtype=float),
        voxel_size=float(voxel_size),
        width=float(size[0]),
        height=float(size[1]),
        depth=float(size[2]),
    )
    return grid

def read_mask(mask_path, threshold=127):
    """
    Read a silhouette mask image and convert it to a binary 0/1 NumPy array.

    Parameters
    ----------
    mask_path : str
        Path to the mask image (e.g., 'mask.jpg').
    threshold : int, optional
        Grayscale threshold in [0, 255] above which pixels are foreground.
        Defaults to 127.

    Returns
    -------
    mask : np.ndarray of shape (H, W), dtype=np.uint8
        Binary mask where foreground pixels are 1 and background are 0.
    """
    # Load as grayscale
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{mask_path}'")
    # Apply threshold
    _, bw = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return bw

def display_voxel_grid(voxel_grid, frames, frame_size=0.1):
    """
    Visualize a VoxelGrid with fixed Z-up orientation (no tilt when rotating).

    Parameters
    ----------
    voxel_grid : open3d.geometry.VoxelGrid
        The voxel grid to display.
    frames : sequence of (t, q)
        Each element is a tuple:
          - t: length-3 iterable (tx, ty, tz) – the frame origin in world coords.
          - q: length-4 iterable (rx, ry, rz, rw) – quaternion (vector part first, scalar last).
    frame_size : float, optional
        Size of the axes for each coordinate frame. Default is 0.1.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)

    # Add each coordinate frame
    for t, q in frames:
        t = np.asarray(t, dtype=float)
        rx, ry, rz, rw = q
        q_o3d = np.array([rw, rx, ry, rz], dtype=float)
        R = o3d.geometry.get_rotation_matrix_from_quaternion(q_o3d)
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        cam_frame.transform(T)
        vis.add_geometry(cam_frame)

    ctr = vis.get_view_control()
    # Set an initial view
    ctr.set_front((0, -0.5, -0.5))
    ctr.set_lookat(voxel_grid.get_center())
    ctr.set_up((0, 0, 1))

    # Run the visualizer loop, enforcing Z-up each frame
    while True:
        if not vis.poll_events():
            break
        ctr.set_up((0, 0, 1))
        vis.update_renderer()

    vis.destroy_window()

def example():
    # 1) create a 1×1×1 m grid centered at origin with 5 cm voxels
    grid = create_dense_voxel_grid(center=[0,0,0],
                                   size=[1.0,1.0,1.0],
                                   voxel_size=0.05)

    # 2) fake a rectangular silhouette in the middle of a 640×480 image
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:380, 200:440] = 1

    # 3) example intrinsics & pose
    camera_intrinsics = (525.0, 525.0, W/2, H/2)
    camera_pose = (
        [0.0, 0.0, -2.0],           # translation (x,y,z)
        [1.0, 0.0, 0.0, 0.0]        # quaternion (w,x,y,z)
    )

    # 4) execute carve
    carved = carve(grid, mask, camera_pose, camera_intrinsics)

    # 5) visualize result
    o3d.visualization.draw_geometries([carved])

def real_example():
    # 1) create a 1×1×1 m grid centered at origin with 5 cm voxels
    grid = create_dense_voxel_grid(center=[1.5,0.5,0.5],
                                   size=[1.0,1.0,0.5],
                                   voxel_size=0.005)

    # 2) fake a rectangular silhouette in the middle of a 640×480 image
    mask_path0 = "/root/workspace/images/moves/cable0_mask.jpg"
    mask_path1 = "/root/workspace/images/moves/cable1_mask.jpg"
    mask_path2 = "/root/workspace/images/moves/cable2_mask.jpg"
    mask_path3 = "/root/workspace/images/moves/cable3_mask.jpg"
    mask_path4 = "/root/workspace/images/moves/cable4_mask.jpg"
    mask_path5 = "/root/workspace/images/moves/cable5_mask.jpg"
    mask_path6 = "/root/workspace/images/moves/cable6_mask.jpg"
    mask0 = read_mask(mask_path0)
    mask1 = read_mask(mask_path1)
    mask2 = read_mask(mask_path2)
    mask3 = read_mask(mask_path3)
    mask4 = read_mask(mask_path4)
    mask5 = read_mask(mask_path5)
    mask6 = read_mask(mask_path6)


    # 3) example intrinsics & pose
    camera_intrinsics = (149.09148, 187.64966, 334.87706, 268.23742)
    camera_pose0 = ([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    camera_pose1 = ([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    camera_pose2 = ([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    camera_pose3 = ([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    camera_pose4 = ([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    camera_pose5 = ([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    camera_pose6 = ([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    origin = ([0, 0, 0], [0, 0, 0, 1])

    # 4) execute carve
    carved0 = carve(grid, mask0, camera_pose0, camera_intrinsics)
    display_voxel_grid(carved0, [origin, camera_pose0])

    carved1 = carve(carved0, mask1, camera_pose1, camera_intrinsics)
    display_voxel_grid(carved1, [origin, camera_pose0, camera_pose1])

    carved2 = carve(carved1, mask2, camera_pose2, camera_intrinsics)
    display_voxel_grid(carved2, [origin, camera_pose0, camera_pose1, camera_pose2])

    carved3 = carve(carved2, mask3, camera_pose3, camera_intrinsics)
    display_voxel_grid(carved3, [origin, camera_pose0, camera_pose1, camera_pose2, camera_pose3])

    carved4 = carve(carved3, mask4, camera_pose4, camera_intrinsics)
    display_voxel_grid(carved4, [origin, camera_pose0, camera_pose1, camera_pose2, camera_pose3, camera_pose4])

    carved5 = carve(carved4, mask5, camera_pose5, camera_intrinsics)
    display_voxel_grid(carved5, [origin, camera_pose0, camera_pose1, camera_pose2, camera_pose3, camera_pose4, camera_pose5])

    carved6 = carve(carved5, mask6, camera_pose6, camera_intrinsics)
    display_voxel_grid(carved6, [origin, camera_pose0, camera_pose1, camera_pose2, camera_pose3, camera_pose4, camera_pose5, camera_pose6])
 
    

if __name__ == "__main__":
    # example()
    real_example()