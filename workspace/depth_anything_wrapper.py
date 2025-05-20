import os
import sys
import cv2
import torch
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d

from tf.transformations import quaternion_matrix
from geometry_msgs.msg import TransformStamped

script_path = os.path.abspath(__file__)

# Import Depth Anything v2
depth_anything_directory = '/root/Depth-Anything-V2'
sys.path.insert(1, depth_anything_directory)
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

class DepthAnythingWrapper():
    def __init__(self, encoder='vitl', dataset='hypersim', max_depth=20, intrinsics: tuple[float, float, float, float] | None = None):
        self.encoder = encoder # or 'vits', 'vitb'
        self.dataset = dataset # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.max_depth = max_depth # 20 for indoor model, 80 for outdoor model
        self.intrinsics = intrinsics
        self.depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        checkpoint_path = f'{depth_anything_directory}/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
        self.depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.depth_anything = self.depth_anything.to(DEVICE).eval()

    def get_depth_map(self, image: NDArray[np.uint8]):
        return self.depth_anything.infer_image(image)
    
    def show_depth_map(self, depth, wait=True):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        cv2.imshow("DepthAnythingWrapper: Depth Map", depth)
        if wait:
            cv2.waitKey(0)

    def get_pointcloud(self,
            depth: NDArray[np.floating] | Image.Image | NDArray[np.uint8],
            stride: int | None = None) -> 'o3d.geometry.PointCloud':
        """Convert *depth map* → Open3D :class:`~open3d.geometry.PointCloud`.

        Parameters
        ----------
        depth
            ``HxW`` depth image in **metres** (``float32``). RGB depth encodings
            will be collapsed to a single channel.
        intrinsics
            Tuple ``(fx, fy, cx, cy)`` in pixels.  *If ``None``*, defaults to a
            simple pin‑hole model with ``fx = fy = max(H, W)``, ``cx = W/2``,
            ``cy = H/2``.
        stride
            Optional pixel stride for pre‑down‑sampling (e.g. ``stride=2`` keeps
            ¼ of the points; often faster than calling ``random_down_sample``
            afterwards).

        Returns
        -------
        o3d.geometry.PointCloud
            3‑D points in the **camera frame**.  Invalid/zero‑depth pixels are
            dropped automatically.
        """
        # --- Prepare depth ----------------------------------------------------
        if isinstance(depth, Image.Image):
            depth = np.asarray(depth).astype(np.float32)
        elif isinstance(depth, np.ndarray):
            depth = depth.astype(np.float32)
        else:
            raise TypeError("depth must be PIL.Image.Image or numpy.ndarray")

        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        # Optional sub‑sampling for speed --------------------------------------
        if stride and stride > 1:
            depth = depth[::stride, ::stride]

        H, W = depth.shape
        fx, fy, cx, cy = (float(max(H, W)),) * 2 + (W / 2.0, H / 2.0) if self.intrinsics is None else [float(v) for v in self.intrinsics]

        # --- Convert to 3‑D ----------------------------------------------------
        i_coords, j_coords = np.indices((H, W), dtype=np.float32)
        z = depth
        x = (j_coords - cx) * z / fx
        y = (i_coords - cy) * z / fy

        pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        valid = pts[:, 2] > 0
        pts = pts[valid].astype(np.float64)

        # --- Pack into Open3D --------------------------------------------------
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        return pc

    def mask_pointcloud(self,
                        pointcloud: 'o3d.geometry.PointCloud',
                        mask: np.ndarray) -> 'o3d.geometry.PointCloud':
        """
        Filter a point cloud using a 2D mask: retains only points whose projection in the image
        falls within the True region of the mask.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The point cloud returned by get_pointcloud().
        mask : np.ndarray
            2D boolean or binary mask of shape (H, W), where True (or 1) indicates pixels to keep.

        Returns
        -------
        o3d.geometry.PointCloud
            A new point cloud containing only the masked points.
        """
        # sanity checks
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("mask_pointcloud expects an open3d.geometry.PointCloud")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a numpy.ndarray")

        # ensure boolean mask
        mask_bool = mask.astype(bool)
        H, W = mask_bool.shape

        # pull out the Nx3 array of points
        pts = np.asarray(pointcloud.points)
        if pts.size == 0:
            return o3d.geometry.PointCloud()  # nothing to keep

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # project back to pixel coords
        j = np.round(x * self.intrinsics[0] / z + self.intrinsics[2]).astype(int)
        i = np.round(y * self.intrinsics[1] / z + self.intrinsics[3]).astype(int)

        # build mask of points whose projection is in-bounds AND True in mask
        keep = (
            (i >= 0) & (i < H) &
            (j >= 0) & (j < W) &
            mask_bool[i, j]
        )

        # filter and rebuild point cloud
        filtered = pts[keep]
        pc_new = o3d.geometry.PointCloud()
        pc_new.points = o3d.utility.Vector3dVector(filtered)
        return pc_new

    def show_pointclouds(
            self,
            pointclouds: 'o3d.geometry.PointCloud',
            max_points: int = 200_000,
            voxel_size: float | None = None,
            axis_size: float = 0.2,
            window_name: str = 'DepthAnything – Point Cloud') -> None:
        """Display a point cloud via *Open3D*'s built‑in viewer, showing axes.

        Parameters
        ----------
        pointcloud
            Cloud to display (camera‑frame coordinates).
        max_points
            Random down‑sample threshold for interactivity.
        voxel_size
            If given, voxel grid down‑sampling size **in metres**.
        axis_size
            Size of the origin coordinate frame (in metres).
        window_name
            Title of the visualiser window.
        """
        if not isinstance(pointclouds[0], o3d.geometry.PointCloud):
            raise TypeError("show_pointcloud expects an open3d.geometry.PointCloud")
        
        # Origin coordinate frame ---------------------------------------------
        geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])]

        for pc in pointclouds:
            if voxel_size is not None and voxel_size > 0:
                pc = pc.voxel_down_sample(voxel_size)
            elif len(pc.points) > max_points:
                pc = pc.random_down_sample(max_points / len(pc.points))
            geometries.append(pc)

        # Viewer ---------------------------------------------------------------
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=720,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False)

    def transform_pointcloud_to_world(self,
                                      pointcloud: o3d.geometry.PointCloud,
                                      camera_pose: TransformStamped) -> o3d.geometry.PointCloud:
        """
        Transform a point cloud from camera coordinates into world coordinates
        using the given camera_pose (a tf2_ros TransformStamped).

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The point cloud in the camera frame.
        camera_pose : tf2_ros.TransformStamped
            The transform from camera frame to world frame.

        Returns
        -------
        o3d.geometry.PointCloud
            A new point cloud expressed in world coordinates.
        """
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

        # Extract translation and quaternion
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        # Build 4x4 homogeneous transformation matrix
        T = quaternion_matrix(quat)        # rotation part
        T[:3, 3] = trans                  # translation part

        # Convert point cloud to Nx4 homogeneous coordinates
        pts = np.asarray(pointcloud.points, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_hom = np.hstack((pts, ones))   # shape (N,4)

        # Apply transform
        pts_world_hom = (T @ pts_hom.T).T  # shape (N,4)
        pts_world = pts_world_hom[:, :3]

        # Build new point cloud in world frame
        pc_world = o3d.geometry.PointCloud()
        pc_world.points = o3d.utility.Vector3dVector(pts_world)
        return pc_world

    def show_pointclouds_with_frames(self,
                                    pointclouds,
                                    frames) -> None:
        """
        Display a point cloud together with coordinate frames.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The point cloud to display.
        frames : Sequence[TransformStamped]
            A list of transforms (camera/world frames) to draw.
        """
        # base origin frame
        geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])]

        # convert each TransformStamped into a mesh frame
        for tf in frames:
            if not isinstance(tf, TransformStamped):
                raise TypeError("Each frame must be a geometry_msgs.msg.TransformStamped")
            # extract translation & rotation
            t = tf.transform.translation
            q = tf.transform.rotation
            trans = np.array([t.x, t.y, t.z], dtype=np.float64)
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            # build homogeneous transform
            T = quaternion_matrix(quat)
            T[:3, 3] = trans
            # create and transform a coordinate frame
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame_mesh.transform(T)
            geometries.append(frame_mesh)

        # add the pointcloud and render
        for pc in pointclouds:
            geometries.append(pc)
        o3d.visualization.draw_geometries(geometries)

    def scale_depth_map(self,
                        depth: np.ndarray,
                        scale: float,
                        shift: float) -> np.ndarray:
        """
        Scale and shift a depth map.

        Parameters
        ----------
        depth : np.ndarray
            Input depth map (H×W) of type float or integer.
        scale : float
            Multiplicative factor to apply to the depth values.
        shift : float
            Additive offset to apply after scaling.

        Returns
        -------
        np.ndarray
            The transformed depth map, with the same shape as input.
        """
        import numpy as np

        if not isinstance(depth, np.ndarray):
            raise TypeError("depth must be a numpy.ndarray")
        if not np.issubdtype(type(scale), np.number):
            raise TypeError("scale must be a numeric type")
        if not np.issubdtype(type(shift), np.number):
            raise TypeError("shift must be a numeric type")

        # perform scale and shift in floating point
        depth_f = depth.astype(np.float32, copy=False)
        return depth_f * scale + shift

    def project_pointcloud(self,
                           pointcloud: o3d.geometry.PointCloud,
                           camera_pose: TransformStamped) -> tuple[np.ndarray, np.ndarray]:
        """
        Project a 3D point cloud (in world coordinates) onto the camera image plane
        to produce a depth map and a black-and-white mask.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The cloud in world coordinates.
        camera_pose : geometry_msgs.msg.TransformStamped
            Transform from camera frame into world frame.

        Returns
        -------
        depth_map : np.ndarray
            Depth map (H×W) in metres, with zero where no points project.
        mask_bw : np.ndarray
            Black-and-white mask (H×W) uint8, 255 where depth_map > 0, else 0.
        """
        # validate inputs
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")
        if not hasattr(self, 'intrinsics'):
            raise AttributeError("DepthAnythingWrapper: self.intrinsics must be set to (fx, fy, cx, cy)")

        fx, fy, cx, cy = self.intrinsics
        W = int(round(2 * cx))
        H = int(round(2 * cy))

        # build camera→world homogeneous matrix
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        T_cam2world = quaternion_matrix(quat)
        T_cam2world[:3, 3] = trans

        # invert to get world→camera
        T_world2cam = np.linalg.inv(T_cam2world)

        # transform points to camera frame
        pts = np.asarray(pointcloud.points, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_hom = np.hstack((pts, ones))
        pts_cam = (T_world2cam @ pts_hom.T).T[:, :3]
        x_cam, y_cam, z_cam = pts_cam.T

        # keep only points in front of camera
        valid = z_cam > 0
        x_cam, y_cam, z_cam = x_cam[valid], y_cam[valid], z_cam[valid]

        # project into pixel coords
        u = x_cam * fx / z_cam + cx
        v = y_cam * fy / z_cam + cy
        j = np.round(u).astype(int)
        i = np.round(v).astype(int)

        # mask out-of-bounds
        in_bounds = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        i, j, z_cam = i[in_bounds], j[in_bounds], z_cam[in_bounds]

        # build depth buffer, taking nearest point per pixel
        depth_map = np.full((H, W), np.inf, dtype=np.float32)
        for ii, jj, zz in zip(i, j, z_cam):
            if zz < depth_map[ii, jj]:
                depth_map[ii, jj] = zz

        # finalize depth map and mask
        mask = depth_map != np.inf
        depth_map[~mask] = 0.0

        # black-and-white mask: 255 where depth>0, else 0
        # mask_bw = (mask.astype(np.uint8) * 255)
        mask_bw = (mask > 0).astype(np.float32)
        # with np.printoptions(threshold=np.inf):
            # print(mask_bw)
        return depth_map, mask_bw

    

if __name__ == '__main__':
    image = cv2.imread('/root/workspace/images/moves/cable0.jpg')

    depth_anything_wrapper = DepthAnythingWrapper()
    depth = depth_anything_wrapper.get_depth_map(image) # HxW depth map in meters in numpy
    # depth_anything_wrapper.show_depth_map(depth)

    pointcloud = depth_anything_wrapper.get_pointcloud(depth)
    depth_anything_wrapper.show_pointclouds([pointcloud])
