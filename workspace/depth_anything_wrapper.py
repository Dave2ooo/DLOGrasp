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
    
    def show_depth_map(self, depth, title="DepthAnythingWrapper: Depth Map", wait=True):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        cv2.imshow(title, depth)
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
            title: str = 'DepthAnything – Point Cloud') -> None:
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
            window_name=title,
            width=1280,
            height=720,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False)

    def transform_pointcloud_to_world(self, pointcloud: o3d.geometry.PointCloud, camera_pose: TransformStamped) -> o3d.geometry.PointCloud:
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

    def transform_pointcloud_from_world(self,
                                        pointcloud: o3d.geometry.PointCloud,
                                        camera_pose: TransformStamped) -> o3d.geometry.PointCloud:
        """
        Transform a point cloud from world coordinates into the camera frame
        using the given camera_pose (camera→world).

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The point cloud in world coordinates.
        camera_pose : tf2_ros.TransformStamped
            The transform from camera frame to world frame.

        Returns
        -------
        o3d.geometry.PointCloud
            A new point cloud expressed in the camera coordinate frame.
        """

        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("pointcloud must be an open3d.geometry.PointCloud")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

        # Extract translation and rotation
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        # Build camera→world matrix then invert for world→camera
        T_cam2world = quaternion_matrix(quat)
        T_cam2world[:3, 3] = trans
        T_world2cam = np.linalg.inv(T_cam2world)

        # Convert point cloud to homogeneous coords
        pts_world = np.asarray(pointcloud.points, dtype=np.float64)
        if pts_world.size == 0:
            return o3d.geometry.PointCloud()
        ones = np.ones((pts_world.shape[0], 1), dtype=np.float64)
        pts_hom = np.hstack((pts_world, ones))  # (N,4)

        # Apply world→camera transform
        pts_cam_hom = (T_world2cam @ pts_hom.T).T  # (N,4)
        pts_cam = pts_cam_hom[:, :3]

        # Build and return new point cloud
        pc_cam = o3d.geometry.PointCloud()
        pc_cam.points = o3d.utility.Vector3dVector(pts_cam)
        return pc_cam

    def transform_point_to_world(self, point, camera_pose: TransformStamped) -> np.ndarray:
        """
        Transform a single 3D point from the camera frame into world coordinates.

        Parameters
        ----------
        point : sequence of three floats
            The [x, y, z] coordinates in the camera frame.
        camera_pose : geometry_msgs.msg.TransformStamped
            The transform from camera frame to world frame.

        Returns
        -------
        np.ndarray
            A length-3 array of the point in world coordinates.
        """
        # Validate inputs
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError("point must be a sequence of three numbers [x, y, z]")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

        # Build the 4×4 camera→world transform
        q = camera_pose.transform.rotation
        t = camera_pose.transform.translation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)

        T = quaternion_matrix(quat)    # 4×4 rotation+1s
        T[:3, 3] = trans               # insert translation

        # Apply to homogeneous point
        hom_pt = np.append(p, 1.0)      # [x, y, z, 1]
        world_hom = T @ hom_pt          # matrix multiplication

        return world_hom[:3]

    def transform_point_from_world(self, point, camera_pose: TransformStamped) -> np.ndarray:
        """
        Transform a single 3D point from world coordinates into the camera frame.

        Parameters
        ----------
        point : sequence of three floats
            The [x, y, z] coordinates in the world frame.
        camera_pose : geometry_msgs.msg.TransformStamped
            The transform from camera frame to world frame.

        Returns
        -------
        np.ndarray
            A length-3 array of the point in camera coordinates.
        """
        # Validate inputs
        p_world = np.asarray(point, dtype=np.float64)
        if p_world.shape != (3,):
            raise ValueError("point must be a sequence of three numbers [x, y, z]")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

        # Build the 4×4 camera→world transform
        q = camera_pose.transform.rotation
        t = camera_pose.transform.translation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        T_cam2world = quaternion_matrix(quat)  # 4×4 rotation+1s
        T_cam2world[:3, 3] = trans              # insert translation

        # Invert to get world→camera
        T_world2cam = np.linalg.inv(T_cam2world)

        # Apply to homogeneous point
        hom_pt = np.append(p_world, 1.0)        # [x, y, z, 1]
        cam_hom = T_world2cam @ hom_pt          # matrix multiplication

        return cam_hom[:3]

    def show_pointclouds_with_frames(self, pointclouds, frames, title='DepthAnything – Point Cloud') -> None:
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
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1280,
            height=720,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False)

    def scale_depth_map(self,
                        depth: np.ndarray,
                        scale: float,
                        shift: float) -> np.ndarray:
        """
        Scale and shift a depth map, preserving any existing mask:
        - If `depth` is a numpy MaskedArray, apply scale/shift to the data
          and retain the mask.
        - Otherwise, treat zeros as masked pixels and ensure they remain zero.

        Parameters
        ----------
        depth : np.ndarray or np.ma.MaskedArray
            Input depth map (H×W), possibly masked.
        scale : float
            Multiplicative factor to apply to the depth values.
        shift : float
            Additive offset to apply after scaling.

        Returns
        -------
        np.ndarray or np.ma.MaskedArray
            The transformed depth map, masked in the same way as input.
        """
        # Numeric checks
        if not np.isscalar(scale) or not np.isscalar(shift):
            raise TypeError("scale and shift must be numeric scalars")

        # Handle masked arrays
        if isinstance(depth, np.ma.MaskedArray):
            # apply scale/shift to data, keep mask
            data = depth.data.astype(np.float32, copy=False)
            scaled = data * scale + shift
            return np.ma.MaskedArray(scaled, mask=depth.mask, copy=False)

        # Otherwise treat zeros as mask
        if not isinstance(depth, np.ndarray):
            raise TypeError("depth must be a numpy.ndarray or MaskedArray")
        depth_f = depth.astype(np.float32, copy=False)
        mask_zero = (depth_f == 0)

        # scale & shift
        scaled = depth_f * scale + shift

        # re‐apply mask: zero out previously masked pixels
        scaled[mask_zero] = 0.0
        return scaled

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
        W = int(2 * cx) + 1   # 2*319.5 = 639 → +1 = 640 cols
        H = int(2 * cy) + 1   # 2*239.5 = 479 → +1 = 480 rows

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

    def get_closest_points(self,
                           pointcloud1: o3d.geometry.PointCloud,
                           pointcloud2: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Find closest‐point correspondences from pointcloud1 to pointcloud2.

        Parameters
        ----------
        pointcloud1 : o3d.geometry.PointCloud
            Source cloud (N₁ points).
        pointcloud2 : o3d.geometry.PointCloud
            Target cloud (N₂ points).

        Returns
        -------
        corres : np.ndarray of shape (M,2), dtype=int
            Each row [i, j] means point i in pointcloud1 corresponds to
            its nearest neighbor j in pointcloud2.
        """
        # build a KD‐tree on the second cloud
        kdtree = o3d.geometry.KDTreeFlann(pointcloud2)

        pts1 = np.asarray(pointcloud1.points)
        corres = []
        for i, p in enumerate(pts1):
            # search for the single nearest neighbor in cloud2
            [_, idx, _] = kdtree.search_knn_vector_3d(p, 1)
            corres.append([i, idx[0]])

        return np.array(corres, dtype=np.int64)

    def project_point(self, point):
        """
        Project a single 3D point in the camera coordinate system onto the image plane.

        Parameters
        ----------
        point : sequence of float or np.ndarray, shape (3,)
            A 3D point [x, y, z] in camera coordinates.

        Returns
        -------
        (u, v) : tuple of float
            The projected pixel coordinates. Raises if z <= 0.
        """
        import numpy as np
        from typing import Tuple

        if not hasattr(self, 'intrinsics'):
            raise AttributeError("self.intrinsics must be set to (fx, fy, cx, cy)")

        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError("point must be a 3-element sequence or array")
        x, y, z = p
        if z <= 0:
            raise ValueError("Point is behind the camera (z <= 0)")

        fx, fy, cx, cy = self.intrinsics
        u = x * fx / z + cx
        v = y * fy / z + cy
        return [u, v]

    def project_point_from_world(self,
                      point,
                      camera_pose: TransformStamped):
        """
        Project a single 3D point in world coordinates onto the image plane
        of the camera defined by camera_pose.

        Parameters
        ----------
        point : sequence of three floats
            The [x, y, z] coordinates in world frame.
        camera_pose : geometry_msgs.msg.TransformStamped
            The transform from camera frame to world frame.

        Returns
        -------
        (u, v) : tuple of float
            The projected pixel coordinates.
        """
        if not hasattr(self, 'intrinsics'):
            raise AttributeError("self.intrinsics must be set to (fx, fy, cx, cy)")
        if not isinstance(camera_pose, TransformStamped):
            raise TypeError("camera_pose must be a geometry_msgs.msg.TransformStamped")

        # world→camera: invert camera→world
        t = camera_pose.transform.translation
        q = camera_pose.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat  = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        T_cam2world = quaternion_matrix(quat)
        T_cam2world[:3, 3] = trans
        T_world2cam = np.linalg.inv(T_cam2world)

        # transform the point
        p_world = np.asarray(point, dtype=np.float64)
        if p_world.shape != (3,):
            raise ValueError("point must be a sequence of three numbers [x, y, z]")
        hom = np.append(p_world, 1.0)
        x_cam, y_cam, z_cam = (T_world2cam @ hom)[:3]

        if z_cam <= 0:
            raise ValueError("Point is behind the camera (z <= 0)")

        # project using intrinsics
        fx, fy, cx, cy = self.intrinsics
        u = x_cam * fx / z_cam + cx
        v = y_cam * fy / z_cam + cy
        return u, v

    def triangulate(self,
                    point1_3d,
                    pose1: TransformStamped,
                    point2_2d,
                    pose2: TransformStamped) -> np.ndarray:
        """
        Given a 3D point on ray1 (from camera1) and a 2D pixel on ray2 (from camera2),
        compute the 3D point on ray1 that is closest to ray2.

        Parameters
        ----------
        point1_3d : sequence of three floats
            A point [x, y, z] on the first ray, in world coordinates.
        pose1 : TransformStamped
            Transform from camera1 frame to world frame.
        point2_2d : sequence of two floats
            A pixel coordinate [u, v] in image2.
        pose2 : TransformStamped
            Transform from camera2 frame to world frame.

        Returns
        -------
        np.ndarray, shape (3,)
            The world‐coordinate point on ray1 nearest to ray2.
        """

        if not hasattr(self, 'intrinsics'):
            raise AttributeError("self.intrinsics must be set to (fx, fy, cx, cy)")
        if not isinstance(pose1, TransformStamped) or not isinstance(pose2, TransformStamped):
            raise TypeError("pose1 and pose2 must be geometry_msgs.msg.TransformStamped")

        # Convert inputs
        p1 = np.asarray(point1_3d, dtype=np.float64)
        if p1.shape != (3,):
            raise ValueError("point1_3d must be [x, y, z]")

        # Camera1 center and ray direction d1
        t1 = pose1.transform.translation
        C1 = np.array([t1.x, t1.y, t1.z], dtype=np.float64)
        v1 = p1 - C1
        norm_v1 = np.linalg.norm(v1)
        if norm_v1 == 0:
            raise ValueError("point1_3d coincides with camera1 center")
        d1 = v1 / norm_v1

        # Camera2 center
        t2 = pose2.transform.translation
        C2 = np.array([t2.x, t2.y, t2.z], dtype=np.float64)

        # Build ray2 direction in world frame
        fx, fy, cx, cy = self.intrinsics
        u, v = point2_2d
        # back‐project pixel into camera‐frame ray
        d_cam = np.array([(u - cx) / fx,
                          (v - cy) / fy,
                          1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        # rotate into world
        q2 = pose2.transform.rotation
        M2 = quaternion_matrix([q2.x, q2.y, q2.z, q2.w])
        R2 = M2[:3, :3]
        d2 = R2.dot(d_cam)

        # Solve for t on ray1 and s on ray2 minimizing ||(C1 + t d1) - (C2 + s d2)||²
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        w0 = C1 - C2
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)
        denom = a * c - b * b

        if abs(denom) < 1e-6:
            # near‐parallel: project C2–C1 onto d1
            t = -d / a
        else:
            t = (b * e - c * d) / denom

        # Closest point on ray1
        return C1 + t * d1

    def count_inliers(self,
                      depth1: np.ndarray,
                      depth2: np.ndarray) -> int:
        """
        Count the number of pixels/points that are “valid” in both inputs.
        Treats any boolean or integer array as a mask (non-zero == valid),
        and any float array as a depth map (value > 0 == valid).

        Parameters
        ----------
        depth1, depth2 : np.ndarray
            Either depth maps (float) or masks (bool or int). Must have the same shape.

        Returns
        -------
        int
            The count of positions where both inputs are valid.
        """
        import numpy as np

        if depth1.shape != depth2.shape:
            raise ValueError("Inputs must have the same shape")

        def make_mask(x):
            # boolean mask stays boolean
            if x.dtype == bool:
                return x
            # integer mask: nonzero is valid
            if np.issubdtype(x.dtype, np.integer):
                return x != 0
            # float depth: positive values are valid
            if np.issubdtype(x.dtype, np.floating):
                return x > 0
            raise TypeError(f"Unsupported dtype for count_inliers: {x.dtype}")

        m1 = make_mask(depth1)
        m2 = make_mask(depth2)

        # count where both are True
        return int(np.count_nonzero(m1 & m2))



if __name__ == '__main__':
    image = cv2.imread('/root/workspace/images/moves/cable0.jpg')

    depth_anything_wrapper = DepthAnythingWrapper()
    depth = depth_anything_wrapper.get_depth_map(image) # HxW depth map in meters in numpy
    # depth_anything_wrapper.show_depth_map(depth)

    pointcloud = depth_anything_wrapper.get_pointcloud(depth)
    depth_anything_wrapper.show_pointclouds([pointcloud])
