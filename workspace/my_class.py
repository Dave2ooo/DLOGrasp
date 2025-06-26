from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from ROS_handler import ROSHandler
from camera_handler import ImageSubscriber
from publisher import *
from my_utils import *
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
import rospy
import cv2
import time
import random
import scipy.optimize as opt

# camera_intrinsics = (203.71833, 203.71833, 319.5, 239.5) # old/wrong
camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)


class MyClass:
    def __init__(self):
        self.depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_parameters)
        self.grounded_sam_wrapper = GroundedSamWrapper()
        pass

    def convert_trans_and_rot_to_stamped_transform(self, translation, rotation):
        transform_stamped = TransformStamped()
        transform_stamped.transform.translation.x = translation[0]
        transform_stamped.transform.translation.y = translation[1]
        transform_stamped.transform.translation.z = translation[2]

        transform_stamped.transform.rotation.x = rotation[0]
        transform_stamped.transform.rotation.y = rotation[1]
        transform_stamped.transform.rotation.z = rotation[2]
        transform_stamped.transform.rotation.w = rotation[3]
        return transform_stamped
    
    def process_image(self, image, transform, show=False):
        print('Processing image...')

        depth = self.depth_anything_wrapper.get_depth_map(image)
        mask = self.grounded_sam_wrapper.get_mask(image, 'cable.')[0][0]
        mask_reduced = reduce_mask(mask, 1)

        depth_masked = mask_depth_map(depth, mask)
        depth_masked_reduced = mask_depth_map(depth, mask_reduced) # Reduce mask border

        pointcloud_masked = convert_depth_map_to_pointcloud(depth_masked, camera_parameters)
        pointcloud_masked_reduced = convert_depth_map_to_pointcloud(depth_masked_reduced, camera_parameters)

        pointcloud_masked_world = transform_pointcloud_to_world(pointcloud_masked, transform)
        pointcloud_masked_world_reduced = transform_pointcloud_to_world(pointcloud_masked_reduced, transform)

        if show:
            show_masks([mask], title="Original Mask")
            show_masks_union(mask, mask_reduced, title="Mask & Reduced mask")
            show_depth_map(depth_masked, title="Original Depth Map Masked")
            show_pointclouds([pointcloud_masked, pointcloud_masked_reduced], title="Original Pointcloud Masked in Camera Frame")
            show_pointclouds_with_frames_and_grid([pointcloud_masked_world, pointcloud_masked_world_reduced], [transform], title="Original & Reduced Pointcloud Masked in World Frame")

        print('Done')
        return depth, mask_reduced, depth_masked_reduced, pointcloud_masked_reduced, pointcloud_masked_world_reduced
    
    def estimate_scale_shift(self, data1, data2, transform1, transform2, show=False):
        """
        Estimates the optimal scale and shift values of data1 to fit data2.
        
        Parameters
        ----------
        data1 : tuple
            A tuple containing the first dataset, including the mask and depth map.
        data2 : tuple
            A tuple containing the second dataset, including the mask.
        transform1 : TransformStamped
            The transformation from the first camera to the world frame.
        transform2 : TransformStamped
            The transformation from the second camera to the world frame.
        show : bool, optional
            If True, display visualizations of the masks and point clouds.
        
        Returns
        -------
        best_alpha : float
            The estimated optimal scale factor.
        best_beta : float
            The estimated optimal shift value.
        best_pointcloud_world : np.ndarray
            The transformed point cloud in world coordinates after applying the scale and shift.
        """

        print('Estimating scale and shift...')
        _, mask1, depth_masked1, _, pointcloud_masked_world1 = data1
        _, mask2, _, _, _ = data2

        max_inliers_counter = 0
        best_num_union = None
        best_alpha, best_beta = 0, 0
        best_pointcloud_world = None
        best_projection = None
        num_skips = 0
        # Loop N times
        for i in range(1000):
            if rospy.is_shutdown(): exit()
            # Pick two random points from pointcloud 1
            pointA_world = random.choice(pointcloud_masked_world1.points)
            pointB_world = random.choice(pointcloud_masked_world1.points)
            if np.all(pointA_world == pointB_world):
                num_skips += 1
                continue

            # Project the points into coordinates of camera 2
            projectionA_cam2 = self.depth_anything_wrapper.project_point_from_world(pointA_world, transform2)
            projectionB_cam2 = self.depth_anything_wrapper.project_point_from_world(pointB_world, transform2)

            # Project the pointcloud into coordinates of camera 2
            projected_pointcloud_depth, projected_pointcloud_mask_2 = project_pointcloud_from_world(pointcloud_masked_world1, transform2)
            # self.grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projectionA_cam2, projectionB_cam2], title="Pointcloud 1 projected onto camera 2 with rand points")

            # Get the closest points between single point (from pointcloud 1) and mask 2
            closestA_cam2 = self.grounded_sam_wrapper.get_closest_point(mask2, projectionA_cam2)
            closestB_cam2 = self.grounded_sam_wrapper.get_closest_point(mask2, projectionB_cam2)

            # Show the closest points
            # self.grounded_sam_wrapper.show_mask_and_points(mask2, [closestA_cam2, closestB_cam2], title="Mask 2 in camera 2 with closest points")
            # self.grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [closestA_cam2, closestB_cam2], title="Pointcloud 1 projected onto camera 2 with closest points")

            # self.grounded_sam_wrapper.show_mask_and_points(mask2, [closestA_cam2, projectionA_cam2], title="projection and closest point A")
            # self.grounded_sam_wrapper.show_mask_and_points(mask2, [closestB_cam2, projectionB_cam2], title="projection and closest point B")

            # Triangulate
            trtiangulatedA_world = self.depth_anything_wrapper.triangulate(pointA_world, transform1, closestA_cam2, transform2)
            trtiangulatedB_world = self.depth_anything_wrapper.triangulate(pointB_world, transform1, closestB_cam2, transform2)
            # print(f'trtiangulatedA_world: {trtiangulatedA_world}')
            # print(f'trtiangulatedB_world: {trtiangulatedB_world}')
            try:
                # Project the triangulated points into coordinates of camera 2
                projected_triangulatedA_cam2 = self.depth_anything_wrapper.project_point_from_world(trtiangulatedA_world, transform1)
                projected_triangulatedB_cam2 = self.depth_anything_wrapper.project_point_from_world(trtiangulatedB_world, transform1)
            except Exception as e:
                # print(f'{e}, Skipping to next iteration')
                num_skips += 1
                continue
            
            # # Show mask 2 and triangulated points
            # self.grounded_sam_wrapper.show_mask_and_points(mask2, [projected_triangulatedA_cam2, projected_triangulatedB_cam2], title="Mask 2 with triangulated points")
            # # Show pointcloud 1 and triangulated points in camera 2
            # self.grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projected_triangulatedA_cam2, projected_triangulatedB_cam2], title="Pointcloud 1 projected onto camera 2 with triangulated points")
            
            # Transform triangulated point into camera 1 coordinates
            triangulated1_A = self.depth_anything_wrapper.transform_point_from_world(trtiangulatedA_world, transform1)
            triangulated1_B = self.depth_anything_wrapper.transform_point_from_world(trtiangulatedB_world, transform1)

            # Get  distance of original pointA_world and pointB_world
            z_A_orig = self.depth_anything_wrapper.transform_point_from_world(pointA_world, transform1)[2]
            z_B_orig = self.depth_anything_wrapper.transform_point_from_world(pointB_world, transform1)[2]

            # Get distance id triangulated point
            z_A_triangulated = triangulated1_A[2]
            z_B_triangulated = triangulated1_B[2]
            # print(f'depth_A_orig: {z_A_orig}')
            # print(f'depth_A_triangulated: {z_A_triangulated}')
            # print(f'depth_B_orig: {z_B_orig}')
            # print(f'depth_B_triangulated: {z_B_triangulated}')
            
            # Calculate scale and shift
            alpha = (z_A_triangulated - z_B_triangulated) / (z_A_orig - z_B_orig)
            beta = z_A_triangulated - alpha * z_A_orig
            # print(f'alpha: {alpha:.2f}, beta: {beta:.2f}')

            # Scale and shift
            depth_new = self.depth_anything_wrapper.scale_depth_map(depth_masked1, scale=alpha, shift=beta)

            # Show original and scaled depth maps
            # self.depth_anything_wrapper.show_depth_map(depth_masked1, title="Original depth map")
            # self.depth_anything_wrapper.show_depth_map(depth_new, title="Scaled depth map")

            # Get scaled/shifted pointcloud
            new_pc_cam1 = convert_depth_map_to_pointcloud(depth_new)

            # Transform scaled pointcloud into world coordinates
            new_pc_world = self.depth_anything_wrapper.transform_pointcloud_to_world(new_pc_cam1, transform1)

            # Get projection of scaled pointcloud into camera 2
            projection_new_pc2_depth, projection_new_pc2_mask = project_pointcloud_from_world(new_pc_world, transform2)
        
            # Show projection of scaled pointcloud in camera 2 and closest points
            # self.grounded_sam_wrapper.show_mask_and_points(projection_new_pc2_depth, [closestA_cam2, closestB_cam2], title="Scaled pointcloud with closest points")

            # # Show original and scaled depth maps and masks
            # self.grounded_sam_wrapper.show_masks([projection_new_pc2_mask, mask2], title="Scaled depth map and mask")
            # # Show original and scaled pointclouds in world coordinates
            # self.depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_masked_world1, new_pc_world], [transform1], title="Original and scaled pointcloud")

            # Count the number of inliers between mask 2 and projection of scaled pointcloud
            num_inliers, num_union = self.depth_anything_wrapper.count_inliers(projection_new_pc2_mask, mask2)
            # print(f'{i}: num_inliers: {num_inliers}')

            if num_inliers > max_inliers_counter and alpha < 0.5 and alpha > 0 and beta < 0.5 and beta > -0.5:
                max_inliers_counter = num_inliers
                best_num_union = num_union
                best_alpha = alpha
                best_beta = beta
                best_pointcloud_world = new_pc_world
                best_projection = projection_new_pc2_depth

        print(f'Max inliers: {max_inliers_counter}, Inliers ratio: {max_inliers_counter/best_num_union} alpha: {best_alpha:.2f}, beta: {best_beta:.2f}, Skipped points: {num_skips}')

        if show: 
            # Show original and scaled depth maps and masks
            # self.grounded_sam_wrapper.show_masks([best_projection, mask2], title="Scaled depth map and mask")
            self.grounded_sam_wrapper.show_mask_union(best_projection, mask2)
            
            # Show original and scaled pointclouds in world coordinates
            self.depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_masked_world1, best_pointcloud_world], [transform1], title="Original and scaled pointcloud")

        return best_alpha, best_beta, best_pointcloud_world

    def estimate_scale_shift_new(self, data1, data2, transform1, transform2, show=False):
        """
        Estimates the optimal scale and shift values of data1 to fit data2.
        
        Parameters
        ----------
        data1 : tuple
            A tuple containing the first dataset, including the mask and depth map.
        data2 : tuple
            A tuple containing the second dataset, including the mask.
        transform1 : TransformStamped
            The transformation from the first camera to the world frame.
        transform2 : TransformStamped
            The transformation from the second camera to the world frame.
        show : bool, optional
            If True, display visualizations of the masks and point clouds.
        
        Returns
        -------
        best_alpha : float
            The estimated optimal scale factor.
        best_beta : float
            The estimated optimal shift value.
        best_pointcloud_world : np.ndarray
            The transformed point cloud in world coordinates after applying the scale and shift.
        """
        print('Estimating scale and shift...')
        _, mask1, depth_masked1, _, pointcloud_masked_world1 = data1
        _, mask2, _, _, _ = data2

        bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

        start = time.perf_counter()
        result = opt.differential_evolution(
            self.inliers_function,
            bounds,
            args=(depth_masked1, transform1, mask2, transform2),
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=1e-2,
            disp=False
        )
        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        # start = time.perf_counter()
        y_opt = -self.inliers_function([alpha_opt, beta_opt], depth_masked1, transform1, mask2, transform2)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        depth_opt = scale_depth_map(depth_masked1, scale=alpha_opt, shift=beta_opt)
        pc_cam1_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam1_opt, transform1)
        projection_pc_depth_cam2_opt, projection_pc_mask_cam2_opt = project_pointcloud_from_world(pc_cam0_opt, transform2, camera_parameters)

        fixed_projection_pc_depth_cam2_opt = fill_mask_holes(projection_pc_mask_cam2_opt)

        # self.depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_cam0_opt], [transform1])

        num_inliers, num_inliers_union = count_inliers(fixed_projection_pc_depth_cam2_opt, mask2)

        print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Inliers: {y_opt}, Inliers ratio: {num_inliers/num_inliers_union:.2f}')


        # Show mask and pointcloud
        if show:
            show_masks([projection_pc_mask_cam2_opt], title='Optimal Projection')
            show_masks([fixed_projection_pc_depth_cam2_opt], title='Projection holes fixed')
            show_masks_union(projection_pc_mask_cam2_opt, fixed_projection_pc_depth_cam2_opt, title="Optimal Projection orig & holes fixed Projection")
            show_masks_union(fixed_projection_pc_depth_cam2_opt, mask2, title='Optimal Projection with holes fixed vs Mask to fit')

        return alpha_opt, beta_opt, pc_cam0_opt, num_inliers, num_inliers_union

    def estimate_scale_shift_from_multiple_cameras(self, datas, transforms, show=False):
        """
        Estimates the optimal scale and shift values of data1 to fit data2.
        
        Parameters
        ----------
        data1 : tuple
            A tuple containing the first dataset, including the mask and depth map.
        data2 : tuple
            A tuple containing the second dataset, including the mask.
        transform1 : TransformStamped
            The transformation from the first camera to the world frame.
        transform2 : TransformStamped
            The transformation from the second camera to the world frame.
        show : bool, optional
            If True, display visualizations of the masks and point clouds.
        
        Returns
        -------
        best_alpha : float
            The estimated optimal scale factor.
        best_beta : float
            The estimated optimal shift value.
        best_pointcloud_world : np.ndarray
            The transformed point cloud in world coordinates after applying the scale and shift.
        """
        print('Estimating scale and shift...')
        # _, mask1, depth_masked1, _, pointcloud_masked_world1 = data1
        _, _, depth_masked_last, _, _ = datas[-1]

        bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

        start = time.perf_counter()
        result = opt.differential_evolution(
            self.inliers_function_multiple_cameras,
            bounds,
            args=(datas, transforms),
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=5e-3,
            disp=False
        )
        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        # start = time.perf_counter()
        y_opt = -self.inliers_function_multiple_cameras([alpha_opt, beta_opt], datas, transforms)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        # num_inliers_list = []
        # num_inliers_union_list = []
        projections_list = []
        num_inliers = 0
        num_inliers_union = 0

        depth_opt = scale_depth_map(depth_masked_last, scale=alpha_opt, shift=beta_opt)
        pc_cam_last_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam_last_opt, transforms[-1])
        for data, transform in zip(datas[:-1], transforms[:-1]):
            _, mask, _, _, _ = data
            # Get projection of scaled pointcloud into camera 2
            _, projection_pc_cam_x_opt_mask = project_pointcloud_from_world(pc_cam0_opt, transform, camera_parameters)
            # Fill holes
            fixed_projection_npc_cam0_opt_mask = fill_mask_holes(projection_pc_cam_x_opt_mask)
            num_inliers_temp, num_inliers_union_temp = count_inliers(fixed_projection_npc_cam0_opt_mask, mask)

            print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Inliers: {y_opt}, Inliers ratio: {num_inliers_temp/num_inliers_union_temp:.2f}')
            # num_inliers_list.append(num_inliers)
            # num_inliers_union_list.append(num_inliers_union)
            projections_list.append(fixed_projection_npc_cam0_opt_mask)
            num_inliers += num_inliers_temp
            num_inliers_union += num_inliers_union_temp

        # Show mask and pointcloud
        if show:
            for i, (data, projection) in enumerate(zip(datas[:-1], projections_list)):
                _, mask, _, _, _ = data
                show_masks_union(projection, mask, title=f'Optimal Projection with holes fixed vs Mask {i} to fit')

        return alpha_opt, beta_opt, pc_cam0_opt, num_inliers, num_inliers_union

    def estimate_scale_shift_new_distance(self, data1, data2, transform1, transform2, show=False):
        """
        Estimates the optimal scale and shift values of data1 to fit data2.
        
        Parameters
        ----------
        data1 : tuple
            A tuple containing the first dataset, including the mask and depth map.
        data2 : tuple
            A tuple containing the second dataset, including the mask.
        transform1 : TransformStamped
            The transformation from the first camera to the world frame.
        transform2 : TransformStamped
            The transformation from the second camera to the world frame.
        show : bool, optional
            If True, display visualizations of the masks and point clouds.
        
        Returns
        -------
        best_alpha : float
            The estimated optimal scale factor.
        best_beta : float
            The estimated optimal shift value.
        best_pointcloud_world : np.ndarray
            The transformed point cloud in world coordinates after applying the scale and shift.
        """
        print('Estimating scale and shift...')
        _, mask1, depth_masked1, _, pointcloud_masked_world1 = data1
        _, mask2, _, _, _ = data2

        bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

        start = time.perf_counter()
        result = opt.differential_evolution(
            self.score_function,
            bounds,
            args=(depth_masked1, transform1, mask2, transform2),
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=1e-2,
            disp=False
        )
        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        # start = time.perf_counter()
        y_opt = -self.score_function([alpha_opt, beta_opt], depth_masked1, transform1, mask2, transform2)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        depth_opt = scale_depth_map(depth_masked1, scale=alpha_opt, shift=beta_opt)
        pc_cam1_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam1_opt, transform1)
        projection_pc_depth_cam2_opt, projection_pc_mask_cam2_opt = project_pointcloud_from_world(pc_cam0_opt, transform2, camera_parameters)

        fixed_projection_pc_depth_cam2_opt = fill_mask_holes(projection_pc_mask_cam2_opt)

        # self.depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_cam0_opt], [transform1])

        score = score_mask_match(fixed_projection_pc_depth_cam2_opt, mask2)

        print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Score: {y_opt}')


        # Show mask and pointcloud
        if show:
            show_masks([projection_pc_mask_cam2_opt], title='Optimal Projection')
            show_masks([fixed_projection_pc_depth_cam2_opt], title='Projection holes fixed')
            show_masks_union(projection_pc_mask_cam2_opt, fixed_projection_pc_depth_cam2_opt, title="Optimal Projection orig & holes fixed Projection")
            show_masks_union(fixed_projection_pc_depth_cam2_opt, mask2, title='Optimal Projection with holes fixed vs Mask to fit')

        return alpha_opt, beta_opt, pc_cam0_opt, score

    def estimate_scale_shift_from_multiple_cameras_distance(self, datas, transforms, decay=0.9, show=False):
        """
        Estimates the optimal scale and shift values of data1 to fit data2.
        
        Parameters
        ----------
        data1 : tuple
            A tuple containing the first dataset, including the mask and depth map.
        data2 : tuple
            A tuple containing the second dataset, including the mask.
        transform1 : TransformStamped
            The transformation from the first camera to the world frame.
        transform2 : TransformStamped
            The transformation from the second camera to the world frame.
        show : bool, optional
            If True, display visualizations of the masks and point clouds.
        
        Returns
        -------
        best_alpha : float
            The estimated optimal scale factor.
        best_beta : float
            The estimated optimal shift value.
        best_pointcloud_world : np.ndarray
            The transformed point cloud in world coordinates after applying the scale and shift.
        """
        print('Estimating scale and shift...')
        # _, mask1, depth_masked1, _, pointcloud_masked_world1 = data1
        _, _, depth_masked_last, _, _ = datas[-1]

        bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

        start = time.perf_counter()
        result = opt.differential_evolution(
            self.score_function_multiple_cameras,
            bounds,
            args=(datas, transforms, 0.9),
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=5e-3,
            disp=False
        )
        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        # start = time.perf_counter()
        y_opt = -self.score_function_multiple_cameras([alpha_opt, beta_opt], datas, transforms, decay)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        # num_inliers_list = []
        # num_inliers_union_list = []
        projections_list = []
        score = 0

        depth_opt = scale_depth_map(depth_masked_last, scale=alpha_opt, shift=beta_opt)
        pc_cam_last_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam_last_opt, transforms[-1])
        for data, transform in zip(datas[:-1], transforms[:-1]):
            _, mask, _, _, _ = data
            # Get projection of scaled pointcloud into camera 2
            _, projection_pc_cam_x_opt_mask = project_pointcloud_from_world(pc_cam0_opt, transform, camera_parameters)
            # Fill holes
            fixed_projection_npc_cam0_opt_mask = fill_mask_holes(projection_pc_cam_x_opt_mask)
            score_temp = score_mask_match(fixed_projection_npc_cam0_opt_mask, mask)

            print(f'Score: {score_temp}')
            # num_inliers_list.append(num_inliers)
            # num_inliers_union_list.append(num_inliers_union)
            projections_list.append(fixed_projection_npc_cam0_opt_mask)
            score += score_temp
        
        print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Score: {y_opt}')

        # Show mask and pointcloud
        if show:
            for i, (data, projection) in enumerate(zip(datas[:-1], projections_list)):
                _, mask, _, _, _ = data
                show_masks_union(projection, mask, title=f'Optimal Projection with holes fixed vs Mask {i} to fit')

        return alpha_opt, beta_opt, pc_cam0_opt, score

    def get_desired_pose(self, position, frame="map"):
        # build Pose
        p = Pose()
        p.position.x = float(position[0])
        p.position.y = float(position[1])
        p.position.z = float(position[2])
        p.orientation.x = float(1)
        p.orientation.y = float(0)
        p.orientation.z = float(0)
        p.orientation.w = float(0)
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = frame
        ps.pose = p
        return ps
    
    def get_highest_point(self, pointcloud: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Return the 3D point in the point cloud with the largest Z coordinate.

        Parameters
        ----------
        pointcloud : o3d.geometry.PointCloud
            The input point cloud.

        Returns
        -------
        np.ndarray
            A length-3 array [x, y, z] of the point with the maximum z value.
        """
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("pointcloud must be an open3d.geometry.PointCloud")

        pts = np.asarray(pointcloud.points, dtype=np.float64)
        if pts.size == 0:
            raise ValueError("Point cloud is empty")

        # find the index of the maximum Z
        idx = np.argmax(pts[:, 2])
        return pts[idx]

    def inliers_function(self, x, depth_masked, transform1, mask, transform2, scoring_function='inliers'):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        # Scale and shift
        depth_new = scale_depth_map(depth_masked, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc1 = convert_depth_map_to_pointcloud(depth_new, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc1, transform1)
        # Get projection of scaled pointcloud into camera 2
        projection_new_pc2_depth, projection_new_pc2_mask = project_pointcloud_from_world(new_pc0, transform2, camera_parameters)
        # Fill holes
        fixed_projection_new_pc2_mask = fill_mask_holes(projection_new_pc2_mask)
        # Count the number of inliers between mask 2 and projection of scaled pointcloud
        score, inlier_union = count_inliers(fixed_projection_new_pc2_mask, mask)
        # print(f'num_inliers: {num_inliers}')
        return -score

    def inliers_function_multiple_cameras(self, x, datas, transforms):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        num_inliers = 0
        _, _, depth_masked_last, _, _ = datas[-1]
        # Scale and shift
        depth_new_last = scale_depth_map(depth_masked_last, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc_camera_last = convert_depth_map_to_pointcloud(depth_new_last, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc_camera_last, transforms[-1])

        for data, transform in zip(datas[:-1], transforms[:-1]):
            _, mask, _, _, _ = data
            # Get projection of scaled pointcloud into camera 2
            projection_new_pc_x_depth, projection_new_pc_x_mask = project_pointcloud_from_world(new_pc0, transform, camera_parameters)
            # Fill holes
            fixed_projection_new_pc_x_mask = fill_mask_holes(projection_new_pc_x_mask)
            # Count the number of inliers between mask 2 and projection of scaled pointcloud
            num_inliers_temp, inlier_union_temp = count_inliers(fixed_projection_new_pc_x_mask, mask)
            # print(f'num_inliers: {num_inliers}')
            num_inliers += num_inliers_temp

        return -num_inliers

    def score_function(self, x, depth_masked, transform1, mask, transform2):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        # Scale and shift
        depth_new = scale_depth_map(depth_masked, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc1 = convert_depth_map_to_pointcloud(depth_new, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc1, transform1)
        # Get projection of scaled pointcloud into camera 2
        projection_new_pc2_depth, projection_new_pc2_mask = project_pointcloud_from_world(new_pc0, transform2, camera_parameters)
        # Fill holes
        fixed_projection_new_pc2_mask = fill_mask_holes(projection_new_pc2_mask)
        # Count the number of inliers between mask 2 and projection of scaled pointcloud
        score = score_mask_match(fixed_projection_new_pc2_mask, mask)
        # print(f'num_inliers: {num_inliers}')
        return -score

    def score_function_multiple_cameras(self, x, datas, transforms, decay):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        score = 0
        _, _, depth_masked_last, _, _ = datas[-1]
        # Scale and shift
        depth_new_last = scale_depth_map(depth_masked_last, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc_camera_last = convert_depth_map_to_pointcloud(depth_new_last, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc_camera_last, transforms[-1])

        for i, (data, transform) in enumerate(zip(datas[:-1], transforms[:-1])):
            _, mask, _, _, _ = data
            # Get projection of scaled pointcloud into camera 2
            projection_new_pc_x_depth, projection_new_pc_x_mask = project_pointcloud_from_world(new_pc0, transform, camera_parameters)
            # Fill holes
            fixed_projection_new_pc_x_mask = fill_mask_holes(projection_new_pc_x_mask)
            # Count the number of inliers between mask 2 and projection of scaled pointcloud
            score_temp = score_mask_match(fixed_projection_new_pc_x_mask, mask)
            # print(f'num_inliers: {num_inliers}')
            # print(f'Initial score: {score_temp}')
            score_temp = score_temp * decay**(len(transforms)-2-i)
            # print(f'decay: {decay**(len(transforms)-2-i)}, new score: {score_temp}')
            score += score_temp

        return -score

    def score_function_bspline(self, x, datas, camera_pose, degree):
        if rospy.is_shutdown(): exit()
        ctrl_points = np.array(x).reshape(-1, 3) 
        _, mask, _, _, _ = datas[-1]

        projected_spline = project_bspline(ctrl_points, camera_pose, camera_parameters, degree=degree)
        score = score_mask_match(mask, projected_spline)
        print(f"x: {x},score inside function: {score}")
        return -score
    

def test1():
    my_class = MyClass()
    ros_handler = ROSHandler()

    image1 = cv2.imread(f'/root/workspace/images/moves/cable0.jpg')
    image2 = cv2.imread(f'/root/workspace/images/moves/cable1.jpg')
    transform1 = my_class.convert_trans_and_rot_to_stamped_transform([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    transform2 = my_class.convert_trans_and_rot_to_stamped_transform([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])

    data1 = my_class.process_image(image1, transform1, show=True)
    data2 = my_class.process_image(image2, transform2, show=True)

    alpha, beta, best_pointcloud_world = my_class.estimate_scale_shift(data1, data2, transform1, transform2, show=True)

def test2():
    my_class = MyClass()
    ros_handler = ROSHandler()

    transform_stamped0 = my_class.convert_trans_and_rot_to_stamped_transform([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    transform_stamped6 = my_class.convert_trans_and_rot_to_stamped_transform([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])

    path = ros_handler.interpolate_poses(transform_stamped0, transform_stamped6, num_steps=10)

    while not rospy.is_shutdown():
        ros_handler.publish_path(path, "/my_path")
        rospy.sleep(1)

def test3():
    my_class = MyClass()
    ros_handler = ROSHandler()

    desired_pose = my_class.get_desired_pose([1, 1, 1])
    while not rospy.is_shutdown():
        ros_handler.publish_pose(desired_pose, "/desired_pose")
        rospy.sleep(1)

def pipeline():
    my_class = MyClass()
    ros_handler = ROSHandler()
    image_subscriber = ImageSubscriber('/hsrb/hand_camera/image_rect_color')
    pose_publisher = PosePublisher("/next_pose")
    path_publisher = PathPublisher("/my_path")

    images = []
    transforms = []
    transforms_palm = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []

    # Take image
    images.append(image_subscriber.get_current_image())
    # Get current transform
    transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
    transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map"))
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=True))
    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()
    # Move arm
    next_pose_stamped = create_pose(z=0.1, pitch=-0.4, reference_frame="hand_palm_link")
    pose_publisher.publish(next_pose_stamped)
    # input("Press Enter when moves are finished…")
    rospy.sleep(5)

    while not rospy.is_shutdown(): 
        # Take image
        images.append(image_subscriber.get_current_image())
        # Get current transform
        transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
        transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map"))
        # Process image
        data.append(my_class.process_image(images[-1], transforms[-1], show=True))
        # Estimate scale and shift
        best_alpha, best_beta, best_pc_world, num_inliers, num_inliers_union = my_class.estimate_scale_shift_new(data[-2], data[-1], transforms[-2], transforms[-1], show=True)
        best_alphas.append(best_alpha)
        best_betas.append(best_beta)
        best_pcs_world.append(best_pc_world)
        # Get highest Point in pointcloud
        target_point = my_class.get_highest_point(best_pcs_world[-1])
        tarrget_point_offset = target_point
        tarrget_point_offset[2] += 0.098 - 0.020 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        target_poses.append(my_class.get_desired_pose(tarrget_point_offset))
        # Calculate Path
        target_path = interpolate_poses(transforms_palm[-1], target_poses[-1], num_steps=3)
        # Move arm a step
        path_publisher.publish(target_path)
        usr_input = input("Go to final Pose? y/[n]: ")
        if usr_input == "y":
            _, final_mask = project_pointcloud_from_world(best_pcs_world[-1], target_poses[-1], camera_parameters)
            # target_point_2D = project_point_from_world(target_point, target_poses[-1], camera_parameters)
            # angle = calculate_angle_from_mask_and_point(final_mask, [640//2, 480//2])
            angle = calculate_angle_from_pointcloud(best_pcs_world[-1], target_point)
            rotated_target_pose = rotate_pose_around_z(target_poses[-1], angle)
            pose_publisher.publish(rotated_target_pose)
            break
        else:
            pose_publisher.publish(target_path[1])
        # input("Press Enter when moves are finished…")
        rospy.sleep(5)
    # Loop End

def pipeline2():
    my_class = MyClass()
    ros_handler = ROSHandler()
    image_subscriber = ImageSubscriber('/hsrb/hand_camera/image_rect_color')
    pose_publisher = PosePublisher("/next_pose")
    path_publisher = PathPublisher("/my_path")
    pointcloud_publisher = PointcloudPublisher(topic="my_pointcloud", frame_id="map")

    images = []
    transforms = []
    transforms_palm = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []

    # Take image
    images.append(image_subscriber.get_current_image())
    # Get current transform
    transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
    transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map"))
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=True))
    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()
    # Move arm
    next_pose_stamped = create_pose(z=0.1, pitch=-0.4, reference_frame="hand_palm_link")
    pose_publisher.publish(next_pose_stamped)
    # input("Press Enter when moves are finished…")
    rospy.sleep(5)

    while not rospy.is_shutdown(): 
        # Take image
        images.append(image_subscriber.get_current_image())
        # Get current transform
        transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
        transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map"))
        # Process image
        data.append(my_class.process_image(images[-1], transforms[-1], show=True))
        # Estimate scale and shift
        # best_alpha, best_beta, best_pc_world, num_inliers, num_inliers_union = my_class.estimate_scale_shift_new(data[-2], data[-1], transforms[-2], transforms[-1], show=True)
        best_alpha, best_beta, best_pc_world, score = my_class.estimate_scale_shift_from_multiple_cameras_distance(data, transforms, show=True)
        best_alphas.append(best_alpha)
        best_betas.append(best_beta)
        best_pcs_world.append(best_pc_world)
        pointcloud_publisher.publish(best_pcs_world[-1])
        # Get highest Point in pointcloud
        target_point = my_class.get_highest_point(best_pcs_world[-1])
        tarrget_point_offset = target_point
        tarrget_point_offset[2] += 0.098 + 0.010# - 0.020 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        target_poses.append(my_class.get_desired_pose(tarrget_point_offset))
        # Calculate Path
        target_path = interpolate_poses(transforms_palm[-1], target_poses[-1], num_steps=3)
        # Move arm a step
        path_publisher.publish(target_path)
        usr_input = input("Go to final Pose? y/[n]: ")
        if usr_input == "y":
            _, final_mask = project_pointcloud_from_world(best_pcs_world[-1], target_poses[-1], camera_parameters)
            # target_point_2D = project_point_from_world(target_point, target_poses[-1], camera_parameters)
            # angle = calculate_angle_from_mask_and_point(final_mask, [640//2, 480//2])
            angle = calculate_angle_from_pointcloud(best_pcs_world[-1], target_point)
            rotated_target_pose = rotate_pose_around_z(target_poses[-1], angle)
            pose_publisher.publish(rotated_target_pose)
            break
        else:
            pose_publisher.publish(target_path[1])
        # input("Press Enter when moves are finished…")
        rospy.sleep(5)
    # Loop End

def pipeline_spline():
    my_class = MyClass()
    # ros_handler = ROSHandler()
    image_subscriber = ImageSubscriber('/hsrb/hand_camera/image_rect_color')
    pose_publisher = PosePublisher("/next_pose")
    path_publisher = PathPublisher("/my_path")
    pointcloud_publisher = PointcloudPublisher(topic="my_pointcloud", frame_id="map")
    grasp_point_publisher = PointStampedPublisher("/grasp_point")

    transform_stamped0 = create_stamped_transform_from_trans_and_rot([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    transform_stamped1 = create_stamped_transform_from_trans_and_rot([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    transform_stamped2 = create_stamped_transform_from_trans_and_rot([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    transform_stamped3 = create_stamped_transform_from_trans_and_rot([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    transform_stamped4 = create_stamped_transform_from_trans_and_rot([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    transform_stamped5 = create_stamped_transform_from_trans_and_rot([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    transform_stamped6 = create_stamped_transform_from_trans_and_rot([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    offline_image_name = "tube"
    images = []
    transforms = []
    transforms_palm = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []
    ctrl_points = []

    # Take image
    # images.append(image_subscriber.get_current_image()) # <- online
    images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{0}.jpg')) # <- offline
    # Get current transform
    # transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    transforms.append(transform_stamped0) # <- offline
    # transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=False))
    data[-1][1][:] = cleanup_mask(data[-1][1])
    # show_masks(data[-1][1])

    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()
    # Move arm
    next_pose_stamped = create_pose(z=0.1, pitch=-0.4, reference_frame="hand_palm_link")
    pose_publisher.publish(next_pose_stamped)
    # input("Press Enter when moves are finished…")
    # rospy.sleep(5)
    # spline_2d = extract_2d_spline(data[-1][1])
    # display_2d_spline_gradient(data[-1][1], spline_2d)
    # exit()

    input("Press Enter when image is correct")

    #region -------------------- Depth Anything --------------------
    # Take image
    # images.append(image_subscriber.get_current_image()) # <- online
    images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{1}.jpg')) # <- offline
    # Get current transform
    # transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    transforms.append(transform_stamped1) # <- offline
    # transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=False))
    data[-1][1][:] = cleanup_mask(data[-1][1])
    # show_masks(data[-1][1])
    # Estimate scale and shift
    # best_alpha, best_beta, best_pc_world, num_inliers, num_inliers_union = my_class.estimate_scale_shift_new(data[-2], data[-1], transforms[-2], transforms[-1], show=True)
    best_alpha, best_beta, best_pc_world, score = my_class.estimate_scale_shift_from_multiple_cameras_distance(data, transforms, show=True)
    best_alphas.append(best_alpha)
    best_betas.append(best_beta)
    best_pcs_world.append(best_pc_world)
    pointcloud_publisher.publish(best_pcs_world[-1])
    #endregion -------------------- Depth Enything --------------------

    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(data[-1][0], best_alpha, best_beta)
    # centerline_pts_cam2 = extract_centerline_from_mask(best_depth, data[-1][1], camera_parameters)
    centerline_pts_cam2 = extract_centerline_from_mask_overlap(best_depth, data[-1][1], camera_parameters)
    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, transforms[-1])
    degree = 3
    # Fit B-spline
    ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20))
    # ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20)[3:-3,:])
    print(f"ctrl_points: {ctrl_points[0]}")

    spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
    pointcloud_publisher.publish(spline_pc)

    visualize_spline_with_pc(best_pc_world, ctrl_points[0], degree, title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(ctrl_points[0], transforms[1], camera_parameters, degree=degree)
    show_masks([data[1][1], projected_spline_cam1], "Projected B-Spline Cam1 - 1")
    
    correct_skeleton = skeletonize_mask(data[-1][1])
    show_masks([data[-1][1], correct_skeleton], "Correct Skeleton")

    show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")
    exit()

    # Get highest Point in pointcloud
    target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
    tarrget_point_offset = target_point.copy()
    tarrget_point_offset[2] += 0.098 + 0.010 # - 0.020 # Make target Pose hover above actual target pose - tested offset
    grasp_point_publisher.publish(target_point)
    # Convert to Pose
    target_poses.append(my_class.get_desired_pose(tarrget_point_offset))
    # Calculate Path
    target_path = interpolate_poses(transforms_palm[-1], target_poses[-1], num_steps=4) # <- online
    # target_path = interpolate_poses(transforms[-1], target_poses[-1], num_steps=4) # <- offline
    # Move arm a step
    path_publisher.publish(target_path)
    pose_publisher.publish(target_path[1])
    # input("Press Enter when moves are finished…")
    # rospy.sleep(5)


    while not rospy.is_shutdown():
        input("Press Enter when image is correct")
        # Take image
        images.append(image_subscriber.get_current_image()) # <- online
        # images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{1}.jpg')) # <- offline
        # Get current transform
        transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
        # transforms.append(transform_stamped2) # <- offline
        transforms_palm.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
        # Process image
        data.append(my_class.process_image(images[-1], transforms[-1], show=False))
        data[-1][1][:] = cleanup_mask(data[-1][1])
        # show_masks(data[-1][1])

        projected_spline_cam1 = project_bspline(ctrl_points[-1], transforms[-2], camera_parameters, degree=degree)
        show_masks([data[-2][1], projected_spline_cam1], "Projected B-Spline Cam1")

        projected_spline_cam2 = project_bspline(ctrl_points[-1], transforms[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region Translate b-spline
        start = time.perf_counter()
        bounds = [(-0.1, 0.1), (-0.1, 0.1)] # [(x_min, x_max), (y_min,  y_max)]
        result_translation = opt.minimize(
            fun=score_bspline_translation,   # returns –score
            x0=[0, 0],
            args=(data[-1], transforms[-1], camera_parameters, degree, ctrl_points[-1]),
            method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
            bounds=bounds,                     # same ±0.5 bounds per coord
            options={
                'maxiter': 1e6,
                'ftol': 1e-5,
                'eps': 0.0005,
                'disp': True
            }
        )
        end = time.perf_counter()
        print(f"B-spline translation took {end - start:.2f} seconds")
        ctrl_points_translated = apply_translation_to_ctrl_points(ctrl_points[-1], result_translation.x, transforms[-1])

        # ctrl_points_translated = shift_control_points(ctrl_points[-1], data[-1][1], transforms[-1], camera_parameters, degree)

        projected_spline_cam2_translated = project_bspline(ctrl_points_translated, transforms[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2_translated], "Projected B-Spline Cam2 Translated")

        #region Coarse optimize control points
        # 1) Precompute once:
        skeletons, interps = precompute_skeletons_and_interps(data)        # 2) Call optimizer with our new score:
        bounds = make_bspline_bounds(ctrl_points_translated, delta=0.5)
        init_x_coarse = ctrl_points_translated.flatten()
        reg_weight = 0 # 500
        decay = 1 # 0..1
        curvature_weight = 1
        # 2) Coarse/fine minimization calls:
        start = time.perf_counter()
        result_coarse = opt.minimize(
            fun=score_function_bspline_reg_multiple_pre,
            x0=init_x_coarse,
            args=(
                transforms,
                camera_parameters,
                degree,
                init_x_coarse,
                reg_weight,
                decay,
                curvature_weight,
                skeletons,
                interps,
                50
            ),
            method='L-BFGS-B', # 'Powell', # 
            bounds=bounds,
            options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-6, 'xtol':1e-4, 'disp':True}
        )
        end = time.perf_counter()
        print(f"Coarse optimization took {end - start:.2f} seconds")
        print(f"result: {result_coarse}")
        ctrl_points_coarse = result_coarse.x.reshape(-1, 3)
        projected_spline_cam2_coarse = project_bspline(ctrl_points_coarse, transforms[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")
        #endregion

        #region Fine optimizer
        init_x_fine = ctrl_points_coarse.flatten()
        start = time.perf_counter()
        result_fine = opt.minimize(
            fun=score_function_bspline_reg_multiple_pre,
            x0=init_x_fine,
            args=(
                transforms,
                camera_parameters,
                degree,
                init_x_fine,
                reg_weight,
                decay,
                curvature_weight,
                skeletons,
                interps,
                200
            ),
            method='L-BFGS-B', # 'Powell', # 
            bounds=bounds,
            options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-8, 'xtol':1e-4, 'disp':True}
        )
        end = time.perf_counter()
        print(f"Fine optimization took {end - start:.2f} seconds")
        print(f"result: {result_fine}")
        ctrl_points_fine = result_fine.x.reshape(-1, 3)
        projected_spline_cam2_fine = project_bspline(ctrl_points_fine, transforms[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2_fine], "Projected B-Spline Cam2 Fine")

        ctrl_points.append(ctrl_points_fine)
        #endregion

        spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
        pointcloud_publisher.publish(spline_pc)

        projected_spline_opt = project_bspline(ctrl_points[-1], transforms[-1], camera_parameters, degree=degree)
        correct_skeleton_cam2 = skeletonize_mask(data[-1][1])
        show_masks([correct_skeleton_cam2, projected_spline_opt])

        visualize_spline_with_pc(best_pc_world, ctrl_points[-1], degree)

        # Movement
        # Get highest Point in pointcloud
        target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
        tarrget_point_offset = target_point.copy()
        tarrget_point_offset[2] += 0.098 + 0.010 - 0.035 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        target_poses.append(my_class.get_desired_pose(tarrget_point_offset))
        # Calculate Path
        target_path = interpolate_poses(transforms_palm[-1], target_poses[-1], num_steps=4) # <- online
        # target_path = interpolate_poses(transforms[-1], target_poses[-1], num_steps=4) # <- offline
        grasp_point_publisher.publish(target_point)
        # Move arm a step
        path_publisher.publish(target_path)
        usr_input = input("Go to final Pose? y/[n] or enter pose index: ").strip().lower()
        if usr_input == "c": exit()
        if usr_input == "y":
            rotated_target_pose = rotate_pose_around_z(target_poses[-1], target_angle)
            pose_publisher.publish(rotated_target_pose)
            break
        else:
            try:
                idx = int(usr_input)
                pose_publisher.publish(target_path[idx])
            except ValueError:
                pose_publisher.publish(target_path[1])
        # input("Press Enter when moves are finished…")
        rospy.sleep(5)

    #endregion -------------------- Spline --------------------


if __name__ == "__main__":
    rospy.init_node("MyClass", anonymous=True)
    # pipeline()
    # pipeline2()
    pipeline_spline()   

    # my_class = MyClass()
    # ros_handler = ROSHandler()


    # pose_publisher = PosePublisher("/next_pose")
    # path_publisher = PathPublisher("/my_path")
    # transforms = []
    # transforms.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
    # desired_pose = my_class.get_desired_pose([1, 1, 1])
    # path = ros_handler.interpolate_poses(transforms[-1], desired_pose, num_steps=10)
    # next_pose = path[1]

    # print("Publishing...")
    # rate = rospy.Rate(1)          # 1 Hz
    # while not rospy.is_shutdown():
    #     pose_publisher.publish(next_pose)
    #     path_publisher.publish(path)
    #     rate.sleep()


