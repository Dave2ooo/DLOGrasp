from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
# from ROS_handler import ROSHandler
from ROS_handler_new import ROSHandler
from camera_handler import ImageSubscriber
from publisher import *
from my_utils import *
from save_data import *
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
import rospy
import cv2
import time
import random
import scipy.optimize as opt

from save_load_numpy import load_numpy_from_file
from save_data import load_pose_stamped

camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)

class ImageProcessing:
    def __init__(self):
        self.depth_anything_wrapper = DepthAnythingWrapper()
        self.grounded_sam_wrapper = GroundedSamWrapper()
    
    def get_mask(self, image, prompt, show: bool = False):
        mask = self.grounded_sam_wrapper.get_mask(image, prompt)
        if mask is None:
            return None
        
        mask = mask[0][0]
        print(f"mask.shape: {mask.shape}")
        if show:
            show_masks([mask], title="Original Mask")
        return mask

    def get_depth_masked(self, image, mask, show: bool = False):
        depth = self.depth_anything_wrapper.get_depth_map(image)
        depth_masked = mask_depth_map(depth, mask)
        if show:
            show_depth_map(depth_masked, title="Original Depth Map Masked")
        return depth_masked

    def get_depth_unmasked(self, image, show: bool = False):
        depth = self.depth_anything_wrapper.get_depth_map(image)
        if show:
            show_depth_map(depth, title="Original Depth Map Masked")
        return depth


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

    def estimate_scale_shift_from_multiple_cameras(self, datas, camera_poses: list[PoseStamped], show=False):
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
            args=(datas, camera_poses),
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
        y_opt = -self.inliers_function_multiple_cameras([alpha_opt, beta_opt], datas, camera_poses)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        # num_inliers_list = []
        # num_inliers_union_list = []
        projections_list = []
        num_inliers = 0
        num_inliers_union = 0

        depth_opt = scale_depth_map(depth_masked_last, scale=alpha_opt, shift=beta_opt)
        pc_cam_last_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam_last_opt, camera_poses[-1])
        for data, transform in zip(datas[:-1], camera_poses[:-1]):
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

    def estimate_scale_shift_from_multiple_cameras_distance(self, depths, masks, camera_poses: list[PoseStamped], decay=0.9, show=False):
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
        depth_masked_last= depths[-1]


        bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

        start = time.perf_counter()
        result = opt.minimize(
            self.score_function_multiple_cameras,
            bounds=bounds,
            args=(depths, masks, camera_poses, 0.9),
            # strategy='best1bin',
            # maxiter=20,
            # popsize=15,
            # tol=5e-6,
            # disp=True,
            x0 = [0, 0],
            method='Powell', # 'L-BFGS-B',
            options={'maxiter':1e6, 'ftol':1e-10, 'eps':1e-8, 'xtol':1e-8, 'disp':True, 'maxfun':1e6}
        )

        # result_coarse = opt.minimize(
        #     fun=score_function_bspline_reg_multiple_pre,
        #     x0=init_x_coarse,
        #     args=(
        #         camera_poses,
        #         camera_parameters,
        #         degree,
        #         init_x_coarse,
        #         reg_weight,
        #         decay,
        #         curvature_weight,
        #         skeletons,
        #         interps,
        #         50
        #     ),
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-6, 'xtol':1e-4, 'disp':True, 'maxfun':1e6}
        # )

        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        # start = time.perf_counter()
        y_opt = -self.score_function_multiple_cameras([alpha_opt, beta_opt], depths, masks, camera_poses, decay)
        # end = time.perf_counter()
        # print(f"One function call took {end - start:.4f} seconds")

        # num_inliers_list = []
        # num_inliers_union_list = []
        projections_list = []
        score = 0

        depth_opt = scale_depth_map(depth_masked_last, scale=alpha_opt, shift=beta_opt)
        pc_cam_last_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
        pc_cam0_opt = transform_pointcloud_to_world(pc_cam_last_opt, camera_poses[-1])
        for mask, camera_pose in zip(masks, camera_poses[:-1]):
            # Get projection of scaled pointcloud into camera 2
            _, projection_pc_cam_x_opt_mask = project_pointcloud_from_world(pc_cam0_opt, camera_pose, camera_parameters)
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
            for i, (mask, projection) in enumerate(zip(masks, projections_list)):
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

    def inliers_function_multiple_cameras(self, x, datas, camera_poses):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        num_inliers = 0
        _, _, depth_masked_last, _, _ = datas[-1]
        # Scale and shift
        depth_new_last = scale_depth_map(depth_masked_last, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc_camera_last = convert_depth_map_to_pointcloud(depth_new_last, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc_camera_last, camera_poses[-1])

        for data, transform in zip(datas[:-1], camera_poses[:-1]):
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

    def score_function_multiple_cameras(self, x, depths, masks, camera_poses, decay):
        if rospy.is_shutdown(): exit()
        alpha, beta = x
        score = 0
        depth_masked_last = depths[-1]
        # Scale and shift
        depth_new_last = scale_depth_map(depth_masked_last, scale=alpha, shift=beta)
        # Get scaled/shifted pointcloud
        new_pc_camera_last = convert_depth_map_to_pointcloud(depth_new_last, camera_parameters)
        # Transform scaled pointcloud into world coordinates
        new_pc0 = transform_pointcloud_to_world(new_pc_camera_last, camera_poses[-1])

        for i, (mask, camera_pose) in enumerate(zip(masks, camera_poses)):
            # Get projection of scaled pointcloud into camera 2
            projection_new_pc_x_depth, projection_new_pc_x_mask = project_pointcloud_from_world(new_pc0, camera_pose, camera_parameters)
            # Fill holes
            fixed_projection_new_pc_x_mask = fill_mask_holes(projection_new_pc_x_mask)
            # Count the number of inliers between mask 2 and projection of scaled pointcloud
            score_temp = score_mask_match(fixed_projection_new_pc_x_mask, mask)
            # print(f'num_inliers: {num_inliers}')
            # print(f'Initial score: {score_temp}')
            score_temp = score_temp * decay**(len(camera_poses)-2-i)
            # print(f'decay: {decay**(len(camera_poses)-2-i)}, new score: {score_temp}')
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
    

def pipeline_spline():
    image_processing = None #ImageProcessing()

    ros_handler = ROSHandler()
    image_subscriber = ImageSubscriber('/hsrb/hand_camera/image_rect_color')
    next_pose_publisher = PosePublisher("/next_pose")
    target_pose_publisher = PosePublisher("/target_pose")
    # path_publisher = PathPublisher("/my_path")
    pointcloud_publisher = PointcloudPublisher(topic="my_pointcloud", frame_id="map")
    grasp_point_publisher = PointStampedPublisher("/grasp_point")
    save_data_class = save_data()

    # offline_folder = '/root/workspace/images/thesis_images/' + '2025_08_04_11-17'
    # folder_name_image = f'{offline_folder}/image'
    # folder_name_camera_pose = f'{offline_folder}/camera_pose'
    # folder_name_palm_pose = f'{offline_folder}/palm_pose'
    # offline_counter = 0

    optimizer_decay=1
    optimizer_num_samples=20
    optimizer_smoothness_weight=1e0 # 1e-1 # 1e-2
    optimizer_verbose=2
    optimizer_symmetric=False

    num_initial_ctrl_points = None #15
    num_interpolate_poses = 5

    camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)
    SAM_prompt = "wire.cable.tube."

    hand_camera_frame = "hand_camera_frame"
    map_frame = "map"
    hand_palm_frame = "hand_palm_link"

    degree = 3

    images = []
    masks = []
    depths_orig = []
    depths = []
    camera_poses = []
    palm_poses = []
    palm_poses = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []
    b_splines = []

    optimization_time_translate = []
    optimization_time_coarse = []
    optimization_time_fine = []
    optimization_cost_translate = []
    optimization_cost_coarse = []
    optimization_cost_fine = []

    starting_pose = ros_handler.get_current_pose("hand_palm_link", "map") # <- online



    # Take image

    # Get current transform
    camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    # camera_poses.append(load_pose_stamped(folder_name_camera_pose, str(offline_counter))) # <- offline
    palm_poses.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    # palm_poses.append(load_pose_stamped(folder_name_palm_pose, str(offline_counter))) # <- offline
    # Process image
    # online:
    usr_input_correct_mask = "n"
    while usr_input_correct_mask == "n":
        temp_mask = None
        while temp_mask is None:
            images = []
            images.append(image_subscriber.get_current_image(show=True))  # <- online
            # images.append(cv2.imread(f'{folder_name_image}/{offline_counter}.png')) # <- offline
            # offline_counter += 1 # <- offline
            image_processing = ImageProcessing()
            temp_mask = image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=True)
            if temp_mask is None:
                usr_input = input("No Object Detected. Repeat? [y]/n: ")
                if usr_input == "n": exit()

        masks.append(temp_mask)
        depths_orig.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
        depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))
        # show_masks(masks[-1])

        usr_input_correct_mask = input("Mask Correct? [y]/n: ")
        if usr_input_correct_mask == "c": exit()


    
    save_data_class.save_all(images[-1], masks[-1], depths_orig[-1], depths[-1], camera_poses[-1], palm_poses[-1], None, None, None)

    input("Capture image with smartphone and press Enter when done…")
    # Move arm
    next_pose_stamped = create_pose(x=0.1, z=0.1, pitch=-0.4, reference_frame="hand_palm_link")
    next_pose_publisher.publish(next_pose_stamped)

    rospy.sleep(5) # <- online

    # spline_2d = extract_2d_spline(masks[-1])
    # display_2d_spline_gradient(masks[-1], spline_2d)
    # exit()

    # Take image
    images.append(image_subscriber.get_current_image()) # <- online
    # images.append(cv2.imread(f'{folder_name_image}/{offline_counter}.png')) # <- offline
    # Get current transform
    camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    # camera_poses.append(load_pose_stamped(folder_name_camera_pose, str(offline_counter))) # <- offline
    # offline_counter += 1 # <- offline
    palm_poses.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    # palm_poses.append(load_pose_stamped(folder_name_palm_pose, str(offline_counter))) # <- offline
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=True))
    depths_orig.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))
    # show_masks(masks[-1])
    # Estimate scale and shift
    # best_alpha, best_beta, best_pc_world, num_inliers, num_inliers_union = my_class.estimate_scale_shift_new(data[-2], data[-1], camera_poses[-2], camera_poses[-1], show=True)

    #region old scale-shift
    # best_alpha, best_beta, best_pc_world, score = my_class.estimate_scale_shift_from_multiple_cameras_distance(depths, masks, camera_poses, show=True)
    # best_alphas.append(best_alpha)
    # best_betas.append(best_beta)
    # best_pcs_world.append(best_pc_world)
    # pointcloud_publisher.publish(best_pcs_world[-1])
    #endregion old scale-shift

    # best_alpha, best_beta, score = interactive_scale_shift(depth1=depths[0], mask2=masks[1], pose1=camera_poses[0], pose2=camera_poses[1], camera_parameters=camera_parameters)
  

    #region new scale-shift
    best_alpha, best_beta, best_pc_world, score = optimize_depth_map(depths=depths, masks=masks, camera_poses=camera_poses, camera_parameters=camera_parameters, show=True)
    best_pcs_world.append(best_pc_world)
    pointcloud_publisher.publish(best_pcs_world[-1])
    #endregion new scale-shift

    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(depths[-1], best_alpha, best_beta)
    # centerline_pts_cam2 = extract_centerline_from_mask(best_depth, masks[-1], camera_parameters)
    centerline_pts_cam2_array = extract_centerline_from_mask_individual(best_depth, masks[-1], camera_parameters, save_data_class, show=True)
    # print(f"centerline_pts_cam2_array: {len(centerline_pts_cam2_array)}: {centerline_pts_cam2_array}")
    centerline_pts_cam2 = max(centerline_pts_cam2_array, key=lambda s: s.shape[0]) # Extract the longest path
    # print(f"centerline_pts_cam2: {centerline_pts_cam2.shape[0]}: {centerline_pts_cam2}")
    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, camera_poses[-1])
    # Fit B-spline
    b_splines.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20, num_ctrl=20))

    save_data_class.save_initial_spline(b_splines[-1])

    spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
    pointcloud_publisher.publish(spline_pc)

    visualize_spline_with_pc(best_pc_world, b_splines[0], title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(b_splines[0], camera_poses[1], camera_parameters)
    show_masks([masks[1], projected_spline_cam1], "Projected B-Spline Cam1 - 1")
    
    correct_skeleton = skeletonize_mask(masks[-1])
    show_masks([masks[-1], correct_skeleton], "Correct Skeleton")

    show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")


    best_pointcloud = convert_depth_map_to_pointcloud(best_depth, camera_parameters)
    best_pointcloud_world = transform_pointcloud_to_world(best_pointcloud, camera_poses[-1])
    save_data_class.save_pointcloud_and_spline(best_pointcloud_world, b_splines[-1], "best_pointcloud_and_spline")

    #region optimize control points - new-pre - funktioniert, arbeite mit dem weiter
    skeletons, interps = precompute_skeletons_and_interps(masks) 
    print("--------------------    Coarse Optimization    --------------------")
    start = time.perf_counter()
    coarse_bspline, opt_cost = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                            camera_parameters=camera_parameters,
                            masks=masks,
                            camera_poses=camera_poses,
                            decay=1,
                            reg_weight=0.2,
                             curvature_weight=0.5e-1, # 0.5e-2,
                            num_samples=50,
                            symmetric=True,
                            translate=False,
                            disp=True)
    end = time.perf_counter()

    optimization_time_coarse.append(end-start)
    optimization_cost_coarse.append(opt_cost)

    b_splines.append(coarse_bspline)


    for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")

    spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
    pointcloud_publisher.publish(spline_pc)


    print("--------------------    Fine Optimization    --------------------")
    start = time.perf_counter()
    fine_bspline, opt_cost = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                            camera_parameters=camera_parameters,
                            masks=masks,
                            camera_poses=camera_poses,
                            decay=1,
                            reg_weight=0.2,
                             curvature_weight=0.5e-1, # 0.5e-2,
                            num_samples=200,
                            symmetric=True,
                            translate=False,
                            disp=True)
    end = time.perf_counter()

    optimization_time_fine.append(end-start)
    optimization_cost_fine.append(opt_cost)

    b_splines.append(fine_bspline)
    

    for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
    #endregion optimize control points - new-pre

    save_data_class.save_all(images[-1], masks[-1], depths_orig[-1], depths[-1], camera_poses[-1], palm_poses[-1], coarse_bspline, fine_bspline, None)

    # input("Capture image with smartphone and press Enter when done…")


    #region Online:
    # Get highest Point in pointcloud
    # target_point, target_angle = get_highest_point_and_angle_spline(b_splines[-1])
    target_point, target_angle = get_midpoint_and_angle_spline(b_splines[-1])
    print(f"target_point: {target_point}", f"target_angle: {target_angle}")
    tarrget_point_offset = target_point.copy()
    tarrget_point_offset[2] += 0.098 - 0.015 # Make target Pose hover above actual target pose - tested offset
    grasp_point_publisher.publish(target_point)
    # Convert to Pose
    base_footprint = ros_handler.get_current_pose("base_footprint", map_frame) # <- online
    target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
    # Calculate Path
    # target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4)
    next_pose = control_law(current_pose=palm_poses[-1], target_pose=target_poses[-1], step_size=0.25)
    # target_path = interpolate_poses(camera_poses[-1], target_poses[-1], num_steps=4) # <- offline
    # Move arm a step
    target_pose_publisher.publish(target_poses[-1])
    next_pose_publisher.publish(next_pose)
    # input("Press Enter when moves are finished…")
    #endregion Online

    for loop in range(3):
    # while not rospy.is_shutdown():
        rospy.sleep(5)
        # Take image
        images.append(image_subscriber.get_current_image()) # <- online
        # images.append(cv2.imread(f'{folder_name_image}/{offline_counter}.png')) # <- offline
        # Get current transform
        camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
        # camera_poses.append(load_pose_stamped(folder_name_camera_pose, str(offline_counter))) # <- offline
        # offline_counter += 1 # <- offline
        palm_poses.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
        # palm_poses.append(load_pose_stamped(folder_name_palm_pose, str(offline_counter))) # <- offline
        # Process image
        masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
        # show_masks(masks[-1])

        projected_spline_cam1 = project_bspline(b_splines[-1], camera_poses[-2], camera_parameters)
        # show_masks([masks[-2], projected_spline_cam1], "Projected B-Spline Cam1")

        projected_spline_cam2 = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
        # show_masks([masks[-1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region Translate b-spline
        start = time.perf_counter()
        bounds = [(-0.3, 0.3), (-0.3, 0.3)] # [(x_min, x_max), (y_min,  y_max)]

        result_translation = opt.minimize(
            fun=score_bspline_translation,   # returns –score
            x0=[0, 0],
            args=(masks[-1], camera_poses[-1], camera_parameters, degree, b_splines[-1]),
            method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
            bounds=bounds,                     # same ±0.5 bounds per coord
            # options={
            #     'maxiter': 1e6,
            #     'ftol': 1e-5,
            #     'eps': 0.0005,
            #     'disp': True
            # }
            options={'maxiter':1e8, 'ftol':1e-2, 'eps':1e-6, 'disp':True, 'maxfun':1e8}
        )
        end = time.perf_counter()
        optimization_time_translate.append(end - start)
        print(f"B-spline translation took {end - start:.2f} seconds")

        opt_cost = score_bspline_translation(result_translation.x, masks[-1], camera_poses[-1], camera_parameters, degree, b_splines[-1])
        optimization_cost_translate.append(opt_cost)

        bspline_translated = apply_translation_to_ctrl_points(b_splines[-1], result_translation.x, camera_poses[-1])
        #endregion Translate b-spline old

        #region optimize control points - old/working
        # # 1) Precompute once:
        # skeletons, interps = precompute_skeletons_and_interps(masks)        # 2) Call optimizer with our new score:
        # bounds = make_bspline_bounds(bspline_translated, delta=0.4)
        # init_x_coarse = bspline_translated.c.flatten()
        # reg_weight = 0 # 500
        # decay = 1 # 0..1
        # curvature_weight = 1
        # # 2) Coarse/fine minimization calls:
        # start = time.perf_counter()
        # result_coarse = opt.minimize(
        #     fun=score_function_bspline_reg_multiple_pre,
        #     x0=init_x_coarse,
        #     args=(
        #         camera_poses,
        #         camera_parameters,
        #         degree,
        #         init_x_coarse,
        #         reg_weight,
        #         decay,
        #         curvature_weight,
        #         skeletons,
        #         interps,
        #         50
        #     ),
        #     jac='3-point',
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-4, 'xtol':1e-4, 'disp':False, 'maxfun':1e6}
        # )
        # end = time.perf_counter()
        # print(f"Coarse optimization took {end - start:.2f} seconds")
        # print(f"result: {result_coarse}")
        # ctrl_points_coarse = result_coarse.x.reshape(-1, 3)
        # spline_coarse = create_bspline(ctrl_points_coarse)
        # projected_spline_cam2_coarse = project_bspline(spline_coarse, camera_poses[-1], camera_parameters)
        # show_masks([masks[-1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")

        # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        # pointcloud_publisher.publish(spline_pc)


        # init_x_fine = ctrl_points_coarse.flatten()
        # start = time.perf_counter()
        # result_fine = opt.minimize(
        #     fun=score_function_bspline_reg_multiple_pre,
        #     x0=init_x_fine,
        #     args=(
        #         camera_poses,
        #         camera_parameters,
        #         degree,
        #         init_x_fine,
        #         reg_weight,
        #         decay,
        #         curvature_weight,
        #         skeletons,
        #         interps,
        #         200
        #     ),
        #     jac='3-point',
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-4, 'xtol':1e-4, 'disp':False, 'maxfun':1e6}
        # )
        # end = time.perf_counter()
        # print(f"Fine optimization took {end - start:.2f} seconds")
        # print(f"result: {result_fine}")
        # ctrl_points_fine = result_fine.x.reshape(-1, 3)
        # b_splines.append(create_bspline(ctrl_points_fine))
        #endregion

        #region optimize control points - new/semi-working
        # b_splines.append(optimize_bspline_custom(
        #     initial_spline=b_splines[-1],
        #     camera_parameters=camera_parameters,
        #     masks=masks,
        #     camera_poses=camera_poses,
        #     decay=optimizer_decay,
        #     num_samples=optimizer_num_samples,
        #     smoothness_weight=optimizer_smoothness_weight,
        #     symmetric=optimizer_symmetric,
        #     translate=True,
        #     verbose=optimizer_verbose,
        # ))
        # visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
        # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        # pointcloud_publisher.publish(spline_pc)

        # b_splines.append(optimize_bspline_custom(
        #     initial_spline=b_splines[-1],
        #     camera_parameters=camera_parameters,
        #     masks=masks,
        #     camera_poses=camera_poses,
        #     decay=optimizer_decay,
        #     num_samples=50,
        #     smoothness_weight=optimizer_smoothness_weight,
        #     symmetric=optimizer_symmetric,
        #     translate=False,
        #     verbose=optimizer_verbose,
        # ))

        # visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
        # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        # pointcloud_publisher.publish(spline_pc)

        # b_splines.append(optimize_bspline_custom(
        #     initial_spline=b_splines[-1],
        #     camera_parameters=camera_parameters,
        #     masks=masks,
        #     camera_poses=camera_poses,
        #     decay=optimizer_decay,
        #     num_samples=200,
        #     smoothness_weight=optimizer_smoothness_weight,
        #     symmetric=optimizer_symmetric,
        #     translate=False,
        #     verbose=optimizer_verbose,
        # ))
        #endregion optimize control points - new/semi-working

        #region optimize control points - least-squares
        # # 1) Precompute once:
        # skeletons, interps = precompute_skeletons_and_interps(masks) 
        # print("--------------------    Coarse Optimization    --------------------")
        # b_splines.append(optimize_bspline_pre_least_squares(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=10,
        #                      curvature_weight=1,
        #                      num_samples=50,
        #                      symmetric=True,
        #                      translate=False,
        #                      disp=True))

        # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")


        # print("--------------------    Fine Optimization    --------------------")
        # b_splines.append(optimize_bspline_pre_least_squares(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=10,
        #                      curvature_weight=1,
        #                      num_samples=200,
        #                      symmetric=True,
        #                      translate=False,
        #                      disp=True))
        

        # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
        #endregion optimize control points - least-squares
        
        #region optimize control points - new-pre - funktioniert, arbeite mit dem weiter
        skeletons, interps = precompute_skeletons_and_interps(masks) 
        print("--------------------    Coarse Optimization    --------------------")
        start = time.perf_counter()
        coarse_bspline, opt_cost = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                             camera_parameters=camera_parameters,
                             masks=masks,
                             camera_poses=camera_poses,
                             decay=1.1,
                             reg_weight=0.2,
                             curvature_weight=0.5e-1, # 0.5e-2,
                             num_samples=50,
                             symmetric=True,
                             translate=False,
                             disp=True)
        end = time.perf_counter()

        optimization_time_coarse.append(end - start)
        optimization_cost_coarse.append(opt_cost)

        b_splines.append(coarse_bspline)

        for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
            projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
            show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")

        spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        pointcloud_publisher.publish(spline_pc)


        print("--------------------    Fine Optimization    --------------------")
        start = time.perf_counter()
        fine_bspline, opt_cost = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                             camera_parameters=camera_parameters,
                             masks=masks,
                             camera_poses=camera_poses,
                             decay=1.1,
                             reg_weight=0.2,
                             curvature_weight=0.5e-1, # 0.5e-2,
                             num_samples=200,
                             symmetric=True,
                             translate=False,
                             disp=True)
        end = time.perf_counter()
        
        optimization_time_fine.append(end - start)
        optimization_cost_fine.append(opt_cost)

        b_splines.append(fine_bspline)
        

        for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
            projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
            show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
        #endregion optimize control points - new-pre

        save_data_class.save_all(images[-1], masks[-1], None, None, camera_poses[-1], palm_poses[-1], coarse_bspline, fine_bspline, None)

        spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        pointcloud_publisher.publish(spline_pc)

        # projected_spline_opt = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
        # correct_skeleton_cam2 = skeletonize_mask(masks[-1])
        # show_masks([correct_skeleton_cam2, projected_spline_opt])

        # visualize_spline_with_pc(best_pc_world, b_splines[-1], degree)

        # input("Capture image with smartphone and press Enter when done…")
        #region Online
        input("Go to next pose…")
        # Get highest Point in pointcloud
        # target_point, target_angle = get_highest_point_and_angle_spline(b_splines[-1])
        target_point, target_angle = get_midpoint_and_angle_spline(b_splines[-1])
        tarrget_point_offset = target_point.copy()
        tarrget_point_offset[2] += 0.098 - 0.015 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        base_footprint = ros_handler.get_current_pose("base_footprint", map_frame)
        target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
        # Calculate Path
        # target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4) # <- online
        next_pose = control_law(current_pose=palm_poses[-1], target_pose=target_poses[-1], step_size=0.25)
        # target_path = interpolate_poses(camera_poses[-1], target_poses[-1], num_steps=4) # <- offline
        grasp_point_publisher.publish(target_point)
        # Move arm a step
        target_pose_publisher.publish(target_poses[-1])
        # usr_input = input("Go to final Pose? y/[n] or enter pose index: ").strip().lower()
        # if usr_input == "c": exit()
        # if usr_input == "y":
        #     rotated_target_pose = rotate_pose_around_z(target_poses[-1], target_angle)
        #     next_pose_publisher.publish(rotated_target_pose)
        #     break

        next_pose_publisher.publish(next_pose)
        # #endregion Online
        rospy.sleep(5)


    next_pose_publisher.publish(target_poses[-1])
    input("Press to continue...")
    rotated_target_pose = rotate_pose_around_z(target_poses[-1], target_angle)
    next_pose_publisher.publish(rotated_target_pose)

    #endregion -------------------- Spline --------------------

    grasp_success = False
    usr_input = input("Grasp successfull? y/[n]: ").strip().lower()
    if usr_input == "y":
        grasp_success = True

    save_data_class.save_misc_params(best_alpha, best_beta, optimization_time_translate, optimization_time_coarse, optimization_time_fine, optimization_cost_translate, optimization_cost_coarse, optimization_cost_fine, grasp_success)

    # exit() # <- offline

    #region -------------------- More Images for Voxel Carving --------------------
    next_pose_stamped = transform_pose_intrinsic_xy(starting_pose, alpha=-1, beta=-0.4, x=0, y=-0.4, z=0.3)
    # next_pose_stamped = transform_pose_intrinsic_xy(starting_pose, alpha=-1, beta=-0.4, x=0, y=0, z=0)
    next_pose_publisher.publish(next_pose_stamped)
    rospy.sleep(10)
    images.append(image_subscriber.get_current_image())
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    palm_poses.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    save_data_class.save_all(images[-1], masks[-1], None, None, camera_poses[-1], palm_poses[-1], None, None, None)

    starting_pose.header.stamp = rospy.Time.now()
    next_pose_publisher.publish(starting_pose)
    rospy.sleep(10)

    next_pose_stamped = transform_pose_intrinsic_xy(starting_pose, alpha=1, beta=-0.4, x=0, y=0.4, z=0.3)
    next_pose_publisher.publish(next_pose_stamped)
    rospy.sleep(10)
    images.append(image_subscriber.get_current_image())
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map")) # <- online
    palm_poses.append(ros_handler.get_current_pose("hand_palm_link", "map")) # <- online
    save_data_class.save_all(images[-1], masks[-1], None, None, camera_poses[-1], palm_poses[-1], None, None, None)

    # # next_pose_stamped = create_pose(x=0, y=-0.4, z=0.1, roll=-np.pi/4, reference_frame="hand_palm_link")
    # next_pose_stamped = create_pose(y=-0.4, z=0.4, roll=-np.pi/4,reference_frame="hand_palm_link")
    # next_pose_publisher.publish(next_pose_stamped)
    # rospy.sleep(5)
    # next_pose_stamped = create_pose(pitch=-0.4,reference_frame="hand_palm_link")
    # next_pose_publisher.publish(next_pose_stamped)
    # rospy.sleep(5)

    # next_pose_publisher.publish(starting_pose)
    # rospy.sleep(5)

    # next_pose_stamped = create_pose(y=0.4, z=0.4, roll=np.pi/4,reference_frame="hand_palm_link")
    # next_pose_publisher.publish(next_pose_stamped)
    # rospy.sleep(5)
    # next_pose_stamped = create_pose(pitch=-0.4,reference_frame="hand_palm_link")
    # next_pose_publisher.publish(next_pose_stamped)
    # rospy.sleep(5)
    #endregion -------------------- More Images for Voxel Carving --------------------


if __name__ == "__main__":
    rospy.init_node("MyClass", anonymous=True)
    # pipeline()
    # pipeline2()
    pipeline_spline()   

    # my_class = MyClass()
    # ros_handler = ROSHandler()


    # pose_publisher = PosePublisher("/next_pose")
    # path_publisher = PathPublisher("/my_path")
    # camera_poses = []
    # camera_poses.append(ros_handler.get_current_pose("hand_camera_frame", "map"))
    # desired_pose = my_class.get_desired_pose([1, 1, 1])
    # path = ros_handler.interpolate_poses(camera_poses[-1], desired_pose, num_steps=10)
    # next_pose = path[1]

    # print("Publishing...")
    # rate = rospy.Rate(1)          # 1 Hz
    # while not rospy.is_shutdown():
    #     pose_publisher.publish(next_pose)
    #     path_publisher.publish(path)
    #     rate.sleep()


