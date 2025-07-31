# Custom Files
from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from ROS_handler_new import ROSHandler
from camera_handler import ImageSubscriber
from publisher import *
from my_utils import *
from save_load_numpy import *

from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
import rospy
import cv2
import time
import random
import scipy.optimize as opt
from datetime import datetime

camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)

def convert_trans_and_rot_to_stamped_pose(translation, rotation):
    pose_stamped = PoseStamped()
    pose_stamped.pose.position.x = translation[0]
    pose_stamped.pose.position.y = translation[1]
    pose_stamped.pose.position.z = translation[2]

    pose_stamped.pose.orientation.x = rotation[0]
    pose_stamped.pose.orientation.y = rotation[1]
    pose_stamped.pose.orientation.z = rotation[2]
    pose_stamped.pose.orientation.w = rotation[3]
    return pose_stamped

class ImageProcessing:
    def __init__(self):
        # self.depth_anything_wrapper = DepthAnythingWrapper()
        # self.grounded_sam_wrapper = GroundedSamWrapper()
        self.depths_folder = f'/root/workspace/images/moves_depths'
        self.masks_folder = f'/root/workspace/images/moves_masks'
        self.offline_image_name = "tube"
    
    def get_mask(self, index, show: bool = False):
        # mask = self.grounded_sam_wrapper.get_mask(image, prompt)[0][0]
        mask = load_numpy_from_file(self.masks_folder, f'{self.offline_image_name}{index}')
        if show:
            show_masks([mask], title="Original Mask")
        return mask

    def get_depth_masked(self, index, mask, show: bool = False):
        # depth = self.depth_anything_wrapper.get_depth_map(image)
        depth_masked = load_numpy_from_file(self.depths_folder, f'{self.offline_image_name}{index}')
        if show:
            show_depth_map(depth_masked, title="Original Depth Map Masked")
        return depth_masked

class MyClass:
    def __init__(self):
        pass
  
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
            options={'maxiter':1e6, 'ftol':1e-8, 'eps':1e-8, 'xtol':1e-6, 'disp':True, 'maxfun':1e6}
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
    

def tube_grasp_pipeline(debug: bool = False):
    camera_pose0 = convert_trans_and_rot_to_stamped_pose([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    camera_pose1 = convert_trans_and_rot_to_stamped_pose([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    camera_pose2 = convert_trans_and_rot_to_stamped_pose([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    camera_pose3 = convert_trans_and_rot_to_stamped_pose([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    camera_pose4 = convert_trans_and_rot_to_stamped_pose([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    camera_pose5 = convert_trans_and_rot_to_stamped_pose([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    camera_pose6 = convert_trans_and_rot_to_stamped_pose([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    all_camera_poses = [camera_pose0, camera_pose1, camera_pose2, camera_pose3, camera_pose4, camera_pose5, camera_pose6]


    degree = 3

    optimizer_decay=1
    optimizer_num_samples=20
    optimizer_smoothness_weight=1e0 # 1e-1 # 1e-2
    optimizer_verbose=2
    optimizer_symmetric=False

    num_initial_ctrl_points = None #15
    num_interpolate_poses = 5

    SAM_prompt = "wire.cable.tube."

    hand_camera_frame = "hand_camera_frame"
    map_frame = "map"
    hand_palm_frame = "hand_palm_link"

    image_processing = ImageProcessing()
    my_class = MyClass()

    images = []
    masks = []
    depths = []
    camera_poses = []
    palm_poses = []
    target_poses = []
    b_splines = []
    ctrl_points = []
    skeletons = []

    offline_index = 0
    camera_poses.append(all_camera_poses[offline_index])
    masks.append(image_processing.get_mask(offline_index, show=False))
    depths.append(image_processing.get_depth_masked(offline_index, mask=masks[-1], show=False))
    skeletons.append(skeletonize_mask(masks[-1]))
    offline_index += 1
    
    camera_poses.append(all_camera_poses[offline_index])
    masks.append(image_processing.get_mask(offline_index, show=False))
    depths.append(image_processing.get_depth_masked(offline_index, mask=masks[-1], show=False))
    skeletons.append(skeletonize_mask(masks[-1]))
    offline_index += 1

    #region -------------------- Depth Anything --------------------
    # Estimate scale and shift
    #region new scale-shift
    best_alpha, best_beta, best_pc_world, score = optimize_depth_map(depths=depths, masks=masks, camera_poses=camera_poses, camera_parameters=camera_parameters, show=False)
    #endregion new scale-shift
    #endregion -------------------- Depth Enything --------------------


    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(depths[-1], best_alpha, best_beta)
    # centerline_pts_cam2 = extract_centerline_from_mask(best_depth, masks[-1], camera_parameters)
    centerline_pts_cam2_array = extract_centerline_from_mask_individual(best_depth, masks[-1], camera_parameters, show=False)
    # print(f"centerline_pts_cam2_array: {len(centerline_pts_cam2_array)}: {centerline_pts_cam2_array}")

    centerline_pts_cam2 = max(centerline_pts_cam2_array, key=lambda s: s.shape[0]) # Extract the longest path
    # print(f"centerline_pts_cam2: {centerline_pts_cam2.shape[0]}: {centerline_pts_cam2}")

    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, camera_poses[-1])

    # Fit B-spline
    b_splines.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20))

    # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])

    # visualize_spline_with_pc(best_pc_world, b_splines[0], title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(b_splines[0], camera_poses[1], camera_parameters)
    correct_skeleton = skeletonize_mask(masks[-1])
    # show_masks([projected_spline_cam1, correct_skeleton], "Correct Skeleton")

    # show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")

    index = 0
    while not rospy.is_shutdown():
        camera_poses.append(all_camera_poses[offline_index])
        masks.append(image_processing.get_mask(offline_index, show=False))
        depths.append(image_processing.get_depth_masked(offline_index, mask=masks[-1], show=False))
        skeletons.append(skeletonize_mask(masks[-1]))
        offline_index += 1

        # projected_spline_cam1 = project_bspline(b_splines[-1], camera_poses[-2], camera_parameters)
        # show_masks([skeletons[-2], projected_spline_cam1], "Projected B-Spline Cam1")

        # projected_spline_cam2 = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
        # show_masks([skeletons[-1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region Translate b-spline
        # print("--------------------    Translating B-spline    --------------------")
        start1 = time.perf_counter()
        bounds = [(-0.3, 0.3), (-0.3, 0.3)] # [(x_min, x_max), (y_min,  y_max)]
        result_translation = opt.minimize(
            fun=score_bspline_translation,   # returns –score
            x0=[0, 0],
            args=(masks[-1], camera_poses[-1], camera_parameters, degree, b_splines[-1]),
            method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
            bounds=bounds,                     # same ±0.5 bounds per coord
            # options={
            #     'maxiter': 1e6,
            #     'ftol': 1e-6,
            #     'eps': 0.0005,
            #     'disp': True
            # }
            options={'maxiter':1e8, 'ftol':1e-2, 'eps':1e-6, 'disp':True, 'maxfun':1e8}
        )
        end1 = time.perf_counter()
        # print(f"B-spline translation took {end1 - start1:.2f} seconds")
        bspline_translated = apply_translation_to_ctrl_points(b_splines[-1], result_translation.x, camera_poses[-1])

        # _, interps = precompute_skeletons_and_interps(masks)
        # translate_score = score_function_bspline_reg_multiple_pre(bspline_translated.c.flatten(), camera_poses, camera_parameters, degree, b_splines[-1].c.flatten(), 0, 1, 1, skeletons, interps, 200)
        # print(f"Translate Score: {translate_score:.2f} and took  {end-start:.2f} seconds")

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
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-4, 'xtol':1e-4, 'disp':False, 'maxfun':1e6}
        # )
        # end = time.perf_counter()
        # print(f"Coarse optimization took {end - start:.2f} seconds")
        # print(f"result: {result_coarse}")
        # ctrl_points_coarse = result_coarse.x.reshape(-1, 3)
        # spline_coarse = create_bspline(ctrl_points_coarse)
        # # projected_spline_cam2_coarse = project_bspline(spline_coarse, camera_poses[-1], camera_parameters)
        # # show_masks([skeletons[-1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")


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

        #region optimize control points - least-squares
        # # 1) Precompute once:
        # print("--------------------    Coarse Optimization    --------------------")
        # coarse_bspline, coarse_score = optimize_bspline_pre_least_squares(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=0,
        #                      curvature_weight=1,
        #                      num_samples=50,
        #                      symmetric=True,
        #                      translate=False,
        #                      disp=1)
        # b_splines.append(coarse_bspline)

        # # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        # #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        # #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")


        # print("--------------------    Fine Optimization    --------------------")
        # fine_bspline, fine_score = optimize_bspline_pre_least_squares(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=0,
        #                      curvature_weight=1,
        #                      num_samples=200,
        #                      symmetric=True,
        #                      translate=False,
        #                      disp=1)
        # b_splines.append(fine_bspline)
        

        # # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        # #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        # #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
        #endregion optimize control points - least-squares

        #region optimize control points - new-pre
        # 1) Precompute once:
        # print("--------------------    Coarse Optimization    --------------------")
        num_samples_coarse = 50
        num_samples_fine = 200
        start2 = time.perf_counter()
        coarse_bspline = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                             camera_parameters=camera_parameters,
                             masks=masks,
                             camera_poses=camera_poses,
                             decay=1,
                             reg_weight=1,
                             curvature_weight=1e-1,
                             num_samples=num_samples_coarse,
                             symmetric=True,
                             translate=False,
                             disp=2)
        end2 = time.perf_counter()
        b_splines.append(coarse_bspline)

        for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
            projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
            show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")


        # print("--------------------    Fine Optimization    --------------------")
        start3 = time.perf_counter()
        fine_bspline = optimize_bspline_pre_working(initial_spline=b_splines[-1],
                             camera_parameters=camera_parameters,
                             masks=masks,
                             camera_poses=camera_poses,
                             decay=1,
                             reg_weight=1,
                             curvature_weight=1e-1,
                             num_samples=num_samples_fine,
                             symmetric=True,
                             translate=False,
                             disp=2)
        end3 = time.perf_counter()
        b_splines.append(fine_bspline)
        

        for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
            projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
            show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
        #endregion optimize control points - new-pre


        #region optimize control points - least-squared - new
        # # 1) Precompute once:
        # # print("--------------------    Coarse Optimization    --------------------")
        # num_samples_coarse = 50
        # num_samples_fine = 200
        # start2 = time.perf_counter()
        # coarse_bspline = optimize_bspline_pre_least_squares_many_residuals(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=0,
        #                      curvature_weight=1,
        #                      num_samples=num_samples_coarse,
        #                     #  symmetric=True,
        #                     #  translate=False,
        #                      disp=0)
        # end2 = time.perf_counter()
        # b_splines.append(coarse_bspline)

        # # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        # #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        # #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Coarse")


        # # print("--------------------    Fine Optimization    --------------------")
        # start3 = time.perf_counter()
        # fine_bspline = optimize_bspline_pre_least_squares_many_residuals(initial_spline=b_splines[-1],
        #                      camera_parameters=camera_parameters,
        #                      masks=masks,
        #                      camera_poses=camera_poses,
        #                      decay=1,
        #                      reg_weight=0,
        #                      curvature_weight=1,
        #                      num_samples=num_samples_fine,
        #                     #  symmetric=True,
        #                     #  translate=False,
        #                      disp=0)
        # end3 = time.perf_counter()
        # b_splines.append(fine_bspline)
        

        # # for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
        # #     projected_spline_cam2_fine = project_bspline(b_splines[-1], camera_pose, camera_parameters)
        # #     show_masks([skeleton, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")
        #endregion optimize control points - least-squared - one residual

        # visualize_spline_with_pc(best_pc_world, b_splines[-1], degree)
        _, interps = precompute_skeletons_and_interps(masks)
        final_score_minimize = score_function_bspline_reg_multiple_pre(b_splines[-1].c.flatten(), camera_poses, camera_parameters, degree, b_splines[-1].c.flatten(), 0, 1, 1, skeletons, interps, 200)
        print(f"{num_samples_coarse}/{num_samples_fine} Score {index}: {final_score_minimize:.2f}, Time: {end3-start3+end2-start2:.2f} seconds")
        # print(f"{num_samples_coarse} Score {index}: {final_score_minimize:.2f}, Time: {end2-start2:.2f} seconds")

        # ls_score = make_ls_score(
        #     initial_spline=b_splines[-1],
        #     camera_parameters=camera_parameters,
        #     masks=masks,
        #     camera_poses=camera_poses,
        #     decay=0.95,
        #     reg_weight=0,
        #     curvature_weight=1,
        #     num_samples=200,          # must match the final optimiser level
        #     loss="soft_l1",           # must match least_squares(loss=...)
        #     f_scale=1.0               # must match least_squares(f_scale=...)
        # )
        # print(f"Least Squares Score: {ls_score(b_splines[-1])}")

        index += 1

        # print(f"Final Score Least Squares: {fine_score}")

        # usr_input = input("Continue: y/[n]").strip().lower()
        # if usr_input != "y": exit()


    #endregion -------------------- Spline --------------------




if __name__ == "__main__":
    # rospy.init_node("TubeGraspPipeline", anonymous=True)
    tube_grasp_pipeline(debug=True)  










