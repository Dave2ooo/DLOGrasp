# Custom Files
from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from ROS_handler_new import ROSHandler
from camera_handler import ImageSubscriber
from publisher import *
from my_utils_new import *

from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
import rospy
import cv2
import time
import random
import scipy.optimize as opt
from datetime import datetime

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
        self.depth_anything_wrapper = DepthAnythingWrapper()
        self.grounded_sam_wrapper = GroundedSamWrapper()
    
    def get_mask(self, image, prompt, show: bool = False):
        mask = self.grounded_sam_wrapper.get_mask(image, prompt)[0][0]
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


def tube_grasp_pipeline(debug: bool = False):
    camera_pose0 = convert_trans_and_rot_to_stamped_pose([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    camera_pose1 = convert_trans_and_rot_to_stamped_pose([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    camera_pose2 = convert_trans_and_rot_to_stamped_pose([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    camera_pose3 = convert_trans_and_rot_to_stamped_pose([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    camera_pose4 = convert_trans_and_rot_to_stamped_pose([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    camera_pose5 = convert_trans_and_rot_to_stamped_pose([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    camera_pose6 = convert_trans_and_rot_to_stamped_pose([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    all_camera_poses = [camera_pose0, camera_pose1, camera_pose2, camera_pose3, camera_pose4, camera_pose5, camera_pose6]
    offline_image_name = "tube"

    camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)
    SAM_prompt = "wire.cable.tube."

    hand_camera_frame = "hand_camera_frame"
    map_frame = "map"
    hand_palm_frame = "hand_palm_link"

    # For saving images (thesis documentation)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M")
    save_image_folder = f'/root/workspace/images/pipeline_saved/{timestamp}'

    image_processing = ImageProcessing()
    # ros_handler = ROSHandler()
    image_subscriber = ImageSubscriber('/hsrb/hand_camera/image_rect_color')
    pose_publisher = PosePublisher("/next_pose")
    path_publisher = PathPublisher("/my_path")
    pointcloud_publisher = PointcloudPublisher(topic="my_pointcloud", frame_id=map_frame)
    grasp_point_publisher = PointStampedPublisher("/grasp_point")

    images = []
    masks = []
    depths_unmasked = []
    depths = []
    camera_poses = []
    palm_poses = []
    target_poses = []
    b_splines = []
    ctrl_points = []
    # Pipeline
    # images.append(image_subscriber.get_current_image(show=False)) # Take image
    images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{0}.jpg')) # <- offline
    # camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    camera_poses.append(camera_pose0)
    # palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))

    # save_depth_map(data[-1][0], save_image_folder, "Original_Depth")
    # save_depth_map(data[-1][2], save_image_folder, "Masked_Depth")
    # save_masks(data[-1][1], save_image_folder, "Mask")


    # usr_input = input("Mask Correct? [y]/n: ")
    # if usr_input == "n": exit()
    # Move arm a predetermined length
    next_pose = create_pose(z=0.1, pitch=-0.4, reference_frame=hand_palm_frame)
    pose_publisher.publish(next_pose)
    # input("Press Enter when moves are finished…")
    rospy.sleep(5)
    # spline_2d = extract_2d_spline(data[-1][1])
    # display_2d_spline_gradient(data[-1][1], spline_2d)
    # exit()

    # usr_input = input("Press Enter when image is correct")
    # if usr_input == "c": exit()
    #region -------------------- Depth Anything --------------------
    # images.append(image_subscriber.get_current_image(show=False)) # Take image
    images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{1}.jpg')) # <- offline
    # camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    camera_poses.append(camera_pose1)
    # palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))

    # Estimate scale and shift
    print(f"Score: {score_function([1, 0], depth1=depths[-2], camera_pose1=camera_poses[-2], mask2=masks[-1], camera_pose2=camera_poses[-1], camera_parameters=camera_parameters)}")
    best_alpha, best_beta, best_pc_world, score = estimate_scale_shift(depth1=depths[-2], mask2=masks[-1], camera_pose1=camera_poses[-2], camera_pose2=camera_poses[-1], camera_parameters=camera_parameters, show=False)
    pointcloud_publisher.publish(best_pc_world)
    # interactive_scale_shift(depth1=depths[-2], mask2=masks[-1], pose1=camera_poses[-2], pose2=camera_poses[-1], camera_parameters=camera_parameters)

    #endregion -------------------- Depth Enything --------------------
    
    # save_masks([data[-1][1], data[-2][1]], save_image_folder, "Masks")

    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(depths[-1], best_alpha, best_beta)
    centerline_pts_cam2_array = extract_centerline_from_mask_individual(best_depth, masks[-1], camera_parameters, show=False)
    centerline_pts_cam2 = max(centerline_pts_cam2_array, key=lambda s: s.shape[0]) # Extract the longest path
    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, camera_poses[-1])
    show_pointclouds([centerline_pts_world])
    degree = 3
    # Fit B-spline
    b_splines.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-3, nest=20))
    
    ctrl_points.append(b_splines[-1].c)
    print(f"Number of controlpoints: {ctrl_points[-1].shape}")
    print(f"ctrl_points: {ctrl_points[-1]}")

    # interactive_bspline_editor(ctrl_points[-1], masks[-1], camera_poses[-1], camera_parameters, degree)

    # save_mask_spline(data[-1][1], ctrl_points[-1], degree, camera_parameters, camera_poses[-1], save_image_folder, "Spline")

    spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
    pointcloud_publisher.publish(spline_pc)

    # visualize_spline_with_pc(best_pc_world, b_splines[-1], title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
    # show_masks([masks[1], projected_spline_cam1], "Projected B-Spline Cam1 - 1")
    
    correct_skeleton = skeletonize_mask(masks[-1])
    # show_masks([masks[-1], correct_skeleton], "Correct Skeleton")

    # show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")

    # skeletons, interps = precompute_skeletons_and_interps(masks)        # 2) Call optimizer with our new score:











    #region Diff Chamfer Test - working but only one view
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # # === EXAMPLE USAGE ===

    # # 1) Prepare inputs:
    # #    mask: H×W np.ndarray binary
    # #    pose_stamped: ROS PoseStamped in "map" frame
    # #    camera_parameters = (fx, fy, cx, cy)
    # #    spline = BSpline(t=knots, c=ctrl_pts, k=degree)  # SciPy object

    # # --- mask → dist map
    # # mask = cv2.imread("mask.png", 0) > 0
    # mask_dist = mask_to_dist_map(masks[-1])  # (H,W) torch.Tensor
    # mask_dist = mask_dist.to(device)           # H×W tensor

    # # --- pose → torch
    # T_map_cam = pose_stamped_to_matrix(camera_poses[-1])  # np (4×4)
    # T_map_cam_t = torch.from_numpy(T_map_cam).float()
    # T_map_cam_t = T_map_cam_t.to(device)       # 4×4 tensor

    # # 2) Build model
    # model = BSplineFitter(b_splines[-1],
    #                       camera_parameters,
    #                       T_map_cam_t,
    #                       mask_dist,
    #                       num_samples=200).to(device)

    # # 3) Optimize: Adam then L-BFGS
    # opt1 = torch.optim.Adam(model.parameters(), lr=1e-2)
    # for epoch in range(1000):
    #     opt1.zero_grad()
    #     loss = model()
    #     loss.backward()
    #     opt1.step()

    # opt2 = torch.optim.LBFGS(model.parameters(),
    #                          lr=1.0,
    #                          max_iter=100,
    #                          line_search_fn="strong_wolfe")
    # def closure():
    #     opt2.zero_grad()
    #     l = model()
    #     l.backward()
    #     return l
    # opt2.step(closure)

    # print("Final loss:", model().item())

    # # 1) detach & move to NumPy
    # optimized_ctrl_pts = model.ctrl_pts.detach().cpu().numpy()  # shape (n_ctrl, 3)
    # knots_np           = model.knots.cpu().numpy()              # shape (n_ctrl+degree+1,)
    # degree            = int(model.degree)

    # # 2) build the SciPy BSpline
    # optimal_spline = BSpline(t=knots_np,
    #                         c=optimized_ctrl_pts,
    #                         k=degree)

    # # Now `optimal_spline(u)` will give you (len(u), 3) points along the fitted curve.

    # visualize_spline_with_pc(best_pc_world, optimal_spline, title="Optimal Spline")

    # projected_spline = project_bspline(optimal_spline, camera_poses[-1], camera_parameters)
    # show_masks([masks[1], projected_spline], "Projected B-Spline Cam1 - 1")

    #endregion

    #region Diff Chamfer Test - multiple views
    # b_splines.append(
    #     optimize_bspline_distance(
    #         spline=b_splines[-1],
    #         camera_parameters=camera_parameters,
    #         masks=masks,
    #         camera_poses=camera_poses,
    #         decay=0.95,              # or 0.95 etc.
    #         num_samples=200,
    #         adam_iters=2000,
    #         lbfgs_iters=100,
    #         show=True
    #     )
    # )

    # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
    # pointcloud_publisher.publish(spline_pc)

    #endregion

    #region Optimize B-spline ChatGPT 4.1
    # print("Optimizing B-spline with distance to masks and camera poses...")
    # def vis_callbacks(bspl, pts3d):
    #     for i, (mask, pose) in enumerate(zip(masks, camera_poses)):
    #         projected, _ = project_points(pts3d, pose, camera_parameters)
    #         show_masks([mask, projected], f"Projected B-Spline Cam {i}")
    #     visualize_spline_with_pc(pointcloud=None, spline=bspl, num_samples=200, title="Spline with PC")

    # b_splines.append(optimize_bspline_distance(
    #     spline=b_splines[-1],
    #     camera_parameters=camera_parameters,
    #     masks=masks,
    #     camera_poses=camera_poses,
    #     decay=0.1,
    #     num_samples=20,
    #     stay_close_weight=1e-3,
    #     smoothness_weight=1e-4,
    #     max_nfev=30,
    #     verbose=True,
    #     visualize_callbacks=[vis_callbacks]
    # ))
    # ctrl_points.append(b_splines[-1].c)
    # print(f"Number of controlpoints: {ctrl_points[-1].shape}")
    # print(f"ctrl_points: {ctrl_points[-1]}")
    # # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
    # # pointcloud_publisher.publish(spline_pc)

    # visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
    # for i, (mask, camera_pose) in enumerate(zip(masks, camera_poses)): 
    #     projected_spline = project_bspline(b_splines[-1], camera_pose, camera_parameters) 
    #     show_masks([mask, projected_spline], f"Projected B-Spline Cam {i}")
    #endregion

    b_splines[-1] = trim_bspline(b_splines[-1], remove_ctrl_lo=2, remove_ctrl_hi=2)
    visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
    ctrl_points.append(b_splines[-1].c)
    print(f"trimmed ctrl_points {ctrl_points[-1].shape}: {ctrl_points[-1]}")

    #region Optimize B-spline custom
    b_splines.append(optimize_bspline_custom(
        initial_spline=b_splines[-1],
        camera_parameters=camera_parameters,
        masks=masks,
        camera_poses=camera_poses,
        decay=0.95,
        num_samples=20,
        stay_close_weight=1e-3,
        smoothness_weight=1e0,
        max_nfev=30,
        verbose=True
    ))
    visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")

    #endregion













    # # Get highest Point in pointcloud
    # target_point, target_angle = get_highest_point_and_angle_spline(b_splines[-1])
    # print(f"Target Point: {target_point}, Target Angle: {target_angle}")
    # tarrget_point_offset = target_point.copy()
    # tarrget_point_offset[2] += 0.098 + 0.010 # - 0.020 # Make target Pose hover above actual target pose - tested offset
    # grasp_point_publisher.publish(target_point)
    # # Convert to Pose
    # base_footprint = ros_handler.get_current_pose("base_footprint", map_frame)
    # target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
    # # Calculate Path
    # target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4)
    # # Move arm a step
    # path_publisher.publish(target_path)
    # pose_publisher.publish(target_path[1])
    # # input("Press Enter when moves are finished…")

    index_loop = 1
    while not rospy.is_shutdown():
        index_loop += 1
        rospy.sleep(5)

        # usr_input = input("Press Enter when image is correct")
        # if usr_input == "c": exit()

        # images.append(image_subscriber.get_current_image(show=False)) # Take image
        images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{index_loop}.jpg')) # <- offline
        # camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
        camera_poses.append(all_camera_poses[index_loop]) # <- offline
        # camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
        # palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
        # Process image
        masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))

        projected_spline_cam1 = project_bspline(b_splines[-1], camera_poses[-2], camera_parameters)
        if debug:
            show_masks([masks[-2], projected_spline_cam1], "Projected B-Spline Cam1")

        projected_spline_cam2 = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
        if debug:
            show_masks([masks[-1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region old optimizing
        # #region Translate b-spline
        # # 1) Precompute once:
        # skeletons, interps = precompute_skeletons_and_interps(masks)        # 2) Call optimizer with our new score:

        # # interactive_translate_bspline(ctrl_points[-1], masks[-1], camera_poses[-1], camera_parameters, degree, num_samples=200, shift_range=0.1)

        # # print(f"Translation Score: {translation_score}")

        # start = time.perf_counter()
        # bounds = [(-1, 1), (-1, 1)] # [(x_min, x_max), (y_min,  y_max)]
        # result_translation = opt.minimize(
        # #     fun=score_function_bspline_point_ray_translation,   # returns –score
        # #     x0=[0, 0],
        # #     args=(
        # #         ctrl_points[-1],
        # #         camera_poses[-1],
        # #         camera_parameters,
        # #         degree,
        # #         skeletons[-1],
        # #         20
        # #     ),
        # #     method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
        # #     bounds=bounds,                     # same ±0.5 bounds per coord
        # #     options={
        # #         'maxiter': 1e6,
        # #         'ftol': 1e-5,
        # #         'eps': 0.0005,
        # #         'disp': True
        # #     }
        # # )
        #     # fun=score_bspline_translation,   # returns –score
        #     # x0=[0, 0],
        #     # args=(masks[-1], camera_poses[-1], camera_parameters, degree, ctrl_points[-1]),
        #     # method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
        #     # bounds=bounds,                     # same ±0.5 bounds per coord
        #     # options={
        #     #     'maxiter': 1e6,
        #     #     'ftol': 1e-8,
        #     #     'eps': 0.0005,
        #     #     'disp': True
        #     # }
        #     fun=score_bspline_translation,   # returns –score
        #     x0=[0, 0],
        #     args=(masks[-1], camera_poses[-1], camera_parameters, degree, ctrl_points[-1]),
        #     method='L-BFGS-B', # 'Powell',                  # a quasi-Newton gradient‐based method
        #     bounds=bounds,                     # same ±0.5 bounds per coord
        #     options={
        #         'maxiter': 1e6,
        #         'ftol': 1e-8,
        #         'eps': 0.0005,
        #         'disp': True
        #     }
        # )
        # end = time.perf_counter()
        # print(f"B-spline translation took {end - start:.2f} seconds")
        # print(f"Translation Score: {result_translation.fun} @ {result_translation.x}")
        # ctrl_points_translated = shift_ctrl_points(ctrl_points[-1], result_translation.x, camera_poses[-1], camera_parameters)

        # # ctrl_points_translated = shift_control_points(ctrl_points[-1], data[-1][1], camera_poses[-1], camera_parameters, degree)

        # projected_spline_cam2_translated = project_bspline(ctrl_points_translated, camera_poses[-1], camera_parameters, degree=degree)
        # show_masks([masks[-1], projected_spline_cam2_translated], "Projected B-Spline Cam2 Translated")

        # #region Coarse optimize control points
        # # bounds = make_bspline_bounds(ctrl_points_translated, delta=0.5)
        # # init_x_coarse = ctrl_points_translated.flatten()
        # bounds = make_bspline_bounds(ctrl_points[-1], delta=0.5)
        # init_x_coarse = ctrl_points[-1].flatten()
        # # 2) Coarse/fine minimization calls:
        # start = time.perf_counter()
        # result_coarse = opt.minimize(
        #     # fun=score_function_bspline_point_ray,
        #     # x0=init_x_coarse,
        #     # args=(
        #     #     camera_poses,
        #     #     camera_parameters,
        #     #     degree,
        #     #     skeletons,
        #     #     10,
        #     #     0.9,
        #     # ),
        #     # fun=score_function_bspline_reg_multiple_pre,
        #     # x0=init_x_coarse,
        #     # args=(
        #     #     camera_poses,
        #     #     camera_parameters,
        #     #     degree,
        #     #     init_x_coarse,
        #     #     0,
        #     #     1,
        #     #     1,
        #     #     skeletons,
        #     #     interps,
        #     #     10
        #     # ),
        #     fun=score_function_bspline_chamfer_diff,
        #     x0=init_x_coarse,
        #     args=(
        #         camera_poses,
        #         camera_parameters,
        #         degree,
        #         init_x_coarse,
        #         0,
        #         1,
        #         1,
        #         skeletons,
        #         interps,
        #         10
        #     ),
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-6, 'xtol':1e-4, 'disp':True}
        # )
        # end = time.perf_counter()
        # print(f"Coarse optimization took {end - start:.2f} seconds")
        # print(f"result: {result_coarse}")
        # ctrl_points_coarse = result_coarse.x.reshape(-1, 3)
        # projected_spline_cam2_coarse = project_bspline(ctrl_points_coarse, camera_poses[-1], camera_parameters, degree=degree)
        # show_masks([masks[-1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")
        # #endregion

        # #region Fine optimizer
        # init_x_fine = ctrl_points_coarse.flatten()
        # start = time.perf_counter()
        # result_fine = opt.minimize(
        #     # fun=score_function_bspline_point_ray,
        #     # x0=init_x_fine,
        #     # args=(
        #     #     camera_poses,
        #     #     camera_parameters,
        #     #     degree,
        #     #     skeletons,
        #     #     50,
        #     #     0.9,
        #     # ),
        #     # fun=score_function_bspline_reg_multiple_pre,
        #     # x0=init_x_fine,
        #     # args=(
        #     #     camera_poses,
        #     #     camera_parameters,
        #     #     degree,
        #     #     init_x_fine,
        #     #     0,
        #     #     1,
        #     #     1,
        #     #     skeletons,
        #     #     interps,
        #     #     50
        #     # ),
        #     fun=score_function_bspline_chamfer_diff,
        #     x0=init_x_fine,
        #     args=(
        #         camera_poses,
        #         camera_parameters,
        #         degree,
        #         init_x_fine,
        #         0,
        #         1,
        #         1,
        #         skeletons,
        #         interps,
        #         50
        #     ),
        #     method='L-BFGS-B', # 'Powell', # 
        #     bounds=bounds,
        #     options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-8, 'xtol':1e-4, 'disp':True}
        # )
        # end = time.perf_counter()
        # print(f"Fine optimization took {end - start:.2f} seconds")
        # print(f"result: {result_fine}")
        # ctrl_points_fine = result_fine.x.reshape(-1, 3)
        # for index, (mask, camera_pose) in enumerate(zip(masks, camera_poses)):
        #     projected_spline_cam2_fine = project_bspline(ctrl_points_fine, camera_pose, camera_parameters, degree=degree)
        #     show_masks([mask, projected_spline_cam2_fine], f"Projected B-Spline Cam {index} Fine")

        # ctrl_points.append(ctrl_points_fine)
        #endregion


        #region Diff Chamfer Test - multiple views
        # b_splines.append(
        #     optimize_bspline_distance(
        #         spline=b_splines[-1],
        #         camera_parameters=camera_parameters,
        #         masks=masks,
        #         camera_poses=camera_poses,
        #         decay=0.95,              # or 0.95 etc.
        #         num_samples=200,
        #         adam_iters=2000,
        #         lbfgs_iters=100,
        #         show=True
        #     )
        # )
        #endregion


        #region Optimize B-spline ChatGPT 4.1
        # print("Optimizing B-spline with distance to masks and camera poses...")
        # def vis_callbacks(bspl, pts3d):
        #     for i, (mask, pose) in enumerate(zip(masks, camera_poses)):
        #         projected, _ = project_points(pts3d, pose, camera_parameters)
        #         show_masks([mask, projected], f"Projected B-Spline Cam {i}")
        #     visualize_spline_with_pc(pointcloud=None, spline=bspl, num_samples=200, title="Spline with PC")

        # b_splines.append(optimize_bspline_distance(
        #     spline=b_splines[-1],
        #     camera_parameters=camera_parameters,
        #     masks=masks,
        #     camera_poses=camera_poses,
        #     decay=0.1,
        #     num_samples=20,
        #     # stay_close_weight=1e-3,
        #     # smoothness_weight=1e-4,
        #     stay_close_weight=0,
        #     smoothness_weight=0,
        #     max_nfev=30,
        #     verbose=True,
        #     visualize_callbacks=[vis_callbacks]
        # ))
        # ctrl_points.append(b_splines[-1].c)
        # print(f"Number of controlpoints: {ctrl_points[-1].shape}")
        # print(f"ctrl_points: {ctrl_points[-1]}")
        # # spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        # # pointcloud_publisher.publish(spline_pc)

        # visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
        # for i, (mask, camera_pose) in enumerate(zip(masks, camera_poses)): 
        #     projected_spline = project_bspline(b_splines[-1], camera_pose, camera_parameters) 
        #     show_masks([mask, projected_spline], f"Projected B-Spline Cam {i}")
        #endregion


        #region Optimize B-spline custom
        b_splines.append(optimize_bspline_custom(
            initial_spline=b_splines[-1],
            camera_parameters=camera_parameters,
            masks=masks,
            camera_poses=camera_poses,
            decay=0.95,
            num_samples=20,
            stay_close_weight=1e-3,
            smoothness_weight=1e0,
            max_nfev=30,
            verbose=True
        ))
        visualize_spline_with_pc(pointcloud=best_pc_world, spline=b_splines[-1], num_samples=200, title="Optimized Spline with PC")
        usr_input = input("Continue? ([y]/n)")
        if usr_input == "n":
            break
        #endregion




        spline_pc = convert_bspline_to_pointcloud(b_splines[-1])
        pointcloud_publisher.publish(spline_pc)

        projected_spline_opt = project_bspline(b_splines[-1], camera_poses[-1], camera_parameters)
        correct_skeleton_cam2 = skeletonize_mask(masks[-1])
        show_masks([correct_skeleton_cam2, projected_spline_opt])

        # visualize_spline_with_pc(best_pc_world, b_splines[-1])

        # Movement
        # # Get highest Point in pointcloud
        # target_point, target_angle = get_highest_point_and_angle_spline(b_splines[-1])
        # tarrget_point_offset = target_point.copy()
        # tarrget_point_offset[2] += 0.098 + 0.010 - 0.03 # Make target Pose hover above actual target pose - tested offset
        # # Convert to Pose
        # base_footprint = ros_handler.get_current_pose("base_footprint", map_frame)
        # target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
        # # Calculate Path
        # target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4)
        # grasp_point_publisher.publish(target_point)
        # # Move arm a step
        # path_publisher.publish(target_path)
        # usr_input = input("Go to final Pose? y/[n] or enter pose index: ").strip().lower()
        # if usr_input == "c": exit()
        # if usr_input == "y":
        #     rotated_target_pose = rotate_pose_around_z(target_poses[-1], target_angle)
        #     pose_publisher.publish(rotated_target_pose)
        #     break
        # else:
        #     try:
        #         idx = int(usr_input)
        #         pose_publisher.publish(target_path[idx])
        #     except ValueError:
        #         pose_publisher.publish(target_path[1])
        # # input("Press Enter when moves are finished…")
        # rospy.sleep(5)

    #endregion -------------------- Spline --------------------



if __name__ == "__main__":
    rospy.init_node("TubeGraspPipeline", anonymous=True)
    tube_grasp_pipeline(debug=True)  










