# Custom Files
from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from ROS_handler import ROSHandler
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

camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)

class ImageProcessing:
    def __init__(self):
        self.depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_parameters)
        self.grounded_sam_wrapper = GroundedSamWrapper()
    
    def get_mask(self, image, prompt: str = "cable.", show: bool = False):
        mask = self.grounded_sam_wrapper.get_mask(image, prompt)[0][0]
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
    SAM_prompt = "cable."

    hand_camera_frame = "hand_camera_frame"
    map_frame = "map"
    hand_palm_frame = "hand_palm_link"

    # For saving images (thesis documentation)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M")
    save_image_folder = f'/root/workspace/images/pipeline_saved/{timestamp}'

    image_processing = ImageProcessing()
    ros_handler = ROSHandler()
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
    ctrl_points = []

    # Pipeline
    images.append(image_subscriber.get_current_image(show=debug)) # Take image
    camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=debug))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=debug))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=debug))

    # save_depth_map(data[-1][0], save_image_folder, "Original_Depth")
    # save_depth_map(data[-1][2], save_image_folder, "Masked_Depth")
    # save_masks(data[-1][1], save_image_folder, "Mask")


    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()
    # Move arm a predetermined length
    next_pose = create_pose(z=0.1, pitch=-0.4, reference_frame=hand_palm_frame)
    pose_publisher.publish(next_pose)
    # input("Press Enter when moves are finished…")
    # rospy.sleep(5)
    # spline_2d = extract_2d_spline(data[-1][1])
    # display_2d_spline_gradient(data[-1][1], spline_2d)
    # exit()

    usr_input = input("Press Enter when image is correct")
    if usr_input == "c": exit()
    #region -------------------- Depth Anything --------------------
    images.append(image_subscriber.get_current_image(show=debug)) # Take image
    camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=debug))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=debug))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=debug))

    # Estimate scale and shift
    best_alpha, best_beta, best_pc_world, score = estimate_scale_shift_new_distance(data, camera_poses, show=debug)
    best_pcs_world.append(best_pc_world)
    pointcloud_publisher.publish(best_pcs_world[-1])
    #endregion -------------------- Depth Enything --------------------

    # ---------------------------------- CONTINUE HERE ----------------------------------------------------
    
    # save_masks([data[-1][1], data[-2][1]], save_image_folder, "Masks")

    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(data[-1][0], best_alpha, best_beta)
    # centerline_pts_cam2 = extract_centerline_from_mask(best_depth, data[-1][1], camera_parameters)
    # centerline_pts_cam2 = extract_centerline_from_mask_overlap(best_depth, data[-1][1], camera_parameters, show=True)
    centerline_pts_cam2_array = extract_centerline_from_mask_individual(best_depth, data[-1][1], camera_parameters, show=True)
    centerline_pts_cam2 = max(centerline_pts_cam2_array, key=lambda s: s.shape[0])

    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, camera_poses[-1])
    show_pointclouds([centerline_pts_world])
    degree = 3
    # Fit B-spline
    # ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20))
    ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20)[3:-3,:]) # Remove first and last three control points
    print(f"Number of controlpoints: {ctrl_points[-1].shape}")
    print(f"ctrl_points: {ctrl_points[0]}")

    # save_mask_spline(data[-1][1], ctrl_points[-1], degree, camera_parameters, camera_poses[-1], save_image_folder, "Spline")

    spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
    pointcloud_publisher.publish(spline_pc)

    visualize_spline_with_pc(best_pc_world, ctrl_points[-1], degree, title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
    show_masks([data[1][1], projected_spline_cam1], "Projected B-Spline Cam1 - 1")
    
    correct_skeleton = skeletonize_mask(data[-1][1])
    show_masks([data[-1][1], correct_skeleton], "Correct Skeleton")

    show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")

    # Get highest Point in pointcloud
    target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
    tarrget_point_offset = target_point.copy()
    tarrget_point_offset[2] += 0.098 + 0.010 # - 0.020 # Make target Pose hover above actual target pose - tested offset
    grasp_point_publisher.publish(target_point)
    # Convert to Pose
    base_footprint = ros_handler.get_current_pose("base_footprint", map_frame))
    target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
    # Calculate Path
    target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4) # <- online
    # target_path = interpolate_poses(camera_poses[-1], target_poses[-1], num_steps=4) # <- offline
    # Move arm a step
    path_publisher.publish(target_path)
    pose_publisher.publish(target_path[1])
    # input("Press Enter when moves are finished…")
    # rospy.sleep(5)


    while not rospy.is_shutdown():
        usr_input = input("Press Enter when image is correct")
        if usr_input == "c": exit()
        # Take image
        images.append(image_subscriber.get_current_image()) # <- online
        # images.append(cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{1}.jpg')) # <- offline
        # Get current transform
        camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # <- online
        # camera_poses.append(transform_stamped2) # <- offline
        palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # <- online
        # Process image
        data.append(my_class.process_image(images[-1], camera_poses[-1], show=False))
        data[-1][1][:] = cleanup_mask(data[-1][1])
        # show_masks(data[-1][1])

        projected_spline_cam1 = project_bspline(ctrl_points[-1], camera_poses[-2], camera_parameters, degree=degree)
        show_masks([data[-2][1], projected_spline_cam1], "Projected B-Spline Cam1")

        projected_spline_cam2 = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region Translate b-spline
        start = time.perf_counter()
        bounds = [(-0.1, 0.1), (-0.1, 0.1)] # [(x_min, x_max), (y_min,  y_max)]
        result_translation = opt.minimize(
            fun=score_bspline_translation,   # returns –score
            x0=[0, 0],
            args=(data[-1], camera_poses[-1], camera_parameters, degree, ctrl_points[-1]),
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
        ctrl_points_translated = apply_translation_to_ctrl_points(ctrl_points[-1], result_translation.x, camera_poses[-1])

        # ctrl_points_translated = shift_control_points(ctrl_points[-1], data[-1][1], camera_poses[-1], camera_parameters, degree)

        projected_spline_cam2_translated = project_bspline(ctrl_points_translated, camera_poses[-1], camera_parameters, degree=degree)
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
                camera_poses,
                camera_parameters,
                degree,
                init_x_coarse,
                reg_weight,
                decay,
                curvature_weight,
                skeletons,
                interps,
                10
            ),
            method='L-BFGS-B', # 'Powell', # 
            bounds=bounds,
            options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-6, 'xtol':1e-4, 'disp':True}
        )
        end = time.perf_counter()
        print(f"Coarse optimization took {end - start:.2f} seconds")
        print(f"result: {result_coarse}")
        ctrl_points_coarse = result_coarse.x.reshape(-1, 3)
        projected_spline_cam2_coarse = project_bspline(ctrl_points_coarse, camera_poses[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")
        #endregion

        #region Fine optimizer
        init_x_fine = ctrl_points_coarse.flatten()
        start = time.perf_counter()
        result_fine = opt.minimize(
            fun=score_function_bspline_reg_multiple_pre,
            x0=init_x_fine,
            args=(
                camera_poses,
                camera_parameters,
                degree,
                init_x_fine,
                reg_weight,
                decay,
                curvature_weight,
                skeletons,
                interps,
                50
            ),
            method='L-BFGS-B', # 'Powell', # 
            bounds=bounds,
            options={'maxiter':1e6, 'ftol':1e-6, 'eps':1e-8, 'xtol':1e-4, 'disp':True}
        )
        end = time.perf_counter()
        print(f"Fine optimization took {end - start:.2f} seconds")
        print(f"result: {result_fine}")
        ctrl_points_fine = result_fine.x.reshape(-1, 3)
        projected_spline_cam2_fine = project_bspline(ctrl_points_fine, camera_poses[-1], camera_parameters, degree=degree)
        show_masks([data[-1][1], projected_spline_cam2_fine], "Projected B-Spline Cam2 Fine")

        ctrl_points.append(ctrl_points_fine)
        #endregion

        spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
        pointcloud_publisher.publish(spline_pc)

        projected_spline_opt = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
        correct_skeleton_cam2 = skeletonize_mask(data[-1][1])
        show_masks([correct_skeleton_cam2, projected_spline_opt])

        visualize_spline_with_pc(best_pc_world, ctrl_points[-1], degree)

        # Movement
        # Get highest Point in pointcloud
        target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
        tarrget_point_offset = target_point.copy()
        tarrget_point_offset[2] += 0.098 + 0.010 - 0.03 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        base_footprint.append(ros_handler.get_current_pose("base_footprint", map_frame))
        target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint[-1]))
        # Calculate Path
        target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4) # <- online
        # target_path = interpolate_poses(camera_poses[-1], target_poses[-1], num_steps=4) # <- offline
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
    rospy.init_node("Tube Grasp Pipeline", anonymous=True)
    tube_grasp_pipeline(debug=True)  










