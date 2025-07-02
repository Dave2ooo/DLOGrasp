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

class ImageProcessing:
    def __init__(self):
        self.depth_anything_wrapper = DepthAnythingWrapper()
        self.grounded_sam_wrapper = GroundedSamWrapper()
    
    def get_mask(self, image, prompt: str = "cable.", show: bool = False):
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

def estimate_scale_shift(depth1, mask2, camera_pose1, camera_pose2, camera_parameters, show=False):
    """
    Estimates the optimal scale and shift values of data1 to fit data2.
    
    Parameters
    ----------
    data1 : tuple
        A tuple containing the first dataset, including the mask and depth map.
    data2 : tuple
        A tuple containing the second dataset, including the mask.
    transform1 : PoseStamped
        The pose from the first camera in world coordinates.
    transform2 : PoseStamped
        The pose from the second camera world coordinates.
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

    if not isinstance(camera_pose1, PoseStamped):
        raise TypeError("camera_pose1 must be a geometry_msgs.msg.PoseStamped")
    if not isinstance(camera_pose2, PoseStamped):
        raise TypeError("camera_pose2 must be a geometry_msgs.msg.PoseStamped")

    bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    start = time.perf_counter()
    result = opt.minimize(
        fun=score_function,   # returns –score
        x0=[0.3, -0.15],
        args=(depth1, camera_pose1, mask2, camera_pose2, camera_parameters),
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
    if show:
        print(f"Scale & Sfhift optimization took {end - start:.2f} seconds")

    alpha_opt, beta_opt = result.x
    y_opt = score_function([alpha_opt, beta_opt], depth1, camera_pose1, mask2, camera_pose2, camera_parameters)

    depth_opt = scale_depth_map(depth1, scale=alpha_opt, shift=beta_opt)
    pc_cam1_opt = convert_depth_map_to_pointcloud(depth_opt, camera_parameters)
    pc_cam0_opt = transform_pointcloud_to_world(pc_cam1_opt, camera_pose1)
    _, projection_pc_mask_cam2_opt = project_pointcloud_from_world(pc_cam0_opt, camera_pose2, camera_parameters)

    fixed_projection_pc_depth_cam2_opt = fill_mask_holes(projection_pc_mask_cam2_opt)

    score = score_mask_chamfer(fixed_projection_pc_depth_cam2_opt, mask2)

    # Show mask and pointcloud
    if show:
        print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Score: {y_opt}')
        show_masks([projection_pc_mask_cam2_opt], title='Optimal Projection')
        show_masks([fixed_projection_pc_depth_cam2_opt], title='Projection holes fixed')
        show_masks_union(projection_pc_mask_cam2_opt, fixed_projection_pc_depth_cam2_opt, title="Optimal Projection orig & holes fixed Projection")
        show_masks_union(fixed_projection_pc_depth_cam2_opt, mask2, title='Optimal Projection with holes fixed vs Mask to fit')

    return alpha_opt, beta_opt, pc_cam0_opt, score

def score_function(x, depth1, camera_pose1, mask2, camera_pose2, camera_parameters):
    if rospy.is_shutdown(): exit()
    alpha, beta = x
    # Scale and shift
    depth_new = scale_depth_map(depth1, scale=alpha, shift=beta)
    # Get scaled/shifted pointcloud
    new_pc1 = convert_depth_map_to_pointcloud(depth_new, camera_parameters)
    # Transform scaled pointcloud into world coordinates
    new_pc0 = transform_pointcloud_to_world(new_pc1, camera_pose1)
    # Get projection of scaled pointcloud into camera 2
    projection_new_pc2_depth, projection_new_pc2_mask = project_pointcloud_from_world(new_pc0, camera_pose2, camera_parameters)
    # Fill holes
    # fixed_projection_new_pc2_mask = fill_mask_holes(projection_new_pc2_mask)
    # Count the number of inliers between mask 2 and projection of scaled pointcloud
    # score = score_mask_chamfer(fixed_projection_new_pc2_mask, mask2)
    score = score_mask_chamfer(projection_new_pc2_mask, mask2)
    # print(f'num_inliers: {num_inliers}')
    return score

def score_mask_chamfer(P: np.ndarray, M: np.ndarray, thresh: float = 0.5) -> float:
    """
    Compute the symmetric Chamfer distance between two binary masks, 
    robust to one mask being empty (e.g., projection outside the image).

    Parameters
    ----------
    P : np.ndarray
        Projected mask (arbitrary numeric or boolean array).
    M : np.ndarray
        Reference mask (same shape as P).
    thresh : float
        Threshold for binarization; values > thresh are considered foreground.

    Returns
    -------
    float
        Robust Chamfer distance (lower is better).
    """
    # ---- binarize masks ---------------------------------------------------
    P_bin = P > thresh if P.dtype != np.bool_ else P
    M_bin = M > thresh if M.dtype != np.bool_ else M

    # ---- distance transforms ----------------------------------------------
    dt_M = cv2.distanceTransform((~M_bin).astype(np.uint8), cv2.DIST_L2, 3)
    dt_P = cv2.distanceTransform((~P_bin).astype(np.uint8), cv2.DIST_L2, 3)

    # ---- one-way distances -----------------------------------------------
    dist_p_to_m = dt_M[P_bin].mean() if P_bin.any() else None
    dist_m_to_p = dt_P[M_bin].mean() if M_bin.any() else None

    # ---- robust symmetric combination ------------------------------------
    distances = []
    if dist_p_to_m is not None:
        distances.append(dist_p_to_m)
    if dist_m_to_p is not None:
        distances.append(dist_m_to_p)

    # If both masks are empty, define distance as 0; otherwise average available
    if not distances:
        return 0.0
    return float(np.mean(distances))

def interactive_scale_shift(depth1: np.ndarray,
                            mask2: np.ndarray,
                            pose1: PoseStamped,
                            pose2: PoseStamped,
                            camera_parameters,
                            scale_limits: tuple[float, float] = (0.005, 0.4),
                            shift_limits: tuple[float, float] = (-1.0, 1.0)
                            ) -> tuple[float, float, int]:
    """
    Open an OpenCV window with two trackbars (“scale” and “shift”) that lets you
    interactively adjust the scaling and offset of `depth1`, regenerate the point
    cloud from camera1, project it into camera2, compare against `mask2`, and
    display the count of inliers as well as the current scale and shift.

    Parameters
    ----------
    depth1 : np.ndarray
        Original depth map from camera1 (H×W).
    mask2 : np.ndarray
        Binary mask from camera2 (H×W).
    pose1 : TransformStamped
        camera1→world transform.
    pose2 : TransformStamped
        camera2→world transform.
    scale_limits : tuple(float, float)
        (min_scale, max_scale) for the scale slider.
    shift_limits : tuple(float, float)
        (min_shift, max_shift) for the shift slider.

    Returns
    -------
    (scale, shift, inliers) : Tuple[float, float, int]
        The final slider values and number of inliers when the window is closed.
    """
    H, W = depth1.shape
    window = "Scale/Shift Tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # unpack limits and compute slider resolution
    smin, smax = scale_limits
    scale_res = 0.001
    s_steps = max(1, int(round((smax - smin) / scale_res)))
    scale_res = (smax - smin) / s_steps

    tmin, tmax = shift_limits
    t_steps = 200
    shift_res = (tmax - tmin) / t_steps

    result = {'scale': None, 'shift': None, 'score': None}

    def update(_=0):
        v_s = cv2.getTrackbarPos("scale", window)
        v_t = cv2.getTrackbarPos("shift", window)

        scale = smin + v_s * scale_res
        shift = tmin + v_t * shift_res
        # 1) scale & shift depth1
        d1 = scale_depth_map(depth1, scale, shift)
        # 2) to world pointcloud
        pc1 = convert_depth_map_to_pointcloud(d1, camera_parameters)
        pc1_world = transform_pointcloud_to_world(pc1, pose1)
        # 3) project into cam2
        _, reproj_mask = project_pointcloud_from_world(pc1_world, pose2, camera_parameters)
        # 4) overlay masks
        m2 = (mask2.astype(bool)).astype(np.uint8) * 255
        rm = (reproj_mask.astype(bool)).astype(np.uint8) * 255
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[m2 > 0]               = (0,   0, 255)
        overlay[rm > 0]               = (0, 255,   0)
        overlay[(m2 > 0) & (rm > 0)]  = (0, 255, 255)
        # 5) count score
        score = score_mask_chamfer(reproj_mask, mask2)

        # 6) annotate scale, shift, score
        cv2.putText(overlay, f"Scale: {scale:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f"Shift: {shift:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f"Inliers: {score}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        result['scale'], result['shift'], result['score'] = scale, shift, score
        cv2.imshow(window, overlay)

    # create trackbars
    cv2.createTrackbar("scale", window, int(s_steps//2), s_steps, update)
    cv2.createTrackbar("shift", window, int(t_steps//2), t_steps, update)

    update()  # initial draw

    # loop until Esc or window closed
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window)
    return result['scale'], result['shift'], result['score']

def tube_grasp_pipeline(debug: bool = False):
    camera_parameters = (149.09148, 187.64966, 334.87706, 268.23742)
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
    images.append(image_subscriber.get_current_image(show=False)) # Take image
    camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))

    # save_depth_map(data[-1][0], save_image_folder, "Original_Depth")
    # save_depth_map(data[-1][2], save_image_folder, "Masked_Depth")
    # save_masks(data[-1][1], save_image_folder, "Mask")


    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()
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
    images.append(image_subscriber.get_current_image(show=False)) # Take image
    camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
    palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
    # Process image
    masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))
    depths_unmasked.append(image_processing.get_depth_unmasked(image=images[-1], show=False))
    depths.append(image_processing.get_depth_masked(image=images[-1], mask=masks[-1], show=False))

    # Estimate scale and shift
    print(f"Score: {score_function([1, 0], depth1=depths[-2], camera_pose1=camera_poses[-2], mask2=masks[-1], camera_pose2=camera_poses[-1], camera_parameters=camera_parameters)}")
    best_alpha, best_beta, best_pc_world, score = estimate_scale_shift(depth1=depths[-2], mask2=masks[-1], camera_pose1=camera_poses[-2], camera_pose2=camera_poses[-1], camera_parameters=camera_parameters, show=False)
    pointcloud_publisher.publish(best_pc_world)
    #endregion -------------------- Depth Enything --------------------

    # ---------------------------------- CONTINUE HERE ----------------------------------------------------
    
    # save_masks([data[-1][1], data[-2][1]], save_image_folder, "Masks")

    #region -------------------- Spline --------------------
    best_depth = scale_depth_map(depths[-1], best_alpha, best_beta)
    centerline_pts_cam2_array = extract_centerline_from_mask_individual(best_depth, masks[-1], camera_parameters, show=False)
    centerline_pts_cam2 = max(centerline_pts_cam2_array, key=lambda s: s.shape[0]) # Extract the longest path

    centerline_pts_world = transform_points_to_world(centerline_pts_cam2, camera_poses[-1])
    show_pointclouds([centerline_pts_world])
    degree = 3
    # Fit B-spline
    # ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20))
    ctrl_points.append(fit_bspline_scipy(centerline_pts_world, degree=degree, smooth=1e-5, nest=20)[3:-3,:]) # Remove first and last three control points
    print(f"Number of controlpoints: {ctrl_points[-1].shape}")
    print(f"ctrl_points: {ctrl_points[0]}")

    interactive_bspline_editor(ctrl_points[-1], masks[-1], camera_poses[-1], camera_parameters, degree)

    # save_mask_spline(data[-1][1], ctrl_points[-1], degree, camera_parameters, camera_poses[-1], save_image_folder, "Spline")

    spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
    pointcloud_publisher.publish(spline_pc)

    visualize_spline_with_pc(best_pc_world, ctrl_points[-1], degree, title="Scaled PointCloud & Spline")

    projected_spline_cam1 = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
    show_masks([masks[1], projected_spline_cam1], "Projected B-Spline Cam1 - 1")
    
    correct_skeleton = skeletonize_mask(masks[-1])
    show_masks([masks[-1], correct_skeleton], "Correct Skeleton")

    show_masks([correct_skeleton, projected_spline_cam1], "Correct Skeleton and Projected Spline (CAM1)")

    # Get highest Point in pointcloud
    target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
    tarrget_point_offset = target_point.copy()
    tarrget_point_offset[2] += 0.098 + 0.010 # - 0.020 # Make target Pose hover above actual target pose - tested offset
    grasp_point_publisher.publish(target_point)
    # Convert to Pose
    base_footprint = ros_handler.get_current_pose("base_footprint", map_frame)
    target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
    # Calculate Path
    target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4)
    # Move arm a step
    path_publisher.publish(target_path)
    pose_publisher.publish(target_path[1])
    # input("Press Enter when moves are finished…")


    while not rospy.is_shutdown():
        rospy.sleep(5)

        # usr_input = input("Press Enter when image is correct")
        # if usr_input == "c": exit()

        images.append(image_subscriber.get_current_image(show=False)) # Take image
        camera_poses.append(ros_handler.get_current_pose(hand_camera_frame, map_frame)) # Get current camera pose
        palm_poses.append(ros_handler.get_current_pose(hand_palm_frame, map_frame)) # Get current hand palm pose
        # Process image
        masks.append(image_processing.get_mask(image=images[-1], prompt=SAM_prompt, show=False))

        projected_spline_cam1 = project_bspline(ctrl_points[-1], camera_poses[-2], camera_parameters, degree=degree)
        if debug:
            show_masks([masks[-2], projected_spline_cam1], "Projected B-Spline Cam1")

        projected_spline_cam2 = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
        if debug:
            show_masks([masks[-1], projected_spline_cam2], "Projected B-Spline Cam2")

        #region Translate b-spline
        start = time.perf_counter()
        bounds = [(-0.1, 0.1), (-0.1, 0.1)] # [(x_min, x_max), (y_min,  y_max)]
        result_translation = opt.minimize(
            fun=score_bspline_translation,   # returns –score
            x0=[0, 0],
            args=(masks[-1], camera_poses[-1], camera_parameters, degree, ctrl_points[-1]),
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
        show_masks([masks[-1], projected_spline_cam2_translated], "Projected B-Spline Cam2 Translated")

        #region Coarse optimize control points
        # 1) Precompute once:
        skeletons, interps = precompute_skeletons_and_interps(masks)        # 2) Call optimizer with our new score:
        # bounds = make_bspline_bounds(ctrl_points_translated, delta=0.5)
        # init_x_coarse = ctrl_points_translated.flatten()
        bounds = make_bspline_bounds(ctrl_points[-1], delta=0.5)
        init_x_coarse = ctrl_points[-1].flatten()
        # 2) Coarse/fine minimization calls:
        start = time.perf_counter()
        result_coarse = opt.minimize(
            fun=score_function_bspline_point_ray,
            x0=init_x_coarse,
            args=(
                camera_poses,
                camera_parameters,
                degree,
                skeletons,
                10,
                0.9,
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
        show_masks([masks[-1], projected_spline_cam2_coarse], "Projected B-Spline Cam2 Coarse")
        #endregion

        #region Fine optimizer
        init_x_fine = ctrl_points_coarse.flatten()
        start = time.perf_counter()
        result_fine = opt.minimize(
            fun=score_function_bspline_point_ray,
            x0=init_x_fine,
            args=(
                camera_poses,
                camera_parameters,
                degree,
                skeletons,
                50,
                0.9,
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
        show_masks([masks[-1], projected_spline_cam2_fine], "Projected B-Spline Cam2 Fine")

        ctrl_points.append(ctrl_points_fine)
        #endregion

        spline_pc = convert_bspline_to_pointcloud(ctrl_points[-1])
        pointcloud_publisher.publish(spline_pc)

        projected_spline_opt = project_bspline(ctrl_points[-1], camera_poses[-1], camera_parameters, degree=degree)
        correct_skeleton_cam2 = skeletonize_mask(masks[-1])
        show_masks([correct_skeleton_cam2, projected_spline_opt])

        visualize_spline_with_pc(best_pc_world, ctrl_points[-1], degree)

        # Movement
        # Get highest Point in pointcloud
        target_point, target_angle = get_highest_point_and_angle_spline(ctrl_points[-1])
        tarrget_point_offset = target_point.copy()
        tarrget_point_offset[2] += 0.098 + 0.010 - 0.03 # Make target Pose hover above actual target pose - tested offset
        # Convert to Pose
        base_footprint = ros_handler.get_current_pose("base_footprint", map_frame)
        target_poses.append(get_desired_pose(tarrget_point_offset, base_footprint))
        # Calculate Path
        target_path = interpolate_poses(palm_poses[-1], target_poses[-1], num_steps=4) # <- online
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
    rospy.init_node("TubeGraspPipeline", anonymous=True)
    tube_grasp_pipeline(debug=True)  










