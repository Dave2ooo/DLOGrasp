from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from ROS_handler import ROSHandler
from my_class import MyClass
from my_utils import *
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
import cv2

import random
import time
import scipy.optimize as opt

from publisher import *

# camera_intrinsics = (203.71833, 203.71833, 319.5, 239.5) # old/wrong
camera_intrinsics = (149.09148, 187.64966, 334.87706, 268.23742)

depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
grounded_sam_wrapper = GroundedSamWrapper()

def get_stamped_transform(translation, rotation):
    transform_stamped = TransformStamped()
    transform_stamped.transform.translation.x = translation[0]
    transform_stamped.transform.translation.y = translation[1]
    transform_stamped.transform.translation.z = translation[2]

    transform_stamped.transform.rotation.x = rotation[0]
    transform_stamped.transform.rotation.y = rotation[1]
    transform_stamped.transform.rotation.z = rotation[2]
    transform_stamped.transform.rotation.w = rotation[3]
    return transform_stamped

def get_relative_transform(transform1: TransformStamped,
                           transform2: TransformStamped) -> TransformStamped:
    """
    T2 = T2_1 * T1
    Compute the transform T_rel such that T2 = T_rel * T1,
    i.e. the relative transform from frame1 to frame2.

    Parameters
    ----------
    transform1 : TransformStamped
        The first transform (T1).
    transform2 : TransformStamped
        The second transform (T2).

    Returns
    -------
    TransformStamped
        A TransformStamped representing T_rel = T1⁻¹ * T2.
        The header.frame_id will be transform1.header.frame_id,
        the child_frame_id will be transform2.child_frame_id,
        and the stamp will be taken from transform2.
    """
    # build 4×4 matrices
    def to_matrix(tf: TransformStamped) -> np.ndarray:
        q = tf.transform.rotation
        t = tf.transform.translation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        M[:3, 3] = [t.x, t.y, t.z]
        return M

    M1 = to_matrix(transform1)
    M2 = to_matrix(transform2)

    # relative matrix
    M_rel = np.linalg.inv(M1) @ M2

    # convert back to quaternion + translation
    q_rel = quaternion_from_matrix(M_rel)
    t_rel = M_rel[:3, 3]

    # populate a new TransformStamped
    rel = TransformStamped()
    rel.header.stamp = transform2.header.stamp
    rel.header.frame_id = transform1.header.frame_id
    rel.child_frame_id  = transform2.child_frame_id

    rel.transform.translation.x = float(t_rel[0])
    rel.transform.translation.y = float(t_rel[1])
    rel.transform.translation.z = float(t_rel[2])
    rel.transform.rotation.x    = float(q_rel[0])
    rel.transform.rotation.y    = float(q_rel[1])
    rel.transform.rotation.z    = float(q_rel[2])
    rel.transform.rotation.w    = float(q_rel[3])

    return rel

def get_rotation_matrix(transform: TransformStamped) -> np.ndarray:
    """
    Extract the 3×3 rotation matrix from a ROS TransformStamped.

    Parameters
    ----------
    transform : TransformStamped
        The input transform containing a quaternion.

    Returns
    -------
    np.ndarray
        A 3×3 rotation matrix.
    """
    if not isinstance(transform, TransformStamped):
        raise TypeError("transform must be a geometry_msgs.msg.TransformStamped")

    q = transform.transform.rotation
    # Construct 4×4 homogeneous matrix from quaternion
    M4 = quaternion_matrix([q.x, q.y, q.z, q.w])
    # Return the top‐left 3×3 rotation block
    return M4[:3, :3]

def get_homogeneous_matrix(transform: TransformStamped) -> np.ndarray:
    """
    Convert a ROS TransformStamped into a 4×4 homogeneous transformation matrix.

    Parameters
    ----------
    transform : TransformStamped
        The input transform containing translation and rotation (quaternion).

    Returns
    -------
    np.ndarray
        A 4×4 homogeneous transformation matrix.
    """
    if not isinstance(transform, TransformStamped):
        raise TypeError("transform must be a geometry_msgs.msg.TransformStamped")

    # extract quaternion and translation
    q = transform.transform.rotation
    t = transform.transform.translation

    # build rotation matrix (4×4) from quaternion
    M = quaternion_matrix([q.x, q.y, q.z, q.w])

    # insert translation into the homogeneous matrix
    M[0, 3] = t.x
    M[1, 3] = t.y
    M[2, 3] = t.z

    return M

def inliers_function(x, depth_masked, transform1, mask, transform2):
    alpha, beta = x
    # Scale and shift
    depth_new = depth_anything_wrapper.scale_depth_map(depth_masked, scale=alpha, shift=beta)
    # Get scaled/shifted pointcloud
    new_pc1 = depth_anything_wrapper.get_pointcloud(depth_new)
    # Transform scaled pointcloud into world coordinates
    new_pc0 = depth_anything_wrapper.transform_pointcloud_to_world(new_pc1, transform1)
    # Get projection of scaled pointcloud into camera 2
    projection_new_pc2_depth, projection_new_pc2_mask = depth_anything_wrapper.project_pointcloud(new_pc0, transform2)
    # Fill holes   
    fixed_projection_new_pc2_mask = grounded_sam_wrapper.fill_mask_holes(projection_new_pc2_mask)
    # Count the number of inliers between mask 2 and projection of scaled pointcloud
    num_inliers, inlier_union = depth_anything_wrapper.count_inliers(fixed_projection_new_pc2_mask, mask)

    return -num_inliers

def test2():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()
    image = cv2.imread(f'/root/workspace/images/moves/cable0.jpg')

    depth_orig = depth_anything_wrapper.get_depth_map(image)
    depth_scale = depth_anything_wrapper.scale_depth_map(depth_orig, scale=0.1, shift=0.0)

    pointcloud_orig = depth_anything_wrapper.get_pointcloud(depth_orig)
    pointcloud_scaled = depth_anything_wrapper.get_pointcloud(depth_scale)
    
    mask = grounded_sam_wrapper.get_mask(image)[0][0]
    
    pointcloud_orig_masked = depth_anything_wrapper.mask_pointcloud(pointcloud_orig, mask)
    pointcloud_scaled_masked = depth_anything_wrapper.mask_pointcloud(pointcloud_scaled, mask)
    depth_anything_wrapper.show_pointclouds([pointcloud_orig_masked, pointcloud_scaled_masked])

    pointcloud_orig_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_orig_masked, transform_stamped0)
    pointcloud_scaled_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_scaled_masked, transform_stamped0)
    depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_orig_masked_world, pointcloud_scaled_masked_world], [transform_stamped0])

def test_least_squares():
    
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    depths = []
    pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:2]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth = depth_anything_wrapper.scale_depth_map(depth, scale=0.1, shift=5)
        depths.append(depth)
        # depth_anything_wrapper.show_depth_map(depth)

        pointcloud = depth_anything_wrapper.get_pointcloud(depth)
        pointclouds.append(pointcloud)
        # depth_anything_wrapper.show_pointclouds([pointcloud])

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        pointcloud_masked = depth_anything_wrapper.mask_pointcloud(pointcloud, mask)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_world])


    projection_pc0_to_cam1_depth, projection_pc0_to_cam1_mask = depth_anything_wrapper.project_pointcloud(pointclouds_masked_world[0], transforms[1])
    # depth_anything_wrapper.show_depth_map(projection_pc1_to_cam2_depth, wait=False)
    # grounded_sam_wrapper.show_mask(projection_pc1_to_cam2_mask, wait=True)
    # grounded_sam_wrapper.show_mask(masks[1], wait=True)

    # grounded_sam_wrapper.show_masks([masks[1], projection_pc0_to_cam1_mask], wait=True)

    # depth_anything_wrapper.show_pointclouds_with_frames(pointclouds_masked_world[0:2], transforms[0:2])    

    corr = depth_anything_wrapper.get_closest_points(pointclouds_masked_world[0], pointclouds_masked_world[1])

    # print(f'corr.shape: {corr.shape}, corr[0]: {corr[0]}, corr[1]: {corr[1]}, corr[2]: {corr[2]}, corr[3]: {corr[3]}, corr[4]: {corr[4]}')

    # print(f'pointclouds_masked_world[0].points[0]: {pointclouds_masked_world[0].points[0]}')
    
    H2_1 = get_homogeneous_matrix(get_relative_transform(transform_stamped0, transform_stamped1))
    # print(f'H2_1: {H2_1}')

    R2_1 = H2_1[0:3, 0:3]
    # print(f'R2_1: {R2_1}')

    d2_1 = H2_1[0:3, 3]
    # print(f'd2_1: {d2_1}')
    # print(f'd2_1.shape: {d2_1.shape}')

    b = None
    A = None

    for c in corr:
        point_1 = pointclouds_masked[0].points[c[0]]
        point_2 = pointclouds_masked[1].points[c[1]]

        P22 = point_2
        # print(f'P22: {P22}')

        a1 = P22
        # print(f'a1: {a1}')

        a2 = a1 / a1[2]
        # print(f'a2: {a2}')

        P11 = point_1
        # print(f'P11: {P11}')

        a3 = - np.matmul(R2_1, P11)
        # print(f'a3: {a3}')

        a4 = a3 / a3[2]
        # print(f'a4: {a4}')

        A_temp = np.column_stack((a1, a2, a3, a4))
        # print(f'A_temp: {A_temp}')

        A = np.row_stack((A, A_temp)) if A is not None else A_temp
        b = np.concatenate((b, d2_1)) if b is not None else d2_1
        # print('-------------------')

    print(f'A.shape: {A.shape}')
    # print(f'A: {A}')
    print(f'b.shape: {b.shape}')
    # print(f'b: {b}')

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    alpha2, beta2, alpha1, beta1 = x
    print(f'alpha2: {alpha2}, beta2: {beta2}, alpha1: {alpha1}, beta1: {beta1}')

    depth_new_0 = depth_anything_wrapper.scale_depth_map(depths[0], scale=alpha1, shift=beta1)
    depth_new_1 = depth_anything_wrapper.scale_depth_map(depths[1], scale=alpha2, shift=beta2)

    pointcloud_new_0 = depth_anything_wrapper.get_pointcloud(depth_new_0)
    pointcloud_new_1 = depth_anything_wrapper.get_pointcloud(depth_new_1)

    pointcloud_new_0_masked = depth_anything_wrapper.mask_pointcloud(pointcloud_new_0, masks[0])
    pointcloud_new_1_masked = depth_anything_wrapper.mask_pointcloud(pointcloud_new_1, masks[1])

    pointcloud_new_0_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_new_0_masked, transform_stamped0)
    pointcloud_new_1_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_new_1_masked, transform_stamped1)

    # depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_new_0_masked_world, pointcloud_new_1_masked_world], transforms[0:2])
    depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_new_0_masked_world], [transforms[0]])
    depth_anything_wrapper.show_pointclouds_with_frames([pointcloud_new_1_masked_world], [transforms[1]])

def ransac():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:2]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth = depth_anything_wrapper.scale_depth_map(depth, scale=0.1, shift=5)
        # depths.append(depth)
        # depth_anything_wrapper.show_depth_map(depth)

        # pointcloud = depth_anything_wrapper.get_pointcloud(depth)
        # pointclouds.append(pointcloud)
        # depth_anything_wrapper.show_pointclouds([pointcloud])

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        depth_anything_wrapper.show_depth_map(depth_masked)

        # pointcloud_masked = depth_anything_wrapper.mask_pointcloud(pointcloud, mask)
        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_world])

    
    H2_1 = get_homogeneous_matrix(get_relative_transform(transform_stamped0, transform_stamped1))
    # print(f'H2_1: {H2_1}')

    R2_1 = H2_1[0:3, 0:3]
    # print(f'R2_1: {R2_1}')

    d2_1 = H2_1[0:3, 3]
    # print(f'd2_1: {d2_1}')
    # print(f'd2_1.shape: {d2_1.shape}')

    H1_2 = get_homogeneous_matrix(get_relative_transform(transform_stamped1, transform_stamped0))
    # print(f'H2_1: {H2_1}')

    R1_2 = H1_2[0:3, 0:3]
    # print(f'R2_1: {R2_1}')

    d1_2 = H1_2[0:3, 3]
    # print(f'd2_1: {d2_1}')
    # print(f'd2_1.shape: {d2_1.shape}')

    index_pointcloud = 0
    current_pointcloud_masked_world = pointclouds_masked_world[index_pointcloud]
    # Loop N times
    for i in range(len(current_pointcloud_masked_world.points)):
        # Pick two random points from pointcloud 1
        point0_A = random.choice(current_pointcloud_masked_world.points)
        point0_B = random.choice(current_pointcloud_masked_world.points)
        if np.all(point0_A == point0_B):
            print(f'point0_A: {point0_A}')
            print(f'point0_B: {point0_B}')
            print('Points are the same')
            continue

        # Project the points into coordinates of camera 2
        projection2_A = depth_anything_wrapper.project_point_from_world(point0_A, transforms[index_pointcloud+1])
        projection2_B = depth_anything_wrapper.project_point_from_world(point0_B, transforms[index_pointcloud+1])

        # Project the pointcloud into coordinates of camera 2
        projected_pointcloud_depth, projected_pointcloud_mask = depth_anything_wrapper.project_pointcloud(pointclouds_masked_world[index_pointcloud], transforms[index_pointcloud+1])
        grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask, [projection2_A, projection2_B], title="Pointcloud 1 projected onto camera 2 with rand points")

        # Get the closest points between single point (from pointcloud 1) and mask 2
        closest2_A = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_A)
        closest2_B = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_B)

        grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, closest2_B], title="Mask 2 in camera 2 with closest points")
        grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask, [closest2_A, closest2_B], title="Pointcloud 1 projected onto camera 2 with closest points")

        grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, projection2_A], title="projection and closest point A")
        grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_B, projection2_B], title="projection and closest point B")

        trtiangulated0_A = depth_anything_wrapper.triangulate(point0_A, transforms[index_pointcloud], closest2_A, transforms[index_pointcloud+1])
        trtiangulated0_B = depth_anything_wrapper.triangulate(point0_B, transforms[index_pointcloud], closest2_B, transforms[index_pointcloud+1])

        print(f'trtiangulated0_A: {trtiangulated0_A}')
        print(f'trtiangulated0_B: {trtiangulated0_B}')
        try:
            projected_triangulated2_A = depth_anything_wrapper.project_point_from_world(trtiangulated0_A, transforms[index_pointcloud+1])
            projected_triangulated2_B = depth_anything_wrapper.project_point_from_world(trtiangulated0_B, transforms[index_pointcloud+1])
        except Exception as e:
            print(e)
            continue
        
        grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [projected_triangulated2_A, projected_triangulated2_B], title="Mask 2 with triangulated points")
        grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask, [projected_triangulated2_A, projected_triangulated2_B], title="Pointcloud 1 projected onto camera 2 with triangulated points")
        
        triangulated1_A = depth_anything_wrapper.transform_point_from_world(trtiangulated0_A, transforms[index_pointcloud])
        triangulated1_B = depth_anything_wrapper.transform_point_from_world(trtiangulated0_B, transforms[index_pointcloud])

        z_A_orig = depth_anything_wrapper.transform_point_from_world(point0_A, transforms[index_pointcloud])[2]
        z_B_orig = depth_anything_wrapper.transform_point_from_world(point0_B, transforms[index_pointcloud])[2]
        z_A_triangulated = triangulated1_A[2]
        z_B_triangulated = triangulated1_B[2]
        print(f'depth_A_orig: {z_A_orig}')
        print(f'depth_A_triangulated: {z_A_triangulated}')
        print(f'depth_B_orig: {z_B_orig}')
        print(f'depth_B_triangulated: {z_B_triangulated}')

        alpha = (z_A_triangulated - z_B_triangulated) / (z_A_orig - z_B_orig)
        beta = z_A_triangulated - alpha * z_A_orig

        print(f'alpha: {alpha}')
        print(f'beta: {beta}')

        grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [projected_triangulated2_A, projected_triangulated2_B], title="Mask 2 with triangulated points")

        depth_new = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha, shift=beta)

        grounded_sam_wrapper.show_mask(depths_masked[index_pointcloud], title="Original depth map")
        grounded_sam_wrapper.show_mask(depth_new, title="Scaled depth map")

        orig_pc = depth_anything_wrapper.get_pointcloud(depths_masked[index_pointcloud])
        new_pc = depth_anything_wrapper.get_pointcloud(depth_new)

        new_pc_2 = depth_anything_wrapper.transform_pointcloud_to_world(new_pc, transforms[index_pointcloud])
        projection_new_pc_2_cam2_depth, projection_new_pc_2_cam2_mask = depth_anything_wrapper.project_pointcloud(new_pc_2, transforms[index_pointcloud+1])

        depth_anything_wrapper.show_pointclouds([orig_pc, new_pc], title="Original and scaled pointcloud")

        pointcloud_new_1 = depth_anything_wrapper.get_pointcloud(depth_new)

        pointcloud_new_0 = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_new_1, transforms[index_pointcloud])
        projection_pointcloud_new_cam2_depth, projection_pointcloud_new_cam2_mask = depth_anything_wrapper.project_pointcloud(pointcloud_new_0, transforms[index_pointcloud+1])

        grounded_sam_wrapper.show_mask_and_points(projection_pointcloud_new_cam2_depth, [closest2_A, closest2_B], title="Scaled pointcloud with closest points")

        num_inliers, _ = depth_anything_wrapper.count_inliers(projection_new_pc_2_cam2_depth, masks[index_pointcloud+1])
        print(f'num_inliers: {num_inliers}')
        grounded_sam_wrapper.show_masks([projection_new_pc_2_cam2_depth, masks[index_pointcloud+1]], title="Scaled depth map and mask")

def ransac2():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_world])


    index_pointcloud = 0
    current_pointcloud_masked_world = pointclouds_masked_world[index_pointcloud]
    max_inliers_counter = 0
    best_alpha, best_beta = 0, 0
    best_pointcloud = None
    best_projection = None
    num_skips = 0
    # Loop N times
    for i in range(len(current_pointcloud_masked_world.points)):
        # Pick two random points from pointcloud 1
        point0_A = random.choice(current_pointcloud_masked_world.points)
        point0_B = random.choice(current_pointcloud_masked_world.points)
        if np.all(point0_A == point0_B):
            # print(f'point0_A: {point0_A}')
            # print(f'point0_B: {point0_B}')
            # print('Points are the same')
            continue

        # Project the points into coordinates of camera 2
        projection2_A = depth_anything_wrapper.project_point_from_world(point0_A, transforms[index_pointcloud+1])
        projection2_B = depth_anything_wrapper.project_point_from_world(point0_B, transforms[index_pointcloud+1])

        # Project the pointcloud into coordinates of camera 2
        projected_pointcloud_depth, projected_pointcloud_mask_2 = depth_anything_wrapper.project_pointcloud(pointclouds_masked_world[index_pointcloud], transforms[index_pointcloud+1])
        # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projection2_A, projection2_B], title="Pointcloud 1 projected onto camera 2 with rand points")

        # Get the closest points between single point (from pointcloud 1) and mask 2
        closest2_A = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_A)
        closest2_B = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_B)

        # Show the closest points
        # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, closest2_B], title="Mask 2 in camera 2 with closest points")
        # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [closest2_A, closest2_B], title="Pointcloud 1 projected onto camera 2 with closest points")

        # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, projection2_A], title="projection and closest point A")
        # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_B, projection2_B], title="projection and closest point B")

        # Triangulate
        trtiangulated0_A = depth_anything_wrapper.triangulate(point0_A, transforms[index_pointcloud], closest2_A, transforms[index_pointcloud+1])
        trtiangulated0_B = depth_anything_wrapper.triangulate(point0_B, transforms[index_pointcloud], closest2_B, transforms[index_pointcloud+1])
        # print(f'trtiangulated0_A: {trtiangulated0_A}')
        # print(f'trtiangulated0_B: {trtiangulated0_B}')
        try:
            # Project the triangulated points into coordinates of camera 2
            projected_triangulated2_A = depth_anything_wrapper.project_point_from_world(trtiangulated0_A, transforms[index_pointcloud+1])
            projected_triangulated2_B = depth_anything_wrapper.project_point_from_world(trtiangulated0_B, transforms[index_pointcloud+1])
        except Exception as e:
            # print(f'{e}, Skipping to next iteration')
            num_skips += 1
            continue
        
        # Show mask 2 and triangulated points
        # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [projected_triangulated2_A, projected_triangulated2_B], title="Mask 2 with triangulated points")
        # Show pointcloud 1 and triangulated points in camera 2
        # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projected_triangulated2_A, projected_triangulated2_B], title="Pointcloud 1 projected onto camera 2 with triangulated points")
        
        # Transform triangulated point into camera 1 coordinates
        triangulated1_A = depth_anything_wrapper.transform_point_from_world(trtiangulated0_A, transforms[index_pointcloud])
        triangulated1_B = depth_anything_wrapper.transform_point_from_world(trtiangulated0_B, transforms[index_pointcloud])

        # Get  distance of original point0_A and point0_B
        z_A_orig = depth_anything_wrapper.transform_point_from_world(point0_A, transforms[index_pointcloud])[2]
        z_B_orig = depth_anything_wrapper.transform_point_from_world(point0_B, transforms[index_pointcloud])[2]

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
        depth_new = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha, shift=beta)

        # Show original and scaled depth maps
        # depth_anything_wrapper.show_depth_map(depths_masked[index_pointcloud], title="Original depth map")
        # depth_anything_wrapper.show_depth_map(depth_new, title="Scaled depth map")

        # Get scaled/shifted pointcloud
        new_pc1 = depth_anything_wrapper.get_pointcloud(depth_new)

        # Transform scaled pointcloud into world coordinates
        new_pc0 = depth_anything_wrapper.transform_pointcloud_to_world(new_pc1, transforms[index_pointcloud])

        # Get projection of scaled pointcloud into camera 2
        projection_new_pc2_depth, projection_new_pc2_mask = depth_anything_wrapper.project_pointcloud(new_pc0, transforms[index_pointcloud+1])
    
        # Show projection of scaled pointcloud in camera 2 and closest points
        # grounded_sam_wrapper.show_mask_and_points(projection_new_pc2_depth, [closest2_A, closest2_B], title="Scaled pointcloud with closest points")

        # # Show original and scaled depth maps and masks
        # grounded_sam_wrapper.show_masks([projection_new_pc2_mask, masks[index_pointcloud+1]], title="Scaled depth map and mask")
        # # Show original and scaled pointclouds in world coordinates
        # depth_anything_wrapper.show_pointclouds_with_frames([pointclouds_masked_world[index_pointcloud], new_pc0], [transforms[index_pointcloud]], title="Original and scaled pointcloud")

        # Count the number of inliers between mask 2 and projection of scaled pointcloud
        num_inliers, _ = depth_anything_wrapper.count_inliers(projection_new_pc2_mask, masks[index_pointcloud+1])
        # print(f'{i}: num_inliers: {num_inliers}')

        if num_inliers > max_inliers_counter and alpha < 1:
            max_inliers_counter = num_inliers
            best_alpha = alpha
            best_beta = beta
            best_pointcloud = new_pc0
            best_projection = projection_new_pc2_depth

    print(f'Max inliers: {max_inliers_counter}, alpha: {best_alpha:.2f}, beta: {best_beta:.2f}, Skipped points: {num_skips}')
    # Show original and scaled depth maps and masks
    grounded_sam_wrapper.show_masks([best_projection, masks[index_pointcloud+1]], title="Scaled depth map and mask")
    # Show original and scaled pointclouds in world coordinates
    depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointclouds_masked_world[index_pointcloud], best_pointcloud], [transforms[index_pointcloud]], title="Original and scaled pointcloud")

def ransac3():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])


    best_pointclouds_world = []
    for index_pointcloud, transform in enumerate(transforms[0:6]):
        # print(f'Processing Pointcloud {iteration}')
        current_pointcloud_masked_world = pointclouds_masked_world[index_pointcloud]
        max_inliers_counter = 0
        best_inlier_ratio = None
        best_alpha, best_beta = 0, 0
        best_pointcloud = None
        best_projection = None
        num_skips = 0
        # Loop N times
        for i in range(len(current_pointcloud_masked_world.points)):
        # for i in range(1000):
            # Pick two random points from pointcloud 1
            point0_A = random.choice(current_pointcloud_masked_world.points)
            point0_B = random.choice(current_pointcloud_masked_world.points)
            if np.all(point0_A == point0_B):
                # print(f'point0_A: {point0_A}')
                # print(f'point0_B: {point0_B}')
                # print('Points are the same')
                continue

            # Project the points into coordinates of camera 2
            projection2_A = depth_anything_wrapper.project_point_from_world(point0_A, transforms[index_pointcloud+1])
            projection2_B = depth_anything_wrapper.project_point_from_world(point0_B, transforms[index_pointcloud+1])

            # Project the pointcloud into coordinates of camera 2
            projected_pointcloud_depth, projected_pointcloud_mask_2 = depth_anything_wrapper.project_pointcloud(pointclouds_masked_world[index_pointcloud], transforms[index_pointcloud+1])
            # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projection2_A, projection2_B], title="Pointcloud 1 projected onto camera 2 with rand points")

            # Get the closest points between single point (from pointcloud 1) and mask 2
            closest2_A = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_A)
            closest2_B = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2_B)

            # Show the closest points
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, closest2_B], title="Mask 2 in camera 2 with closest points")
            # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [closest2_A, closest2_B], title="Pointcloud 1 projected onto camera 2 with closest points")

            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_A, projection2_A], title="projection and closest point A")
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2_B, projection2_B], title="projection and closest point B")

            # Triangulate
            trtiangulated0_A = depth_anything_wrapper.triangulate(point0_A, transforms[index_pointcloud], closest2_A, transforms[index_pointcloud+1])
            trtiangulated0_B = depth_anything_wrapper.triangulate(point0_B, transforms[index_pointcloud], closest2_B, transforms[index_pointcloud+1])
            # print(f'trtiangulated0_A: {trtiangulated0_A}')
            # print(f'trtiangulated0_B: {trtiangulated0_B}')
            try:
                # Project the triangulated points into coordinates of camera 2
                projected_triangulated2_A = depth_anything_wrapper.project_point_from_world(trtiangulated0_A, transforms[index_pointcloud+1])
                projected_triangulated2_B = depth_anything_wrapper.project_point_from_world(trtiangulated0_B, transforms[index_pointcloud+1])
            except Exception as e:
                # print(f'{e}, Skipping to next iteration')
                num_skips += 1
                continue
            
            # Show mask 2 and triangulated points
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [projected_triangulated2_A, projected_triangulated2_B], title="Mask 2 with triangulated points")
            # Show pointcloud 1 and triangulated points in camera 2
            # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projected_triangulated2_A, projected_triangulated2_B], title="Pointcloud 1 projected onto camera 2 with triangulated points")
            
            # Transform triangulated point into camera 1 coordinates
            triangulated1_A = depth_anything_wrapper.transform_point_from_world(trtiangulated0_A, transforms[index_pointcloud])
            triangulated1_B = depth_anything_wrapper.transform_point_from_world(trtiangulated0_B, transforms[index_pointcloud])

            # Get  distance of original point0_A and point0_B
            z_A_orig = depth_anything_wrapper.transform_point_from_world(point0_A, transforms[index_pointcloud])[2]
            z_B_orig = depth_anything_wrapper.transform_point_from_world(point0_B, transforms[index_pointcloud])[2]

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
            depth_new = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha, shift=beta)

            # Show original and scaled depth maps
            # depth_anything_wrapper.show_depth_map(depths_masked[index_pointcloud], title="Original depth map")
            # depth_anything_wrapper.show_depth_map(depth_new, title="Scaled depth map")

            # Get scaled/shifted pointcloud
            new_pc1 = depth_anything_wrapper.get_pointcloud(depth_new)

            # Transform scaled pointcloud into world coordinates
            new_pc0 = depth_anything_wrapper.transform_pointcloud_to_world(new_pc1, transforms[index_pointcloud])

            # Get projection of scaled pointcloud into camera 2
            projection_new_pc2_depth, projection_new_pc2_mask = depth_anything_wrapper.project_pointcloud(new_pc0, transforms[index_pointcloud+1])
        
            # Show projection of scaled pointcloud in camera 2 and closest points
            # grounded_sam_wrapper.show_mask_and_points(projection_new_pc2_depth, [closest2_A, closest2_B], title="Scaled pointcloud with closest points")

            # # Show original and scaled depth maps and masks
            # grounded_sam_wrapper.show_masks([projection_new_pc2_mask, masks[index_pointcloud+1]], title="Scaled depth map and mask")
            # # Show original and scaled pointclouds in world coordinates
            # depth_anything_wrapper.show_pointclouds_with_frames([pointclouds_masked_world[index_pointcloud], new_pc0], [transforms[index_pointcloud]], title="Original and scaled pointcloud")

            # Count the number of inliers between mask 2 and projection of scaled pointcloud
            num_inliers, inlier_union = depth_anything_wrapper.count_inliers(projection_new_pc2_mask, masks[index_pointcloud+1])
            # print(f'{i}: num_inliers: {num_inliers}')

            if num_inliers > max_inliers_counter:
                max_inliers_counter = num_inliers
                best_inlier_ratio = num_inliers/inlier_union
                best_alpha = alpha
                best_beta = beta
                best_pointcloud = new_pc0
                best_projection = projection_new_pc2_depth

        print(f'Max inliers: {max_inliers_counter}, Inlier Ratio: {best_inlier_ratio:.2f}, alpha: {best_alpha:.2f}, beta: {best_beta:.2f}, Skipped points: {num_skips}')
        # Show original and scaled depth maps and masks
        # grounded_sam_wrapper.show_masks([best_projection], title="Projection of scaled PC")
        # grounded_sam_wrapper.show_masks([masks[index_pointcloud+1]], title="Mask")
        grounded_sam_wrapper.show_mask_union(best_projection, masks[index_pointcloud+1])
        # Show original and scaled pointclouds in world coordinates
        # depth_anything_wrapper.show_pointclouds_with_frames([pointclouds_masked_world[index_pointcloud], best_pointcloud], [transforms[index_pointcloud]], title="Original and scaled pointcloud")
        best_pointclouds_world.append(best_pointcloud)

    depth_anything_wrapper.show_pointclouds_with_frames_and_grid(best_pointclouds_world, transforms, title='Best pointclouds')

def differential_evolution():
    images = []
    depths_masked = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        mask_reduced = grounded_sam_wrapper.reduce_mask(mask, 1)
        masks.append(mask_reduced)
        # grounded_sam_wrapper.show_mask(mask)
        grounded_sam_wrapper.show_mask_union(mask, mask_reduced)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])
        
    # index_pointcloud = 0
    bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    pointclouds_world_opt = []
    for index_pointcloud in range(6):
        start = time.perf_counter()
        result = opt.differential_evolution(
            inliers_function,
            bounds,
            args=(depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud+1], transforms[index_pointcloud+1]),
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=1e-2,
            disp=True
        )
        end = time.perf_counter()
        print(f"Optimization took {end - start:.2f} seconds")

        alpha_opt, beta_opt = result.x
        start = time.perf_counter()
        y_opt = inliers_function([alpha_opt, beta_opt], depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud+1], transforms[index_pointcloud+1])
        end = time.perf_counter()
        print(f"One function call took {end - start:.4f} seconds")
        print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Inliers: {y_opt}')

        depth_opt = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha_opt, shift=beta_opt)
        pc1_opt = depth_anything_wrapper.get_pointcloud(depth_opt)
        pointclouds_world_opt.append(depth_anything_wrapper.transform_pointcloud_to_world(pc1_opt, transforms[index_pointcloud]))

        # depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointclouds_world_opt[-1]], [transforms[index_pointcloud]], title='Best pointclouds')
    depth_anything_wrapper.show_pointclouds_with_frames_and_grid(pointclouds_world_opt, transforms, title='Best pointclouds')

def differential_evolution_single():
    images = []
    depths_masked = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        mask_reduced = grounded_sam_wrapper.reduce_mask(mask, 1)
        masks.append(mask_reduced)
        # masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)



        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])

        # Reduced
        # mask_reduced = grounded_sam_wrapper.reduce_mask(mask, 1)
        depth_masked_reduced = grounded_sam_wrapper.mask_depth_map(depth, mask_reduced)
        pointcloud_masked_reduced = depth_anything_wrapper.get_pointcloud(depth_masked_reduced)
        pointcloud_masked_reduced_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked_reduced, transform)

        # Show reduced
        # grounded_sam_wrapper.show_mask_union(mask, mask_reduced)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_reduced_world])
        # depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointcloud_masked_world, pointcloud_masked_reduced_world])
        
    index_pointcloud = 0
    index_pointcloud_next = 5
    bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    start = time.perf_counter()
    result = opt.differential_evolution(
        inliers_function,
        bounds,
        args=(depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud_next], transforms[index_pointcloud_next]),
        strategy='best1bin',
        maxiter=20,
        popsize=15,
        tol=1e-2,
        disp=True
    )
    end = time.perf_counter()
    print(f"Optimization took {end - start:.2f} seconds")

    alpha_opt, beta_opt = result.x
    start = time.perf_counter()
    y_opt = inliers_function([alpha_opt, beta_opt], depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud_next], transforms[index_pointcloud_next])
    end = time.perf_counter()
    print(f"One function call took {end - start:.4f} seconds")
    print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Inliers: {y_opt}')

    depth_opt = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha_opt, shift=beta_opt)
    pc1_opt = depth_anything_wrapper.get_pointcloud(depth_opt)
    pc_world_opt = depth_anything_wrapper.transform_pointcloud_to_world(pc1_opt, transforms[index_pointcloud])

    projection_pc2_depth_opt, projection_pc2_mask_opt = depth_anything_wrapper.project_pointcloud(pc_world_opt, transforms[index_pointcloud_next])
    fixed_projection_pc2_mask_world = grounded_sam_wrapper.fill_mask_holes(projection_pc2_depth_opt)
    grounded_sam_wrapper.show_masks([fixed_projection_pc2_mask_world], title="Projection")
    grounded_sam_wrapper.show_mask_union(fixed_projection_pc2_mask_world, masks[index_pointcloud_next])

    # depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointclouds_world_opt[-1]], [transforms[index_pointcloud]], title='Best pointclouds')
    depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_world_opt], transforms, title='Best pointclouds')   

def icp():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:2]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_world])


    index_pointcloud = 0
    max_inliers_counter = 0
    best_alpha, best_beta = 0, 0
    best_pointcloud = None
    best_projection = None
    A = None
    b = []
    icp_depths = [depths_masked[index_pointcloud]]
    icp_pointclouds0 = [pointclouds_masked_world[index_pointcloud]]
    # Loop N times
    for i in range(10):
        for pc_point_index, pc_point in enumerate(icp_pointclouds0[-1].points):
            # print(f'Index: {pc_point_index}, pc_point: {pc_point}')

            # Project the point into coordinates of camera 2
            projection2 = depth_anything_wrapper.project_point_from_world(pc_point, transforms[index_pointcloud+1])
            # grounded_sam_wrapper.show_mask_and_points(projection2, [projection2], title="Pointcloud 1 projected onto camera 2 with rand points")

            # Get the closest points between single point (from pointcloud 1) and mask 2
            closest2 = grounded_sam_wrapper.get_closest_point(masks[index_pointcloud+1], projection2)

            # Show the closest points
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2], title="Mask 2 in camera 2 with closest points")
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [closest2, projection2], title="Projection and closest point A")

            # Triangulate
            trtiangulated0 = depth_anything_wrapper.triangulate(pc_point, transforms[index_pointcloud], closest2, transforms[index_pointcloud+1])
            # print(f'trtiangulated0_A: {trtiangulated0}')
            try:
                # Project the triangulated points into coordinates of camera 2
                projected_triangulated2 = depth_anything_wrapper.project_point_from_world(trtiangulated0, transforms[index_pointcloud+1])
            except Exception as e:
                print(f'{e}, Skipping to next iteration')
                continue
        
            # Show mask 2 and triangulated points
            # grounded_sam_wrapper.show_mask_and_points(masks[index_pointcloud+1], [projected_triangulated2, projected_triangulated2], title="Mask 2 with triangulated points")
            # Show pointcloud 1 and triangulated points in camera 2
            # grounded_sam_wrapper.show_mask_and_points(projected_pointcloud_mask_2, [projected_triangulated2, projected_triangulated2], title="Pointcloud 1 projected onto camera 2 with triangulated points")
        
            # Transform triangulated point into camera 1 coordinates
            triangulated1 = depth_anything_wrapper.transform_point_from_world(trtiangulated0, transforms[index_pointcloud])

            # Get  distance of original point0_A and point0_B
            z_orig = depth_anything_wrapper.transform_point_from_world(pc_point, transforms[index_pointcloud])[2]

            # Get distance id triangulated point
            z_triangulated = triangulated1[2]
            # print(f'depth_orig: {z_orig}')
            # print(f'depth_triangulated: {z_triangulated}')
        
            # Construct A and b
            A_temp = np.matrix([[z_orig, 1]])
            # print(f'A_temp: {A_temp}')

            A = np.row_stack((A, A_temp)) if A is not None else A_temp
            # b = np.concatenate((b, z_triangulated)) if b is not None else [z_triangulated]
            b.append(z_triangulated)
            # print(f'A.shape: {A.shape}')
            # print(f'b.shape: {len(b)}')

        x, res, rank, s = np.linalg.lstsq(A, b)
        alpha, beta = x[0], x[1]
        print(f'alpha: {alpha}')
        print(f'beta: {beta}')

        # Scale and shift
        depth_new = depth_anything_wrapper.scale_depth_map(icp_depths[-1], scale=alpha, shift=beta)
        icp_depths.append(depth_new)

        # Show original and scaled depth maps
        # depth_anything_wrapper.show_depth_map(depths_masked[index_pointcloud], title="Original depth map")
        # depth_anything_wrapper.show_depth_map(depth_new, title="Scaled depth map")

        # Get scaled/shifted pointcloud
        new_pc1 = depth_anything_wrapper.get_pointcloud(depth_new)

        # Transform scaled pointcloud into world coordinates
        new_pc0 = depth_anything_wrapper.transform_pointcloud_to_world(new_pc1, transforms[index_pointcloud])
        icp_pointclouds0.append(new_pc0)

        # Get projection of scaled pointcloud into camera 2
        projection_new_pc2_depth, projection_new_pc2_mask = depth_anything_wrapper.project_pointcloud(new_pc0, transforms[index_pointcloud+1])
    
        # Show projection of scaled pointcloud in camera 2 and closest points
        # grounded_sam_wrapper.show_mask_and_points(projection_new_pc2_depth, [closest2_A, closest2_B], title="Scaled pointcloud with closest points")

        # # Show original and scaled depth maps and masks
        # grounded_sam_wrapper.show_masks([projection_new_pc2_mask, masks[index_pointcloud+1]], title="Scaled depth map and mask")
        # # Show original and scaled pointclouds in world coordinates
        # depth_anything_wrapper.show_pointclouds_with_frames([pointclouds_masked_world[index_pointcloud], new_pc0], [transforms[index_pointcloud]], title="Original and scaled pointcloud")

        # Count the number of inliers between mask 2 and projection of scaled pointcloud
        num_inliers, _ = depth_anything_wrapper.count_inliers(projection_new_pc2_mask, masks[index_pointcloud+1])
        print(f'{i}: num_inliers: {num_inliers}')

    depth_anything_wrapper.show_pointclouds(icp_pointclouds0, title="ICP pointclouds")

def interactive_scale_shift():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_world])

    scale, shift, num_inliers = depth_anything_wrapper.interactive_scale_shift(depths_masked[5], masks[6], transforms[5], transforms[6])
    print(f'scale: {scale:.3f}, shift: {shift:.3f}, num_inliers: {num_inliers}')
    depth_new = depth_anything_wrapper.scale_depth_map(depths_masked[0], scale=scale, shift=shift)
    pc_new1 = depth_anything_wrapper.get_pointcloud(depth_new)
    pc_new0 = depth_anything_wrapper.transform_pointcloud_to_world(pc_new1, transforms[5])
    depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_new0], [transforms[5]])

def test3():
    depth_anything_wrapper = DepthAnythingWrapper(intrinsics=camera_intrinsics)
    grounded_sam_wrapper = GroundedSamWrapper()
    ROS_handler = ROSHandler()

    images = []
    # depths = []
    depths_masked = []
    # pointclouds = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)

        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])

        pointcloud_message = ROS_handler.create_pointcloud_message(pointcloud_masked_world, "odom")
        print(f'pointcloud_message: {pointcloud_message}')

def calculate_angle_test():
    images = []
    depths_masked = []
    masks = []
    pointclouds_masked = []
    pointclouds_masked_world = []

    for i, transform in enumerate(transforms[0:7]):
        print(f'Processing image {i}')
        image = cv2.imread(f'/root/workspace/images/moves/cable{i}.jpg')
        images.append(image)

        depth = depth_anything_wrapper.get_depth_map(image)
        # depth_anything_wrapper.show_depth_map(depth)

        mask = grounded_sam_wrapper.get_mask(image)[0][0]
        mask_reduced = grounded_sam_wrapper.reduce_mask(mask, 1)
        masks.append(mask_reduced)
        # masks.append(mask)
        # grounded_sam_wrapper.show_mask(mask)

        depth_masked = grounded_sam_wrapper.mask_depth_map(depth, mask)
        depths_masked.append(depth_masked)
        # depth_anything_wrapper.show_depth_map(depth_masked)



        pointcloud_masked = depth_anything_wrapper.get_pointcloud(depth_masked)
        pointclouds_masked.append(pointcloud_masked)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked])

        pointcloud_masked_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked, transform)
        pointclouds_masked_world.append(pointcloud_masked_world)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])

        # Reduced
        # mask_reduced = grounded_sam_wrapper.reduce_mask(mask, 1)
        depth_masked_reduced = grounded_sam_wrapper.mask_depth_map(depth, mask_reduced)
        pointcloud_masked_reduced = depth_anything_wrapper.get_pointcloud(depth_masked_reduced)
        pointcloud_masked_reduced_world = depth_anything_wrapper.transform_pointcloud_to_world(pointcloud_masked_reduced, transform)

        # Show reduced
        # grounded_sam_wrapper.show_mask_union(mask, mask_reduced)
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_world])
        # depth_anything_wrapper.show_pointclouds([pointcloud_masked_reduced_world])
        # depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointcloud_masked_world, pointcloud_masked_reduced_world])
        
    index_pointcloud = 0
    index_pointcloud_next = 1
    bounds = [(0.05, 0.4), (-0.3, 0.3)] # [(alpha_min, alpha_max), (beta_min,  beta_max)]

    start = time.perf_counter()
    result = opt.differential_evolution(
        inliers_function,
        bounds,
        args=(depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud_next], transforms[index_pointcloud_next]),
        strategy='best1bin',
        maxiter=20,
        popsize=15,
        tol=1e-2,
        disp=True
    )
    end = time.perf_counter()
    print(f"Optimization took {end - start:.2f} seconds")

    alpha_opt, beta_opt = result.x
    start = time.perf_counter()
    y_opt = inliers_function([alpha_opt, beta_opt], depths_masked[index_pointcloud], transforms[index_pointcloud], masks[index_pointcloud_next], transforms[index_pointcloud_next])
    end = time.perf_counter()
    print(f"One function call took {end - start:.4f} seconds")
    print(f'Optimal alpha: {alpha_opt:.2f}, beta: {beta_opt:.2f}, Inliers: {y_opt}')

    depth_opt = depth_anything_wrapper.scale_depth_map(depths_masked[index_pointcloud], scale=alpha_opt, shift=beta_opt)
    pc1_opt = depth_anything_wrapper.get_pointcloud(depth_opt)
    pc_world_opt = depth_anything_wrapper.transform_pointcloud_to_world(pc1_opt, transforms[index_pointcloud])

    projection_pc2_depth_opt, projection_pc2_mask_opt = depth_anything_wrapper.project_pointcloud(pc_world_opt, transforms[index_pointcloud_next])
    fixed_projection_pc2_mask_world = grounded_sam_wrapper.fill_mask_holes(projection_pc2_depth_opt)
    grounded_sam_wrapper.show_masks([fixed_projection_pc2_mask_world], title="Projection")
    grounded_sam_wrapper.show_mask_union(fixed_projection_pc2_mask_world, masks[index_pointcloud_next])

    # depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pointclouds_world_opt[-1]], [transforms[index_pointcloud]], title='Best pointclouds')
    depth_anything_wrapper.show_pointclouds_with_frames_and_grid([pc_world_opt], transforms, title='Best pointclouds')   

    pose_publisher = PosePublisher("/next_pose")
    path_publisher = PathPublisher("/my_path")


    target_point = get_highest_point(pc_world_opt)
    target_point_offset = target_point
    target_point_offset[2] += 0.098 - 0.0015 # Make target Pose hover above actual target pose - tested offset
    # Convert to Pose
    target_pose = get_desired_pose(target_point_offset)
    # Calculate Path
    target_path = interpolate_poses(transforms[1], target_pose, num_steps=3)
    # Move arm a step
    
    _, final_mask = depth_anything_wrapper.project_pointcloud(pc_world_opt, target_pose)
    # target_point_2D = depth_anything_wrapper.project_point_world(target_point, target_pose)
    angle = calculate_angle_from_mask_and_point(final_mask, [640//2, 480//2])
    rotated_target_pose = rotate_pose_around_z(target_pose, angle)

    while not rospy.is_shutdown():
        pose_publisher.publish(rotated_target_pose)
        path_publisher.publish(target_path)
        rospy.sleep(1)

    # point = (186, 349)
    # calculate_angle_from_mask_and_point(mask, point)
    # grounded_sam_wrapper.show_mask_and_points(mask, [point])

def pipeline_offline_test(offline_transforms):
    my_class = MyClass()

    image = cv2.imread(f'/root/workspace/images/moves/cable{0}.jpg')

    images = []
    transforms = []
    transforms_palm = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []

    # Take image
    images.append(cv2.imread(f'/root/workspace/images/moves/cable{0}.jpg'))
    # Get current transform
    transforms.append(offline_transforms[0])
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=True))
    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()

    index = 0
    while not rospy.is_shutdown(): 
        index += 1
        # Take image
        images.append(cv2.imread(f'/root/workspace/images/moves/cable{index}.jpg'))
        # Get current transform
        transforms.append(offline_transforms[index])
        # Process image
        data.append(my_class.process_image(images[-1], transforms[-1], show=True))
        # Estimate scale and shift
        best_alpha, best_beta, best_pc_world, score = my_class.estimate_scale_shift_new_distance(data[-2], data[-1], transforms[-2], transforms[-1], show=True)
        best_alphas.append(best_alpha)
        best_betas.append(best_beta)
        best_pcs_world.append(best_pc_world)

    # Loop End

def pipeline2_offline_test(offline_transforms):
    my_class = MyClass()

    image = cv2.imread(f'/root/workspace/images/moves/cable{0}.jpg')

    images = []
    transforms = []
    transforms_palm = []
    data = [] # depth, mask, depth_masked, pointcloud_masked, pointcloud_masked_world
    target_poses = []
    best_alphas = []
    best_betas = []
    best_pcs_world = []

    # Take image
    images.append(cv2.imread(f'/root/workspace/images/moves/cable{0}.jpg'))
    # Get current transform
    transforms.append(offline_transforms[0])
    # Process image
    data.append(my_class.process_image(images[-1], transforms[-1], show=True))
    usr_input = input("Mask Correct? [y]/n: ")
    if usr_input == "n": exit()

    index = 0
    while not rospy.is_shutdown(): 
        index += 1
        # Take image
        images.append(cv2.imread(f'/root/workspace/images/moves/cable{index}.jpg'))
        # Get current transform
        transforms.append(offline_transforms[index])
        # Process image
        data.append(my_class.process_image(images[-1], transforms[-1], show=True))
        # Estimate scale and shift
        best_alpha, best_beta, best_pc_world, num_inliers = my_class.estimate_scale_shift_from_multiple_cameras_distance(data, transforms, show=True)
        best_alphas.append(best_alpha)
        best_betas.append(best_beta)
        best_pcs_world.append(best_pc_world)

    # Loop End




if __name__ == '__main__':
    rospy.init_node('pointcloud_test', anonymous=True)
    transform_stamped0 = get_stamped_transform([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    transform_stamped1 = get_stamped_transform([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    transform_stamped2 = get_stamped_transform([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    transform_stamped3 = get_stamped_transform([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    transform_stamped4 = get_stamped_transform([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    transform_stamped5 = get_stamped_transform([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    transform_stamped6 = get_stamped_transform([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    transforms = [transform_stamped0, transform_stamped1, transform_stamped2, transform_stamped3, transform_stamped4, transform_stamped5, transform_stamped6]

    # test_least_squares()
    # test2()
    # ransac()
    # ransac2()
    # ransac3()
    # icp()
    # interactive_scale_shift()
    # test3()
    # differential_evolution()
    # differential_evolution_single()
    # calculate_angle_test()
    # pipeline_offline_test(offline_transforms=transforms)
    pipeline2_offline_test(offline_transforms=transforms)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
