from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from geometry_msgs.msg import TransformStamped

import cv2

camera_intrinsics = (203.71833, 203.71833, 319.5, 239.5)

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

def test1():
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

        depth = depth_anything_wrapper.get_depth_map(image)
        depth = depth_anything_wrapper.scale_depth_map(depth, scale=0.1, shift=5)
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

    grounded_sam_wrapper.show_masks([masks[1], projection_pc0_to_cam1_mask], wait=True)

    depth_anything_wrapper.show_pointclouds_with_frames(pointclouds_masked_world[0:2], transforms[0:2])    


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


if __name__ == '__main__':
    transform_stamped0 = get_stamped_transform([0.767, 0.495, 0.712], [0.707, 0.001, 0.707, -0.001])
    transform_stamped1 = get_stamped_transform([0.839, 0.493, 0.727], [0.774, 0.001, 0.633, -0.001])
    transform_stamped2 = get_stamped_transform([0.903, 0.493, 0.728], [0.834, 0.001, 0.552, -0.000])
    transform_stamped3 = get_stamped_transform([0.953, 0.493, 0.717], [0.885, 0.000, 0.466, -0.000])
    transform_stamped4 = get_stamped_transform([0.991, 0.492, 0.698], [0.927, 0.001, 0.376, -0.000])
    transform_stamped5 = get_stamped_transform([1.014, 0.493, 0.672], [0.960, 0.001, 0.282, -0.000])
    transform_stamped6 = get_stamped_transform([1.025, 0.493, 0.643], [0.983, 0.001, 0.184, -0.000])
    transforms = [transform_stamped0, transform_stamped1, transform_stamped2, transform_stamped3, transform_stamped4, transform_stamped5, transform_stamped6]

    test1()
    # test2()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
