
from save_load_numpy import load_numpy_from_file
from save_data import load_pose_stamped

from my_utils import *
from save_data import *


# offline_folder = '/root/workspace/images/thesis_images/' + '2025_08_04_11-17'
# offline_folder = '/root/workspace/images/thesis_images/' + '2025_08_22_11-40'
offline_folder = '/root/workspace/images/thesis_images/' + '2025_08_22_12-44'
folder_name_image = f'{offline_folder}/image'
folder_name_camera_pose = f'{offline_folder}/camera_pose'
folder_name_palm_pose = f'{offline_folder}/palm_pose'
folder_name_spline_fine = f'{offline_folder}/spline_fine'
folder_name_masks = f'{offline_folder}/mask_numpy'
folder_name_projections_thesis_sampled = f'{offline_folder}/projections_thesis_sampled'
offline_counter = 0


camera_poses = []
masks = []

spline_initial = load_bspline(offline_folder, "initial_spline")

for i in range(4):
    print(i)
    camera_poses.append(load_pose_stamped(folder_name_camera_pose, str(i))) # <- offline
    masks.append(load_numpy_from_file(folder_name_masks, str(i)))

camera_poses_display = [camera_poses[0], camera_poses[2]]

# show_bspline_with_frames_and_grid(spline_initial, camera_poses_display, num_samples=20, line_width_px=5, tube_radius=0.005, show_grid=False)


#region save projections
# skeletons, interps = precompute_skeletons_and_interps(masks) 

# projected_spline_cam2_initial = []
# num_samples = 40
# for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
#     projected_spline_cam2_initial.append(project_bspline_points(spline_initial, camera_pose, camera_parameters, num_samples=num_samples))
#     # show_masks([skeleton, projected_spline_cam2_initial[-1]], f"Projected B-Spline Cam {index} Coarse")

# save_masks([skeletons[0], projected_spline_cam2_initial[0]], folder_name_projections_thesis_sampled, f'initial_spline_projected_onto_0', invert_color=True)
# save_masks([skeletons[2], projected_spline_cam2_initial[2]], folder_name_projections_thesis_sampled, f'initial_spline_projected_onto_2', invert_color=True)


# spline_fine = load_bspline(folder_name_spline_fine, "2")
# projected_spline_cam2_fine = []
# for index, (skeleton, camera_pose) in enumerate(zip(skeletons, camera_poses)):
#     projected_spline_cam2_fine.append(project_bspline_points(spline_fine, camera_pose, camera_parameters, num_samples=num_samples))
#     # show_masks([skeleton, projected_spline_cam2_fine[-1]], f"Projected B-Spline Cam {index} Coarse")

# save_masks([skeletons[0], projected_spline_cam2_fine[0]], folder_name_projections_thesis_sampled, f'spline_fine_projected_onto_0', invert_color=True)
# save_masks([skeletons[2], projected_spline_cam2_fine[2]], folder_name_projections_thesis_sampled, f'spline_fine_projected_onto_2', invert_color=True)
#endregion save projections



skeletons, interps = precompute_skeletons_and_interps(masks) 

# for mask in masks:
    # show_masks(skeleton_distance_transform(mask), f"Interp Cam 0")



height = 480
width = 640
left = 220
bottom = 120
right = 300
top = 160

dt = skeleton_distance_transform(masks[0])
dt_copped = dt[top:height-bottom, left:width-right]
dt_color = dt_to_color(dt_copped, cmap='plasma')  # returns H×W×3 uint8 BGR
cv2.imshow('DT coloured', dt_color); cv2.waitKey(0)

left = 255
bottom = 65
right = 220
top = 140

dt = skeleton_distance_transform(masks[2])
dt_copped = dt[top:height-bottom, left:width-right]
dt_color = dt_to_color(dt_copped, cmap='plasma')  # returns H×W×3 uint8 BGR
cv2.imshow('DT coloured', dt_color); cv2.waitKey(0)