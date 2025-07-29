# Custom Files
from depth_anything_wrapper import DepthAnythingWrapper
from grounded_sam_wrapper import GroundedSamWrapper
from publisher import *
from my_utils import *

import numpy as np
import cv2
import os



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



def save_numpy_to_file(depth_map: np.ndarray, folder_name: str, file_name: str) -> None:
    """
    Save a depth map to a .npy file.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map array of shape (H, W).
    folder_name : str
        Path to the directory where the file will be saved.
    file_name : str
        Name of the file (with or without .npy extension).

    Raises
    ------
    TypeError
        If depth_map is not a numpy.ndarray or folder_name/file_name are not strings.
    OSError
        If the directory cannot be created or file cannot be written.
    """
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a numpy.ndarray")
    if not isinstance(folder_name, str) or not isinstance(file_name, str):
        raise TypeError("folder_name and file_name must be strings")

    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Ensure .npy extension
    if not file_name.lower().endswith('.npy'):
        file_name = file_name + '.npy'

    path = os.path.join(folder_name, file_name)
    # Save using NumPy
    np.save(path, depth_map)

def load_numpy_from_file(folder_name: str, file_name: str) -> np.ndarray:
    """
    Load a depth map from a .npy file.

    Parameters
    ----------
    folder_name : str
        Path to the directory containing the file.
    file_name : str
        Name of the file (with or without .npy extension).

    Returns
    -------
    np.ndarray
        The loaded depth map array.

    Raises
    ------
    TypeError
        If folder_name/file_name are not strings.
    FileNotFoundError
        If the .npy file does not exist.
    OSError
        If the file cannot be read.
    """
    if not isinstance(folder_name, str) or not isinstance(file_name, str):
        raise TypeError("folder_name and file_name must be strings")

    # Ensure .npy extension
    if not file_name.lower().endswith('.npy'):
        file_name = file_name + '.npy'

    path = os.path.join(folder_name, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: '{path}'")

    # Load using NumPy
    depth_map = np.load(path)
    return depth_map



if __name__ == "__main__":
    image_processing = ImageProcessing()


    depths_folder = f'/root/workspace/images/moves_depths'
    masks_folder = f'/root/workspace/images/moves_masks'

    offline_image_name = "cable"

    SAM_prompt = "wire.cable.tube."

    
    # Save Masks and Depths
    for i in range(7):
        print(f"Processing image {i}")
        image = cv2.imread(f'/root/workspace/images/moves/{offline_image_name}{i}.jpg')
        mask = image_processing.get_mask(image, SAM_prompt, show=True)
        depth_masked = image_processing.get_depth_masked(image, mask, show=True)

        save_numpy_to_file(mask, masks_folder, f'{offline_image_name}{i}')
        save_numpy_to_file(depth_masked, depths_folder, f'{offline_image_name}{i}')
