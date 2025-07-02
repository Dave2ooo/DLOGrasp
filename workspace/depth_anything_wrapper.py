import os
import sys
import cv2
import torch
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import open3d as o3d
from my_utils import *

from tf.transformations import quaternion_matrix
from geometry_msgs.msg import TransformStamped, PoseStamped

script_path = os.path.abspath(__file__)

# Import Depth Anything v2
depth_anything_directory = '/root/Depth-Anything-V2'
sys.path.insert(1, depth_anything_directory)
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

import copy

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

class DepthAnythingWrapper():
    def __init__(self, encoder='vitl', dataset='hypersim', max_depth=20):
        self.encoder = encoder # or 'vits', 'vitb'
        self.dataset = dataset # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.max_depth = max_depth # 20 for indoor model, 80 for outdoor model
        self.depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        checkpoint_path = f'{depth_anything_directory}/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
        self.depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.depth_anything = self.depth_anything.to(DEVICE).eval()

    def get_depth_map(self, image: NDArray[np.uint8]):
        return self.depth_anything.infer_image(image)