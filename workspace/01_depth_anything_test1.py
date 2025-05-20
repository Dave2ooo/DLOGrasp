import os
import sys
import cv2
import torch
import numpy as np

script_path = os.path.abspath(__file__)
print(script_path)

depth_anything_directory = '/root/Depth-Anything-V2'
sys.path.insert(1, depth_anything_directory)
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
checkpoint_path = f'{depth_anything_directory}/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
# model.eval()
depth_anything = depth_anything.to(DEVICE).eval()


raw_img = cv2.imread('./images/moves/cable0.jpg')
depth = depth_anything.infer_image(raw_img) # HxW depth map in meters in numpy

depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)

cv2.imshow("image", depth)
cv2.waitKey(0)