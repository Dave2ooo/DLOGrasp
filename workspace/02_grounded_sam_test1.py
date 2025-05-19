import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image

grounded_sam_directory = '/root/grounded_sam2'
# sys.path.insert(1, grounded_sam_directory)
sys.path.append(grounded_sam_directory)
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# import matplotlib.pyplot as plt

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = f'{grounded_sam_directory}/checkpoints/sam2_hiera_large.pt'
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
# model_id = "IDEA-Research/grounding-dino-tiny"
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "tube."

img_path = '/root/workspace/images/image1.jpg'
image = Image.open(img_path)

# run Grounding DINO on the image
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.25,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)






# --- Visualisation with OpenCV (no Matplotlib required) ----------------------
# Paste this after you already have `masks`, `input_boxes`, and `image`.

import cv2
import numpy as np
import os

# Convert PIL → NumPy RGB
img_np = np.array(image.convert("RGB"))  # (H, W, 3)
H, W = img_np.shape[:2]

# ---------- NORMALISE MASKS --------------------------------------------------
mask_list: list[np.ndarray] = []
if masks.ndim == 2:
    mask_list = [masks]
elif masks.ndim == 3:
    mask_list = [m for m in masks]
elif masks.ndim == 4:
    mask_list = [m.squeeze(0) for m in masks]
else:
    raise ValueError(f"Unexpected mask ndim {masks.ndim}")

mask_list_resized = []
for m in mask_list:
    if m.shape != (H, W):
        m_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        m_resized = m.astype(np.uint8)
    mask_list_resized.append(m_resized)

# ---------- 1️⃣  SEGMENTED OVERLAY ------------------------------------------
segmented = img_np.copy()
overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # green
alpha = 0.45
for m in mask_list_resized:
    mask_bool = m.astype(bool)
    segmented[mask_bool] = ((1 - alpha) * segmented[mask_bool] + alpha * overlay_color).astype(np.uint8)

# ---------- 2️⃣  BOUNDING-BOX IMAGE -----------------------------------------
box_img = img_np.copy()
for box in input_boxes:
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(box_img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red

# ---------- 3️⃣  MASK VISUAL (first mask) -----------------------------------
mask_vis = cv2.cvtColor(mask_list_resized[0] * 255, cv2.COLOR_GRAY2BGR)  # white mask

# ---------- DISPLAY WITH OPENCV --------------------------------------------
cv2.imshow("Segmented", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
cv2.imshow("Bounding Boxes", cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR))
cv2.imshow("Mask", mask_vis)
print("[Info] Press any key in an image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
