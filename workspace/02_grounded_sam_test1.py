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





# Try to switch to an interactive backend (if a GUI is available). If none work, we fall back to Agg.
for _backend in ("QtAgg", "TkAgg", "GTK3Agg", "MacOSX"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        pass  # backend not available; continue searching

import matplotlib.pyplot as plt  # noqa: E402  # must come **after** backend selection


def visualise_and_save(img_pil, masks, boxes, out_dir="visualisations", alpha: float = 0.45):
    """Create visualisations (segmented overlay, raw mask, bbox image), display them if possible,
    and always save them to *out_dir*.

    Args:
        img_pil (PIL.Image): Input image.
        masks (np.ndarray): Boolean/integer mask array with shape (N, H, W).
        boxes (np.ndarray): Bounding boxes, shape (N, 4) in (x0, y0, x1, y1).
        out_dir (str): Folder where images will be written.
        alpha (float): Transparency of the mask overlay when composing the segmented view.
    """

    os.makedirs(out_dir, exist_ok=True)

    img_np = np.array(img_pil.convert("RGB"))

    # ── 1️⃣  segmented overlay ────────────────────────────────────────────────
    overlay_colour = np.array([0, 255, 0], dtype=np.uint8)  # green
    segmented = img_np.copy()
    for m in masks:
        segmented[m.astype(bool)] = (
            (1 - alpha) * segmented[m.astype(bool)] + alpha * overlay_colour
        ).astype(np.uint8)

    # ── 2️⃣  bounding‑box image ───────────────────────────────────────────────
    box_img = img_np.copy()
    for (x0, y0, x1, y1) in boxes.astype(int):
        cv2.rectangle(box_img, (x0, y0), (x1, y1), (255, 0, 0), 2)  # red box

    # ── 3️⃣  raw mask (first mask only for display) ───────────────────────────
    mask_img = (masks[0] * 255).astype(np.uint8)

    # ── Save individual outputs ───────────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "segmented.png"), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "bbox.png"), cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask_img)

    # ── Create and save combined figure ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(segmented)
    axes[0].set_title("Segmented Overlay")
    axes[0].axis("off")

    axes[1].imshow(mask_img, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(box_img)
    axes[2].set_title("Bounding Boxes")
    axes[2].axis("off")

    plt.tight_layout()
    combined_path = os.path.join(out_dir, "summary.png")
    fig.savefig(combined_path, dpi=300)

    # Attempt to display if we're on an interactive backend --------------------
    try:
        plt.show(block=False)
        plt.pause(0.001)  # give the window a chance to render
    except Exception:
        print("Non‑interactive Matplotlib backend detected; figure saved to:", combined_path)

    print("\n[✓] Outputs written to:")
    for f in ("segmented.png", "bbox.png", "mask.png", "summary.png"):
        print(" └─", os.path.join(out_dir, f))


# ── Call the helper with your results ─────────────────────────────────────────
visualise_and_save(image, masks, input_boxes)
