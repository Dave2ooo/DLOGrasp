import os
import sys
import cv2
import torch
import numpy as np
from numpy.typing import NDArray
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


class GroundedSamWrapper:
    def __init__(self):
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
        self.sam2_checkpoint = f'{grounded_sam_directory}/checkpoints/sam2_hiera_large.pt'
        self.model_cfg = "sam2_hiera_l.yaml"

        # self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        sam2_image_model = build_sam2(self.model_cfg, self.sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # init grounding dino model from huggingface
        # model_id = "IDEA-Research/grounding-dino-tiny"
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def get_mask(self, image: NDArray[np.uint8], prompt="tube.cable."):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        text = prompt

        # run Grounding DINO on the image
        inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image_pil.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"].cpu().numpy()
        OBJECTS = results[0]["labels"]

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # print(f'masks.shape = {masks.shape}')
        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        # elif masks.ndim == 4:
            # masks = masks.squeeze(1)

        # print(f'masks.shape = {masks.shape}')
        return masks
    
    def show_mask(self, mask: NDArray[np.ndarray], wait = True):
        """
        Display a binary or float mask using OpenCV, with nothing else overlaid.
        Blocks until any key is pressed.
        """
        # if mask is (N, H, W), take the first one
        if mask.ndim == 3:
            mask = mask[0]

        # convert boolean or float mask to uint8 0–255
        mask_uint8 = (mask.astype(np.float32) * 255).astype(np.uint8)

        # create a resizable window (you can switch to WINDOW_AUTOSIZE if you prefer)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", mask_uint8)
        if wait:
            cv2.waitKey(0)

    def show_masks(self, masks, wait=True) -> None:
        """
        Display multiple binary or float masks in one window, each colored differently.
        Blocks until any key is pressed.

        Parameters
        ----------
        masks : np.ndarray or sequence of np.ndarray
            If a single 2D mask (H×W), it will be wrapped in a list.
            If an array of shape (N, H, W), each slice along axis 0 is treated as one mask.
            Or you can pass a list/tuple of 2D masks.
        """
        import cv2
        import numpy as np

        # Normalize input to a list of 2D masks
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                mask_list = [masks]
            elif masks.ndim == 3:
                mask_list = [masks[i] for i in range(masks.shape[0])]
            else:
                raise ValueError("masks array must have 2 or 3 dimensions")
        elif isinstance(masks, (list, tuple)):
            mask_list = list(masks)
        else:
            raise TypeError("masks must be a numpy array or a list/tuple of arrays")

        # All masks must have the same shape
        H, W = mask_list[0].shape

        # Pre‐defined distinct BGR colors
        colors = [
            (0, 0, 255),    # red
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
            (128, 0, 128),  # purple
            (0, 128, 128),  # teal
            (128, 128, 0),  # olive
        ]

        # Create an empty color canvas
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Overlay each mask in its color
        for idx, mask in enumerate(mask_list):
            # If mask has extra dims (e.g. (1, H, W)), take the first channel
            if mask.ndim == 3:
                mask = mask[0]
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized.astype(bool)
            color = colors[idx % len(colors)]
            canvas[mask_bool] = color

        # Display the combined masks
        cv2.namedWindow("masks", cv2.WINDOW_NORMAL)
        cv2.imshow("masks", canvas)
        if wait:
            cv2.waitKey(0)

    
if __name__ == "__main__":
    image = cv2.imread('/root/workspace/images/moves/cable0.jpg')

    grounded_sam_wrapper = GroundedSamWrapper()
    masks = grounded_sam_wrapper.get_mask(image)
    grounded_sam_wrapper.show_mask(masks[0])