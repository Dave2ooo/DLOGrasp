"""
Grounded‑SAM / Grounding‑DINO tuning helper – batch‑folder version
================================================================
💡 **NEW**: the magenta mask now always corresponds to the *highest‑confidence
bounding‑box* that Grounding‑DINO produced for the current image.
That eliminates the “wrong box shows up in `sam_mask`” problem.

Usage is unchanged: put your test frames in a folder, set `FOLDER_PATH` and
`PROMPT`, run the file, and step through with any key (Esc exits).

───────────────────────────────────────────────────────────────────────────────
TUNING ORDER (unchanged)
───────────────────────────────────────────────────────────────────────────────
1. Adjust `BOX_THRESHOLD`, `TEXT_THRESHOLD` until green boxes look right.
2. Tune `BOX_SHRINK_PCT`, `MASK_THRESHOLD`, `MULTIMASK_OUTPUT` for masks.
3. Fine‑tune only if both stages plateau.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ───────────── CONFIG ─────────────────────────────────────────────────────────
MODEL_ID   = "IDEA-Research/grounding-dino-base"
SAM2_CKPT  = "/root/grounded_sam2/checkpoints/sam2_hiera_large.pt"
SAM2_CFG   = "sam2_hiera_l.yaml"

# Detector params – tune *first*
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

# Mask params – tune *second*
MASK_THRESHOLD   = 1.0
BOX_SHRINK_PCT   = 0.00
MULTIMASK_OUTPUT = True

# ───────────── MODEL BUILD ────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_models():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    sam2 = build_sam2(SAM2_CFG, SAM2_CKPT).to(device)
    predictor = SAM2ImagePredictor(sam2, mask_threshold=MASK_THRESHOLD)
    return processor, dino, predictor

# ───────────── HELPER FUNCTIONS ───────────────────────────────────────────────

def run_dino(img_bgr, prompt, processor, dino):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil.size[::-1]],
    )
    boxes  = res[0]["boxes"].cpu().numpy()
    labels = res[0]["labels"]
    scores = res[0]["scores"].cpu().numpy()
    return boxes, labels, scores


def shrink_box(box, pct):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pct, (y2 - y1) * pct
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def run_sam(img_bgr, boxes, scores, predictor):
    """Return mask for the box with *highest confidence*."""
    if len(boxes) == 0:
        return None

    idx_max = int(np.argmax(scores))          # highest‑conf box
    box_one = boxes[idx_max:idx_max + 1]      # shape (1,4)

    # optional shrink
    if BOX_SHRINK_PCT > 0:
        box_one = np.vstack([shrink_box(b, BOX_SHRINK_PCT) for b in box_one])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)
        masks, m_scores, _ = predictor.predict(
            box=box_one,
            multimask_output=MULTIMASK_OUTPUT,
        )

    # pick best variant by (score / area) if multimask
    if MULTIMASK_OUTPUT and masks.ndim == 4:
        best = max(range(3), key=lambda j: m_scores[0][j] / masks[0, j].sum())
        mask = masks[0, best]
    else:
        mask = masks[0]
    return mask


def viz_boxes(img, boxes, labels, scores):
    vis = img.copy()
    for (x1, y1, x2, y2), lab, sc in zip(boxes.astype(int), labels, scores):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{lab}:{sc:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("dino_boxes", vis)


def viz_mask(mask):
    cv2.imshow("sam_mask", (mask.astype(np.float32) * 255).astype(np.uint8))

# ───────────── USER SETTINGS ──────────────────────────────────────────────────
if __name__ == "__main__":
    FOLDER_PATH = "/root/workspace/images/tube_images/only_ceiling_lights"  # folder with images
    PROMPT      = "medical plastic hose ."

    exts = {".jpg", ".jpeg", ".png"}
    img_files = [os.path.join(FOLDER_PATH, f) for f in sorted(os.listdir(FOLDER_PATH))
                 if os.path.splitext(f.lower())[1] in exts]
    if not img_files:
        raise RuntimeError(f"No images found in {FOLDER_PATH}")

    processor, dino, predictor = build_models()

    for path in img_files:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] skipping unreadable file: {path}")
            continue

        boxes, labels, scores = run_dino(img, PROMPT, processor, dino)
        viz_boxes(img, boxes, labels, scores)

        mask = run_sam(img, boxes, scores, predictor)
        if mask is not None:
            viz_mask(mask)
        else:
            cv2.imshow("sam_mask", np.zeros_like(img[:, :, 0]))
            print(f"[info] no boxes for {os.path.basename(path)} – adjust thresholds")

        print(f"Viewing: {os.path.basename(path)}  –  press any key for next, Esc to exit.")
        key = cv2.waitKey(0)
        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()
