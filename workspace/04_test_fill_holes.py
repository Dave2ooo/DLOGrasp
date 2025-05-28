from grounded_sam_wrapper import GroundedSamWrapper
import cv2

grounded_sam_wrapper = GroundedSamWrapper()

if __name__ == "__main__":
    # === Parameters ===
    input_path = '/root/workspace/images/fill_holes/orig.png'      # Path to your test PNG mask
    closing_radius = 5                 # Adjust radius for closing
    kernel_shape = 'disk'              # 'disk' or 'square'
    # ==================

    # Load the mask as grayscale
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    # Apply hole filling
    # filled_mask = grounded_sam_wrapper.fill_mask_holes(mask, closing_radius, kernel_shape)
    # filled_mask = grounded_sam_wrapper.fill_mask_holes(mask)
    filled_mask = grounded_sam_wrapper.fill_mask_holes(mask)

    # Display original and filled masks
    cv2.imshow("Original Mask", mask)
    cv2.imshow("Filled Mask", filled_mask)
    print(f"Applied fill_mask_holes with radius={closing_radius}, shape='{kernel_shape}'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()