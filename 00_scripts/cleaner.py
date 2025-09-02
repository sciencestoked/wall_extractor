#!/usr/bin/env python3
"""
Floor Plan Cleaning Tool (v5 - Mask and Reconstruct)
Precisely removes noise by creating a 'noise mask' and subtracting it
from the original binarized image, preserving original wall shapes.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# --- Configuration (Derived from your statistical analysis) ---
CONFIG = {
    # Any shape with an area SMALLER than this is considered noise.
    "noise_max_area": 50,

    # Any shape with an aspect ratio OUTSIDE this range is considered noise.
    # e.g., 1/15 to 15. This removes very long/thin lines.
    "noise_aspect_ratio_threshold": 15.0,
}
# --- End Configuration ---

def clean_image(image_path: Path, config: dict) -> np.ndarray:
    """
    Applies a mask-and-reconstruct cleaning pipeline.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Stage 1: Create a clean binarized version of the source ---
    # This will be our base image that we subtract noise from.
    # Lines are white for processing.
    source_mask = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # --- Stage 2: Identify and create a 'Noise Mask' ---
    # Find all contours in the original binary image.
    contours, _ = cv2.findContours(source_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    noise_mask = np.zeros_like(source_mask)
    ar_thresh = config["noise_aspect_ratio_threshold"]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Condition 1: Is the shape too small?
        is_small_noise = area < config["noise_max_area"]
        
        # Condition 2: Is the shape too long or thin?
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        is_line_noise = (aspect_ratio > ar_thresh) or (aspect_ratio < 1/ar_thresh)
        
        # If it meets either noise condition, add it to the noise mask.
        if is_small_noise or is_line_noise:
            cv2.drawContours(noise_mask, [cnt], -1, 255, -1)
    
    # --- Stage 3: Subtract the Noise from the Source ---
    # This is the key step. We remove all pixels that were part of the noise mask.
    walls_only_mask = cv2.subtract(source_mask, noise_mask)

    # --- Final Step: Invert to Black-on-White ---
    final_image = cv2.bitwise_not(walls_only_mask)

    return final_image


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python clean_floorplan_v5.py path/to/your/image.png")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("🔧 Cleaning image with Mask and Reconstruct method:")
    for key, value in CONFIG.items():
        print(f"   - {key}: {value}")
    
    try:
        cleaned_image = clean_image(input_path, CONFIG)
        original_image_bgr = cv2.imread(str(input_path))
        cleaned_image_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side comparison
        h_orig, w_orig, _ = original_image_bgr.shape
        h_clean, w_clean = cleaned_image.shape
        if h_orig != h_clean or w_orig != w_clean:
            original_image_bgr = cv2.resize(original_image_bgr, (w_clean, h_clean))

        comparison = np.hstack([original_image_bgr, cleaned_image_bgr])
        
        window_title = "Original (Left) vs. Cleaned (Right) -- Press 's' to save, 'q' to quit"
        cv2.imshow(window_title, comparison)
        
        print("\n✅ Processing complete. Displaying result.")
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            output_dir = Path("./cleaned_images")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_cleaned_v5.png"
            cv2.imwrite(str(output_path), cleaned_image)
            print(f"💾 Image saved to: {output_path}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
