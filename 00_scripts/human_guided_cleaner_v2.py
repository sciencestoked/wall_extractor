#!/usr/bin/env python3
"""
Floor Plan Cleaning Tool (v14 - Zoom Select & Debug)
Adds a zoom feature for precise thickness measurement and a debug view
for the binarization step to diagnose line detection issues.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import math

# --- Configuration ---
CONFIG = {
    "thickness_tolerance": 0.3,
    "min_line_length": 15,
    "show_binarization_debug": True # Set to True to see the binarization result
}
# --- End Configuration ---

def get_user_defined_thickness(image: np.ndarray) -> float:
    """
    Guides the user through a zoom-and-click process to define wall thickness.
    """
    roi = None
    clone_full = image.copy()
    window_name_full = "Step 1a: Select an area to zoom in on"
    
    print("\n--- Step 1: Define Wall Thickness ---")
    print("  1. Draw a rectangle around a clear wall segment.")
    print("  -> Click and drag your mouse, then press 'c' to confirm.")

    # Let user select an ROI
    roi = cv2.selectROI(window_name_full, clone_full, showCrosshair=False)
    cv2.destroyWindow(window_name_full)
    
    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("ROI selection was cancelled or invalid.")

    # Crop the selected ROI and create a zoomed view
    x, y, w, h = roi
    roi_cropped = image[y:y+h, x:x+w]
    zoom_factor = 5
    zoomed_roi = cv2.resize(roi_cropped, (w*zoom_factor, h*zoom_factor), interpolation=cv2.INTER_NEAREST)

    points = []
    window_name_zoom = "Step 1b: Click on both edges of the wall"
    
    def mouse_callback(event, zx, zy, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((zx, zy))
            cv2.circle(zoomed_roi, (zx, zy), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(zoomed_roi, points[0], points[1], (0, 0, 255), 2)
            cv2.imshow(window_name_zoom, zoomed_roi)

    cv2.namedWindow(window_name_zoom)
    cv2.setMouseCallback(window_name_zoom, mouse_callback)
    
    print("\n  2. In the NEW zoomed window, click on both edges of the wall.")
    print("  -> Press 'c' to confirm your selection.")

    while True:
        cv2.imshow(window_name_zoom, zoomed_roi)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(points) == 2: break
            else: print("  -> Please select exactly two points.")
    
    cv2.destroyAllWindows()
    
    # Calculate distance in the zoomed image and scale it back down
    p1_zoom, p2_zoom = points
    dist_zoom = math.sqrt((p2_zoom[0] - p1_zoom[0])**2 + (p2_zoom[1] - p1_zoom[1])**2)
    original_thickness = dist_zoom / zoom_factor
    
    return original_thickness

def get_line_thickness(line: np.ndarray, img: np.ndarray, num_samples: int = 5) -> float:
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0: return 0
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    perp_dx, perp_dy = -dy, dx
    thicknesses = []
    for i in range(1, num_samples + 1):
        sample_x = x1 + dx * (length * i / (num_samples + 1))
        sample_y = y1 + dy * (length * i / (num_samples + 1))
        c, c2 = -1, -1
        while True:
            c += 1
            px, py = int(round(sample_x + perp_dx * c)), int(round(sample_y + perp_dy * c))
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]) or img[py, px] == 0: break
        while True:
            c2 += 1
            px, py = int(round(sample_x - perp_dx * c2)), int(round(sample_y - perp_dy * c2))
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]) or img[py, px] == 0: break
        thicknesses.append(c + c2)
    return np.median(thicknesses) if thicknesses else 0

def clean_image(image_path: Path, config: dict) -> np.ndarray:
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if img_gray is None: raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Stage 1: Human-Guided Thickness Measurement ---
    user_thickness = get_user_defined_thickness(img_bgr)
    print(f"  -> Measured wall thickness: {user_thickness:.2f} pixels.")
    min_thick = user_thickness * (1 - config["thickness_tolerance"])
    max_thick = user_thickness * (1 + config["thickness_tolerance"])
    print(f"  -> Setting valid thickness range to: {min_thick:.2f} - {max_thick:.2f} pixels.")

    # --- Stage 2: Binarization and Line Detection ---
    print("\n--- Step 2: Auto-detecting all line segments ---")
    binary_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    if config["show_binarization_debug"]:
        print("  -> Displaying binarization result for debugging. Check for broken lines.")
        cv2.imshow("Binarization Result", binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(binary_img)
    if lines is None: return cv2.bitwise_not(np.zeros_like(binary_img))
    print(f"  -> Found {len(lines)} total line segments.")

    # --- Stage 3: Filter Lines Based on User's Thickness ---
    print("\n--- Step 3: Filtering lines based on thickness ---")
    final_image_mask = np.zeros_like(binary_img)
    kept_lines_count = 0
    
    for line_arr in lines:
        line = line_arr[0]
        length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        if length > config["min_line_length"]:
            measured_thickness = get_line_thickness(line, binary_img)
            if min_thick <= measured_thickness <= max_thick:
                kept_lines_count += 1
                x1, y1, x2, y2 = [int(p) for p in line]
                cv2.line(final_image_mask, (x1, y1), (x2, y2), 255, int(round(user_thickness)))

    print(f"  -> Kept {kept_lines_count} lines matching the specified thickness.")
    return cv2.bitwise_not(final_image_mask)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/your/image.png")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("🔧 Starting Human-Guided Floor Plan Cleaner:")
    try:
        cleaned_image = clean_image(input_path, CONFIG)
        print("\n✅ Processing complete.")
        original_image_bgr = cv2.imread(str(input_path))
        cleaned_image_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        
        h_orig, w_orig, _ = original_image_bgr.shape
        h_clean, w_clean = cleaned_image.shape
        if h_orig != h_clean or w_orig != w_clean:
            original_image_bgr = cv2.resize(original_image_bgr, (w_clean, h_clean))

        comparison = np.hstack([original_image_bgr, cleaned_image_bgr])
        window_title = "Original (Left) vs. Final Cleaned (Right) -- Press 's' to save, 'q' to quit"
        cv2.imshow(window_title, comparison)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            output_dir = Path("./cleaned_images")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_cleaned_final.png"
            cv2.imwrite(str(output_path), cleaned_image)
            print(f"💾 Image saved to: {output_path}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
