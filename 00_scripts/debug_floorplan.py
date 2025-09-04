#!/usr/bin/env python3
"""
Floor Plan Debugging Tool (for v9 - Line Segment Analysis)
Visualizes each step of the cleaning process and prints intermediate data
to help with understanding and parameter tuning.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# --- Configuration ---
# These are the parameters you can tune after observing the debug output
CONFIG = {
    "thickness_histogram_prominence": 10,
    "min_line_length": 15, # in pixels
}
# --- End Configuration ---

def display_image(window_name: str, image: np.ndarray):
    """Helper function to display an image and wait for a key press."""
    print(f"  -> Displaying '{window_name}'. Press any key to continue...")
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

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

def clean_and_debug_image(image_path: Path, config: dict):
    """
    Applies a line segment analysis pipeline with debugging outputs at each step.
    """
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if img_gray is None: raise FileNotFoundError(f"Could not read image: {image_path}")
    print(f"--- Starting Debug Analysis for: {image_path.name} ---")

    # --- Step 1: Binarization ---
    print("\n[Step 1/5] Binarizing image with Adaptive Threshold...")
    binary_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    display_image("Step 1 - Binarized Image", binary_img)

    # --- Step 2: Line Segment Detection ---
    print("\n[Step 2/5] Detecting all line segments...")
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(binary_img)
    if lines is None: print("-> No lines detected. Exiting."); return
    
    # Visualize all detected lines
    all_lines_vis = img_bgr.copy()
    for line_arr in lines:
        x1, y1, x2, y2 = [int(p) for p in line_arr[0]]
        cv2.line(all_lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    print(f"-> Found {len(lines)} total line segments.")
    display_image("Step 2 - All Detected Lines (in Red)", all_lines_vis)

    # --- Step 3: Line Property Analysis ---
    print("\n[Step 3/5] Measuring length and thickness of significant lines...")
    line_data = []
    for line_arr in lines:
        line = line_arr[0]
        length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        if length > config["min_line_length"]:
            thickness = get_line_thickness(line, binary_img)
            if thickness > 0:
                line_data.append({'line': line, 'thickness': thickness, 'length': length})
    
    print(f"-> Kept {len(line_data)} lines longer than {config['min_line_length']}px.")
    print("-> Sample of measured line data:")
    for i, ld in enumerate(line_data[:5]): # Print first 5 samples
        print(f"  - Line {i+1}: Length={ld['length']:.1f}px, Thickness={ld['thickness']:.1f}px")

    # --- Step 4: Determine Thickness Threshold ---
    print("\n[Step 4/5] Determining wall thickness threshold...")
    thicknesses = [ld['thickness'] for ld in line_data]
    median_thickness, wall_thickness_peak = np.median(thicknesses), np.median(thicknesses)
    
    try:
        from scipy.signal import find_peaks
        hist, bins = np.histogram(thicknesses, bins=int(max(thicknesses)))
        peaks, _ = find_peaks(hist, prominence=config["thickness_histogram_prominence"])
        if len(peaks) > 0: wall_thickness_peak = bins[peaks[-1]]
        else: print("  -> No prominent peaks found. Using median thickness as fallback.")
    except ImportError: print("  -> SciPy not found. Using median thickness.")

    thickness_threshold = wall_thickness_peak * 0.8
    print(f"-> Wall thickness identified around {wall_thickness_peak:.1f}px.")
    print(f"-> Filter threshold set to {thickness_threshold:.1f}px.")

    # Visualize the lines that passed the filter
    wall_lines_vis = img_bgr.copy()
    kept_lines_count = 0
    for ld in line_data:
        if ld['thickness'] >= thickness_threshold:
            x1, y1, x2, y2 = [int(p) for p in ld['line']]
            cv2.line(wall_lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            kept_lines_count += 1
    print(f"-> {kept_lines_count} lines were classified as walls.")
    display_image("Step 4 - Filtered Wall Lines (in Green)", wall_lines_vis)
    
    # --- Step 5: Reconstruct Final Image ---
    print("\n[Step 5/5] Reconstructing final clean image...")
    final_image_mask = np.zeros_like(binary_img)
    for ld in line_data:
        if ld['thickness'] >= thickness_threshold:
            x1, y1, x2, y2 = [int(p) for p in ld['line']]
            cv2.line(final_image_mask, (x1, y1), (x2, y2), 255, int(round(ld['thickness'])))
    
    final_image = cv2.bitwise_not(final_image_mask)
    print("-> Final image created.")
    display_image("Step 5 - Final Reconstructed Image", final_image)
    print("\n--- Debug Session Complete ---")

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_floorplan.py path/to/your/image.png")
        sys.exit(1)
    
    try: from scipy.signal import find_peaks
    except ImportError:
        print("Warning: SciPy not found. Peak detection will be skipped (pip install scipy)")

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    clean_and_debug_image(input_path, CONFIG)

if __name__ == "__main__":
    main()
