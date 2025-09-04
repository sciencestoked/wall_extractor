#!/usr/bin/env python3
"""
Floor Plan Cleaning Tool (v9 - Robust Line Analysis)
Fixes a crash when no histogram peaks are found and improves fallback logic.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# --- Configuration ---
CONFIG = {
    "thickness_histogram_prominence": 10,
    "min_line_length": 15, # in pixels
}
# --- End Configuration ---

def get_line_thickness(line: np.ndarray, img: np.ndarray, num_samples: int = 5) -> float:
    """
    Measures the thickness of a line segment by sampling perpendicular transects.
    """
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0: return 0

    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    perp_dx, perp_dy = -dy, dx

    thicknesses = []
    for i in range(1, num_samples + 1):
        sample_x = x1 + dx * (length * i / (num_samples + 1))
        sample_y = y1 + dy * (length * i / (num_samples + 1))
        c = -1
        while True:
            c += 1
            px, py = int(round(sample_x + perp_dx * c)), int(round(sample_y + perp_dy * c))
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]) or img[py, px] == 0:
                break
        c2 = -1
        while True:
            c2 += 1
            px, py = int(round(sample_x - perp_dx * c2)), int(round(sample_y - perp_dy * c2))
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]) or img[py, px] == 0:
                break
        thicknesses.append(c + c2)
    
    return np.median(thicknesses) if thicknesses else 0

def clean_image(image_path: Path, config: dict) -> np.ndarray:
    """
    Applies a line segment analysis and filtering pipeline.
    """
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Stage 1: Binarization ---
    binary_img = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # --- Stage 2: Line Segment Detection ---
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(binary_img)
    if lines is None:
        print("   -> No lines detected in the image.")
        return cv2.bitwise_not(np.zeros_like(binary_img))

    # --- Stage 3: Line Property Analysis ---
    line_data = []
    print(f"   -> Analyzing {len(lines)} detected line segments...")
    for line_arr in lines:
        line = line_arr[0]
        length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        if length > config["min_line_length"]:
            thickness = get_line_thickness(line, binary_img)
            if thickness > 0:
                line_data.append({'line': line, 'thickness': thickness, 'length': length})

    if not line_data:
        print("   -> No significant lines found after initial filtering.")
        return cv2.bitwise_not(np.zeros_like(binary_img))

    # --- Stage 4: Determine Thickness Threshold (FIXED LOGIC) ---
    thicknesses = [ld['thickness'] for ld in line_data]
    
    # Establish a safe fallback value first
    median_thickness = np.median(thicknesses)
    wall_thickness_peak = median_thickness # Default to median
    
    try:
        from scipy.signal import find_peaks
        hist, bins = np.histogram(thicknesses, bins=int(max(thicknesses)))
        peaks, _ = find_peaks(hist, prominence=config["thickness_histogram_prominence"])
        
        if len(peaks) > 1:
            # If multiple peaks, assume the last major peak represents the walls
            wall_thickness_peak = bins[peaks[-1]]
        elif len(peaks) == 1:
            # If one peak, use it as the reference
            wall_thickness_peak = bins[peaks[0]]
        else:
            print("   -> No prominent thickness peaks found. Relying on median thickness.")

    except ImportError:
        # Fallback if SciPy is not installed
        print("   -> SciPy not found. Relying on median thickness.")

    # Set the threshold slightly below the determined wall peak (or median)
    thickness_threshold = wall_thickness_peak * 0.8
    print(f"   -> Wall thickness identified around {wall_thickness_peak:.1f}px. Setting filter threshold to {thickness_threshold:.1f}px.")

    # --- Stage 5: Filter Lines and Reconstruct Image ---
    final_image_mask = np.zeros_like(binary_img)
    for ld in line_data:
        if ld['thickness'] >= thickness_threshold:
            x1, y1, x2, y2 = [int(p) for p in ld['line']]
            cv2.line(final_image_mask, (x1, y1), (x2, y2), 255, int(round(ld['thickness'])))

    return cv2.bitwise_not(final_image_mask)

def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_floorplan_v9.py path/to/your/image.png")
        sys.exit(1)
        
    try:
        from scipy.signal import find_peaks
    except ImportError:
        print("\nWarning: SciPy is not installed. Peak detection will be skipped.")
        print("For better results, please install it by running: pip install scipy\n")

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("🔧 Cleaning image with Line Segment Analysis:")
    for key, value in CONFIG.items():
        print(f"   - {key}: {value}")
    
    try:
        cleaned_image = clean_image(input_path, CONFIG)
        original_image_bgr = cv2.imread(str(input_path))
        cleaned_image_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        
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
            output_path = output_dir / f"{input_path.stem}_cleaned_v9.png"
            cv2.imwrite(str(output_path), cleaned_image)
            print(f"💾 Image saved to: {output_path}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
