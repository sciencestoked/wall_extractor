#!/usr/bin/env python3
"""
Floor Plan Cleaning Tool (v12 - Robust Sandwich Method)
Identifies walls by finding parallel lines and validating that the
entire area between them is sufficiently solid.
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path
import math

CONFIG_FILE = "config.json"

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("🔧 Configuration loaded successfully from config.json.")
    return config

def display_image(window_name: str, image: np.ndarray):
    print(f"  -> Displaying '{window_name}'. Press any key to continue...")
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def get_line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def find_wall_segments(lines: list, binary_img: np.ndarray, config: dict) -> list:
    """Finds wall segments by pairing parallel lines and validating the area between them."""
    wall_segments = []
    paired_indices = set()
    wall_cfg = config['wall_finding']
    
    for i in range(len(lines)):
        if i in paired_indices: continue
        line1 = lines[i][0]
        angle1 = get_line_angle(line1)
        
        for j in range(i + 1, len(lines)):
            if j in paired_indices: continue
            line2 = lines[j][0]
            angle2 = get_line_angle(line2)

            angle_diff = abs(angle1 - angle2)
            if angle_diff < wall_cfg['parallel_angle_tolerance'] or abs(angle_diff - 180) < wall_cfg['parallel_angle_tolerance']:
                mid1 = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
                mid2 = ((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2)
                dist = math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)

                if wall_cfg['min_wall_thickness'] < dist < wall_cfg['max_wall_thickness']:
                    # --- NEW ROBUST VALIDATION LOGIC ---
                    # 1. Define the quadrilateral area between the two lines.
                    quad = np.array([
                        [line1[0], line1[1]], [line1[2], line1[3]],
                        [line2[2], line2[3]], [line2[0], line2[1]]
                    ], dtype=np.int32)
                    
                    # 2. Create a mask for this specific area.
                    mask = np.zeros_like(binary_img)
                    cv2.fillConvexPoly(mask, quad, 255)
                    
                    # 3. Calculate the solidity (percentage of white pixels).
                    area_of_interest = cv2.bitwise_and(binary_img, binary_img, mask=mask)
                    white_pixels = cv2.countNonZero(area_of_interest)
                    total_pixels = cv2.countNonZero(mask)
                    solidity = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    
                    if solidity >= wall_cfg['min_solidity_percentage']:
                        # This is a valid wall segment.
                        wall_segments.append(quad)
                        paired_indices.add(i)
                        paired_indices.add(j)
                        break # Move to next unpaired line
    return wall_segments

def clean_image(image_path: Path, config: dict):
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    print("\n[Step 1/4] Binarizing image...")
    binary_img = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        config["binarization"]["block_size"], config["binarization"]["c_value"]
    )
    if config['debugging']['show_steps']: display_image("Step 1 - Binarized Image", binary_img)

    print("\n[Step 2/4] Detecting all line segments...")
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(binary_img)
    if lines is None: lines = []
    
    min_len = config["line_detection"]["min_line_length"]
    long_lines = [l for l in lines if np.sqrt((l[0][2] - l[0][0])**2 + (l[0][3] - l[0][1])**2) > min_len]
    print(f"-> Found {len(long_lines)} lines longer than {min_len}px.")
    
    if config['debugging']['show_steps']:
        all_lines_vis = img_bgr.copy()
        for line_arr in long_lines:
            x1, y1, x2, y2 = [int(p) for p in line_arr[0]]
            cv2.line(all_lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        display_image("Step 2 - Detected Long Lines", all_lines_vis)
        
    print("\n[Step 3/4] Finding Wall Segments (Robust Sandwich Method)...")
    wall_segments = find_wall_segments(long_lines, binary_img, config)
    print(f"-> Identified {len(wall_segments)} wall segments.")
    
    if config['debugging']['show_steps']:
        wall_segments_vis = img_bgr.copy()
        cv2.drawContours(wall_segments_vis, wall_segments, -1, (0, 255, 0), 2)
        display_image("Step 3 - Identified Wall Segments", wall_segments_vis)
        
    print("\n[Step 4/4] Reconstructing final clean image...")
    final_mask = np.zeros_like(binary_img)
    cv2.fillPoly(final_mask, wall_segments, 255)
    
    final_image = cv2.bitwise_not(final_mask)
    if config['debugging']['show_steps']: display_image("Step 4 - Final Reconstructed Image", final_image)
    
    return final_image

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/image.png"); sys.exit(1)

    input_path = Path(sys.argv[1])
    config_path = Path(CONFIG_FILE)
    
    try:
        config = load_config(config_path)
        cleaned_image = clean_image(input_path, config)
        
        original_image_bgr = cv2.imread(str(input_path))
        cleaned_image_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        h_orig, w_orig, _ = original_image_bgr.shape
        h_clean, w_clean = cleaned_image.shape
        if h_orig != h_clean:
            original_image_bgr = cv2.resize(original_image_bgr, (int(w_orig * h_clean/h_orig), h_clean))
        comparison = np.hstack([original_image_bgr, cleaned_image_bgr])
        
        window_title = "Original (Left) vs. Final Cleaned (Right) -- Press 's' to save, 'q' to quit"
        cv2.imshow(window_title, comparison)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            output_dir = Path("./cleaned_images")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_cleaned_v12.png"
            cv2.imwrite(str(output_path), cleaned_image)
            print(f"💾 Image saved to: {output_path}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
