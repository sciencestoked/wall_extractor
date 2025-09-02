#!/usr/bin/env python3
"""
Floor Plan Image Analyzer (EDA Tool)
Performs a detailed statistical and visual analysis of shapes within a
floor plan image to inform a data-driven cleaning strategy.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_image(image_path: Path, output_dir: Path):
    """
    Performs a comprehensive analysis of the input image.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"--- Analyzing Image: {image_path.name} ({img.shape[1]}x{img.shape[0]}) ---")

    # --- 1. Grayscale Intensity Analysis ---
    print("\n[1/4] Analyzing grayscale intensity histogram...")
    plt.figure(figsize=(10, 5))
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Grayscale Intensity Histogram')
    plt.xlabel('Intensity (0=Black, 255=White)')
    plt.ylabel('Pixel Count')
    hist_path = output_dir / f"{image_path.stem}_histogram.png"
    plt.savefig(hist_path)
    print(f"  -> Saved histogram to {hist_path}")
    plt.close()

    # --- 2. Contour Detection ---
    print("\n[2/4] Detecting all contours...")
    # Binarize the image to find contours
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  -> Found {len(contours)} initial contours.")

    # --- 3. Statistical Analysis of Contours ---
    print("\n[3/4] Calculating statistics for each contour...")
    stats = {
        'area': [], 'perimeter': [], 'aspect_ratio': [], 'extent': [], 'solidity': []
    }

    if not contours:
        print("  -> No contours found to analyze.")
        return

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:  # Ignore tiny noise
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        rect_area = w * h
        extent = area / float(rect_area) if rect_area > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0

        stats['area'].append(area)
        stats['perimeter'].append(perimeter)
        stats['aspect_ratio'].append(aspect_ratio)
        stats['extent'].append(extent)
        stats['solidity'].append(solidity)
        
    # --- Print Statistical Summary ---
    print("\n--- Statistical Summary of Shapes ---")
    print("Property      |    Mean |     Std |     Min |     25% |     50% |     75% |     Max")
    print("--------------|---------|---------|---------|---------|---------|---------|---------")
    for key, values in stats.items():
        if not values: continue
        arr = np.array(values)
        mean, std, p0, p25, p50, p75, p100 = np.mean(arr), np.std(arr), np.min(arr), np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75), np.max(arr)
        print(f"{key:<14}| {mean:>7.2f} | {std:>7.2f} | {p0:>7.2f} | {p25:>7.2f} | {p50:>7.2f} | {p75:>7.2f} | {p100:>7.2f}")
    
    # --- 4. Visual Analysis ---
    print("\n[4/4] Generating visual analysis images...")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Visualize contours colored by AREA
    vis_area = img_bgr.copy()
    q75_area = np.percentile(stats['area'], 75) if stats['area'] else 1000
    q25_area = np.percentile(stats['area'], 25) if stats['area'] else 100
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue
        if area > q75_area: color = (0, 255, 0) # Large (Green)
        elif area < q25_area: color = (0, 0, 255) # Small (Red)
        else: color = (0, 255, 255) # Medium (Yellow)
        cv2.drawContours(vis_area, [cnt], -1, color, 1)
    area_path = output_dir / f"{image_path.stem}_analysis_by_area.png"
    cv2.imwrite(str(area_path), vis_area)
    print(f"  -> Saved area visualization to {area_path}")

    # Visualize contours colored by ASPECT RATIO
    vis_aspect = img_bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 20: continue
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio > 5 or aspect_ratio < 0.2: color = (0, 0, 255) # Long/Tall (Red)
        else: color = (0, 255, 0) # Squarish (Green)
        cv2.drawContours(vis_aspect, [cnt], -1, color, 1)
    aspect_path = output_dir / f"{image_path.stem}_analysis_by_aspect_ratio.png"
    cv2.imwrite(str(aspect_path), vis_aspect)
    print(f"  -> Saved aspect ratio visualization to {aspect_path}")


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_floorplan.py path/to/your/image.png")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
        
    # Create an output directory for the analysis results
    output_dir = Path("./analysis_results") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        analyze_image(input_path, output_dir)
        print(f"\n✅ Analysis complete. Results saved in: {output_dir}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
