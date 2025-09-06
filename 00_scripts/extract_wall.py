import cv2
import numpy as np
import argparse
import os


def extract_walls(image_path, output_path=None):
    # Load image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Could not open {image_path}")

    # Thresholding to extract darker lines (walls)
    _, thresh = cv2.threshold(original_image, 180, 255, cv2.THRESH_BINARY_INV)

    # Morphology to remove small text/noise
    kernel = np.ones((3, 3), np.uint8)
    walls_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Invert back → black walls on white background
    walls_only = cv2.bitwise_not(walls_only)

    # Save result
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_walls.png"

    cv2.imwrite(output_path, walls_only)
    print(f"✅ Walls extracted and saved to: {output_path}")

    # Show output window
    cv2.imshow("Walls Only", walls_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract walls from CAD drawing image")
    parser.add_argument("image_path", help="Path to input image (PNG/JPG)")
    parser.add_argument("-o", "--output", help="Optional path for output image")

    args = parser.parse_args()
    extract_walls(args.image_path, args.output)
