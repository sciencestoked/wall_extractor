#!/usr/bin/env python3
"""
Interactive Floor Plan Cleaner (v17 - Typo Fix)
Features a clean image-only display with all controls and parameter
states managed through the terminal for an uncluttered experience.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import os

# Attempt to import pdf2image
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Warning: pdf2image not found. PDF support will be disabled.")
    print("Install it with: pip install pdf2image")
    convert_from_path = None

class InteractiveCleaner:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.original_image = self._load_input()
        
        self.params = {
            "binarization_block_size": {"val": 21, "min": 3, "max": 51, "step": 2},
            "binarization_c_value": {"val": 10, "min": -10, "max": 30, "step": 1},
            "opening_kernel_size": {"val": 3, "min": 0, "max": 20, "step": 1},
            "closing_kernel_size": {"val": 3, "min": 0, "max": 20, "step": 1}
        }
        
        self.param_keys = list(self.params.keys())
        self.active_param_idx = 0
        self.window_name = "Original (Left) vs. Cleaned (Right)"
        self.needs_update = True

    def _load_input(self) -> np.ndarray:
        if self.input_path.suffix.lower() == '.pdf':
            if convert_from_path is None:
                raise ImportError("PDF processing requires pdf2image and poppler.")
            images = convert_from_path(str(self.input_path), first_page=1, last_page=1, dpi=300)
            if not images: raise ValueError("Could not extract any pages from the PDF.")
            
            # --- FIX IS HERE: Corrected the typo from COLOR_RGB_BGR to COLOR_RGB2BGR ---
            img_bgr = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(str(self.input_path), cv2.IMREAD_GRAYSCALE)
            if img is None: raise FileNotFoundError(f"Could not read image: {self.input_path}")
            return img

    def clean_image(self) -> np.ndarray:
        binary_img = cv2.adaptiveThreshold(
            self.original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.params["binarization_block_size"]["val"], self.params["binarization_c_value"]["val"]
        )
        k_open = self.params["opening_kernel_size"]["val"]
        if k_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        k_close = self.params["closing_kernel_size"]["val"]
        if k_close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_not(binary_img)

    def print_ui(self):
        """Prints the current parameters and controls to the terminal."""
        os.system('cls' if os.name == 'nt' else 'clear') # Clear the terminal
        print("--- 🛠️ Interactive Floor Plan Cleaner ---")
        print("\nCONTROLS (on image window):")
        print("  [W] / [S] : Cycle through parameters")
        print("  [+] / [-] : Adjust parameter value")
        print("  [SHIFT+S] : Save cleaned image")
        print("  [Q]       : Quit")
        print("\nPARAMETERS:")
        for i, (key, data) in enumerate(self.params.items()):
            prefix = ">>" if i == self.active_param_idx else "  "
            print(f"  {prefix} {key:<25}: {data['val']}")
        print("\n-------------------------------------------")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            if self.needs_update:
                self.print_ui()
                cleaned_image = self.clean_image()
                
                original_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
                cleaned_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
                
                h_orig, w_orig, _ = original_bgr.shape
                h_clean, w_clean, _ = cleaned_bgr.shape
                if h_orig != h_clean:
                    original_bgr = cv2.resize(original_bgr, (int(w_orig * h_clean/h_orig), h_clean))

                comparison = np.hstack([original_bgr, cleaned_bgr])
                cv2.imshow(self.window_name, comparison)
                self.needs_update = False

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): break
            elif key == ord('w'):
                self.active_param_idx = (self.active_param_idx - 1) % len(self.param_keys)
                self.needs_update = True
            elif key == ord('s'):
                self.active_param_idx = (self.active_param_idx + 1) % len(self.param_keys)
                self.needs_update = True
            elif key == ord('S'):
                output_dir = Path("./cleaned_images")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{self.input_path.stem}_cleaned_interactive.png"
                cv2.imwrite(str(output_path), cleaned_image)
                print(f"\n💾 Image saved to: {output_path}")

            active_param_name = self.param_keys[self.active_param_idx]
            param_to_adjust = self.params[active_param_name]
            if key == ord('+') or key == ord('='):
                param_to_adjust['val'] = min(param_to_adjust['max'], param_to_adjust['val'] + param_to_adjust['step'])
                self.needs_update = True
            elif key == ord('-'):
                param_to_adjust['val'] = max(param_to_adjust['min'], param_to_adjust['val'] - param_to_adjust['step'])
                self.needs_update = True
        
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/image_or.pdf"); sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}"); sys.exit(1)
        
    try:
        cleaner = InteractiveCleaner(input_path)
        cleaner.run()
    except Exception as e:
        print(f"\nAn error occurred: {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
