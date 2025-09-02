#!/usr/bin/env python3
"""
Snapper Tool - Interactive Shape Detection and Point Annotation
Part of Wall Extractor Tool Suite
VERSION 2.1: Fixed JSON serialization bug and added startup instructions.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class DetectedShape:
    """Container for detected shape information."""
    center: Tuple[int, int]
    shape_type: str
    contour: np.ndarray
    area: float
    bounds: Tuple[int, int, int, int]  # x, y, width, height

class ShapeDetector:
    """Handles shape detection within a specified region by analyzing full-image contours."""

    def __init__(self):
        self.circle_params = {
            'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 15,
            'minRadius': 3, 'maxRadius': 50
        }
        self.full_image_contours = []
        self.image_binary = None

    def preprocess_image(self, image: np.ndarray):
        """
        Pre-processes the entire image to find all potential contours once.
        This is a major optimization.
        """
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image_binary = cv2.bitwise_not(binary)
        self.full_image_contours, _ = cv2.findContours(
            self.image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    def detect_shapes(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> List[DetectedShape]:
        """
        Detects shapes that intersect with the specified region by analyzing
        the pre-computed full-image contours.
        """
        rx, ry, rw, rh = region
        roi = image[ry:ry+rh, rx:rx+rw]
        
        if roi.size == 0:
            return []

        shapes = []

        # --- 1. Detect Circles (still best done on the ROI) ---
        circles = self._detect_circles(roi)
        for circle in circles:
            cx, cy, radius = circle
            center = (cx + rx, cy + ry)
            shapes.append(DetectedShape(
                center=center,
                shape_type="circle",
                contour=np.array([[center]]), # Placeholder
                area=np.pi * radius**2,
                bounds=(center[0] - radius, center[1] - radius, 2*radius, 2*radius)
            ))

        # --- 2. Detect Polygons by checking for intersection with the ROI ---
        for contour in self.full_image_contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            if (rx < bx + bw and rx + rw > bx and ry < by + bh and ry + rh > by):
                area = cv2.contourArea(contour)
                if 25 < area < 5000:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        shape_type = self._classify_polygon(contour)
                        shapes.append(DetectedShape(
                            center=(cx, cy), shape_type=shape_type, contour=contour,
                            area=area, bounds=(bx, by, bw, bh)
                        ))
        return shapes

    def _detect_circles(self, roi: np.ndarray) -> List[np.ndarray]:
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, **self.circle_params)
        return np.round(circles[0, :]).astype("int") if circles is not None else []

    def _classify_polygon(self, contour: np.ndarray) -> str:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        if vertices == 4:
            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            return "square" if 0.8 < aspect_ratio < 1.2 else "rectangle"
        elif vertices == 3: return "triangle"
        elif vertices > 4: return f"polygon_{vertices}"
        else: return "polygon"

class Snapper:
    """Main Snapper tool class."""

    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = None
        self.display_image = None
        self.search_window = [100, 100, 200, 200]
        self.search_window_size = 200
        self.mouse_pos = [300, 300]
        self.shape_detector = ShapeDetector()
        self.detected_shapes = []
        self.recorded_points = []
        self.window_name = f"Snapper - {self.image_path.name}"
        self._load_image()

    def _load_image(self) -> bool:
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None: return False
        self.shape_detector.preprocess_image(self.image)
        h, w = self.image.shape
        self.search_window[0] = w // 2 - self.search_window_size // 2
        self.search_window[1] = h // 2 - self.search_window_size // 2
        self.search_window[2] = self.search_window_size
        self.search_window[3] = self.search_window_size
        return True

    def _update_display(self):
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        x, y, w, h = self.search_window
        cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center_x, center_y = x + w//2, y + h//2
        cv2.line(self.display_image, (center_x-10, center_y), (center_x+10, center_y), (0, 255, 255), 1)
        cv2.line(self.display_image, (center_x, center_y-10), (center_x, center_y+10), (0, 255, 255), 1)
        self.detected_shapes = self.shape_detector.detect_shapes(self.image, tuple(self.search_window))
        for shape in self.detected_shapes:
            if shape.shape_type == "circle":
                radius = int(np.sqrt(shape.area / np.pi))
                cv2.circle(self.display_image, shape.center, radius, (0, 0, 255), 2)
            else:
                cv2.drawContours(self.display_image, [shape.contour], -1, (0, 0, 255), 2)
            cv2.circle(self.display_image, shape.center, 6, (0, 255, 0), -1)
            cv2.circle(self.display_image, shape.center, 8, (0, 200, 0), 2)
            label = f"{shape.shape_type} ({int(shape.area)}px²)"
            cv2.putText(self.display_image, label,
                        (shape.center[0] + 10, shape.center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for i, point_data in enumerate(self.recorded_points):
            pos = point_data['position']
            cv2.circle(self.display_image, pos, 8, (255, 0, 0), -1)
            cv2.circle(self.display_image, pos, 10, (255, 255, 255), 2)
            cv2.putText(self.display_image, str(i+1),
                        (pos[0] + 12, pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _resize_search_window(self, delta: int):
        new_size = max(20, min(300, self.search_window_size + delta))
        self.search_window_size = new_size
        self.search_window[2] = new_size
        self.search_window[3] = new_size
        h, w = self.image.shape
        self.search_window[0] = max(0, min(w - new_size, self.search_window[0]))
        self.search_window[1] = max(0, min(h - new_size, self.search_window[1]))

    def _handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            min_dist, closest_shape = float('inf'), None
            for shape in self.detected_shapes:
                dist = np.sqrt((x - shape.center[0])**2 + (y - shape.center[1])**2)
                if dist < min_dist:
                    min_dist, closest_shape = dist, shape

            if closest_shape and min_dist < 50:
                is_duplicate = any(np.linalg.norm(np.array(p['position']) - np.array(closest_shape.center)) < 5 for p in self.recorded_points)
                if is_duplicate:
                    print(f"⚠️ Point already recorded at {closest_shape.center}")
                    return
                
                # --- FIX IS HERE: Convert all NumPy types to standard Python types ---
                point_data = {
                    'position': (int(closest_shape.center[0]), int(closest_shape.center[1])),
                    'shape_type': str(closest_shape.shape_type),
                    'area': float(closest_shape.area),
                    'click_position': (x, y),
                    'snap_distance': float(min_dist),
                    'timestamp': len(self.recorded_points)
                }
                self.recorded_points.append(point_data)
                print(f"✅ Snapped #{len(self.recorded_points)}: {closest_shape.center} ({closest_shape.shape_type}, {int(closest_shape.area)}px²)")
            else:
                print(f"❌ No valid shape centroid found near click at ({x}, {y})")

    def run(self):
        """Main execution loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_event)

        # --- FIX IS HERE: Added print instructions ---
        print(f"\n🎯 Snapper Tool - {self.image_path.name}")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")
        print("Controls:")
        print("  Mouse Hover: Move search window")
        print("  +/-: Resize search window")
        print("  Mouse Click: Snap to nearest green centroid")
        print("  'r': Reset recorded points")
        print("  's': Save points to file")
        print("  'q': Quit")
        print(f"Search window: {self.search_window_size}x{self.search_window_size} pixels\n")

        while True:
            self.search_window[0] = max(0, min(self.mouse_pos[0] - self.search_window_size // 2, self.image.shape[1] - self.search_window_size))
            self.search_window[1] = max(0, min(self.mouse_pos[1] - self.search_window_size // 2, self.image.shape[0] - self.search_window_size))
            self._update_display()
            cv2.imshow(self.window_name, self.display_image)

            shapes_count = len(self.detected_shapes)
            window_title = f"{self.window_name} | Win: {self.search_window_size}x{self.search_window_size} | Shapes: {shapes_count} | Pts: {len(self.recorded_points)}"
            cv2.setWindowTitle(self.window_name, window_title)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('+'), ord('=')): self._resize_search_window(10)
            elif key == ord('-'): self._resize_search_window(-10)
            elif key == ord('r'): self.recorded_points.clear(); print("🔄 Reset points")
            elif key == ord('s'): self._save_points()
            elif key == ord('q'): print("👋 Exiting"); break
        cv2.destroyAllWindows()

    def _save_points(self):
        """Save recorded points to JSON file."""
        output_file = self.image_path.with_suffix('.json')
        data = {
            'image_path': str(self.image_path),
            'image_size': (int(self.image.shape[1]), int(self.image.shape[0])),
            'points': self.recorded_points,
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.recorded_points)} points to {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python snapper.py path/to/image.jpg"); sys.exit(1)
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}"); sys.exit(1)
    
    snapper = Snapper(image_path)
    snapper.run()

if __name__ == "__main__":
    main()
