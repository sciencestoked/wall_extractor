#!/usr/bin/env python3
"""
Snapper Tool - Interactive Shape Detection and Point Annotation
Part of Wall Extractor Tool Suite
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple
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
    """Handles shape detection within a specified region."""
    
    def __init__(self):
        self.circle_params = {
            'dp': 1,
            'minDist': 20,
            'param1': 50,
            'param2': 15,
            'minRadius': 3,
            'maxRadius': 50
        }
        
    def detect_shapes(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> List[DetectedShape]:
        """
        Detect circles and polygons in specified region.
        
        Args:
            image: Grayscale image
            region: (x, y, width, height) of search region
            
        Returns:
            List of detected shapes with their properties
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return []
            
        shapes = []
        
        # Detect circles
        circles = self._detect_circles(roi)
        for circle in circles:
            cx, cy = circle[:2]
            # Convert to full image coordinates
            center = (cx + x, cy + y)
            shapes.append(DetectedShape(
                center=center,
                shape_type="circle",
                contour=np.array([[cx + x, cy + y]]),
                area=np.pi * circle[2]**2,
                bounds=(cx + x - circle[2], cy + y - circle[2], 2*circle[2], 2*circle[2])
            ))
        
        # Detect polygons
        polygons = self._detect_polygons(roi, x, y)
        shapes.extend(polygons)
        
        return shapes
    
    def _detect_circles(self, roi: np.ndarray) -> List[np.ndarray]:
        """Detect circles in ROI using HoughCircles."""
        circles = cv2.HoughCircles(
            roi, 
            cv2.HOUGH_GRADIENT,
            **self.circle_params
        )
        
        if circles is not None:
            return np.round(circles[0, :]).astype("int")
        return []
    
    def _detect_polygons(self, roi: np.ndarray, offset_x: int, offset_y: int) -> List[DetectedShape]:
        """Detect closed polygonal shapes in ROI."""
        # Binary threshold for contour detection
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)  # Invert for dark shapes on light background
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (reasonable size for architectural features)
            if 25 < area < 5000:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    
                    # Adjust contour coordinates to full image
                    adjusted_contour = contour + [offset_x, offset_y]
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Determine shape type based on properties
                    shape_type = self._classify_polygon(contour, area)
                    
                    shapes.append(DetectedShape(
                        center=(cx, cy),
                        shape_type=shape_type,
                        contour=adjusted_contour,
                        area=area,
                        bounds=(x + offset_x, y + offset_y, w, h)
                    ))
        
        return shapes
    
    def _classify_polygon(self, contour: np.ndarray, area: float) -> str:
        """Classify polygon type based on geometric properties."""
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        if vertices == 4:
            # Check if it's roughly square/rectangular
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.8 < aspect_ratio < 1.2:
                return "square"
            else:
                return "rectangle"
        elif vertices == 3:
            return "triangle"
        elif vertices > 4:
            return f"polygon_{vertices}"
        else:
            return "polygon"


class Snapper:
    """Main Snapper tool class."""
    
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = None
        self.display_image = None
        
        # Search window properties
        self.search_window = [100, 100, 100, 100]  # x, y, width, height
        self.search_window_size = 100
        
        # Detection and recording
        self.shape_detector = ShapeDetector()
        self.detected_shapes: List[DetectedShape] = []
        self.recorded_points: List[Dict] = []
        
        # UI state
        self.window_name = f"Snapper - {self.image_path.name}"
        
        # Load image
        if not self._load_image():
            raise ValueError(f"Could not load image: {image_path}")
    
    def _load_image(self) -> bool:
        """Load and prepare image for processing."""
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            return False
            
        # Initialize search window at center
        h, w = self.image.shape
        self.search_window[0] = w // 2 - self.search_window_size // 2
        self.search_window[1] = h // 2 - self.search_window_size // 2
        self.search_window[2] = self.search_window_size
        self.search_window[3] = self.search_window_size
        
        return True
    
    def _update_display(self):
        """Update the display with current search window and detected shapes."""
        # Convert to color for visualization
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # Draw search window (yellow rectangle with crosshair)
        x, y, w, h = self.search_window
        cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Add crosshair in center of search window
        center_x, center_y = x + w//2, y + h//2
        cv2.line(self.display_image, (center_x-10, center_y), (center_x+10, center_y), (0, 255, 255), 1)
        cv2.line(self.display_image, (center_x, center_y-10), (center_x, center_y+10), (0, 255, 255), 1)
        
        # Detect shapes in current search window
        self.detected_shapes = self.shape_detector.detect_shapes(
            self.image, 
            tuple(self.search_window)
        )
        
        # Draw detected shapes
        valid_shapes = 0
        for shape in self.detected_shapes:
            # Check if shape is mostly within search window
            if self._is_shape_valid(shape):
                valid_shapes += 1
                
                # Draw shape outline in red
                if shape.shape_type == "circle":
                    radius = int(np.sqrt(shape.area / np.pi))
                    cv2.circle(self.display_image, shape.center, radius, (0, 0, 255), 2)
                else:
                    cv2.drawContours(self.display_image, [shape.contour], -1, (0, 0, 255), 2)
                
                # Draw centroid in green (larger for better visibility)
                cv2.circle(self.display_image, shape.center, 6, (0, 255, 0), -1)
                cv2.circle(self.display_image, shape.center, 8, (0, 200, 0), 2)
                
                # Add shape label
                label = f"{shape.shape_type} ({int(shape.area)}px²)"
                cv2.putText(self.display_image, label, 
                           (shape.center[0] + 10, shape.center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw recorded points as permanent blue markers
        for i, point_data in enumerate(self.recorded_points):
            pos = point_data['position']
            cv2.circle(self.display_image, pos, 8, (255, 0, 0), -1)
            cv2.circle(self.display_image, pos, 10, (255, 255, 255), 2)
            
            # Add point number
            cv2.putText(self.display_image, str(i+1), 
                       (pos[0] + 12, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _is_shape_valid(self, shape: DetectedShape) -> bool:
        """Check if shape is sufficiently within search window."""
        sx, sy, sw, sh = self.search_window
        bx, by, bw, bh = shape.bounds
        
        # Calculate overlap percentage
        overlap_x = max(0, min(sx + sw, bx + bw) - max(sx, bx))
        overlap_y = max(0, min(sy + sh, by + bh) - max(sy, by))
        overlap_area = overlap_x * overlap_y
        
        shape_area = bw * bh
        if shape_area == 0:
            return False
            
        overlap_ratio = overlap_area / shape_area
        return overlap_ratio > 0.8  # 80% overlap threshold
    
    def _move_search_window(self, dx: int, dy: int):
        """Move search window by specified offset."""
        h, w = self.image.shape
        
        new_x = max(0, min(w - self.search_window[2], self.search_window[0] + dx))
        new_y = max(0, min(h - self.search_window[3], self.search_window[1] + dy))
        
        self.search_window[0] = new_x
        self.search_window[1] = new_y
    
    def _resize_search_window(self, delta: int):
        """Resize search window by specified amount."""
        new_size = max(20, min(300, self.search_window_size + delta))
        self.search_window_size = new_size
        self.search_window[2] = new_size
        self.search_window[3] = new_size
        
        # Adjust position to keep window in bounds
        h, w = self.image.shape
        self.search_window[0] = max(0, min(w - new_size, self.search_window[0]))
        self.search_window[1] = max(0, min(h - new_size, self.search_window[1]))
    
    def _handle_click(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest green centroid within valid shapes
            min_dist = float('inf')
            closest_shape = None
            
            for shape in self.detected_shapes:
                if self._is_shape_valid(shape):
                    dist = np.sqrt((x - shape.center[0])**2 + (y - shape.center[1])**2)
                    if dist < min_dist and dist < 25:  # Within 25 pixels of centroid
                        min_dist = dist
                        closest_shape = shape
            
            if closest_shape:
                # Check if this point was already recorded (avoid duplicates)
                for existing in self.recorded_points:
                    if np.linalg.norm(np.array(existing['position']) - np.array(closest_shape.center)) < 5:
                        print(f"⚠️  Point already recorded at {closest_shape.center}")
                        return
                
                # Record the point
                point_data = {
                    'position': closest_shape.center,
                    'shape_type': closest_shape.shape_type,
                    'area': closest_shape.area,
                    'search_window_center': (self.search_window[0] + self.search_window[2]//2, 
                                           self.search_window[1] + self.search_window[3]//2),
                    'timestamp': len(self.recorded_points),
                    'detection_confidence': min_dist  # Distance from click to centroid
                }
                self.recorded_points.append(point_data)
                print(f"✅ Recorded #{len(self.recorded_points)}: {closest_shape.center} ({closest_shape.shape_type}, {int(closest_shape.area)}px²)")
            else:
                print(f"❌ No valid shape centroid near click at ({x}, {y})")
    
    def run(self):
        """Main execution loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_click)
        
        step_size = 10  # Pixels per arrow key press
        
        print(f"\n🎯 Snapper Tool - {self.image_path.name}")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")
        print("Controls:")
        print("  Arrow Keys (↑↓←→): Move search window")
        print("  +/-: Resize search window")
        print("  Mouse Click: Snap to green centroid")
        print("  'r': Reset recorded points")
        print("  's': Save points to file")
        print("  'q': Quit")
        print(f"Search window: {self.search_window_size}x{self.search_window_size} pixels")
        print()
        
        while True:
            self._update_display()
            cv2.imshow(self.window_name, self.display_image)
            
            # Show current status
            shapes_count = len([s for s in self.detected_shapes if self._is_shape_valid(s)])
            window_title = f"{self.window_name} | Window: {self.search_window_size}x{self.search_window_size} | Shapes: {shapes_count} | Points: {len(self.recorded_points)}"
            cv2.setWindowTitle(self.window_name, window_title)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Arrow keys for movement (multiple key codes for compatibility)
            if key in [82, 65362, ord('w')]:  # Up arrow / W
                self._move_search_window(0, -step_size)
            elif key in [84, 65364, ord('s')]:  # Down arrow / S  
                self._move_search_window(0, step_size)
            elif key in [81, 65361, ord('a')]:  # Left arrow / A
                self._move_search_window(-step_size, 0)
            elif key in [83, 65363, ord('d')]:  # Right arrow / D
                self._move_search_window(step_size, 0)
            
            # Resize window
            elif key == ord('+') or key == ord('='):
                self._resize_search_window(10)
                print(f"Search window size: {self.search_window_size}x{self.search_window_size}")
            elif key == ord('-'):
                self._resize_search_window(-10)
                print(f"Search window size: {self.search_window_size}x{self.search_window_size}")
            
            # Reset points
            elif key == ord('r'):
                self.recorded_points.clear()
                print("🔄 Reset recorded points")
            
            # Save points
            elif key == ord('s'):
                self._save_points()
            
            # Quit
            elif key == ord('q'):
                print("👋 Exiting Snapper")
                break
        
        cv2.destroyAllWindows()
    
    def _save_points(self):
        """Save recorded points to JSON file."""
        output_file = self.image_path.with_suffix('.json')
        
        data = {
            'image_path': str(self.image_path),
            'image_size': self.image.shape[::-1],  # (width, height)
            'points': self.recorded_points,
            'total_points': len(self.recorded_points)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(self.recorded_points)} points to {output_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python snapper.py path/to/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        snapper = Snapper(image_path)
        snapper.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()