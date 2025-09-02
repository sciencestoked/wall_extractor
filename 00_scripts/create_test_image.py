#!/usr/bin/env python3
"""
Creates a simple test image with circles and squares for testing the Snapper tool.
"""

import cv2
import numpy as np

def create_test_image():
    """Create a test image with various shapes."""
    # Create white canvas
    img = np.ones((600, 800), dtype=np.uint8) * 255
    
    # Draw some circles (black on white)
    cv2.circle(img, (150, 150), 25, 0, -1)  # Filled circle
    cv2.circle(img, (300, 200), 30, 0, 3)   # Circle outline
    cv2.circle(img, (500, 150), 20, 0, -1)  # Small circle
    
    # Draw some squares/rectangles
    cv2.rectangle(img, (100, 300), (150, 350), 0, -1)  # Filled square
    cv2.rectangle(img, (250, 320), (320, 380), 0, 3)   # Square outline
    cv2.rectangle(img, (450, 300), (550, 340), 0, -1)  # Rectangle
    
    # Draw some other shapes
    # Triangle
    triangle = np.array([[400, 450], [350, 520], [450, 520]], np.int32)
    cv2.fillPoly(img, [triangle], 0)
    
    # Polygon (hexagon)
    center = (200, 480)
    radius = 40
    points = []
    for i in range(6):
        angle = i * np.pi / 3
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append([x, y])
    polygon = np.array(points, np.int32)
    cv2.fillPoly(img, [polygon], 0)
    
    # Add some noise/lines to simulate real floor plan
    cv2.line(img, (0, 100), (800, 100), 0, 2)
    cv2.line(img, (200, 0), (200, 600), 0, 1)
    cv2.line(img, (600, 0), (600, 600), 0, 1)
    
    # Save test image
    cv2.imwrite('test_shapes.png', img)
    print("Created test_shapes.png with various shapes for testing")

if __name__ == "__main__":
    create_test_image()