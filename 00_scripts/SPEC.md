# Wall Extractor Tool Suite - Specifications

## Project Vision
Modular computer vision tools for architectural floor plan annotation and analysis.

## Tool 1: Snapper

### Purpose
Interactive shape detection and point annotation tool for precise coordinate capture on floor plan images.

### Core Functionality

#### Image Display
- Open any image file (PNG, JPG, JPEG, TIFF, BMP)
- Display with zoom/pan capabilities
- Resize window appropriately for screen

#### Search Window Control
- **Manual Control**: Arrow keys move search window
- **Visual Feedback**: Yellow rectangular overlay showing active search area
- **Size**: Configurable (default 100x100 pixels)
- **Position**: Keyboard controlled, mouse position independent

#### Shape Detection
- **Target Shapes**: Circles and closed polygons (squares, rectangles, etc.)
- **Detection Area**: Only within current search window
- **Visual Output**:
  - Detected shapes outlined in **red**
  - Shape centroids marked with **green dots**
- **Real-time**: Updates as search window moves

#### Click-to-Snap Functionality
- **Target**: Click on green centroid dots
- **Action**: Record exact centroid coordinates
- **Validation**: Shape must be completely (or mostly) within search window
- **Feedback**: Visual confirmation + coordinate logging

#### Boundary Validation Logic
- **Complete Containment**: Preferred - entire shape within search window
- **Partial Tolerance**: Allow if >80% of shape perimeter is within window
- **Decision**: Prioritize precision over strictness

### Controls
- **Arrow Keys**: Move search window (↑↓←→)
- **+/-**: Resize search window
- **Mouse Click**: Snap to nearest green centroid
- **'r'**: Reset recorded points
- **'s'**: Save points to file
- **'q'**: Quit

### Output
- **Console**: Logged coordinates with shape type
- **File**: JSON format with image path and point list
- **Visual**: Permanent markers on recorded points

### Technical Requirements
- **Dependencies**: OpenCV, NumPy, JSON
- **Performance**: Real-time shape detection (<100ms updates)
- **Memory**: Efficient for large architectural images
- **Modularity**: Clean class design for integration into larger tools

## Future Tools (Planned)
- **Tool 2**: Wall Tracer - Connect snapped points to trace wall segments
- **Tool 3**: Mask Generator - Convert traced walls to binary masks
- **Tool 4**: Validator - Quality check extracted wall data

## Design Principles
- **Single Responsibility**: Each tool does one thing well
- **Modularity**: Tools work independently and together
- **User Experience**: Intuitive controls, immediate visual feedback
- **Precision**: Accurate coordinate capture for architectural work
- **Performance**: Real-time responsiveness for interactive use