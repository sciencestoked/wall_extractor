# Wall Extractor Tool Suite

Clean, modular computer vision tools for architectural floor plan annotation.

## Tool 1: Snapper

Interactive shape detection and point annotation tool for precise coordinate capture.

### Quick Start

```bash
python snapper.py path/to/your/floorplan.jpg
```

### How It Works

1. **Open Image**: Loads your floor plan image
2. **Search Window**: Yellow rectangle shows detection area
3. **Shape Detection**: Finds circles and polygons in real-time
4. **Visual Feedback**: 
   - Red outlines = detected shapes
   - Green dots = clickable centroids
   - Blue markers = recorded points
5. **Click to Snap**: Click green dots to record precise coordinates

### Controls

| Key | Action |
|-----|--------|
| `↑↓←→` | Move search window |
| `+/-` | Resize search window |
| `Mouse Click` | Snap to nearest green centroid |
| `r` | Reset all recorded points |
| `s` | Save points to JSON file |
| `q` | Quit |

### Features

- **Smart Detection**: Finds circles and closed polygons
- **Boundary Validation**: Only records shapes ≥80% within search window
- **Duplicate Prevention**: Won't record same point twice
- **Real-time Feedback**: Live shape detection as window moves
- **Precise Coordinates**: Snaps to mathematical centroid
- **Visual Labels**: Shows shape type and area
- **Point Numbering**: Tracks recording sequence

### Output

Saves to `image_name.json`:
```json
{
  "image_path": "/path/to/image.jpg",
  "image_size": [width, height],
  "points": [
    {
      "position": [x, y],
      "shape_type": "circle",
      "area": 314.16,
      "timestamp": 0
    }
  ],
  "total_points": 1
}
```

### Use Cases

- **Floor Plan Annotation**: Mark architectural features
- **Quality Control**: Identify structural elements
- **Data Collection**: Gather precise coordinates for analysis
- **Preprocessing**: Prepare data for automated analysis

### Dependencies

```bash
pip install opencv-python numpy
```

### Tips

- Start with small search window for precision
- Use arrow keys to systematically scan image
- Green dots = clickable, red outlines = detected shape boundaries
- Check terminal for feedback on recorded points
- Save frequently to avoid losing work

## Future Tools

- **Wall Tracer**: Connect snapped points to trace wall segments  
- **Mask Generator**: Convert traced walls to binary masks
- **Validator**: Quality check extracted data

## Design Philosophy

Each tool has **single responsibility** and works both **independently** and **together** for complex workflows.