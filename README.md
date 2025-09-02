# Wall Extractor - Setup & Usage Guide

## 🚀 Quick Setup

### 1. Environment Setup
```bash
# Navigate to project directory
cd wall_extractor

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Test setup (optional)
python setup.py
```

### 2. File Structure
```
wall_extractor/
├── 00_scripts/           # Python scripts
├── 01_raw_images/        # PUT YOUR IMAGES HERE
├── 02_aligned_images/    # Aligned output
├── 03_pre_masks/         # Auto-generated masks
├── 04_final_masks/       # Final refined masks
├── PROJECT_SPEC.md       # Detailed specification
├── PROGRESS_TODO.md      # Progress tracking
└── README.md            # This file
```

### 3. Image Naming
Place floor plan pairs in `01_raw_images/`:
- First floor: `building_1F.png` 
- Second floor: `building_2F.png`
- Supported: PNG, JPG, JPEG, TIFF, BMP

## 🔄 Usage Workflow

### Step 1: Image Alignment
```bash
cd 00_scripts
python align_images.py
```
**Interactive Process:**
1. Image window opens
2. Click TOP-LEFT corner → BOTTOM-RIGHT corner  
3. Press `c` to confirm, `r` to reset, `q` to skip
4. Repeat for matching floor plan
5. Script saves aligned pairs automatically

### Step 2: Generate Pre-masks  
```bash
python create_premasks.py
```
**Automatic Process:**
- Converts images to black/white masks
- **Adaptive kernel sizing** - automatically adjusts for any image resolution/line thickness
- Removes thin elements (text, furniture) 
- Keeps thick walls
- Saves rough masks for refinement

### Step 3: Parameter Tuning (Optional)
```bash
python create_premasks.py --tune ../02_aligned_images/sample.png
```
**Interactive Tuning:**
- Press `a`/`d` to change kernel size
- Press `s` to save optimal settings
- Press `q` when satisfied
- **Note:** Usually not needed due to adaptive sizing

### Step 4: Manual Refinement (Label Studio)
1. **Install Label Studio:** Follow their docs
2. **Create Project:** Image Segmentation template
3. **Import Images:** Load from `02_aligned_images/`
4. **Import Masks:** Load corresponding files from `03_pre_masks/`
5. **Refine:** Use eraser (remove noise) + brush (fill gaps)
6. **Export:** VOC format → place in `04_final_masks/`

## ⚙️ Configuration

Edit `00_scripts/config.py` for:
- **Window size:** Display dimensions
- **Kernel size:** Morphological operations (default 10x10)
- **Detection window:** Corner refinement area (default 50px)

## 🔧 Common Issues

### "No images found"
- Check files are in `01_raw_images/`
- Verify naming: `*_1F.*` and `*_2F.*`
- Ensure supported file formats

### "Import error" / "Module not found"
```bash
# Ensure virtual environment is active
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Pre-masks too noisy/missing walls
```bash
# Usually auto-resolved with adaptive kernel sizing
# But if needed, run parameter tuning
python create_premasks.py --tune ../02_aligned_images/your_image.png

# Or force specific kernel size
python create_premasks.py --kernel-size 15 15  # Larger for thicker lines
python create_premasks.py --kernel-size 8 8    # Smaller for thinner lines
```

### Alignment not accurate
- Ensure floor plans have clear corner features
- Click as close to corners as possible
- Use `r` to reset and try again
- Adjust `feature_detection_window` in config.py

## 📊 Progress Tracking

Check `PROGRESS_TODO.md` for:
- Current project status
- Completed tasks checklist  
- Next steps and goals
- Known issues and fixes

## 🎯 Expected Results

**After Step 1:** Perfectly aligned floor plan pairs
**After Step 2:** Rough but clean wall masks  
**After Step 3:** Optimized parameters for your drawings
**After Step 4:** Pixel-perfect masks ready for ML training

## 🔍 Quality Check

**Good alignment:** Floor plans overlay perfectly
**Good pre-masks:** Walls are white, everything else black
**Good final masks:** Clean wall boundaries, no noise

---

**Need help?** Check `PROJECT_SPEC.md` for detailed methodology or `PROGRESS_TODO.md` for current status.