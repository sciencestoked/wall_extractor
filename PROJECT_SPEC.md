# Project Plan: Data Preparation for Wall Alignment Ratio (WAR) Model

## Objective
To produce a high-quality, pixel-perfect dataset of annotated floor plans. This dataset will be the foundation for training a machine learning model to automatically extract structural walls from architectural drawings.

## Core Strategy
We will employ a "Human-in-the-Loop" workflow. This combines fast, automated pre-processing using computer vision scripts with efficient manual refinement. This approach is significantly faster and more scalable than pure manual annotation.

## Phase 1: Setup and Tooling
**Goal**: To establish a consistent and efficient environment for our data preparation tasks.

### 1.1. Annotation Tool Selection:
**Tool**: Label Studio

**Why**: It is the best choice for this project because it's free, open-source, and powerful. It runs locally, ensuring our data remains private. Crucially, it supports the brush and eraser tools needed for our efficient "erasing" workflow, and it can import pre-generated masks, which is central to our strategy.

### 1.2. Scripting Environment:
**Tool**: Python with the OpenCV library.

**Why**: This is the industry standard for computer vision tasks. OpenCV contains all the necessary functions for image manipulation, and the vast amount of documentation and community support makes development straightforward.

### 1.3. Actionable Steps:
1. **Set up the Environment**: On a designated machine, install the latest versions of Python and OpenCV.

2. **Install Label Studio**: Follow the official documentation to install and run Label Studio, preferably using their recommended Docker setup for ease of use.

3. **Create the Project**:
   - Launch Label Studio and create a new project.
   - In the "Labeling Setup" screen, choose the "Image Segmentation" template.
   - Define a single label class named `wall`. This will be the only object we annotate.

## Phase 2: Image Alignment
**Goal**: To ensure all floor plans are in a common coordinate system before annotation. This step is critical for preventing errors and is a prerequisite for the final WAR calculation.

### Methodology: The "Smart Diagonal Click" Method
**Why**: Fully automated alignment is complex and prone to errors. This pragmatic approach uses a human to provide two guiding points per image, while the computer calculates the pixel-perfect transformation. It is fast, incredibly accurate, and robust against common issues like different export margins or image dimensions.

### Actionable Steps:
1. **Organize Raw Data**: Create a folder named `01_raw_images` and place all original first and second-floor plan images inside.

2. **Develop the Alignment Script**:
   - Create a Python script using OpenCV.
   - The script's logic will be:
     a. For each pair of floor plans (1F and 2F):
     b. Display the first-floor image.
     c. Prompt the user (via the console or a simple window title) to "Click the TOP-LEFT exterior corner."
     d. Capture the mouse click coordinates. In a 50x50 pixel area around the click, use `cv2.goodFeaturesToTrack` to find the precise corner coordinate. Store this point.
     e. Prompt the user to "Click the BOTTOM-RIGHT exterior corner" and repeat the refinement process.
     f. Repeat steps c-e for the second-floor image.
     g. You now have two precise points for each image. Use `cv2.getPerspectiveTransform` to calculate the transformation matrix that maps the second-floor points to the first-floor points.
     h. Use `cv2.warpPerspective` to apply this transformation to the entire second-floor image.
     i. Save the original first-floor image and the newly aligned second-floor image into a new folder named `02_aligned_images`.

## Phase 3: The Annotation Workflow
**Goal**: To efficiently generate the final, pixel-perfect segmentation masks for every aligned image.

### Methodology: The "Automate-and-Refine" Workflow
**Why**: This is the core of our efficiency strategy. Instead of manually drawing, we will automatically generate a "good enough" mask and then have a human quickly correct it. Correcting is 5-10 times faster than creating from scratch.

### Actionable Steps:
1. **Develop the Pre-masking Script**:
   - Create a new Python/OpenCV script.
   - The script's logic will be:
     a. Take an image from the `02_aligned_images` folder as input.
     b. Convert it to grayscale.
     c. Apply a binary threshold to create a black-and-white image. `cv2.THRESH_OTSU` is recommended as it finds an optimal threshold value automatically.
     d. Perform a morphological opening (`cv2.morphologyEx` with `cv2.MORPH_OPEN`). This involves first eroding and then dilating the image. The goal is to make all thin lines (text, furniture, dimensions) disappear while keeping the thick structural walls.
     e. **Tune the Kernel Size**: The size of the kernel used in the opening operation is the key parameter. It should be tuned so that it successfully removes noise without destroying the walls.
     f. Save the resulting black-and-white image to a new folder named `03_pre_masks`. The filename should correspond to the original (e.g., plan1.png -> plan1_premask.png).

2. **Bulk Pre-masking**: Run this script on all images in the `02_aligned_images` folder to generate a complete set of rough masks.

3. **Import to Label Studio**: In your Label Studio project, import all the images from `02_aligned_images`.

4. **Manual Refinement (The Annotation Task)**:
   - For each image in the Label Studio interface, the annotator will import the corresponding rough mask from the `03_pre_masks` folder.
   - The annotator will now see the original image with the imperfect mask overlaid.
   - Their task is simple and fast:
     - Select the **Eraser Tool** to remove any leftover non-wall elements (e.g., thick furniture, logos).
     - Select the **Brush Tool** to fill in any small gaps in the walls that the script may have created.

5. **Export the Final Dataset**:
   - Once all images have been refined and marked as complete in Label Studio, use the "Export" function.
   - **Export Format**: Choose the "VOC" format (for Pascal VOC). This will generate a folder named `SegmentationClass` containing your final PNG masks.
   - Rename this folder to `04_final_masks`.

## Final Deliverable
Upon completion of this plan, we will have two critical folders:

- `02_aligned_images`: The clean, aligned source images.
- `04_final_masks`: The corresponding pixel-perfect, black-and-white segmentation masks.

This dataset is the final output of the preparation phase and is now ready to be used for training the machine learning model.