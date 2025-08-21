# Class-Based Keypoint System Documentation

## Overview
The updated code now properly handles the YOLO model output format where different molecule classes provide different numbers of meaningful keypoints.

## Model Output Format
The `best_0820.pt` model returns data in the following format:
```
[class, x, y, w, h, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y]
```

## Class-Based Keypoint Rules

### Class 1 Molecules
- **Meaningful Keypoints**: All 4 keypoints (KP1, KP2, KP3, KP4)
- **Visualization**: Shows 4 different colored keypoints
- **Spectroscopy**: Can use up to 4 keypoints for measurements
- **Colors**: 
  - KP1: Yellow
  - KP2: Magenta  
  - KP3: Green
  - KP4: Cyan

### Class 0 & Class 2 Molecules
- **Meaningful Keypoints**: Only the first keypoint (KP1)
- **Visualization**: Shows only 1 keypoint (KP1 in yellow)
- **Spectroscopy**: Uses only KP1 for measurements
- **Note**: KP2-KP4 data exists but is unreliable/meaningless

## Code Changes Made

### 1. Visualization Updates (`test_simulation_standalone.py`)
- Added class-based keypoint rendering
- Updated legend to clarify which keypoints are meaningful for each class
- Enhanced title to show class information

### 2. Test Script Updates (`test_simulation.py`)
- Modified keypoint extraction to respect class rules
- Updated spectroscopy point selection based on molecule class
- Enhanced JSON export with class-specific keypoint information

### 3. Main Workflow Updates (`AI_Spectroscopy_main.py`)
- Updated spectroscopy point determination to use class-based logic
- Added class checking before using keypoints for measurements

## Key Functions Modified

### `visualize_molecules_on_image()`
```python
# Determine keypoints to show based on class
mol_class = int(molecule[0])
if mol_class == 1:
    num_keypoints = 4  # Show all 4 keypoints
else:
    num_keypoints = 1  # Show only KP1
```

### Spectroscopy Point Selection
```python
# Extract meaningful keypoints based on class
if mol_class == 1:
    # Class 1: All 4 keypoints are meaningful
    for kp_idx in range(4):
        # Extract all keypoints
else:
    # Class 0 and 2: Only first keypoint is meaningful
    keypoints.append((molecule[5], molecule[6]))  # Only KP1
```

## Benefits

1. **Accuracy**: Only uses reliable keypoints for each molecule class
2. **Clarity**: Visual representation clearly shows which keypoints are meaningful
3. **Efficiency**: Prevents wasted measurements on unreliable keypoints
4. **Flexibility**: Can easily adapt to different model outputs or class definitions

## Usage

### Running Tests
```bash
# Using virtual environment
D:/HKUST/research/AI/.venv/Scripts/python.exe test_simulation.py
D:/HKUST/research/AI/.venv/Scripts/python.exe test_class_keypoints.py
```

### Expected Output
- Class 1 molecules will show 4 colored keypoints (Yellow, Magenta, Green, Cyan)
- Class 0 & 2 molecules will show only 1 keypoint (Yellow)
- Legend clearly indicates which keypoints are available for each class
- Spectroscopy points will be selected appropriately based on class

## Virtual Environment Setup
The project now uses a virtual environment located at:
```
D:/HKUST/research/AI/.venv/
```

All required packages are installed:
- opencv-python (4.12.0.88)
- matplotlib (3.10.3)
- numpy (2.2.6)
- torch (2.7.1+cu128)
- scikit-image (0.25.2)

To run any Python script, use the full path to the virtual environment Python:
```
D:/HKUST/research/AI/.venv/Scripts/python.exe <script_name>.py
```
