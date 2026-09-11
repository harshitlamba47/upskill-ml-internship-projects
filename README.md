# Up Skill Campus ML Internship Projects

This package contains the two machine-learning projects completed during a four-week internship:

1. Crop and Weed Detection with YOLOv8
2. NASA C-MAPSS Turbofan Engine RUL Prediction

Both projects are organized as standalone VS Code/GitHub repositories.

## Recommended GitHub approach

Create two repositories:

- `crop-weed-detection-yolov8`
- `turbofan-rul-prediction`

Keep the large datasets and trained weights out of GitHub. The supplied `.gitignore` files already exclude them.

## What was implemented

### Project 1
Dataset splitting, exploration, YOLO-format validation, Albumentations augmentation, YOLOv8 transfer learning, inference, bounding-box extraction, and a weed-only spray decision layer.

### Project 2
C-MAPSS exploration, RUL calculation and capping, rolling statistics, MinMax scaling, Random Forest, Gradient Boosting, PyTorch neural network, ensemble prediction, and maintenance alerts.
