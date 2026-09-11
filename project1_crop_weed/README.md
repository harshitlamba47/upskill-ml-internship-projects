# Crop and Weed Detection with YOLOv8

A computer-vision project for detecting **Crop** and **Weed** objects from field images and converting weed detections into a simple **smart-spray decision**.

## Project workflow

1. Inspect the YOLO dataset.
2. Split image/label pairs into train, validation, and test sets.
3. Apply optional offline Albumentations augmentation.
4. Fine-tune a pretrained YOLOv8n model at 512×512.
5. Run inference on unseen images.
6. Output weed bounding boxes and their pixel-center coordinates.
7. Recommend spraying only when a weed is detected.

Ultralytics supports loading pretrained YOLO models, custom training, validation, and image prediction through its Python API. citeturn675077search1turn675077search0

## Folder structure

```text
project1_crop_weed/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── labels/
│   └── yolo/
├── results/
├── models/
├── augment.py
├── data.yaml
├── explore.py
├── predict.py
├── split_dataset.py
├── train.py
├── requirements.txt
└── README.md
```

## Dataset format

Place the original YOLO-format dataset here:

```text
data/raw/images/
data/raw/labels/
```

Each label file should contain:

```text
class_id x_center y_center width height
```

with normalized YOLO coordinates.

Class mapping:

- `0` = Crop
- `1` = Weed

## Installation in VS Code

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the project

### 1. Split the dataset

```bash
python split_dataset.py
```

### 2. Explore the dataset

```bash
python explore.py
```

This creates:

```text
results/class_distribution.png
```

### 3. Optional offline augmentation

```bash
python augment.py
```

Ultralytics training also supports built-in augmentation parameters, so the supplied `train.py` includes flips, rotation, scaling, and HSV-based color augmentation.

### 4. Train

```bash
python train.py
```

The best model is normally saved at:

```text
runs/crop_weed_v1/weights/best.pt
```

### 5. Inference and smart spray decision

```bash
python predict.py
```

Annotated images and `detections.csv` are written to:

```text
results/predictions/
```

The system prints:

```text
image_001.jpg: SPRAY | weed centers: (241,185)
image_002.jpg: NO SPRAY NEEDED | weed centers: none
```

## Notes

Training is intentionally configurable because CPU, CUDA, and Apple Silicon hardware have different practical batch sizes. The script automatically selects CUDA, Apple MPS, or CPU.

Do not commit datasets or large trained weights to GitHub. Use Git LFS or release assets when model files need to be shared.

## Internship project scope

This repository implements the workflow documented during the first two weeks of the ML internship: dataset exploration, preprocessing, augmentation, transfer-learning-based object detection, evaluation/inference, and a weed-only spray decision layer.
