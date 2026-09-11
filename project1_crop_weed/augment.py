"""Optional offline augmentation for a YOLO dataset.

Creates additional training images while transforming bounding boxes safely.
Ultralytics also performs training-time augmentation; this script is useful
when a project specifically requires an explicit Albumentations pipeline.
"""

from pathlib import Path
import cv2
import numpy as np
import albumentations as A

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.Rotate(limit=18, border_mode=cv2.BORDER_REFLECT_101, p=0.45),
        A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.08, p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15),
)

def read_yolo(label_path):
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        classes.append(int(parts[0]))
        boxes.append([float(x) for x in parts[1:]])
    return boxes, classes

def write_yolo(label_path, boxes, classes):
    with label_path.open("w") as f:
        for cid, box in zip(classes, boxes):
            f.write(str(cid) + " " + " ".join(f"{v:.6f}" for v in box) + "\n")

def main():
    src_images = Path("data/yolo/train/images")
    src_labels = Path("data/yolo/train/labels")
    out_images = Path("data/yolo/train_augmented/images")
    out_labels = Path("data/yolo/train_augmented/labels")
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    generated = 0
    for image_path in src_images.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        boxes, classes = read_yolo(src_labels / f"{image_path.stem}.txt")
        if not boxes:
            continue

        for idx in range(1):  # one augmented copy per training image
            transformed = TRANSFORM(image=image, bboxes=boxes, class_labels=classes)
            aug = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_classes = transformed["class_labels"]

            if not aug_boxes:
                continue

            stem = f"{image_path.stem}_aug{idx + 1}"
            cv2.imwrite(str(out_images / f"{stem}.jpg"), aug)
            write_yolo(out_labels / f"{stem}.txt", aug_boxes, aug_classes)
            generated += 1

    print(f"Generated {generated} augmented training images.")
    print("To use them, either merge train_augmented into train or point a separate YAML at it.")

if __name__ == "__main__":
    main()
