"""Split an existing YOLO-format dataset into train/val/test.

Expected input:
    data/raw/images/*.jpg
    data/raw/labels/*.txt

Each label row:
    class_id x_center y_center width height

The script copies matching image/label pairs and preserves YOLO coordinates.
"""

from pathlib import Path
import random
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_pairs(image_dir: Path, label_dir: Path):
    pairs = []
    for img in sorted(image_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs

def copy_split(items, split, output_root):
    image_out = output_root / split / "images"
    label_out = output_root / split / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    for img, label in items:
        shutil.copy2(img, image_out / img.name)
        shutil.copy2(label, label_out / label.name)

def main():
    random.seed(42)
    root = Path("data/raw")
    image_dir = root / "images"
    label_dir = root / "labels"
    output_root = Path("data/yolo")

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Put the raw YOLO dataset in data/raw/images and data/raw/labels."
        )

    pairs = find_pairs(image_dir, label_dir)
    if not pairs:
        raise RuntimeError("No image/label pairs were found.")

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    for split, items in (("train", train), ("val", val), ("test", test)):
        copy_split(items, split, output_root)

    print(f"Total images: {n}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

if __name__ == "__main__":
    main()
