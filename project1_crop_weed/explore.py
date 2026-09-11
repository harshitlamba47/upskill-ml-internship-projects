"""Dataset exploration for Crop & Weed Detection."""

from pathlib import Path
from collections import Counter
import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = {0: "Crop", 1: "Weed"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def read_labels(label_path):
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                labels.append(int(parts[0]))
            except ValueError:
                pass
    return labels

def main():
    dataset = Path("data/yolo")
    results = Path("results")
    results.mkdir(exist_ok=True)

    images = collect_images(dataset)
    counts = Counter()

    widths, heights = [], []
    bad_labels = []

    for img in images:
        image = cv2.imread(str(img))
        if image is not None:
            h, w = image.shape[:2]
            widths.append(w)
            heights.append(h)

        label = dataset / img.relative_to(dataset)
        label = label.parent.parent / "labels" / f"{img.stem}.txt"
        ids = read_labels(label)
        for cid in ids:
            counts[cid] += 1

        if image is None or not label.exists():
            bad_labels.append(img.name)

    names = [CLASS_NAMES.get(cid, f"Class {cid}") for cid in sorted(counts)]
    values = [counts[cid] for cid in sorted(counts)]

    plt.figure(figsize=(7, 5))
    plt.bar(names, values)
    plt.title("Crop and Weed Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Bounding-box instances")
    plt.tight_layout()
    plt.savefig(results / "class_distribution.png", dpi=160)
    plt.close()

    print(f"Images found: {len(images)}")
    print(f"Average image size: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
    print("Class counts:", {CLASS_NAMES.get(k, k): v for k, v in counts.items()})
    if bad_labels:
        print("Files needing review:", bad_labels)

if __name__ == "__main__":
    main()
