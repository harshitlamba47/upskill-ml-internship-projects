"""Crop/weed inference + smart spray decision system."""

from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

MODEL_PATH = Path("runs/crop_weed_v1/weights/best.pt")
SOURCE = Path("data/yolo/test/images")
OUTPUT = Path("results/predictions")
CONFIDENCE = 0.40
WEED_CLASS_ID = 1

def center_xy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first or update MODEL_PATH."
        )
    if not SOURCE.exists():
        raise FileNotFoundError(f"Input directory not found: {SOURCE}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    rows = []

    image_files = sorted([p for p in SOURCE.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])

    for result in model.predict(source=str(SOURCE), imgsz=512, conf=CONFIDENCE, save=False, verbose=False):
        img_name = Path(result.path).name
        annotated = result.plot()
        cv2.imwrite(str(OUTPUT / img_name), annotated)

        weed_centers = []
        boxes = result.boxes
        if boxes is not None:
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.conf.cpu().numpy(),
                                      boxes.cls.cpu().numpy()):
                cid = int(cls)
                if cid == WEED_CLASS_ID:
                    weed_centers.append(center_xy(box))

                rows.append({
                    "image": img_name,
                    "class": "Weed" if cid == WEED_CLASS_ID else "Crop",
                    "confidence": round(float(conf), 4),
                    "x1": round(float(box[0]), 1),
                    "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1),
                    "y2": round(float(box[3]), 1),
                })

        action = "SPRAY" if weed_centers else "NO SPRAY NEEDED"
        centers = "; ".join(f"({x},{y})" for x, y in weed_centers)
        print(f"{img_name}: {action} | weed centers: {centers or 'none'}")

    with (OUTPUT / "detections.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image","class","confidence","x1","y1","x2","y2"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved annotated predictions and detection report to {OUTPUT}")

if __name__ == "__main__":
    main()
