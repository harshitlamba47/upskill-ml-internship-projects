"""Train a YOLOv8n crop/weed detector using transfer learning."""

from pathlib import Path
import torch
from ultralytics import YOLO

DATA_YAML = "data.yaml"
MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 512
EPOCHS = 50
BATCH = 8

def select_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    Path("models").mkdir(exist_ok=True)
    device = select_device()
    print(f"Training device: {device}")

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=device,
        project="runs",
        name="crop_weed_v1",
        patience=15,
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        degrees=15,
        translate=0.10,
        scale=0.40,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        plots=True,
    )

    print("Training completed.")
    print("Best checkpoint:", Path(results.save_dir) / "weights" / "best.pt")

if __name__ == "__main__":
    main()
