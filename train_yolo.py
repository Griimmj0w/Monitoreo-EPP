"""Small helper to train a YOLOv8 model using the ultralytics package.

Usage:
    .venv\Scripts\python.exe train_yolo.py --data data_template.yaml --epochs 50 --model yolov8n.pt

Dataset: Provide data in YOLOv8 format (images + labels) and update `data_template.yaml`.
"""
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data yaml (train/val/nc/names)")
    p.add_argument("--model", default="yolov8n.pt", help="Base model to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="exp")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ultralytics not available. Install it in your venv: pip install ultralytics")
        raise

    model = YOLO(args.model)
    print(f"Starting training: model={args.model}, data={args.data}, epochs={args.epochs}")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
