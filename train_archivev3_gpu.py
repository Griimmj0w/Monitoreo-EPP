#!/usr/bin/env python3
"""Train a YOLOv8 model using archivev3 dataset with GPU support."""

import os
import sys
from pathlib import Path

def main():
    # Asegurar que estamos en el directorio correcto
    os.chdir(Path(__file__).parent)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics no está instalado")
        print("Instálalo con: python -m pip install ultralytics")
        sys.exit(1)

    # Crear modelo YOLO
    model = YOLO("yolov8n.pt")

    # Entrenar con archivev3 usando GPU
    print("=" * 80)
    print("Iniciando entrenamiento con dataset archivev3 - GPU ACTIVADA")
    print("=" * 80)

    results = model.train(
        data="data_archivev3.yaml",
        epochs=100,
        imgsz=640,
        batch=32,  # Batch size para GPU (Quadro P4000 con 8GB VRAM)
        device=0,  # GPU device 0
        patience=20,
        save=True,
        project="runs/train",
        name="archivev3_model_gpu",
        pretrained=True,
        verbose=True,
        workers=0,  # Windows: usar 0 workers para evitar problemas de multiprocessing
    )

    print("=" * 80)
    print("Entrenamiento completado")
    print(f"Resultados guardados en: {results}")
    print("=" * 80)

if __name__ == '__main__':
    main()
