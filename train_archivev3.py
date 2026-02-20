#!/usr/bin/env python3
"""Train a YOLOv8 model using archivev3 dataset."""

import os
import sys
from pathlib import Path

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

# Entrenar con archivev3
print("=" * 60)
print("Iniciando entrenamiento con dataset archivev3")
print("=" * 60)

results = model.train(
    data="data_archivev3.yaml",
    epochs=100,
    imgsz=640,
    batch=4,  # Batch pequeño para CPU
    device="cpu",  # CPU only
    patience=20,
    save=True,
    project="runs/train",
    name="archivev3_model",
    pretrained=True,
    verbose=True,
    workers=0,  # CPU: usar 0 workers para evitar problemas
    amp=False,  # Desactivar mixed precision en CPU
    optimizer="SGD"  # Usar SGD en lugar de Adam para conservar memoria
)

print("=" * 60)
print("Entrenamiento completado")
print(f"Resultados guardados en: {results}")
print("=" * 60)
