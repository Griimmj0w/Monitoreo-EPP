# Evaluación del modelo (YOLOv8)

Este proyecto usa **detección de objetos** con **YOLOv8** (Ultralytics). Para reportes de investigación, lo más estándar es **mAP** y la **curva Precision–Recall (PR)**, junto con la **matriz de confusión**.

## 1) Gráficas solicitadas

### Curva Precision–Recall (PR)
- **Recomendada** para detección (sensible al desbalance de clases y no depende de “true negatives”).
- YOLOv8 la genera automáticamente al validar con `plots=True`.

### Matriz de confusión
- Útil para ver **en qué clases se confunde** el modelo.
- YOLOv8 la genera automáticamente al validar con `plots=True`.

### Curva ROC (Aclaración importante)
- En **detección** (localización) la ROC **no es la métrica típica**, porque el concepto de **true negatives** no está bien definido.
- Si aun así te piden ROC, una alternativa razonable es calcularla a nivel **presencia por imagen**: “¿la clase aparece en la imagen sí/no?”, usando como score la **máxima confianza** detectada de esa clase.
- El script de este repo genera esta variante como `roc_curve_presence.png`.
- Si tu dataset incluye clases negativas tipo `NO-*` / `Sin *`, el modo `--epp-only` las excluye aunque contengan palabras como "Hardhat" o "Vest".

## 2) Cómo generar los plots

Ejecuta (recomendado para quedarte solo con **persona + casco + chaleco**):

- `python scripts/evaluate_model.py --model RUTA/AL/MODELO.pt --data archive_data.yaml --roc --core-only`

Si quieres solo casco + chaleco (sin persona):

- `python scripts/evaluate_model.py --model RUTA/AL/MODELO.pt --data archive_data.yaml --roc --epp-only`

Alternativa (filtro manual por IDs de clase):

- `python scripts/evaluate_model.py --model RUTA/AL/MODELO.pt --data archive_data.yaml --roc --classes 0,7`

Salida:
- En `runs/eval/exp/` (o el `--name` que elijas) se guardan los PNG.
- Ultralytics suele guardar (según versión) archivos como `PR_curve.png`, `confusion_matrix.png`, etc.
- El script además guarda:
  - `roc_curve_presence.png`
  - `roc_auc_presence.csv`

## 2.1) Resultados ya generados (core_only_eval2)

Para evitar confusiones al revisar, en este repo se dejó como carpeta final de resultados:

- `runs/eval/core_only_eval2/`

Comando usado:

- `python scripts/evaluate_model.py --model best.pt --data archive_data.yaml --roc --core-only --name core_only_eval`

Resumen (validación, 114 imágenes):

| Clase | P | R | mAP50 | mAP50-95 |
|------|---:|---:|------:|---------:|
| all | 0.792 | 0.522 | 0.666 | 0.340 |
| Hardhat | 0.831 | 0.620 | 0.749 | 0.407 |
| Person | 0.755 | 0.482 | 0.625 | 0.303 |
| Safety Vest | 0.792 | 0.463 | 0.624 | 0.311 |

ROC (presencia por imagen) — AUC por clase (desde `roc_auc_presence.csv`):

| Clase | AUC |
|------|----:|
| Hardhat | 0.926587 |
| Person | 0.880952 |
| Safety Vest | 0.875415 |

## 3) ¿Cuál conviene usar (recomendación para tu informe)?

- **Primero (más recomendado en detección):** Curva **PR** + **mAP@0.5** y **mAP@0.5:0.95** (los reporta Ultralytics en la validación).
- **Complemento:** **Matriz de confusión** para analizar errores por clase.
- **ROC:** solo si te lo exigen; reporta explícitamente que es **ROC de presencia por imagen** (no mide calidad de localización de cajas).

## 4) Arquitectura básica (implementación en este proyecto)

Flujo general:

1. **Dataset etiquetado (YOLO format)** → imágenes + labels + `data_*.yaml`
2. **Entrenamiento**: `train_yolo.py` (fine-tuning desde `yolov8n.pt`) → produce `best.pt`
3. **Evaluación**: `scripts/evaluate_model.py` → PR + matriz de confusión + (ROC presencia)
4. **Aplicación**: `app.py` (Streamlit)
   - Captura fuente (Webcam/RTSP/Video)
   - Inferencia + tracking: `core/detector.py` (YOLOv8 + BoT-SORT/ByteTrack)
   - Asociación persona–EPP y reglas: `core/association.py`
   - Eventos/cooldowns: `core/events.py`

Si necesitas, puedo adaptar el diagrama a tu tesis (p. ej. con “Entradas → Procesamiento → Salidas” y módulos).