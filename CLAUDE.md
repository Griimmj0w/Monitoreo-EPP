# CLAUDE.md — Monitoreo de EPP (Monitoreo-EPP)

## Qué es este proyecto

Sistema académico (tesis UPC) de monitoreo automático de Equipo de Protección
Personal (EPP) en una planta metalmecánica peruana, usando detección de objetos
con YOLOv8, tracking y una app web en Streamlit.

**Estado real: prototipo experimental / MVP de investigación (NO producción).**

## Decisiones de hardware y despliegue

- **Entrenamiento**: PC Windows con GPU NVIDIA (GTX 1050 Ti, 4GB VRAM; driver 582.x,
  CUDA 13.0). Solo se entrena aquí; nunca en el edge. PyTorch instalado con wheels
  **cu124** (soporta Pascal sm_61 y Python 3.13).
- **Edge (inferencia)**: Raspberry Pi 5 + Coral USB Accelerator (Edge TPU).
  - El modelo debe convertirse a formato compatible con Edge TPU (TFLite int8 /
    `ultralytics export format="edgetpu"`), no TensorRT.
  - En la Pi se usa solo inferencia (runtime TFLite / pycoral), sin entrenar.
- La app de desarrollo sigue corriendo en PC con `streamlit run app.py`.

## Alcance funcional

- Fuentes de video: webcam, archivo de video/imagen, RTSP (básico).
- Tracking: BoT-SORT / ByteTrack vía `model.track()`.
- EPP objetivo final: **helmet, vest, gloves**. Los guantes están **en pausa**
  (poca visibilidad en el video CCTV de la demo); el foco actual es casco+chaleco.
  El pipeline v4 (`scripts/pseudo_label_v4.py`) tiene flag `--with-gloves` para reactivarlos.
- La evaluación de cumplimiento es **por persona**, asociando cada EPP a la
  persona detectada. No depender de clases negativas (`NO-*`, `Sin *`).
- El próximo dataset debe normalizarse a: `person, helmet, vest, gloves`.

## Métricas y verdad científica

- Baseline actual (val 114 imgs): mAP@0.5 = 0.666, P = 0.792, R = 0.522.
- Objetivos de tesis: mAP@0.5 ≥ 0.90, ≥ 15 FPS, latencia detección→alerta ≤ 5 s.
- **Nunca reportar métricas no medidas.** Si el README dice "esperado", no
  presentarlo como obtenido. La ROC del repo es presencia por imagen (auxiliar).

## Reglas de desarrollo

- `app.py` (raíz) es la app vigente. La app legada `epp_streamlit_app/app.py`
  fue retirada (oct 2026); no volver a introducir una segunda app.
- `core/` contiene la lógica (detector, asociación, eventos, epp_logic, media).
  Seguir separando lógica de UI.
- Cambios pequeños, verificables y reversibles; un tema por commit.
- No eliminar funcionalidad existente sin justificarlo.
- Mantener compatibilidad con Streamlit y el flujo PC→edge.

## Comandos útiles

- App: `.venv\Scripts\python.exe -m streamlit run app.py`
- Entrenar: `.venv\Scripts\python.exe train_yolo.py --data <yaml> --epochs N --model yolov8n.pt`
- Evaluar: `.venv\Scripts\python.exe scripts/evaluate_model.py --model <best.pt> --data <yaml> --roc --core-only`
- Extraer frames de videos CCTV: `.venv\Scripts\python.exe scripts/extract_training_frames.py`
- Instalación: `setup.ps1` (venv Python 3.13 + PyTorch cu124/cu126)
