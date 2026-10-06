sw# Monitoreo de Equipo de Protección Personal (EPP) - Detección con YOLOv8

Aplicación completa de **detección automática de EPP** usando **YOLOv8** con interfaz web en **Streamlit**, entrenamiento en GPU y análisis de resultados en tiempo real.

## 📊 Descripción del Proyecto

Este proyecto implementa un sistema inteligente de monitoreo de **Equipo de Protección Personal (EPP)** en entornos laborales mediante:

1. 🏋️**Entrenamiento Personalizado** - Modelos fine-tuned con dataset propio (cctv_epp_v4) y Ultralytics HUB
2. 🚀 **Optimización GPU** - Soporte para entrenamiento y inferencia acelerada con CUDA
3. 🎥 **Detección en Tiempo Real** - YOLOv8 para identificación de personas y EPP
4. 📱 **Interfaz Web** - Dashboard en Streamlit para visualización y análisis

**Objeto de Detección:**
- 🧑 Personas
- 🪖 Cascos/Casco de seguridad
- 🦺 Chalecos de seguridad
- ❌ Personas **sin** equipo de protección

El objetivo es **monitorear en tiempo real** el cumplimiento de normas de seguridad en áreas de trabajo de alto riesgo, generando alertas automáticas y reportes de incidencias.

## 🎯 Objetivos

- Detectar automáticamente la presencia/ausencia de EPP en trabajadores
- Entrenar modelos YOLOv8 con datasets etiquetados personalizados
- Generar alertas en tiempo real para trabajadores sin equipamiento completo
- Proveer dashboard web intuitivo para monitoreo y análisis
- Optimizar inferencia usando GPU (CUDA) para procesamiento en vivo
- Exportar reportes y estadísticas de conformidad

## 📁 Estructura del Proyecto

```
epp_streamlit_app/
├── 🎨 APLICACIÓN PRINCIPAL
│   ├── app.py                          # App Streamlit principal
│   ├── requirements.txt                # Dependencias Python
│   └── setup.ps1                       # Script de instalación (Windows)
│
├── 🤖 MODELOS Y ENTRENAMIENTOS
│   ├── train_yolo.py                   # Trainer genérico (local)
│   ├── yolov8n.pt                      # Modelo YOLOv8 nano base
│   ├── best.pt                         # Modelo baseline histórico (10 clases)
│   └── exp-2.pt                        # Modelo vigente (v4: person/helmet/vest)
│
├── 📊 DATOS Y CONFIGURACIÓN
│   ├── datasets/cctv_epp_v4/           # Dataset v4 activo (train/val/test + yaml)
│   ├── scripts/pseudo_label_v4.py      # Pseudo-etiquetado + split temporal
│   ├── scripts/package_hub_zip.py      # Empaquetado para Ultralytics HUB
│   └── docs/guia_etiquetado_v4.md      # Guía de etiquetado
│
├── 📁 DATASETS
│   └── datasets/cctv_epp_v4/           # Dataset v4 activo (person, helmet, vest)
│       ├── train/                      # 260 imgs (70% cronológico)
│       ├── val/                        # 74 imgs (20%)
│       └── test/                       # 38 imgs (10%)
│
├── 🔍 MONITOREO Y DEBUG
│   └── check_gpu.py                    # Verificar CUDA y GPU disponible
│
├── 🚀 SCRIPTS DE EJECUCIÓN
│   ├── scripts/
│   │   ├── start_training_detached.ps1 # Lanza entrenamiento en ventana separada
│   │   ├── start_streamlit.ps1         # Inicia app web
│   │   ├── check_streamlit.ps1         # Verifica estado streamlit
│   │   ├── restart_streamlit.ps1       # Reinicia aplicación
│   │   ├── run_inference.py            # Script inferencia standalone
│   │   └── test_camera.py              # Prueba de cámara web
│
├── 🧠 MÓDULOS CORE
│   ├── core/
│   │   ├── detector.py                 # Lógica de detección YOLO
│   │   ├── config.py                   # Configuración global
│   │   ├── association.py              # Asociación de detecciones
│   │   ├── events.py                   # Generación de eventos/alertas
│   │   ├── id_reader.py                # Lectura de IDs/matrícula
│   │   └── __pycache__/               # Cache compilado
│
├── 🖼️ IMÁGENES Y SALIDAS
│   ├── images/test/                    # Imágenes de prueba
│   ├── runs/                           # Salidas de entrenamiento (auto-generado)
│   │   └── detect/                     # Inferencias realizadas
│   └── *.jpg                           # Resultados de detección
│
├── 📦 ENTORNOS VIRTUALES
│   └── .venv/                          # Entorno virtual principal
│
├── 📝 DOCUMENTACIÓN
│   ├── README.md                       # Este archivo
│   ├── training_log.txt                # Log de entrenamientos
│   └── training_log_*.txt              # Logs con timestamp
│
├── 📦 ARCHIVOS COMPRIMIDOS
│   └── *.rar, *.zip                    # Backups y archivos de respaldo
│
└── 🗂️ DATASETS HISTÓRICOS (retirados oct 2026)
    Se eliminaron archive/, archive(2)/ y archive2_auto/ del repo.
```

## 🛠️ Tecnologías Utilizadas

- **Python** 3.13
- **YOLOv8** - Detección de objetos (ultralytics)
- **Streamlit** - Interfaz web interactiva
- **PyTorch** - Framework deep learning (con soporte CUDA)
- **OpenCV** - Procesamiento de imágenes
- **NumPy, Pandas** - Análisis de datos
- **GPU Support** - NVIDIA GTX 1050 Ti (4GB VRAM), driver 582.x, PyTorch wheels cu124

## 📋 Requisitos

### Sistema Operativo
- **Windows 10/11** (PowerShell)
- **Python 3.13** (recomendado)
- **GPU NVIDIA** con soporte CUDA (opcional pero recomendado para entrenamiento)

### Instalación Rápida

```bash
# Opción 1: Script automático (Windows)
.\setup.ps1

# Opción 2: Manual
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Verificar GPU disponible
python check_gpu.py
```

## 🚀 Uso

### 1️⃣ Entrenar Modelo

**Opción A (recomendada): Ultralytics HUB** — empaquetar el dataset con
`scripts/package_hub_zip.py`, subirlo y entrenar con créditos de la
plataforma o con la GPU local (Bring Your Own Agent).

**Opción B: Entrenamiento local directo**

```powershell
# Fine-tune con el dataset v4
.\.venv\Scripts\python.exe train_yolo.py --data datasets/cctv_epp_v4/data_cctv_v4.yaml --epochs 80 --model yolov8n.pt --imgsz 640 --batch 8 --device 0

# O en ventana separada (sigue corriendo aunque cierre VS Code)
.\scripts\start_training_detached.ps1 -DataYaml "datasets/cctv_epp_v4/data_cctv_v4.yaml" -RunName "cctv_v4" -Epochs 80 -Batch 8 -Device "0"
```

**Parámetros disponibles:**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--data` | `datasets/cctv_epp_v4/data_cctv_v4.yaml` | Configuración del dataset |
| `--epochs` | `80` | Número de epochs de entrenamiento |
| `--imgsz` | `640` | Tamaño de imagen |
| `--batch` | `8` | Batch size (ajustar según VRAM) |
| `--device` | `0` | GPU device (0 = primera GPU, "cpu" = CPU) |
| `--workers` | `0` | Workers de datos (Windows: usar 0) |
| `--project` | `runs/train` | Directorio de salida |
| `--name` | `cctv_v4` | Nombre del experimento |

### 2️⃣ Ejecutar Aplicación Web

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Iniciar Streamlit
streamlit run app.py

# O usando PowerShell
.\scripts\start_streamlit.ps1
```

**Se abrirá en:** `http://localhost:8501`

**Funcionalidades del Dashboard:**
- 📹 Carga de imagen o video para inferencia
- 🎯 Visualización de detecciones en tiempo real
- 📊 Estadísticas de confianza por clase
- 🔔 Alertas de incidencias
- 💾 Exportación de resultados

### 3️⃣ Monitorear Entrenamiento

```bash
# Verificar GPU disponible
python check_gpu.py
```

### 4️⃣ Inferencia Standalone

```bash
# Correr detección en imagen/video
python scripts/run_inference.py

# Prueba de cámara
python scripts/test_camera.py
```

## 📈 Resultados Principales

### 📊 Modelos Entrenados

**Modelo Base:** YOLOv8 Nano (yolov8n.pt)
- **Parámetros**: ~3.3M
- **Tamaño**: ~12 MB
- **Velocidad**: ~80ms por frame (GPU)

**Modelo vigente:** exp-2.pt (entrenado en Ultralytics HUB)
- **Dataset**: cctv_epp_v4 (person, helmet, vest)
- **Clases**: Persona, Casco, Chaleco
- **Hardware entrenamiento**: GPU NVIDIA en la nube (HUB)

**Métricas reales medidas (baseline histórico, val 114 imgs):**
- 🎯 **mAP@50**: 0.666
- 📈 **Precisión**: 0.792
- 🔍 **Recall**: 0.522
- (Objetivo de tesis: mAP@0.5 ≥ 0.90 — el baseline es la referencia a superar)

### 📁 Salidas de Entrenamiento

Cada entrenamiento genera:

```
runs/train/cctv_v4/
├── weights/
│   ├── best.pt              # Mejor modelo (época de mínimo loss)
│   └── last.pt              # Último checkpoint (para reanudar)
├── results.csv              # Métricas por epoch
├── confusion_matrix.png     # Matriz de confusión
├── results.png              # Gráficos de entrenamiento
└── val_*.jpg                # Ejemplos de validación
```

## 🎯 Metodología

### Flujo de Entrenamiento

1. **Preparación de Datos**
   - Datasets en formato YOLOv8 (imágenes + txt con anotaciones)
   - Configuración en archivos `.yaml`
   - Split train/val/test

2. **Entrenamiento**
   - Augmentación automática (rotación, zoom, flip)
   - Optimización con Adam
   - Early stopping basado en mAP

3. **Evaluación**
   - Validación en cada epoch
   - Matriz de confusión
   - Curvas de precisión-recall

4. **Exportación**
   - Modelo `.pt` listo para inferencia
   - Logs de entrenamiento
   - Visualizaciones de resultados

### Flujo de Inferencia

1. Carga de modelo (`.pt`)
2. Preprocesamiento de imagen/frame
3. Forward pass en la red
4. Post-procesamiento (NMS)
5. Generación de bounding boxes y confianzas
6. Visualización de resultados

## 📊 Datos

### Dataset activo: cctv_epp_v4

- **Imágenes**: 372 frames de video CCTV (3200x1800), split temporal
  260 train / 74 val / 38 test
- **Clases**: `person`, `helmet`, `vest` (guantes en pausa)
- **Formato**: YOLOv8 (txt con coordenadas normalizadas)
- **Etiquetas**: pseudo-etiquetadas con `scripts/pseudo_label_v4.py`
  y corregidas a mano (ver `docs/guia_etiquetado_v4.md`)

## 🔍 Interpretabilidad

- **Confidence Score (0-1)** - Confianza de cada detección
- **Bounding Boxes** - Ubicación exacta de objetos en imagen
- **Class Predictions** - Etiqueta asignada a cada detección
- **Visualización** - Anotaciones dibujadas directamente en imágenes/videos

## 🎓 Casos de Uso

### Para Empresas/Plantas Industriales

1. **Monitoreo de Seguridad**
   - Verificación automática en tiempo real
   - Alertas por incumplimiento
   - Auditorías digitales

2. **Reportes de Conformidad**
   - Estadísticas de uso de EPP
   - Incidentes por zona/turno
   - Tendencias y mejoras

3. **Capacitación**
   - Análisis de puntos críticos
   - Retroalimentación a trabajadores
   - Histórico de cumplimiento

### Para Investigadores

1. **Validación de Modelos**
   - Benchmarking de YOLOv8 en dominio específico
   - Evaluación de trade-offs entre velocidad/precisión
   - Experimentación con datasets personalizados

2. **Extensiones Posibles**
   - Tracking multi-persona
   - Análisis de comportamiento
   - Predicción de riesgos

## 📋 Configuración del Entorno

### Variables de Entorno (opcional)

```powershell
# Configuración de GPU
$env:CUDA_VISIBLE_DEVICES = "0"        # Seleccionar GPU 0
$env:TF_CPP_MIN_LOG_LEVEL = "3"        # Reducir logs
$env:PYTHONIOENCODING = "utf-8"        # Encoding UTF-8

# Para PowerShell:
chcp 65001 | Out-Null                  # UTF-8 console
```

### Puntos de Montaje/Rutas Importantes

| Ruta | Descripción |
|------|-------------|
| `.venv/` | Entorno virtual principal |

> Nota: el entorno recomendado es `.venv` (Python 3.13).
| `runs/train/` | Modelos entrenados (salida) |
| `core/` | Módulos de detección y eventos |
| `scripts/` | Scripts de ejecución rápida |

## 🐛 Troubleshooting

### Error: "CUDA no disponible"
```bash
python check_gpu.py
python -c "import torch; print(torch.cuda.is_available())"
```

**Solución:** Instalar PyTorch con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error: "Carácteres inentendibles en consola"
Se ejecuta automáticamente en `start_training_detached.ps1`, pero también puedes:
```powershell
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
```

### Memoria GPU insuficiente
Reducir `--batch`:
```bash
python train_yolo.py --data data_archivev3.yaml --batch 16 --device 0
```

### Modelo no mejora
- Aumentar epochs: `--epochs 150`
- Validar anotaciones del dataset
- Usar data augmentation más agresiva
- Ajustar learning rate

## 📄 Documentación Adicional

Para información más detallada, consulta:

- **`app.py`** - Código principal de Streamlit
- **`core/detector.py`** - Lógica de detección
- **`requirements.txt`** - Todas las dependencias
- **`setup.ps1`** - Script de instalación completo

## 📊 Principales Hallazgos

1. **YOLOv8 Nano** es suficientemente rápido para tiempo real (~80ms/frame GPU)
2. **Dataset de calidad** impacta directamente en precisión
3. **GPU es esencial** para entrenamiento (10x más rápido que CPU)
4. **Batch size de 32** es balanceado para GPUs con 8GB VRAM
5. **Early stopping** evita overfitting alrededor de epoch 80-100
6. **Detecciones falsas** disminuyen significativamente con fine-tuning

## 👥 Autor

Desarrollado como sistema de monitoreo industrial de EPP.

## 📄 Licencia

Proyecto de uso comercial/industrial. Uso sujeto a cumplimiento de regulaciones locales.

---

**Last Updated:** 20 Febrero 2026  
**Versión:** 1.0 (Initial Release)  
**Status:** ✅ En producción

