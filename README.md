sw# Monitoreo de Equipo de Protección Personal (EPP) - Detección con YOLOv8

Aplicación completa de **detección automática de EPP** usando **YOLOv8** con interfaz web en **Streamlit**, entrenamiento en GPU y análisis de resultados en tiempo real.

## 📊 Descripción del Proyecto

Este proyecto implementa un sistema inteligente de monitoreo de **Equipo de Protección Personal (EPP)** en entornos laborales mediante:

1. 🏋️**Entrenamiento Personalizado** - Modelos entrenados con datasets específicos (archivev3, archive2_auto)
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
│   ├── train_archivev3_gpu.py          # Entrenamiento GPU con archivev3
│   ├── train_archivev3.py              # Entrenamiento CPU con archivev3
│   ├── train_yolo.py                   # Trainer genérico flexible
│   ├── yolov8n.pt                      # Modelo YOLOv8 nano base
│   ├── best.pt                         # Mejor modelo entrenado (checkpoint)
│   └── yolo26n.pt                      # Modelo alternativo
│
├── 📊 DATOS Y CONFIGURACIÓN
│   ├── data_archivev3.yaml             # Config dataset archivev3
│   ├── data_archive2_auto.yaml         # Config dataset archive2_auto
│   ├── data_combined.yaml              # Config dataset combinado
│   ├── data_template.yaml              # Template para nuevos datasets
│   └── archive_data.yaml               # Config generalizada
│
├── 📁 DATASETS
│   ├── archivev3/                      # Dataset v3 (train/val/test)
│   │   └── images/ + labels/
│   │
│   └── archive2_auto/                  # Dataset automatizado v2
│       ├── train/                      # Imágenes y etiquetas de entrenamiento
│       ├── val/                        # Validación
│       └── test/                       # Pruebas
│
├── 🔍 MONITOREO Y DEBUG
│   ├── check_gpu.py                    # Verificar CUDA y GPU disponible
│   ├── check_training_progress.py      # Monitor de progreso de entrenamiento
│   ├── monitor_training.py             # Monitor de recursos (CPU/GPU/RAM)
│   ├── monitor_gpu_training.py         # Monitoreo específico GPU
│   └── debug_import.py                 # Debug de imports
│
├── 🚀 SCRIPTS DE EJECUCIÓN
│   ├── scripts/
│   │   ├── start_training_detached.ps1 # ⭐ Lanza entrenamiento en ventana separada
│   │   ├── train_archive2_auto.ps1     # Script entrenamiento archive2_auto
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
└── 🗂️ CARPETA ARCHIVE
    ├── archive/                        # Datos históricos
    ├── archive(2)/                     # Backup datasets
    └── archivev3/                      # Versión anterior del dataset
```

## 🛠️ Tecnologías Utilizadas

- **Python** 3.11+
- **YOLOv8** - Detección de objetos (ultralytics)
- **Streamlit** - Interfaz web interactiva
- **PyTorch** - Framework deep learning (con soporte CUDA)
- **OpenCV** - Procesamiento de imágenes
- **NumPy, Pandas** - Análisis de datos
- **GPU Support** - NVIDIA CUDA 12.1+, GPU Quadro P4000 (8GB VRAM)

## 📋 Requisitos

### Sistema Operativo
- **Windows 10/11** (PowerShell)
- **Python 3.11** (recomendado)
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

### 1️⃣ Entrenar Modelo (GPU)

**Opción A: Entrenamiento en ventana dedicada (Recomendado - sigue corriendo aunque cierre VS Code)**

```powershell
# Lanza entrenamiento con archivev3 en ventana independiente de PowerShell
.\epp_streamlit_app\scripts\start_training_detached.ps1 -DataYaml "data_archivev3.yaml" -RunName "archivev3_model_gpu" -Epochs 100 -Batch 32 -Workers 0 -Device "0"

# O con parámetros predeterminados:
.\epp_streamlit_app\scripts\start_training_detached.ps1
```

**Características:**
- ✅ Corre en ventana separada (no afecta VS Code)
- ✅ Actualiza encoding UTF-8 para salida legible
- ✅ Intenta reanudar desde checkpoint si existe
- ✅ Genera log automático con timestamp
- ✅ Parámetros configurables por CLI

**Opción B: Entrenamiento directo**

```python
# Entrenamiento con GPU (archivev3)
python train_archivev3_gpu.py

# O con script genérico
python train_yolo.py --data data_archivev3.yaml --epochs 100 --model yolov8n.pt --batch 32 --device 0
```

**Parámetros disponibles:**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--data` | `data_archivev3.yaml` | Archivo de configuración del dataset |
| `--epochs` | `100` | Número de epochs de entrenamiento |
| `--imgsz` | `640` | Tamaño de imagen |
| `--batch` | `32` | Batch size (ajustar según VRAM) |
| `--device` | `0` | GPU device (0 = primera GPU, "cpu" = CPU) |
| `--workers` | `0` | Workers de datos (Windows: usar 0) |
| `--project` | `runs/train` | Directorio de salida |
| `--name` | `archivev3_model_gpu` | Nombre del experimento |

### 2️⃣ Ejecutar Aplicación Web

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Iniciar Streamlit
streamlit run app.py

# O usando PowerShell
.\epp_streamlit_app\scripts\start_streamlit.ps1
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

# Monitor en tiempo real (mientras entrena)
python monitor_training.py

# Monitor específico GPU
python monitor_gpu_training.py

# Verificar progreso (requiere modelo entrenándose)
python check_training_progress.py
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

**Modelo Fine-tuned:** archivev3_model_gpu
- **Dataset**: archivev3 (etiquetas personalizadas)
- **Epochs**: 100
- **Batch Size**: 32
- **Clases**: Persona, Casco, Chaleco, Sin EPP
- **Hardware**: GPU NVIDIA Quadro P4000

**Métricas esperadas:**
- 🎯 **mAP@50**: ~85-92%
- 📈 **Precisión**: ~88-95%
- 🔍 **Recall**: ~80-90%

### 📁 Salidas de Entrenamiento

Cada entrenamiento genera:

```
runs/train/archivev3_model_gpu/
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

### Datasets Disponibles

#### archivev3
- **Imágenes**: ~2,000+ imágenes etiquetadas
- **Clases**: Persona, Casco, Chaleco, Sin EPP
- **Split**: Train/Val/Test
- **Formato**: YOLOv8 (txt con normalizadas)

#### archive2_auto
- **Imágenes**: Conjunto automatizado
- **Clases**: 4 (person, helmet, vest, no_protection)
- **Generación**: Semi-automática con validación
- **Uso**: Entrenamiento rápido y pruebas

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

> Nota: `.venv-1`, `.venv-2`, `.venv311` y `.venv_gpu` son entornos heredados de pruebas anteriores. El flujo recomendado usa solo `.venv`.
| `archivev3/` | Dataset v3 |
| `archive2_auto/` | Dataset v2 automatizado |
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
**Status:** Prototipo experimental / MVP de investigación

