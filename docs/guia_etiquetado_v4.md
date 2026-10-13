# Guia de etiquetado dataset v4 (person, helmet, vest)

El pseudo-etiquetado ya genero cajas para `person`, `helmet` y `vest`.
Tu tarea: **corregir esas cajas**.

> Nota: la clase `gloves` quedo en pausa (poca visibilidad en este video CCTV).
> Se reactivara luego con grabaciones mas cercanas usando
> `scripts/pseudo_label_v4.py --with-gloves`.

## 1. Donde estan los archivos

```text
datasets/cctv_epp_v4/train/images  + train/labels    (260 imagenes)
datasets/cctv_epp_v4/val/images    + val/labels      (75 imagenes)
datasets/cctv_epp_v4/test/images   + test/labels     (37 imagenes)
```

Cada imagen tiene su `.txt` con el mismo nombre (formato YOLO).

## 2. Clases (IDs fijos)

```text
0 person
1 helmet
2 vest
```

El archivo `predefined_classes_v4.txt` de la raiz tiene esta lista (para LabelImg).

## 3. Herramienta: LabelImg (recomendado)

Instalar en el venv nuevo:

```powershell
.\.venv\Scripts\pip.exe install labelImg pyqt5 lxml
.\.venv\Scripts\python.exe -m labelImg
```

En LabelImg:

1. `Open Dir` -> `datasets\cctv_epp_v4\train\images`
2. `Change Save Dir` -> `datasets\cctv_epp_v4\train\labels`
3. Formato **YOLO** (menu izquierdo).
4. Cargar las clases desde `predefined_classes_v4.txt`
   (o crear las 3 en ese orden exacto: person, helmet, vest).
5. Las cajas existentes se cargan solas desde los `.txt`: **corrige, no dupliques**.
6. Guardar cada imagen con `Ctrl + S`.

Alternativa web si LabelImg falla: https://www.makesense.ai/ (importa labels YOLO).

## 4. Que corregir en cada imagen

- **Eliminar cajas falsas** (detecciones que no corresponden a nada real).
- **Ajustar cajas mal ubicadas** (persona/casco/chaleco).
- **Agregar lo que el modelo no detecto**: personas, cascos o chalecos visibles sin caja.
- Imagenes sin personas/EPP: dejar el `.txt` vacio (sin cajas).

## 5. Prioridades (no hace falta repasar las 372)

1. **Imagenes con trabajadores cerca de la camara** -> donde casco y chaleco se ven claros.
2. Imagenes con errores obvios del pseudo-etiquetado (cajas flotantes, cascos duplicados).
3. El resto puede quedar como esta si las cajas se ven razonables.

Meta sugerida: al menos 150-200 imagenes revisadas, con la mayoria de personas
visibles bien etiquetadas.

## 6. Cuando termines

Avisar para lanzar el entrenamiento:

```powershell
.\.venv\Scripts\python.exe train_yolo.py --data datasets/cctv_epp_v4/data_cctv_v4.yaml --epochs 80 --model runs/detect/runs/train/css_v28_plus/weights/best.pt --imgsz 640 --batch 8 --name cctv_v4_finetune
```
