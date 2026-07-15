# Guia para etiquetar imagenes de CCTV EPP

Esta guia sirve para preparar el dataset de la Camara 9 y reentrenar el modelo para reducir falsos positivos como personas con casco/chaleco marcadas como `Sin EPP`.

## 1. Ubicacion de las imagenes extraidas

Las imagenes ya extraidas del video estan en:

```text
C:\Users\SISTEMAS\Documents\epp_streamlit_app\datasets\cctv_epp\raw_frames
```

Las etiquetas se deben guardar inicialmente en:

```text
C:\Users\SISTEMAS\Documents\epp_streamlit_app\datasets\cctv_epp\raw_labels
```

Si la carpeta `raw_labels` no existe, crearla.

## 2. Clases del modelo

El modelo actual usa estas 10 clases:

```text
0 Hardhat
1 Mask
2 NO-Hardhat
3 NO-Mask
4 NO-Safety Vest
5 Person
6 Safety Cone
7 Safety Vest
8 machinery
9 vehicle
```

Para esta primera mejora, etiquetar principalmente:

```text
5 Person
0 Hardhat
7 Safety Vest
```

Usar estas clases negativas solo cuando sea muy claro:

```text
2 NO-Hardhat
4 NO-Safety Vest
```

## 3. Reglas de etiquetado

Etiquetar cada persona visible:

- `Person`: caja alrededor del cuerpo visible. No incluir sombra.
- `Hardhat`: caja ajustada alrededor del casco.
- `Safety Vest`: caja alrededor del chaleco visible.

Casos importantes:

- Si la persona tiene casco blanco, etiquetar `Hardhat`.
- Si la persona tiene chaleco naranja reflectivo, etiquetar `Safety Vest`.
- Si una persona esta parcialmente tapada, etiquetar solo la parte visible.
- No etiquetar sombras.
- No etiquetar objetos parecidos a casco si no son casco.
- No etiquetar chaleco si esta completamente oculto.
- Si una persona claramente no tiene casco, se puede etiquetar `NO-Hardhat`.
- Si una persona claramente no tiene chaleco, se puede etiquetar `NO-Safety Vest`.

Objetivo inicial:

```text
100 imagenes etiquetadas = prueba rapida
300 imagenes etiquetadas = primera mejora seria
500 a 800 imagenes = mejora recomendada
```

## 4. Opcion A: LabelImg

### Instalar

Desde PowerShell, dentro del proyecto:

```powershell
.\.venv311\Scripts\pip.exe install labelImg pyqt5 lxml
```

### Abrir viendo errores

No abrir con doble clic. Ejecutar desde PowerShell para ver el error si se cierra:

```powershell
.\.venv311\Scripts\python.exe -m labelImg
```

Si ese comando falla, probar:

```powershell
.\.venv311\Scripts\labelImg.exe
```

### Configurar formato YOLO

Dentro de LabelImg:

1. Clic en `Open Dir`.
2. Elegir:

```text
C:\Users\SISTEMAS\Documents\epp_streamlit_app\datasets\cctv_epp\raw_frames
```

3. Clic en `Change Save Dir`.
4. Elegir o crear:

```text
C:\Users\SISTEMAS\Documents\epp_streamlit_app\datasets\cctv_epp\raw_labels
```

5. Cambiar formato a `YOLO`.
6. Dibujar cajas.
7. Guardar cada imagen con `Ctrl + S`.

Cada imagen debe generar un archivo `.txt` con el mismo nombre.

Ejemplo:

```text
cam9_video_000020.jpg
cam9_video_000020.txt
```

## 5. Si LabelImg se cierra solo

Probar estos pasos en orden.

### Paso 1: Ejecutarlo desde consola

```powershell
.\.venv311\Scripts\python.exe -m labelImg
```

Si se cierra, PowerShell debe mostrar el error real.

### Paso 2: Reinstalar dependencias

```powershell
.\.venv311\Scripts\pip.exe uninstall -y labelImg pyqt5 pyqt5-qt5 pyqt5-sip
.\.venv311\Scripts\pip.exe install labelImg pyqt5==5.15.10 lxml
```

Luego abrir:

```powershell
.\.venv311\Scripts\python.exe -m labelImg
```

### Paso 3: Ejecutar desde Python 3.11

El proyecto tiene `.venv311`, usar siempre:

```powershell
.\.venv311\Scripts\python.exe -m labelImg
```

Evitar mezclar con `.venv` si tiene otra version de Python.

### Paso 4: Alternativa si sigue fallando

Usar una herramienta web/local como:

- Roboflow Annotate
- CVAT
- makesense.ai

Para este proyecto, exportar siempre en formato:

```text
YOLOv8 / YOLO txt
```

## 6. Opcion B: makesense.ai

Si LabelImg sigue cerrandose, esta es la opcion mas simple.

1. Abrir:

```text
https://www.makesense.ai/
```

2. Cargar las imagenes de:

```text
datasets\cctv_epp\raw_frames
```

3. Crear estas clases:

```text
Hardhat
Mask
NO-Hardhat
NO-Mask
NO-Safety Vest
Person
Safety Cone
Safety Vest
machinery
vehicle
```

4. Etiquetar las imagenes.
5. Exportar como `YOLO`.
6. Colocar los `.txt` exportados en:

```text
datasets\cctv_epp\raw_labels
```

Importante: mantener los nombres de los `.txt` iguales a los `.jpg`.

## 7. Validacion antes de entrenar

Antes de entrenar, revisar:

- Cada `.jpg` etiquetado tiene su `.txt`.
- Los `.txt` no estan vacios si la imagen tiene personas/EPP.
- Las clases usadas son correctas.
- No se mezclaron nombres de clases diferentes.

## 8. Division del dataset

Despues de etiquetar, dividir:

```text
datasets/cctv_epp/train/images
datasets/cctv_epp/train/labels
datasets/cctv_epp/valid/images
datasets/cctv_epp/valid/labels
datasets/cctv_epp/test/images
datasets/cctv_epp/test/labels
```

Proporcion recomendada:

```text
70% train
20% valid
10% test
```

Ejemplo con 300 imagenes:

```text
210 train
60 valid
30 test
```

## 9. Entrenamiento recomendado

Cuando el dataset este dividido:

```powershell
.\.venv311\Scripts\python.exe train_yolo.py --data data_cctv_finetune.yaml --epochs 80 --model runs/detect/runs/train/css_v28_plus/weights/best.pt --imgsz 960 --batch 8 --name cctv_cam9_finetune
```

Si falta memoria GPU:

```powershell
.\.venv311\Scripts\python.exe train_yolo.py --data data_cctv_finetune.yaml --epochs 80 --model runs/detect/runs/train/css_v28_plus/weights/best.pt --imgsz 640 --batch 8 --name cctv_cam9_finetune
```

O:

```powershell
.\.venv311\Scripts\python.exe train_yolo.py --data data_cctv_finetune.yaml --epochs 80 --model runs/detect/runs/train/css_v28_plus/weights/best.pt --imgsz 960 --batch 4 --name cctv_cam9_finetune
```

## 10. Usar el modelo entrenado

Cuando termine el entrenamiento, usar en la app:

```text
runs/train/cctv_cam9_finetune/weights/best.pt
```

Si Ultralytics guarda dentro de `runs/detect/runs/train`, usar:

```text
runs/detect/runs/train/cctv_cam9_finetune/weights/best.pt
```

