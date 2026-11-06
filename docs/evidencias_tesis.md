# Evidencias técnicas del prototipo (sustento para la tesis)

Este documento reúne fragmentos de código reales del repositorio y salidas
capturadas de ejecuciones reales, listos para citarse como evidencia de
funcionamiento. Las rutas corresponden al repositorio
`Griimmj0w/Monitoreo-EPP`.

## 1. Inferencia y seguimiento

`core/detector.py` — envoltura de YOLOv8 con tracking BoT-SORT/ByteTrack:

```python
def infer(self, frame, conf=0.25, iou=0.5, tracker="botsort.yaml", persist=True, classes=None):
    results = self.model.track(frame, conf=conf, iou=iou,
                               tracker=tracker, persist=persist, ...)
    # normaliza cada deteccion a: clase, caja, track_id y confianza
    dets.append({"cls": cls_name, "bbox": (x1, y1, x2, y2),
                 "track_id": tid, "conf": score})
```

## 2. Asociación EPP–persona

`core/association.py`:

```python
def associate_items_to_persons(person_bbox, items, min_iou):
    matched = []
    relaxed_person_bbox = expanded_box(person_bbox)
    for ib, ic in items:
        if (inside(ib, person_bbox)
                or center_inside(ib, relaxed_person_bbox)
                or overlap_ratio(ib, relaxed_person_bbox) >= 0.45
                or iou(ib, person_bbox) >= min_iou):
            matched.append((ib, ic))
    return matched
```

## 3. Motor de reglas de cumplimiento

`app.py` — estados por persona según EPP detectados:

```python
if has_helmet and has_vest:
    status = "EPP CORRECTO"
elif not has_helmet and not has_vest:
    status = "Sin EPP"
else:
    status = "FALTA CASCO" if not has_helmet else "FALTA CHALECO"
```

## 4. Mitigación de falsos positivos

`core/events.py` — confirmación temporal (5 de 7 frames) y OK pegajoso:

```python
class TemporalConfirmer:
    def update(self, person_id, status):
        hist = self.history.setdefault(person_id, [])
        hist.append(status)
        if len(hist) > self.window:
            hist.pop(0)
        return self.confirmed(person_id)  # None si no se confirma

class EventManager:
    def should_emit(self, person_id, event_type=None):
        # cooldown por (persona, tipo de evento); evita spam
```

Los eventos se emiten únicamente en la transición OK → violación confirmada
(un incumplimiento continuo genera un único registro), y las personas con
confianza menor a 0.30 no se evalúan (puerta anti-detecciones fantasma).

## 5. Consolidación para análisis (Power BI)

`core/analytics.py` — un episodio por persona/estado (no una fila por frame):

```python
ANALYTICS_COLUMNS = ["event_id", "timestamp_inicio", "timestamp_fin", "fecha",
    "hora", "camera_id", "area", "track_id", "worker_id", "casco", "chaleco",
    "estado_epp", "detalle", "confidence", "duracion_evento_seg"]
# estado_epp ∈ {COMPLETO, PARCIAL, SIN EPP}; se escribe
# runs/analytics/ppe_monitoring_YYYYMMDD.csv (acumulativo por dia)
```

## 6. Video demo grabado

`app.py` — grabación del video procesado (detecciones dibujadas):

```python
demo_writer = cv2.VideoWriter(demo_path,
    cv2.VideoWriter_fourcc(*"mp4v"), max(1, int(fps_limit)), demo_out_size)
# ...por cada frame procesado:
demo_writer.write(out_frame)  # imagen con cajas, etiquetas y estados
```

## 7. Dataset v4 sin fuga temporal

`scripts/pseudo_label_v4.py` — particiones cronológicas, no aleatorias:

```python
# Split temporal: ordenar por indice de frame dentro del video y repartir
# en bloques cronologicos continuos (train -> val -> test).
images = sorted(images, key=frame_index)
# 70% train (frames 0-5160) | 20% val | 10% test
```

## 8. Salidas reales capturadas

**Inferencia real** (`exp-3.pt`, imgsz 960, imagen del split test):

```
person: 0.84 | vest: 0.80 | person: 0.74 | person: 0.67
helmet: 0.67 | vest: 0.66 | person: 0.63 | vest: 0.47 | helmet: 0.38
```

**CSV analítico real** (`runs/analytics/ppe_monitoring_20261006.csv`,
1,167 episodios consolidados):

```csv
event_id,timestamp_inicio,timestamp_fin,fecha,hora,track_id,casco,chaleco,estado_epp,detalle,confidence,duracion_evento_seg
EVT-000001,2026-10-06 09:43:14,2026-10-06 09:43:15,2026-10-06,09:43,25,SI,SI,COMPLETO,USO ADECUADO,0.39,1
EVT-000002,2026-10-06 09:43:15,2026-10-06 09:43:19,2026-10-06,09:43,25,NO,NO,SIN EPP,SIN CASCO Y CHALECO,0.75,4
EVT-000003,2026-10-06 09:43:23,2026-10-06 09:43:24,2026-10-06,09:43,37,SI,SI,COMPLETO,USO ADECUADO,0.57,1
```

**Métricas medidas** (validación del modelo `exp-3.pt` en Ultralytics HUB):

| Métrica | Baseline histórico | exp-3 (v4) |
|---|---|---|
| mAP@0.5 | 66.6 | **87.7** |
| mAP@0.5:0.95 | 34.0 | **44.5** |
| Precision | 79.2 | **91.2** |
| Recall | 52.2 | **84.3** |

**Video demo grabado por la aplicación** (`runs/demo/`): archivos
`demo_20261013_100106.mp4` (936 KB), `demo_20261013_100138.mp4` (936 KB)
y `demo_20261013_105104.mp4` (93 MB, sesión completa con detecciones).

## 9. Alcance futuro (declarado, no implementado)

Para citarse en la tesis como trabajo futuro:

- Persistencia en base de datos (SQLite) para eventos históricos.
- Canal externo de entrega de alertas (Telegram/Teams/correo).
- Controles de acceso y autenticación de usuarios del dashboard.
- Despliegue Edge: Raspberry Pi 5 + Coral USB (conversión TFLite int8 y
  compilación Edge TPU del modelo `exp-3.pt`).
- Clase `gloves` (en pausa por visibilidad en el video CCTV disponible).
- Estabilización del tracker (reducción de cambios de `track_id`).
