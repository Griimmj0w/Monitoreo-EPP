# -*- coding: utf-8 -*-

import streamlit as st
import cv2
import pandas as pd
import time
import os
import re

from core.config import Config, CONFIG
from core.detector import YoloTracker
from core.association import split_detections, associate_items_to_persons, iou as bbox_iou
from core.id_reader import HelmetTagReader
from core.events import EventManager


st.set_page_config(page_title="Monitoreo Inteligente de EPP", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top, rgba(46, 204, 113, 0.16), transparent 26%),
            linear-gradient(180deg, #0e1722 0%, #0c1520 45%, #08111a 100%);
        color: #edf4ff;
    }

    section[data-testid="stSidebar"] {
        background: #0b141f;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        color: #ffffff;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        color: rgba(237, 244, 255, 0.7);
        margin-bottom: 1rem;
    }

    .status-banner {
        border-radius: 18px;
        padding: 1rem 1.2rem;
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.24);
        margin-bottom: 1rem;
    }

    .status-ok {
        background: linear-gradient(90deg, #2ea44f 0%, #1f7a39 100%);
        color: white;
    }

    .status-alert {
        background: linear-gradient(90deg, #d12d34 0%, #a4171d 100%);
        color: white;
    }

    .summary-panel, .history-panel, .action-panel {
        background: rgba(15, 24, 37, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
    }

    .summary-item {
        font-size: 1rem;
        padding: 0.2rem 0;
        color: rgba(238, 245, 255, 0.95);
    }

    .summary-item strong {
        color: #ffffff;
        font-weight: 800;
    }

    .history-title {
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
        color: rgba(238, 245, 255, 0.9);
    }

    .history-log {
        max-height: 260px;
        overflow-y: auto;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.65rem;
        font-family: Consolas, Monaco, 'Courier New', monospace;
        font-size: 0.88rem;
        color: rgba(232, 239, 248, 0.95);
        white-space: pre-wrap;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, #e53137 0%, #b41017 100%);
        color: white;
        font-weight: 800;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.2);
    }

    div.stButton > button:hover {
        background: linear-gradient(180deg, #f03a40 0%, #c5131b 100%);
        border-color: rgba(255, 255, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">MONITOREO INTELIGENTE DE EPP\'S</div>', unsafe_allow_html=True)

STATUS_OK = "EPP CORRECTO"
STATUS_NO_EPP = "Sin EPP"
STATUS_NO_HELMET = "FALTA CASCO"
STATUS_NO_VEST = "FALTA CHALECO"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_upload_name(filename):
    base = os.path.basename(filename or "archivo")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "archivo"


def media_kind(path):
    ext = os.path.splitext(path or "")[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


class StaticImageCapture:
    """Pequeno adaptador para procesar una imagen como si fuera una fuente de video."""

    def __init__(self, path):
        self.frame = cv2.imread(path)
        self._served = False

    def isOpened(self):
        return self.frame is not None

    def read(self):
        if self.frame is None or self._served:
            return False, None
        self._served = True
        return True, self.frame.copy()

    def release(self):
        pass


def resize_for_inference(frame, max_width):
    if not max_width or max_width <= 0:
        return frame, 1.0, 1.0
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, width / float(resized.shape[1]), height / float(resized.shape[0])


def scale_detections(dets, scale_x, scale_y):
    if scale_x == 1.0 and scale_y == 1.0:
        return dets
    scaled = []
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        item = dict(det)
        item["bbox"] = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )
        scaled.append(item)
    return scaled

# Funcion para detectar camaras disponibles
@st.cache_data(ttl=60)
def detect_available_cameras(max_cameras=10):
    """Detecta las camaras disponibles en el sistema"""
    available_cameras = []
    detected_indices = set()
    
    # Probar cada indice con el mejor backend para Windows (DSHOW primero)
    backends_to_try = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    
    for i in range(max_cameras):
        if i in detected_indices:
            continue
            
        for backend_name, backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Obtener informacion de la camara
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # Determinar tipo de camara basado en el indice
                        cam_type = "Integrada" if i == 0 else f"USB Externa"
                        
                        camera_info = {
                            "index": i,
                            "backend": backend_name,
                            "resolution": f"{width}x{height}",
                            "fps": fps,
                            "label": f"Camara {i} - {cam_type} ({backend_name}) - {width}x{height}"
                        }
                        available_cameras.append(camera_info)
                        detected_indices.add(i)
                        cap.release()
                        break  # Si funciona con este backend, no probar otros
                    cap.release()
            except Exception as e:
                continue
    
    return available_cameras

with st.sidebar:
    st.header("Configuracion")
    model_path = st.text_input(
        "Ruta del modelo YOLOv8 (.pt)",
        value="runs/detect/runs/train/css_v28_plus/weights/best.pt"
    )


    source_type = st.selectbox("Fuente", ["Webcam", "RTSP", "Video / imagen (archivo)"])

    rstp_url = ""
    video_path = ""
    cam_index = 0
    cam_backend = "DSHOW"

    if source_type == "Webcam":
        st.markdown("**Deteccion de camaras:**")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Buscar camaras disponibles"):
                st.cache_data.clear()
        
        with col_btn2:
            show_debug = st.checkbox("Modo debug", value=False)
        
        available_cameras = detect_available_cameras()
        
        if available_cameras:
            st.success(f"[OK] {len(available_cameras)} camara(s) detectada(s)")
            
            # Modo debug: mostrar detalles de todas las camaras
            if show_debug:
                st.write("**Camaras encontradas:**")
                for cam in available_cameras:
                    st.text(f"Indice: {cam['index']}, Backend: {cam['backend']}, "
                           f"Res: {cam['resolution']}, FPS: {cam['fps']}")
            
            # Crear opciones para el selectbox
            camera_options = {}
            for cam in available_cameras:
                camera_options[cam["label"]] = cam
            
            selected_camera_label = st.selectbox(
                "Seleccionar camara",
                options=list(camera_options.keys()),
                help="Indice 0 = Camara integrada | Indice 1+ = Camaras USB/Externas"
            )
            
            selected_cam = camera_options[selected_camera_label]
            cam_index = selected_cam["index"]
            cam_backend = selected_cam["backend"]
            
            # Mostrar info de la camara seleccionada
            cam_type_desc = "Camara integrada del laptop" if cam_index == 0 else f"Camara USB externa (puerto {cam_index})"
            st.info(f">> CAMARA ACTIVA: Indice {cam_index}\n\n"
                   f"Tipo: {cam_type_desc}\n\n"
                   f"Backend: {cam_backend}\n\n"
                   f"Resolucion: {selected_cam['resolution']}\n\n"
                   f"FPS: {selected_cam['fps']}\n\n"
                   f"Usa 'Test camara' abajo para verificar cual camara es")
        else:
            st.warning("[!] No se detectaron camaras. Configuracion manual:")
            cam_index = st.number_input("Indice de camara", min_value=0, max_value=10, value=0, step=1)
            cam_backend = st.selectbox("Backend camara", ["AUTO", "DSHOW", "MSMF"], index=1)
        
        if st.button("Test camara"):
            st.write(f"Probando camara con indice {cam_index} usando backend {cam_backend}...")
            backend = cv2.CAP_ANY
            if cam_backend == "DSHOW":
                backend = cv2.CAP_DSHOW
            elif cam_backend == "MSMF":
                backend = cv2.CAP_MSMF
            
            cap_test = cv2.VideoCapture(int(cam_index), backend)

            ok = cap_test.isOpened()
            frame = None
            if ok:
                ok, frame = cap_test.read()
            if ok and frame is not None:
                cam_desc = "integrada" if cam_index == 0 else f"externa USB (indice {cam_index})"
                st.success(f"[OK] Camara {cam_desc} funcionando!")
                st.caption("Si esta NO es la camara correcta, selecciona otra de la lista arriba")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            else:
                st.error(f"[ERROR] No se pudo leer la camara {cam_index}. Prueba otro indice o backend.")
            cap_test.release()
    elif source_type == "RTSP":
        rstp_url = st.text_input("URL RTSP", value="rtsp://username:password@ip_address:port/stream")
    else:
        uploaded_media = st.file_uploader(
            "Subir video o imagen",
            type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)],
            help="Puedes subir un video grabado o una imagen para probar la deteccion de EPP.",
        )
        if uploaded_media is not None:
            upload_dir = os.path.join("runs", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            safe_name = safe_upload_name(uploaded_media.name)
            video_path = os.path.join(upload_dir, safe_name)
            with open(video_path, "wb") as fh:
                fh.write(uploaded_media.getbuffer())
            st.session_state.uploaded_media_path = video_path
            st.success(f"Archivo cargado: {safe_name}")

        saved_upload = st.session_state.get("uploaded_media_path", "")
        default_media_path = saved_upload if saved_upload else "video.mp4"
        video_path = st.text_input(
            "O escribe la ruta del video / imagen",
            value=default_media_path,
            help="Ejemplo: C:\\videos\\prueba.mp4 o una imagen .jpg/.png",
        )

    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"])
    
    st.markdown("---")
    st.markdown("**Ajustes de deteccion:**")
    conf = st.slider("Confianza (conf)", 0.05, 0.90, float(Config.DEFAULT_CONF), 0.05,
                    help="Menor = detecta mas pero con mas falsos positivos. Mayor = mas preciso pero puede perder objetos")
    iou = st.slider("IOU NMS,(iou)", 0.10, 0.90, float(Config.DEFAULT_IOU), 0.05,
                   help="Umbral para eliminar detecciones duplicadas")

    min_iou_item = st.slider("Iou minimo item-persona", 0.00, 0.20, float(Config.MIN_IOU_PERSON_ITEM), 0.01)
    
    st.markdown("---")
    st.markdown("**Ajustes de camara (solo Webcam/USB):**")
    adjust_camera = st.checkbox("Ajustar propiedades de camara", value=False,
                                help="Permite modificar brillo, contraste, etc. de la camara")
    
    cam_brightness = 128
    cam_contrast = 128
    cam_saturation = 128
    cam_auto_exposure = True
    
    if adjust_camera and source_type == "Webcam":
        cam_brightness = st.slider("Brillo", 0, 255, 128,
                                   help="Ajusta si la imagen esta muy oscura o muy clara")
        cam_contrast = st.slider("Contraste", 0, 255, 128,
                                 help="Aumenta la diferencia entre zonas claras y oscuras")
        cam_saturation = st.slider("Saturacion", 0, 255, 128,
                                   help="Intensidad de los colores")
        cam_auto_exposure = st.checkbox("Auto exposicion", value=True,
                                       help="Dejar que la camara ajuste la exposicion automaticamente")
    
    st.markdown("---")

    enable_qr = st.checkbox("Leer QR en etiqueta (helmet_tag)", value=False)
    preview_enabled = st.checkbox("Mostrar camara en vivo", value=False)
    show_items_without_person = st.checkbox("Mostrar items sin persona", value=False)
    show_all_dets = st.checkbox("Mostrar todas las detecciones", value=False)
    show_debug_info = st.checkbox("Mostrar info de debug", value=False,
                                  help="Muestra estadisticas de deteccion en tiempo real")

    cooldown = st.number_input("Cooldown eventos (segundos)", min_value=0.0, max_value=60.0, value=float(Config.EVENT_COOLDOWN_SEC), step=0.5)
    fps_limit = st.number_input("FPS limite", min_value=1, max_value=60, value=10, step=1)
    process_every_n = st.number_input(
        "Procesar cada N frames",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Mayor valor = video mas fluido y avance mas rapido, pero menos frames analizados.",
    )
    inference_width = st.select_slider(
        "Ancho maximo de inferencia",
        options=[320, 480, 640, 800, 960, 1280],
        value=640,
        help="Menor valor = mas velocidad. Si pierdes cascos/chalecos pequenos, subelo a 800 o 960.",
    )

    st.markdown("---")
    st.markdown("**Tamaño del visor:**")
    viewer_width = st.slider(
        "Ancho de la ventana de deteccion (px)",
        min_value=400,
        max_value=1200,
        value=700,
        step=50,
        help="Ajusta el ancho de la imagen mostrada sin cambiar la resolucion de deteccion.",
    )

    # Dataset helper: detect 'archive' folder and prepare data yaml
    if os.path.isdir("archive"):
        st.markdown("**Dataset encontrado:** `archive`")
        ds_nc = st.number_input("Numero de clases (nc)", min_value=1, max_value=50, value=2, step=1)
        ds_names = st.text_input("Nombres de clases (separados por coma)", value="person,helmet")
        if st.button("Preparar data_template.yaml desde 'archive'"):
            # detect common structure
            archive = "archive"
            if os.path.isdir(os.path.join(archive, "train", "images")) and os.path.isdir(os.path.join(archive, "val", "images")):
                train_p = f"{archive}/train/images"
                val_p = f"{archive}/val/images"
            elif os.path.isdir(os.path.join(archive, "images")):
                train_p = f"{archive}/images"
                val_p = f"{archive}/val/images" if os.path.isdir(os.path.join(archive, "val", "images")) else train_p
            else:
                # fallback to archive root
                train_p = f"{archive}"
                val_p = f"{archive}"

            names_list = [n.strip() for n in ds_names.split(",") if n.strip()]
            # write simple yaml
            yaml_content = []
            yaml_content.append(f"train: {train_p}")
            yaml_content.append(f"val: {val_p}")
            yaml_content.append("")
            yaml_content.append(f"nc: {int(ds_nc)}")
            yaml_content.append("")
            names_str = ", ".join(f'"{n}"' for n in names_list)
            yaml_content.append(f"names: [{names_str}]")

            with open("data_template.yaml", "w", encoding="utf-8") as fh:
                fh.write("\n".join(yaml_content))

            st.success(f"data_template.yaml escrito. train={train_p} val={val_p} nc={ds_nc}")

if "running" not in st.session_state:
    st.session_state.running = False
if "events" not in st.session_state:
    st.session_state.events = []
if "track_to_worker" not in st.session_state:
    st.session_state.track_to_worker = {}
if "preview_enabled" not in st.session_state:
    st.session_state.preview_enabled = True
if "track_status" not in st.session_state:
    st.session_state.track_status = {}
if "last_export_path" not in st.session_state:
    st.session_state.last_export_path = ""
if "dashboard_state" not in st.session_state:
    st.session_state.dashboard_state = {}
if "uploaded_media_path" not in st.session_state:
    st.session_state.uploaded_media_path = ""

# Zona principal: deteccion y registro siempre visibles en la parte superior.
# El control de tamaño modifica la proporcion de las columnas; la imagen se
# ajusta a su columna para que nunca invada la tabla.
viewer_col, registry_col = st.columns([viewer_width, 500], gap="large")
with viewer_col:
    st.subheader("Deteccion en vivo")
    frame_slot = st.empty()
with registry_col:
    st.subheader("Registro de detecciones")
    table_slot = st.empty()

summary_placeholder = st.empty()
history_placeholder = st.empty()

st.markdown('<div class="action-panel">', unsafe_allow_html=True)
start = st.button("Iniciar camara", key="start_camera", use_container_width=True)
load_media = st.button("Cargar video / imagen", key="load_media", use_container_width=True)
export_evidence = st.button("Exportar Evidencia", key="export_evidence", use_container_width=True)
stop = st.button("Detener", key="stop_camera", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if start:
    st.session_state.running = True
    st.session_state.preview_enabled = True
if stop:
    st.session_state.running = False
    st.session_state.preview_enabled = False
if load_media:
    if source_type == "Webcam":
        st.warning("Selecciona 'Iniciar camara' para usar la webcam.")
    elif source_type == "RTSP" and not rstp_url.strip():
        st.warning("Ingresa una URL RTSP antes de cargar la fuente.")
    elif source_type == "Video / imagen (archivo)" and not video_path.strip():
        st.warning("Sube un archivo o ingresa la ruta del video / imagen antes de cargarlo.")
    elif source_type == "Video / imagen (archivo)" and not os.path.isfile(video_path.strip()):
        st.warning("No se encontro el archivo. Revisa la ruta o vuelve a subirlo.")
    elif source_type == "Video / imagen (archivo)" and media_kind(video_path.strip()) == "unknown":
        st.warning("Formato no reconocido. Usa video mp4/avi/mov/mkv/wmv/m4v o imagen jpg/png/bmp/webp.")
    else:
        st.session_state.running = True
        st.session_state.preview_enabled = True
if preview_enabled and not st.session_state.running:
    st.session_state.preview_enabled = True
elif not st.session_state.running:
    st.session_state.preview_enabled = False

def export_events_csv():
    export_dir = os.path.join("runs", "evidence")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"evidence_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(st.session_state.events).to_csv(export_path, index=False, encoding="utf-8-sig")
    return export_path


def render_summary_panel(state):
    persons_count = state.get("persons_count", 0)
    helmet_state = state.get("helmet_state", "N/D")
    vest_state = state.get("vest_state", "N/D")
    current_time = state.get("current_time", time.strftime("%H:%M:%S"))
    overall_status = state.get("overall_status", "ESTADO: ESPERANDO CAMARA")
    status_class = state.get("status_class", "alert")

    return f"""
    <div class="status-banner status-{status_class}">{overall_status}</div>
    <div class="summary-panel">
        <div class="summary-item"><strong>Personas detectadas:</strong> {persons_count}</div>
        <div class="summary-item"><strong>Casco:</strong> {helmet_state}</div>
        <div class="summary-item"><strong>Chaleco:</strong> {vest_state}</div>
        <div class="summary-item"><strong>Hora:</strong> {current_time}</div>
    </div>
    """


def render_history_panel(events_list):
    if not events_list:
        body = "Sin alertas registradas todavía."
    else:
        recent = events_list[-8:]
        body_lines = []
        for event in recent:
            timestamp = event.get("timestamp", "")
            status = event.get("status", "")
            worker_id = event.get("worker_id", "")
            track_id = event.get("track_id", "")
            line = f"[{timestamp}] Falta: {status}"
            if worker_id:
                line += f" | Trabajador: {worker_id}"
            if track_id != "":
                line += f" | ID: {track_id}"
            body_lines.append(line)
        body = "\n".join(body_lines)

    return f"""
    <div class="history-panel">
        <div class="history-title">Historial de alertas</div>
        <div class="history-log">{body}</div>
    </div>
    """


if export_evidence:
    if st.session_state.events:
        st.session_state.last_export_path = export_events_csv()
        st.success(f"Evidencia exportada en {st.session_state.last_export_path}")
    else:
        st.warning("No hay alertas para exportar.")

summary_placeholder.markdown(
    render_summary_panel(
        {
            "persons_count": 0,
            "helmet_state": "N/D",
            "vest_state": "N/D",
            "current_time": time.strftime("%H:%M:%S"),
            "overall_status": "ESTADO: ESPERANDO CAMARA",
            "status_class": "alert",
        }
    ),
    unsafe_allow_html=True,
)
history_placeholder.markdown(render_history_panel(st.session_state.events), unsafe_allow_html=True)

# Slot para informacion de debug
debug_slot = st.empty() if show_debug_info else None

def load_model(path):
    return YoloTracker(path)

def draw_status_box(frame, bbox, label, color, thickness=2):
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    pad = 4
    text_x1 = x1
    text_y1 = max(0, y1 - th - baseline - (pad * 2))
    text_x2 = min(frame.shape[1] - 1, x1 + tw + (pad * 2))
    text_y2 = y1

    cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), color, -1)
    text_org = (text_x1 + pad, text_y2 - baseline - pad)
    cv2.putText(frame, label, text_org, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

try:
    yolo = load_model(model_path)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

tag_reader = HelmetTagReader()
events = EventManager(cooldown_sec=float(cooldown))

with st.sidebar:
    st.markdown("**EPPs a detectar**")
    model_names = []
    try:
        if getattr(yolo, "names", None):
            model_names = [str(v) for k, v in sorted(yolo.names.items())]
    except Exception:
        model_names = []

    def _match_any(name, keywords):
        n = name.lower()
        return any(k in n for k in keywords)

    HELMET_KEYS = ["helmet", "hardhat", "hard hat", "casco"]
    VEST_KEYS = ["vest", "chaleco"]

    def _pick_one(names, keywords, fallback):
        for n in names:
            if _match_any(n, keywords):
                return n
        return fallback if fallback in names else None

    def _is_negative(name):
        n = name.lower().replace("_", " ").replace("-", " ")
        return n.startswith("no ") or n.startswith("sin ")

    def _pick_classes(names, keywords):
        return [n for n in names if _match_any(n, keywords) and not _is_negative(n)]

    cls_person = _pick_one(model_names, ["person", "persona", "worker", "trabajador"], Config.CLASS_PERSON)
    cls_tag = _pick_one(model_names, ["tag", "qr", "label", "etiqueta"], Config.CLASS_TAG)

    if model_names:
        # Solo mantener EPPs positivos relevantes (casco y chaleco) y ocultar el resto.
        all_candidates = [n for n in model_names if n not in {cls_person, cls_tag}]

        helmet_only = _pick_classes(all_candidates, HELMET_KEYS)
        vest_only = _pick_classes(all_candidates, VEST_KEYS)
        epp_candidates = list(dict.fromkeys(helmet_only + vest_only))

        # Fallback: si el modelo no tiene clases compatibles, no bloquear la app.
        if not epp_candidates:
            epp_candidates = all_candidates if all_candidates else model_names
            st.warning("No se encontraron clases tipo casco/chaleco en el modelo; mostrando todas las clases disponibles.")

        selected_epps = st.multiselect("EPPs a detectar", epp_candidates, default=epp_candidates)
    else:
        st.caption("No se pudieron leer las clases del modelo.")
        selected_epps = []

    filter_inference = True
    filter_classes = []
    if model_names:
        filter_classes = list(dict.fromkeys(selected_epps))
        if cls_person and cls_person not in filter_classes:
            filter_classes.append(cls_person)
        if enable_qr and cls_tag and cls_tag not in filter_classes:
            filter_classes.append(cls_tag)

    cls_helmet = _pick_classes(selected_epps, HELMET_KEYS)
    cls_vest = _pick_classes(selected_epps, VEST_KEYS)

    prioritize_positive = st.checkbox("Priorizar clases positivas", value=True)

def _suppress_negative_overlaps(dets, positive_names, keywords, iou_thresh=0.2):
    if not positive_names:
        return dets
    pos_boxes = [d["bbox"] for d in dets if d["cls"] in positive_names]
    if not pos_boxes:
        return dets
    filtered = []
    for d in dets:
        name = d["cls"]
        if _is_negative(name) and _match_any(name, keywords):
            if any(bbox_iou(d["bbox"], pb) >= iou_thresh for pb in pos_boxes):
                continue
        filtered.append(d)
    return filtered

def open_capture():
    if source_type == "Webcam":
        backend = cv2.CAP_ANY
        if cam_backend == "DSHOW":
            backend = cv2.CAP_DSHOW
        elif cam_backend == "MSMF":
            backend = cv2.CAP_MSMF
        cap = cv2.VideoCapture(int(cam_index), backend)
        
        # Aplicar ajustes basicos de camara
        if cap.isOpened():
            try:
                # Intentar configurar resolucion mas alta
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Si adjust_camera esta activado, aplicar ajustes personalizados
                if adjust_camera:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, cam_brightness)
                    cap.set(cv2.CAP_PROP_CONTRAST, cam_contrast)
                    cap.set(cv2.CAP_PROP_SATURATION, cam_saturation)
                    if cam_auto_exposure:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    else:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except:
                pass
        
        return cap
    if source_type == "RTSP":
        return cv2.VideoCapture(rstp_url)
    selected_path = video_path.strip()
    if media_kind(selected_path) == "image":
        return StaticImageCapture(selected_path)
    return cv2.VideoCapture(selected_path)

cap = None
if st.session_state.running or st.session_state.preview_enabled:
    cap = open_capture()
    if cap is None or not cap.isOpened():
        st.error("No se pudo abrir la fuente de video. Revisa el indice, permisos o si otra app usa la camara.")
        st.session_state.running = False
        st.session_state.preview_enabled = False
    else:
        # Dar tiempo a la camara para inicializar (especialmente USB)
        time.sleep(0.5)
        # Leer y descartar los primeros frames (pueden estar vacios)
        for _ in range(5):
            cap.read()

last_time = 0.0
fail_count = 0
consecutive_fails = 0
frame_index = 0

while st.session_state.running or st.session_state.preview_enabled:
    ok, frame = cap.read()
    if not ok or frame is None:
        if source_type == "Video / imagen (archivo)":
            st.info("Se termino de procesar el archivo cargado.")
            st.session_state.running = False
            st.session_state.preview_enabled = False
            break
        consecutive_fails += 1
        # Solo mostrar warning despues de varios fallos consecutivos
        if consecutive_fails >= 3:
            fail_count += 1
            if fail_count == 1:
                st.warning("No se pudo leer el frame. Verifique la fuente de video.")
            if cap is not None:
                cap.release()
            time.sleep(0.5)
            cap = open_capture()
            # Dar tiempo a reinicializar
            time.sleep(0.3)
            # Descartar primeros frames
            for _ in range(3):
                cap.read()
            consecutive_fails = 0
            if fail_count >= 5:
                st.error("No se pudo leer el frame despues de varios intentos. Revisa la fuente.")
                break
        continue
    frame_index += 1
    if st.session_state.running and process_every_n > 1 and (frame_index % int(process_every_n)) != 0:
        continue
    
    # Frame leido exitosamente
    fail_count = 0
    consecutive_fails = 0
    now = time.time()
    if now - last_time < (1.0 / max(1, fps_limit)):
        continue
    last_time = now

    if st.session_state.running:
        now_ts = time.time()
        class_name_to_id = {}
        if getattr(yolo, "names", None):
            try:
                class_name_to_id = {str(v): int(k) for k, v in yolo.names.items()}
            except Exception:
                class_name_to_id = {}
        class_ids = []
        if filter_inference and filter_classes and class_name_to_id:
            class_ids = [class_name_to_id[n] for n in filter_classes if n in class_name_to_id]

        inference_frame, scale_x, scale_y = resize_for_inference(frame, int(inference_width))
        dets = yolo.infer(
            inference_frame,
            conf=float(conf),
            iou=float(iou),
            tracker=tracker,
            persist=True,
            classes=class_ids if class_ids else None,
        )
        dets = scale_detections(dets, scale_x, scale_y)

        if prioritize_positive:
            dets = _suppress_negative_overlaps(dets, cls_helmet, HELMET_KEYS)
            dets = _suppress_negative_overlaps(dets, cls_vest, VEST_KEYS)
        
        # Mostrar info de debug si esta habilitado
        if debug_slot is not None:
            frame_h, frame_w = frame.shape[:2]
            
            # Contar tipos de detecciones
            det_types = {}
            for d in dets:
                cls_name = d["cls"]
                det_types[cls_name] = det_types.get(cls_name, 0) + 1
            
            det_summary = ", ".join([f"{k}: {v}" for k, v in det_types.items()]) if det_types else "Ninguna"
            
            debug_info = f"""
            **Info de Debug - Frame actual:**
            - Resolucion: {frame_w}x{frame_h}
            - Total detecciones: {len(dets)}
            - Detecciones por tipo: {det_summary}
            - Confianza minima (conf): {conf}
            - IOU threshold: {iou}
            - Fuente: {source_type if source_type != "Webcam" else f"Camara indice {cam_index} ({cam_backend})"}
            - Procesando cada {int(process_every_n)} frame(s)
            - Ancho inferencia: {int(inference_width)} px
            
            **Sugerencias:**
            - Si no detecta personas: Baja 'conf' a 0.25-0.30
            - Si imagen oscura: Activa 'Ajustar propiedades' y sube Brillo
            """
            debug_slot.info(debug_info)

        persons, helmets, vests, tags = split_detections(dets,
            cls_person,
            cls_helmet,
            cls_vest,
            cls_tag
        )

        require_helmet = len(cls_helmet) > 0
        require_vest = len(cls_vest) > 0
        use_epp = len(persons) > 0 and (require_helmet or require_vest)

        if use_epp:
            if show_all_dets:
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    label = f'{d["cls"]} {d["conf"]:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for tid, pb, pconfig in persons:
                matched_helmets = associate_items_to_persons(
                    pb, helmets, min_iou_item)
                matched_vests = associate_items_to_persons(
                    pb, vests, min_iou_item)
                matched_tags = associate_items_to_persons(
                    pb, tags, min_iou_item)

                has_helmet = len(matched_helmets) > 0 if require_helmet else True
                has_vest = len(matched_vests) > 0 if require_vest else True

                worker_id = st.session_state.track_to_worker.get(tid)

                if enable_qr and worker_id is None and len(matched_tags) > 0:
                    tb, tconf = matched_tags[0]
                    x1, y1, x2, y2 = tb
                    crop = frame[max(0, y1): max(0, y2), max(0, x1): max(0, x2)]
                    decoded = tag_reader.read_qr_from_crop(crop)
                    if decoded:
                        st.session_state.track_to_worker[tid] = decoded
                        worker_id = decoded

                if not require_helmet and not require_vest:
                    status = STATUS_OK
                    box_color = (0, 255, 0)
                elif require_helmet and not require_vest:
                    status = STATUS_OK if has_helmet else STATUS_NO_HELMET
                    box_color = (0, 255, 0) if has_helmet else (0, 0, 255)
                elif require_vest and not require_helmet:
                    status = STATUS_OK if has_vest else STATUS_NO_VEST
                    box_color = (0, 255, 0) if has_vest else (0, 0, 255)
                else:
                    if has_helmet and has_vest:
                        status = STATUS_OK
                        box_color = (0, 255, 0)
                    elif not has_helmet and not has_vest:
                        status = STATUS_NO_EPP
                        box_color = (0, 0, 255)
                    else:
                        status = STATUS_NO_HELMET if not has_helmet else STATUS_NO_VEST
                        box_color = (0, 165, 255)

                if tid >= 0 and status != STATUS_OK and events.should_emit(tid, status):
                    st.session_state.events.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "track_id": tid,
                        "worker_id": worker_id or "",
                        "status": status
                    })
                if tid >= 0:
                    st.session_state.track_status[tid] = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "track_id": tid,
                        "worker_id": worker_id or "",
                        "status": status,
                        "last_seen": now_ts,
                    }

                x1, y1, x2, y2 = pb
                label = f"ID:{tid}"
                if worker_id:
                    label += f" Trabajador:{worker_id}"
                label += f" {status}"

                draw_status_box(frame, (x1, y1, x2, y2), label, box_color, thickness=2)

            if show_items_without_person:
                for ib, _ in helmets:
                    x1, y1, x2, y2 = ib
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "casco", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                for ib, _ in vests:
                    x1, y1, x2, y2 = ib
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "chaleco", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                label = f'{d["cls"]} {d["conf"]:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if st.session_state.running:
        if st.session_state.track_status:
            pruned = {
                tid: v for tid, v in st.session_state.track_status.items()
                if now_ts - v.get("last_seen", now_ts) <= 3.0
            }
            st.session_state.track_status = pruned
        status_rows = [
            {k: v for k, v in row.items() if k != "last_seen"}
            for row in st.session_state.track_status.values()
        ]
        if status_rows:
            df = pd.DataFrame(status_rows).sort_values("track_id")
            table_slot.dataframe(df, use_container_width=True, height=600)
        else:
            table_slot.info("No hay personas detectadas.")

        persons_count = len(persons)
        helmet_state = "OK" if not require_helmet or all(len(associate_items_to_persons(pb, helmets, min_iou_item)) > 0 for _, pb, _ in persons) else "FALTA"
        vest_state = "OK" if not require_vest or all(len(associate_items_to_persons(pb, vests, min_iou_item)) > 0 for _, pb, _ in persons) else "FALTA"

        if persons_count == 0:
            overall_status = "ESTADO: SIN PERSONAS"
            status_class = "alert"
        elif all(row.get("status") == STATUS_OK for row in st.session_state.track_status.values()):
            overall_status = "ESTADO: CUMPLE EPP CRITICO"
            status_class = "ok"
        else:
            overall_status = "ESTADO: ALERTA EPP CRITICO"
            status_class = "alert"

        st.session_state.dashboard_state = {
            "persons_count": persons_count,
            "helmet_state": helmet_state,
            "vest_state": vest_state,
            "current_time": time.strftime("%H:%M:%S"),
            "overall_status": overall_status,
            "status_class": status_class,
        }

        summary_placeholder.markdown(
            render_summary_panel(st.session_state.dashboard_state),
            unsafe_allow_html=True,
        )
        history_placeholder.markdown(render_history_panel(st.session_state.events), unsafe_allow_html=True)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if st.session_state.running:
        # Agregar informacion en el frame
        cv2.putText(rgb, f"Detecciones: {len(dets)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        source_label = f"Camara: {cam_index} ({cam_backend})" if source_type == "Webcam" else "Archivo cargado"
        cv2.putText(rgb, source_label, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar advertencia si no hay detecciones
        if len(dets) == 0:
            cv2.putText(rgb, "SIN DETECCIONES - Revisa conf/iluminacion", 
                       (10, rgb.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    frame_slot.image(rgb, channels="RGB", use_column_width=True)

if cap is not None:
    cap.release()
