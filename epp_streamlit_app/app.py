
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

# Keywords para identificar clases de cascos y chalecos
HELMET_KEYS = ["helmet", "casco", "hard hat", "hardhat", "hard-hat", "gorro"]
VEST_KEYS = ["vest", "chaleco", "reflective", "safety vest", "safetyvest"]
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
    """Adaptador simple para analizar una imagen dentro del flujo de video."""

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


st.set_page_config(
    page_title="Monitoreo de EPPs",
    layout="wide",  # Crucial para el diseño de escritorio
    initial_sidebar_state="collapsed",
)

st.title("Deteccion de EPP (Casco + Chaleco) con YOLOv8")
st.caption("MVP Streamlit: tracking + reglas + identificacion por tag (QR) opcional")

# Estilo industrial y botones
st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.stButton>button { border-radius: 5px; height: 3em; font-weight: bold; }
.stImage { border: 2px solid #262730; border-radius: 10px; }
.action-panel { background: rgba(15,24,37,0.92); border-radius:12px; padding:8px; }
.status-banner {
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 1.1rem;
    font-weight: 900;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.status-ok {
    background: linear-gradient(90deg, #2ea44f 0%, #1f7a39 100%);
    color: #ffffff;
}
.status-alert {
    background: linear-gradient(90deg, #d12d34 0%, #a4171d 100%);
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Ajustes responsivos adicionales para que el video se ajuste a la pantalla
st.markdown("""
<style>
/* Asegura que las imágenes renderizadas por st.image escalen correctamente */
.stImage img { width: 100% !important; height: auto !important; max-height: calc(100vh - 260px) !important; object-fit: contain; }
/* Si hay un contenedor de video, limitar su altura y permitir scroll interno si es necesario */
.video-container { max-height: calc(100vh - 260px); overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Layout responsivo: video a la izquierda, métricas a la derecha (se apilará en móvil)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Video Feed en Tiempo Real")
    video_placeholder = st.empty()
with col2:
    m1, m2 = st.columns(2)
    status_placeholder = st.empty()
    persons_metric_placeholder = m1.empty()
    infractions_metric_placeholder = m2.empty()
    st.markdown("### Historial de Detecciones Recientes")
    try:
        df_detecciones = pd.DataFrame(st.session_state.get("events", []))
        st.dataframe(df_detecciones, use_container_width=True)
    except Exception:
        st.write("No hay detecciones aún")
    debug_placeholder = st.empty()

st.markdown(
    """
    <style>
    .action-panel {
        background: rgba(15, 24, 37, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 0.8rem 0.8rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
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

    if source_type == "Webcam":
        cam_index = st.number_input("indice de camara", min_value=0, max_value=10, value=0, step=1)
        cam_backend = st.selectbox("Backend camara", ["AUTO", "DSHOW", "MSMF"], index=1)
        if st.button("Test camara"):
            backend = 0
            if cam_backend == "DSHOW":
                backend = cv2.CAP_DSHOW
            elif cam_backend == "MSMF":
                backend = cv2.CAP_MSMF
            if backend:
                cap_test = cv2.VideoCapture(int(cam_index), backend)
            else:
                cap_test = cv2.VideoCapture(int(cam_index))

            ok = cap_test.isOpened()
            frame = None
            if ok:
                ok, frame = cap_test.read()
            if ok and frame is not None:
                st.success("Camara OK")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            else:
                st.error("No se pudo leer la camara. Prueba otro indice o backend.")
            cap_test.release()
    elif source_type == "RTSP":
        rstp_url = st.text_input("URL RTSP", value="rtsp://username:password@ip_address:port/stream")
    else:
        uploaded_media = st.file_uploader(
            "Subir video o imagen",
            type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)],
            help="Sube un video grabado o una imagen para probar la deteccion de EPP.",
        )
        if uploaded_media is not None:
            save_dir = os.path.join("runs", "uploads")
            os.makedirs(save_dir, exist_ok=True)
            safe_name = safe_upload_name(uploaded_media.name)
            save_path = os.path.join(save_dir, safe_name)
            with open(save_path, "wb") as fh:
                fh.write(uploaded_media.getbuffer())
            st.session_state.uploaded_media_path = save_path
            st.success(f"Archivo cargado: {safe_name}")

        saved_upload = st.session_state.get("uploaded_media_path", "")
        video_path = st.text_input(
            "O escribe la ruta del video / imagen",
            value=saved_upload if saved_upload else "video.mp4",
        )

    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"])
    conf = st.slider("Confianza (conf)", 0.05, 0.90, float(Config.DEFAULT_CONF), 0.05)
    iou = st.slider("IOU NMS,(iou)", 0.10, 0.90, float(Config.DEFAULT_IOU), 0.05)

    min_iou_item = st.slider("Iou minimo item-persona", 0.00, 0.20, float(Config.MIN_IOU_PERSON_ITEM), 0.01)

    enable_qr = st.checkbox("Leer QR en etiqueta (helmet_tag)", value=False)
    preview_enabled = st.checkbox("Mostrar camara en vivo", value=True)
    show_items_without_person = st.checkbox("Mostrar items sin persona", value=False)
    show_all_dets = st.checkbox("Mostrar todas las detecciones", value=False)
    show_debug_info = st.checkbox("Mostrar debug de clases", value=False)

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
if "last_export_path" not in st.session_state:
    st.session_state.last_export_path = ""
if "uploaded_media_path" not in st.session_state:
    st.session_state.uploaded_media_path = ""

st.session_state.preview_enabled = bool(preview_enabled)

# Usar el placeholder creado arriba para el frame y la tabla
frame_slot = video_placeholder
table_slot = col2.empty()

def render_status_banner(status_text, status_class):
    return f"""
    <div class="status-banner status-{status_class}">
        ESTADO: {status_text}
    </div>
    """

status_placeholder.markdown(render_status_banner("ESPERANDO CAMARA", "alert"), unsafe_allow_html=True)
persons_metric_placeholder.metric("Personas", "0")
infractions_metric_placeholder.metric("Infracciones", "0", delta_color="inverse")
debug_placeholder.info("Debug desactivado")


def export_events_csv():
    export_dir = os.path.join("runs", "evidence")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"evidence_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(st.session_state.events).to_csv(export_path, index=False, encoding="utf-8-sig")
    return export_path


def load_model(path):
    return YoloTracker(path)

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
        all_candidates = [n for n in model_names if n not in {cls_person, cls_tag}]
        helmet_defaults = _pick_classes(all_candidates, HELMET_KEYS)
        vest_defaults = _pick_classes(all_candidates, VEST_KEYS)

        if not helmet_defaults:
            st.warning("No se detectó una clase positiva de casco en el modelo.")
        if not vest_defaults:
            st.warning("No se detectó una clase positiva de chaleco en el modelo.")

        # Detectar solo clases positivas de EPP para evitar ruido con Mask/NO-*.
        cls_helmet = helmet_defaults
        cls_vest = vest_defaults

        selected_epps = list(dict.fromkeys(cls_helmet + cls_vest))

        st.caption("Clases del modelo: " + ", ".join(model_names))
        st.caption("Casco: " + (", ".join(cls_helmet) if cls_helmet else "N/D"))
        st.caption("Chaleco: " + (", ".join(cls_vest) if cls_vest else "N/D"))
    else:
        st.caption("No se pudieron leer las clases del modelo.")
        selected_epps = []
        cls_helmet = []
        cls_vest = []

    filter_inference = True
    filter_classes = []
    if model_names:
        filter_classes = list(dict.fromkeys(selected_epps))
        if cls_person and cls_person not in filter_classes:
            filter_classes.append(cls_person)
        if enable_qr and cls_tag and cls_tag not in filter_classes:
            filter_classes.append(cls_tag)

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

# Barra de controles inferior (agrupada)
with st.container():
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("▶️ Iniciar", use_container_width=True):
        st.session_state.running = True
        st.session_state.preview_enabled = True
    if c2.button("📁 Cargar", use_container_width=True):
        if source_type == "Webcam":
            st.warning("Selecciona una fuente de archivo para usar este boton, o usa Iniciar para la webcam.")
        elif source_type == "RTSP" and not rstp_url.strip():
            st.warning("Ingresa una URL RTSP antes de cargar la fuente.")
        elif source_type == "Video / imagen (archivo)" and not video_path.strip():
            st.warning("Sube un archivo o ingresa la ruta del video / imagen.")
        elif source_type == "Video / imagen (archivo)" and not os.path.isfile(video_path.strip()):
            st.warning("No se encontro el archivo. Revisa la ruta o vuelve a subirlo.")
        elif source_type == "Video / imagen (archivo)" and media_kind(video_path.strip()) == "unknown":
            st.warning("Formato no reconocido. Usa video mp4/avi/mov/mkv/wmv/m4v o imagen jpg/png/bmp/webp.")
        else:
            st.session_state.running = True
            st.session_state.preview_enabled = True
    if c3.button("📥 Exportar", use_container_width=True):
        # Generar reporte de personas con EPP incompleto
        bad_events = [e for e in st.session_state.events if e.get("status") and e.get("status") != "EPP OK"]
        if not bad_events:
            st.warning("No se detectaron personas sin EPP completo.")
        else:
            # Crear DataFrame resumen por worker_id o track_id
            df_bad = pd.DataFrame(bad_events)
            # Normalize missing worker_id
            df_bad["worker_id"] = df_bad.get("worker_id", "")
            df_bad["track_id"] = df_bad.get("track_id", "")
            grp = df_bad.groupby(["worker_id", "track_id"]).agg(
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
                occurrences=("timestamp", "count"),
                statuses=("status", lambda s: ", ".join(sorted(set(s))))
            ).reset_index()
            # Guardar CSV en memoria y ofrecer descarga
            csv_bytes = grp.to_csv(index=False).encode("utf-8-sig")
            report_name = f"report_no_epp_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button("Descargar reporte (CSV)", data=csv_bytes, file_name=report_name, mime="text/csv")
            # También guardar en runs/reports
            os.makedirs(os.path.join("runs", "reports"), exist_ok=True)
            path = os.path.join("runs", "reports", report_name)
            grp.to_csv(path, index=False, encoding="utf-8-sig")
            st.success(f"Reporte guardado en: {path}")
    if c4.button("🛑 Detener", use_container_width=True):
        st.session_state.running = False
        st.session_state.preview_enabled = False

    # Mostrar uploader si se solicitó
    if st.session_state.get("show_uploader"):
        with st.container():
            st.markdown("**Subir video o imagen para analisis**")
            uploaded_file = st.file_uploader(
                "Selecciona un archivo",
                type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)],
                key="inline_media_upload",
            )
            if uploaded_file is not None:
                save_dir = os.path.join("runs", "uploads")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, safe_upload_name(uploaded_file.name))
                with open(save_path, "wb") as fh:
                    fh.write(uploaded_file.getbuffer())
                st.success(f"Archivo guardado en: {save_path}")
                # Cambiar la fuente para que el pipeline lo procese
                source_type = "Video / imagen (archivo)"
                video_path = save_path
                st.session_state.uploaded_media_path = save_path
                st.session_state.show_uploader = False
                st.session_state.running = True
                st.session_state.preview_enabled = True

def open_capture():
    if source_type == "Webcam":
        backend = 0
        if "cam_backend" in globals():
            if cam_backend == "DSHOW":
                backend = cv2.CAP_DSHOW
            elif cam_backend == "MSMF":
                backend = cv2.CAP_MSMF
        if backend:
            return cv2.VideoCapture(int(cam_index), backend)
        return cv2.VideoCapture(int(cam_index))
    if source_type == "RTSP":
        return cv2.VideoCapture(rstp_url)
    selected_path = video_path.strip()
    if media_kind(selected_path) == "image":
        return StaticImageCapture(selected_path)
    return cv2.VideoCapture(selected_path)

cap = None
current_persons_count = 0
current_infractions_count = 0
current_status_text = "ESPERANDO CAMARA"
current_status_class = "alert"

if st.session_state.running or st.session_state.preview_enabled:
    cap = open_capture()
    if cap is None or not cap.isOpened():
        st.error("No se pudo abrir la fuente de video. Revisa el indice, permisos o si otra app usa la camara.")
        st.session_state.running = False
        st.session_state.preview_enabled = False

last_time = 0.0
fail_count = 0
frame_index = 0

while st.session_state.running or st.session_state.preview_enabled:
    ok, frame = cap.read()
    if not ok:
        if source_type == "Video / imagen (archivo)":
            st.info("Se termino de procesar el archivo cargado.")
            st.session_state.running = False
            st.session_state.preview_enabled = False
            break
        fail_count += 1
        if fail_count == 1:
            st.warning("No se pudo leer el frame. Verifique la fuente de video.")
        if cap is not None:
            cap.release()
        time.sleep(0.5)
        cap = open_capture()
        if fail_count >= 5:
            st.error("No se pudo leer el frame despues de varios intentos. Revisa la fuente.")
            break
        continue
    frame_index += 1
    if st.session_state.running and process_every_n > 1 and (frame_index % int(process_every_n)) != 0:
        continue
    fail_count = 0
    now = time.time()
    if now - last_time < (1.0 / max(1, fps_limit)):
        continue
    last_time = now

    if st.session_state.running:
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

        det_types = {}
        for det in dets:
            det_types[det["cls"]] = det_types.get(det["cls"], 0) + 1

        if show_debug_info:
            debug_lines = [
                f"Modelo: {os.path.basename(model_path)}",
                f"Clases filtradas: {', '.join(filter_classes) if filter_classes else 'ninguna'}",
                f"Clases casco: {', '.join(cls_helmet) if cls_helmet else 'N/D'}",
                f"Clases chaleco: {', '.join(cls_vest) if cls_vest else 'N/D'}",
                f"Detecciones totales: {len(dets)}",
                "Por clase: " + (", ".join(f"{k}:{v}" for k, v in det_types.items()) if det_types else "ninguna"),
                f"Procesando cada {int(process_every_n)} frame(s)",
                f"Ancho inferencia: {int(inference_width)} px",
            ]
            debug_placeholder.info("\n".join(debug_lines))
        else:
            debug_placeholder.info("Debug desactivado")

        persons, helmets, vests, tags = split_detections(dets,
            cls_person,
            cls_helmet,
            cls_vest,
            cls_tag
        )

        current_persons_count = len(persons)
        current_infractions_count = 0
        current_missing_helmet = False
        current_missing_vest = False

        require_helmet = len(cls_helmet) > 0
        require_vest = len(cls_vest) > 0
        use_epp = len(persons) > 0 and (require_helmet or require_vest)

        if use_epp:
            if show_all_dets:
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    label = f'{d["cls"]} {d["conf"]:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)

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
                    status = "EPP OK"
                elif require_helmet and not require_vest:
                    status = "EPP OK" if has_helmet else "FALTA CASCO"
                elif require_vest and not require_helmet:
                    status = "EPP OK" if has_vest else "FALTA CHALECO"
                elif has_helmet and has_vest:
                    status = "EPP OK"
                elif has_helmet and not has_vest:
                    status = "FALTA CHALECO"
                elif has_vest and not has_helmet:
                    status = "FALTA CASCO"
                else:
                    status = "Sin EPP"

                if status != "EPP OK":
                    current_missing_helmet = current_missing_helmet or (status in ("FALTA CASCO", "Sin EPP"))
                    current_missing_vest = current_missing_vest or (status in ("FALTA CHALECO", "Sin EPP"))

                if status != "EPP OK" and events.should_emit(tid):
                    current_infractions_count += 1
                    st.session_state.events.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "track_id": tid,
                        "worker_id": worker_id or "",
                        "status": status
                    })
                elif status != "EPP OK":
                    current_infractions_count += 1

                x1, y1, x2, y2 = pb
                label = f"ID:{tid}"
                if worker_id:
                    label += f" Worker:{worker_id}"
                label += f" {status}"

                ok_color = (0, 255, 0)
                bad_color = (0, 0, 255)
                box_color = ok_color if status == "EPP OK" else bad_color
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            if show_items_without_person:
                for ib, _ in helmets:
                    x1, y1, x2, y2 = ib
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "casco", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for ib, _ in vests:
                    x1, y1, x2, y2 = ib
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(frame, "chaleco", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        else:
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                label = f'{d["cls"]} {d["conf"]:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)

        if current_persons_count > 0 and current_infractions_count == 0:
            current_status_text = "CUMPLE EPP CRITICO"
            current_status_class = "ok"
        elif current_persons_count > 0 and current_infractions_count > 0:
            if current_missing_helmet and current_missing_vest:
                current_status_text = "NO CUMPLE ALERTA CRITICA"
            elif current_missing_helmet:
                current_status_text = "FALTA CASCO"
            elif current_missing_vest:
                current_status_text = "FALTA CHALECO"
            else:
                current_status_text = "NO CUMPLE ALERTA CRITICA"
            current_status_class = "alert"
        else:
            current_status_text = "ESPERANDO CAMARA"
            current_status_class = "alert"

    if st.session_state.running:
        if len(st.session_state.events) > 0:
            df = pd.DataFrame(st.session_state.events).tail(200)
            table_slot.dataframe(df, use_container_width=True, height=600)
        else:
            table_slot.info("No hay eventos registrados.")

    status_placeholder.markdown(
        render_status_banner(current_status_text, current_status_class),
        unsafe_allow_html=True,
    )
    persons_metric_placeholder.metric("Personas", str(current_persons_count))
    infractions_metric_placeholder.metric("Infracciones", str(current_infractions_count), delta_color="inverse")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if st.session_state.running:
        cv2.putText(rgb, f"Detecciones: {len(dets)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    frame_slot.image(rgb, channels="RGB", use_column_width=True)

if cap is not None:
    cap.release()
