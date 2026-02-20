
import streamlit as st
import cv2
import pandas as pd
import time
import os

from core.config import Config, CONFIG
from core.detector import YoloTracker
from core.association import split_detections, associate_items_to_persons
from core.id_reader import HelmetTagReader
from core.events import EventManager


st.set_page_config(page_title="Deteccion EPP (YOLOv8)", layout="wide")

st.title("Deteccion de EPP (Casco + Chaleco) con YOLOv8")
st.caption("MVP Streamlit: tracking + reglas + identificacion por tag (QR) opcional")


with st.sidebar:
    st.header("Configuracion")
    model_path = st.text_input(
        "Ruta del modelo YOLOv8 (.pt)",
        value="runs/detect/runs/train/archive2_auto_exp/weights/best.pt"
    )


    source_type = st.selectbox("Fuente", ["Webcam", "RTSP", "Video (archivo)"])

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
        video_path = st.text_input("Ruta del video", value="video.mp4")

    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"])
    conf = st.slider("Confianza (conf)", 0.05, 0.90, float(Config.DEFAULT_CONF), 0.05)
    iou = st.slider("IOU NMS,(iou)", 0.10, 0.90, float(Config.DEFAULT_IOU), 0.05)

    min_iou_item = st.slider("Iou minimo item-persona", 0.00, 0.20, float(Config.MIN_IOU_PERSON_ITEM), 0.01)

    enable_qr = st.checkbox("Leer QR en etiqueta (helmet_tag)", value=False)
    preview_enabled = st.checkbox("Mostrar camara en vivo", value=True)
    show_items_without_person = st.checkbox("Mostrar items sin persona", value=False)
    show_all_dets = st.checkbox("Mostrar todas las detecciones", value=False)

    cooldown = st.number_input("Cooldown eventos (segundos)", min_value=0.0, max_value=60.0, value=float(Config.EVENT_COOLDOWN_SEC), step=0.5)
    fps_limit = st.number_input("FPS limite", min_value=1, max_value=60, value=10, step=1)

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

start = st.button("INICIAR DETECCION")
stop = st.button("DETENER DETECCION")

if "running" not in st.session_state:
    st.session_state.running = False
if "events" not in st.session_state:
    st.session_state.events = []
if "track_to_worker" not in st.session_state:
    st.session_state.track_to_worker = {}
if "preview_enabled" not in st.session_state:
    st.session_state.preview_enabled = True

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False
st.session_state.preview_enabled = bool(preview_enabled)

col1, col2 = st.columns([1.4, 1.0])
frame_slot = col1.empty()
table_slot = col2.empty()

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

    cls_person = _pick_one(model_names, ["person", "persona", "worker", "trabajador"], Config.CLASS_PERSON)
    cls_tag = _pick_one(model_names, ["tag", "qr", "label", "etiqueta"], Config.CLASS_TAG)

    if model_names:
        epp_candidates = [n for n in model_names if n not in {cls_person, cls_tag}]
        if not epp_candidates:
            epp_candidates = model_names
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

    def _is_negative(name):
        n = name.lower().replace("_", " ").replace("-", " ")
        return n.startswith("no ") or n.startswith("sin ")

    def _pick_classes(names, keywords):
        return [n for n in names if _match_any(n, keywords) and not _is_negative(n)]

    cls_helmet = _pick_classes(selected_epps, ["helmet", "hardhat", "hard hat", "casco"])
    cls_vest = _pick_classes(selected_epps, ["vest", "chaleco"])

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
    return cv2.VideoCapture(video_path)

cap = None
if st.session_state.running or st.session_state.preview_enabled:
    cap = open_capture()
    if cap is None or not cap.isOpened():
        st.error("No se pudo abrir la fuente de video. Revisa el indice, permisos o si otra app usa la camara.")
        st.session_state.running = False
        st.session_state.preview_enabled = False

last_time = 0.0
fail_count = 0

while st.session_state.running or st.session_state.preview_enabled:
    ok, frame = cap.read()
    if not ok:
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

        dets = yolo.infer(
            frame,
            conf=float(conf),
            iou=float(iou),
            tracker=tracker,
            persist=True,
            classes=class_ids if class_ids else None,
        )

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
                else:
                    status = "EPP OK" if has_helmet and has_vest else (
                        "FALTA CASCO" if (not has_helmet and has_vest) else 
                        "FALTA CHALECO" if (has_helmet and not has_vest) else
                        "Sin EPP"
                    )

                if status != "EPP OK" and events.should_emit(tid):
                    st.session_state.events.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "track_id": tid,
                        "worker_id": worker_id or "",
                        "status": status
                    })

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

    if st.session_state.running:
        if len(st.session_state.events) > 0:
            df = pd.DataFrame(st.session_state.events).tail(200)
            table_slot.dataframe(df, use_container_width=True, height=600)
        else:
            table_slot.info("No hay eventos registrados.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if st.session_state.running:
        cv2.putText(rgb, f"Detecciones: {len(dets)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    frame_slot.image(rgb, channels="RGB", use_column_width=True)

if cap is not None:
    cap.release()
