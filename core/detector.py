try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False


class YoloTracker:
    """Light wrapper around ultralytics' YOLO tracker.

    If the `ultralytics` package is not available, the class still
    exists but will raise an informative error when instantiated.
    """

    def __init__(self, model_path: str):
        if not _HAS_ULTRALYTICS:
            raise RuntimeError(
                "ultralytics package not found. Install it or set up the environment: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        # model.names may contain class id->name mapping
        self.names = getattr(self.model, "names", {}) or {}

    def infer(self, frame, conf=0.25, iou=0.5, tracker="botsort.yaml", persist=True, classes=None):
        kwargs = {
            "conf": conf,
            "iou": iou,
            "tracker": tracker,
            "persist": persist,
            "verbose": False,
        }
        if classes:
            kwargs["classes"] = classes

        results = self.model.track(frame, **kwargs)

        if not results:
            return []

        r = results[0]
        dets = []

        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return dets

        for b in boxes:
            # b.cls, b.xyxy, b.id, b.conf can be tensors/arrays
            try:
                cls_id = int(b.cls[0])
            except Exception:
                cls_id = -1

            cls_name = self.names.get(cls_id, str(cls_id))

            try:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            except Exception:
                # fallback if xyxy is already a list/tuple
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy)
                except Exception:
                    continue

            tid = int(b.id[0]) if getattr(b, "id", None) is not None else -1
            score = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0

            dets.append({
                "cls": cls_name,
                "bbox": (x1, y1, x2, y2),
                "track_id": tid,
                "conf": score,
            })

        return dets


