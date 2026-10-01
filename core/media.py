import os
import re

import cv2


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
