from core.association import iou as bbox_iou


HELMET_KEYS = ["helmet", "casco", "hard hat", "hardhat", "hard-hat", "gorro"]
VEST_KEYS = ["vest", "chaleco", "reflective", "safety vest", "safetyvest"]


def match_any(name, keywords):
    normalized = name.lower()
    return any(keyword in normalized for keyword in keywords)


def pick_one(names, keywords, fallback):
    for name in names:
        if match_any(name, keywords):
            return name
    return fallback if fallback in names else None


def is_negative(name):
    normalized = name.lower().replace("_", " ").replace("-", " ")
    return normalized.startswith("no ") or normalized.startswith("sin ")


def pick_classes(names, keywords):
    return [name for name in names if match_any(name, keywords) and not is_negative(name)]


def suppress_negative_overlaps(dets, positive_names, keywords, iou_thresh=0.2):
    if not positive_names:
        return dets
    pos_boxes = [det["bbox"] for det in dets if det["cls"] in positive_names]
    if not pos_boxes:
        return dets

    filtered = []
    for det in dets:
        name = det["cls"]
        if is_negative(name) and match_any(name, keywords):
            if any(bbox_iou(det["bbox"], pos_box) >= iou_thresh for pos_box in pos_boxes):
                continue
        filtered.append(det)
    return filtered
