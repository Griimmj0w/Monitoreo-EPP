def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def inside(child, parent):
    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent
    return cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2

def expanded_box(box, x_ratio=0.12, y_top_ratio=0.18, y_bottom_ratio=0.04):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return (
        x1 - int(w * x_ratio),
        y1 - int(h * y_top_ratio),
        x2 + int(w * x_ratio),
        y2 + int(h * y_bottom_ratio),
    )

def center_inside(child, parent):
    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent
    cx = (cx1 + cx2) / 2.0
    cy = (cy1 + cy2) / 2.0
    return px1 <= cx <= px2 and py1 <= cy <= py2

def overlap_ratio(child, parent):
    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent
    inter_x1, inter_y1 = max(cx1, px1), max(cy1, py1)
    inter_x2, inter_y2 = min(cx2, px2), min(cy2, py2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    child_area = max(0, cx2 - cx1) * max(0, cy2 - cy1) + 1e-9
    return inter / child_area

def split_detections(dets, cls_person, cls_helmet, cls_vest, cls_tag):
    def _match(name, target):
        if isinstance(target, (list, tuple, set)):
            return name in target
        return name == target

    persons, helmets, vests, tags = [], [], [], []
    for d in dets:
        if _match(d["cls"], cls_person):
            persons.append((d["track_id"], d["bbox"], d["conf"]))
        elif _match(d["cls"], cls_helmet):
            helmets.append((d["bbox"], d["conf"]))
        elif _match(d["cls"], cls_vest):
            vests.append((d["bbox"], d["conf"]))
        elif _match(d["cls"], cls_tag):
            tags.append((d["bbox"], d["conf"]))
    return persons, helmets, vests, tags


def associate_items_to_persons(person_bbox, items, min_iou):

    matched = []
    relaxed_person_bbox = expanded_box(person_bbox)
    for ib, ic in items:
        if (
            inside(ib, person_bbox)
            or center_inside(ib, relaxed_person_bbox)
            or overlap_ratio(ib, relaxed_person_bbox) >= 0.45
            or iou(ib, person_bbox) >= min_iou
        ):
            matched.append((ib, ic))
    return matched

