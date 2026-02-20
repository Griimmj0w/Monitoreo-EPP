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
    for ib, ic in items:
        if inside(ib,person_bbox) or iou(ib,person_bbox) >= min_iou:
            matched.append((ib, ic))
    return matched

