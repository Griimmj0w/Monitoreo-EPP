"""Pseudo-etiqueta frames de CCTV con el modelo actual y arma el dataset v4.

Mapeo de clases (modelo css_v28_plus -> dataset v4):
    5 Person       -> 0 person
    0 Hardhat      -> 1 helmet
    7 Safety Vest  -> 2 vest
    (gloves = 3) no se genera: debe etiquetarse a mano.

Las etiquetas generadas son un punto de partida que luego se corrige
manualmente (LabelImg / makesense.ai), agregando los guantes.

Ejemplo:
    .venv\\Scripts\\python.exe scripts\\pseudo_label_v4.py --images datasets/cctv_epp/raw_frames --out datasets/cctv_epp_v4
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CLASS_MAP = {5: 0, 0: 1, 7: 2}  # cls modelo -> cls v4
NAMES_V4 = ["person", "helmet", "vest"]  # gloves en pausa: activar con --with-gloves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Carpeta con los frames extraidos")
    p.add_argument("--out", default="datasets/cctv_epp_v4", help="Raiz del dataset v4")
    p.add_argument("--model", default="runs/detect/runs/train/css_v28_plus/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0", help="0 para GPU, cpu para CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--link", choices=["copy", "hardlink"], default="hardlink")
    p.add_argument(
        "--with-gloves",
        action="store_true",
        help="Incluir la clase 'gloves' (3) en el dataset (etiquetado manual posterior)",
    )
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            import os
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    names = list(NAMES_V4)
    if args.with_gloves:
        names.append("gloves")

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")

    ratios = [args.train, args.val, args.test]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit("No images found in --images")
    random.Random(args.seed).shuffle(images)

    n_total = len(images)
    n_train = int(n_total * args.train)
    n_val = int(n_total * args.val)
    n_test = n_total - n_train - n_val
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split in splits:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception:
        raise SystemExit("ultralytics not available. Install it first: pip install ultralytics")

    model = YOLO(args.model)
    print(f"Pseudo-etiquetando {n_total} imagenes -> {out_dir}")
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    box_counts = {name: 0 for name in names}
    empty_imgs = 0
    for i, img_path in enumerate(images, 1):
        split = next(s for s, imgs in splits.items() if img_path in imgs)
        img_dst = out_dir / split / "images" / img_path.name
        label_dst = out_dir / split / "labels" / (img_path.stem + ".txt")
        link_or_copy(img_path, img_dst, args.link)

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        r = results[0] if results else None
        lines = []
        if r is not None and getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            xywhn = r.boxes.xywhn.cpu().tolist()
            cls_ids = [int(c) for c in r.boxes.cls.cpu().tolist()]
            for cls_id, (x, y, w, h) in zip(cls_ids, xywhn):
                v4_cls = CLASS_MAP.get(cls_id)
                if v4_cls is None:
                    continue
                lines.append(f"{v4_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                box_counts[names[v4_cls]] += 1
        if not lines:
            empty_imgs += 1
        label_dst.write_text("\n".join(lines), encoding="utf-8")
        if i % 50 == 0:
            print(f"  {i}/{n_total}...", flush=True)

    yaml_lines = [
        f"train: {out_dir.as_posix()}/train/images",
        f"val: {out_dir.as_posix()}/val/images",
        f"test: {out_dir.as_posix()}/test/images",
        "",
        f"nc: {len(names)}",
        "",
        "names: " + ", ".join(f'\"{n}\"' for n in names),
    ]
    (out_dir / "data_cctv_v4.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    print("\n[OK] Dataset v4 listo.")
    print("Cajas generadas por clase:", {k: v for k, v in box_counts.items()})
    print(f"Imagenes sin ninguna caja: {empty_imgs}/{n_total}")
    if args.with_gloves:
        print("Siguiente paso: corregir cajas y agregar 'gloves' a mano en train/val/test.")
    else:
        print("Siguiente paso: corregir las cajas de person/helmet/vest a mano en train/val/test.")


if __name__ == "__main__":
    main()
