import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Folder with images to auto-label")
    p.add_argument("--out-dir", required=True, help="Output dataset root")
    p.add_argument("--model", required=True, help="Path to .pt model for pseudo-labeling")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0", help="Device for inference, e.g. 0 or cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--link", choices=["copy", "hardlink"], default="copy")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def gather_images(images_dir: Path):
    images = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")

    ratios = [args.train, args.val, args.test]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    images = gather_images(images_dir)
    if not images:
        raise SystemExit("No images found to label.")

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
        ensure_dir(out_dir / split / "images")
        ensure_dir(out_dir / split / "labels")

    try:
        from ultralytics import YOLO
    except Exception:
        raise SystemExit("ultralytics not available. Install it first: pip install ultralytics")

    model = YOLO(args.model)

    print(f"Auto-labeling {n_total} images -> {out_dir}")
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    idx = 0
    for split, imgs in splits.items():
        for img_path in imgs:
            idx += 1
            rel_name = img_path.name
            img_dst = out_dir / split / "images" / rel_name
            label_dst = out_dir / split / "labels" / (img_path.stem + ".txt")

            if not args.overwrite and img_dst.exists() and label_dst.exists():
                continue

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
            if r is not None and getattr(r, "boxes", None) is not None:
                boxes = r.boxes
                if len(boxes) > 0:
                    xywhn = boxes.xywhn.cpu().tolist()
                    cls_ids = boxes.cls.cpu().tolist()
                    for c, (x, y, w, h) in zip(cls_ids, xywhn):
                        lines.append(
                            f"{int(c)} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}"
                        )

            label_dst.write_text("\n".join(lines), encoding="utf-8")

            if idx % 200 == 0:
                print(f"Processed {idx}/{n_total}")

    # write data yaml for this auto-labeled dataset
    names = []
    try:
        if getattr(model, "names", None):
            names = [str(v) for k, v in sorted(model.names.items())]
    except Exception:
        names = []

    if names:
        yaml_lines = [
            f"train: {out_dir.as_posix()}/train/images",
            f"val: {out_dir.as_posix()}/val/images",
            f"test: {out_dir.as_posix()}/test/images",
            "",
            f"nc: {len(names)}",
            "",
            "names: [" + ", ".join(f'\"{n}\"' for n in names) + "]",
        ]
        (out_dir / "data_auto.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
