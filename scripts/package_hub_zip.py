"""Empaqueta un dataset YOLO local como ZIP listo para Ultralytics HUB.

Requisitos de HUB:
- El zip contiene una carpeta raiz con el dataset (p. ej. cctv_v4/).
- Dentro de esa carpeta va el dataset YAML con el MISMO nombre (cctv_v4.yaml)
  y con rutas RELATIVAS (train: train/images, etc.).
- El zip, la carpeta y el YAML comparten nombre.

Ejemplo:
    .venv\\Scripts\\python.exe scripts\\package_hub_zip.py --dataset datasets/cctv_epp_v4 --name cctv_v4 --out cctv_v4.zip
"""
import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Carpeta del dataset (train/val/test dentro)")
    p.add_argument("--name", required=True, help="Nombre del dataset en HUB (carpeta + yaml + zip)")
    p.add_argument("--out", default=None, help="Ruta del zip (default: <name>.zip en la raiz del repo)")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"dataset not found: {dataset_dir}")

    name = args.name
    out_zip = Path(args.out) if args.out else Path(f"{name}.zip")

    yaml_lines = [
        f"train: train/images",
        f"val: val/images",
        f"test: test/images",
        "",
        "nc: 3",
        "",
        'names: ["person", "helmet", "vest"]',
    ]
    yaml_text = "\n".join(yaml_lines) + "\n"

    # Verificar estructura antes de empaquetar
    for split in ("train", "val", "test"):
        if not (dataset_dir / split / "images").exists():
            raise SystemExit(f"falta {split}/images en {dataset_dir}")

    n_imgs = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/{name}.yaml", yaml_text)
        for split in ("train", "val", "test"):
            for img in sorted((dataset_dir / split / "images").glob("*")):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                zf.write(img, f"{name}/{split}/images/{img.name}")
                n_imgs += 1
                label = dataset_dir / split / "labels" / (img.stem + ".txt")
                if label.exists():
                    zf.write(label, f"{name}/{split}/labels/{img.stem}.txt")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] {out_zip} ({size_mb:.1f} MB, {n_imgs} imagenes)")
    print(f"YAML interno:\n{yaml_text}")
    print("Subir en HUB: Datasets -> Upload Dataset -> adjuntar este zip.")


if __name__ == "__main__":
    main()
