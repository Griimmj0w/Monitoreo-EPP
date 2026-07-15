"""Extract representative frames from EPP videos for manual YOLO labeling.

Example:
    .venv311\\Scripts\\python.exe scripts\\extract_training_frames.py --source runs\\uploads\\video.mp4 --out datasets\\cctv_epp\\raw_frames --every-sec 1
"""
import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video file or directory with videos")
    parser.add_argument("--out", default="datasets/cctv_epp/raw_frames", help="Output folder")
    parser.add_argument("--every-sec", type=float, default=1.0, help="Seconds between extracted frames")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum frames per video")
    parser.add_argument("--prefix", default="", help="Optional prefix for output image names")
    return parser.parse_args()


def iter_videos(source):
    source_path = Path(source)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    if source_path.is_file():
        return [source_path]
    return sorted(p for p in source_path.rglob("*") if p.suffix.lower() in video_exts)


def extract_frames(video_path, out_dir, every_sec, max_frames, prefix):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(fps * every_sec))
    stem_prefix = f"{prefix}_" if prefix else ""
    stem = f"{stem_prefix}{video_path.stem}"

    saved = 0
    frame_idx = 0
    print(
        f"[INFO] Procesando {video_path.name} | fps={fps:.2f} | frames={frame_count} | step={step}",
        flush=True,
    )

    while saved < max_frames and (frame_count <= 0 or frame_idx < frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        out_path = out_dir / f"{stem}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
        if saved == 1 or saved % 25 == 0:
            print(f"[INFO] {video_path.name}: {saved} frames extraidos...", flush=True)
        frame_idx += step

    cap.release()
    print(f"[OK] {video_path}: {saved} frames -> {out_dir}")
    return saved


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(args.source)
    if not videos:
        raise SystemExit(f"No se encontraron videos en: {args.source}")

    total = 0
    for video_path in videos:
        total += extract_frames(video_path, out_dir, args.every_sec, args.max_frames, args.prefix)

    print(f"Total frames extraidos: {total}")
    print("Siguiente paso: etiqueta estas imagenes en formato YOLO y colocalas en train/valid/test.")


if __name__ == "__main__":
    main()
