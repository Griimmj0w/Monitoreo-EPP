import time
from pathlib import Path
import subprocess
import sys

# Paths
BASE = Path("runs/detect/runs/train/archive_exp")
RESULTS = BASE / "results.csv"
WEIGHTS = BASE / "weights" / "best.pt"


def tail(n=5):
    if not RESULTS.exists():
        print("results.csv not yet created")
        return []
    with RESULTS.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [l for l in f.read().strip().splitlines() if l.strip()]
    for line in lines[-n:]:
        print(line)
    return lines


def gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True)
        print("GPU:", out.strip())
    except Exception as e:
        print("nvidia-smi error:", e)


def last_epoch_from_results():
    if not RESULTS.exists():
        return 0
    with RESULTS.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [l for l in f.read().strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        return 0
    last = lines[-1]
    parts = last.split(",")
    try:
        return int(parts[0])
    except Exception:
        return 0


def main(interval=60, target_epochs=50):
    print("Starting monitor: watching", RESULTS)
    print(f"Will notify when weights exist or epoch >= {target_epochs}")
    try:
        while True:
            print("\n===", time.strftime("%Y-%m-%d %H:%M:%S"), "===")
            tail(5)
            gpu_stats()

            if WEIGHTS.exists():
                print("Training finished: weights found at", WEIGHTS)
                break

            last_epoch = last_epoch_from_results()
            print("Last epoch in results.csv:", last_epoch)
            if last_epoch >= target_epochs:
                print(f"Training appears complete (epoch {last_epoch} >= {target_epochs})")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitor interrupted by user")


if __name__ == '__main__':
    # allow optional args: interval, target_epochs
    try:
        interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        target_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    except Exception:
        interval = 60
        target_epochs = 50
    main(interval=interval, target_epochs=target_epochs)
