"""Generate charts and a summary from the complete CCTV v4 YOLO dataset."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "datasets" / "cctv_epp_v4"
OUTPUT_DIR = PROJECT_DIR / "runs" / "reports"
CLASS_NAMES = ["person", "helmet", "vest"]
SPLITS = ["train", "val", "test"]
CLASS_COLORS = ["#2457a6", "#16805c", "#d97706"]


def read_dataset_counts() -> dict[str, dict[str, int]]:
    counts = {}
    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"
        class_counts = Counter()
        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                fields = line.split()
                if fields and fields[0].isdigit():
                    class_id = int(fields[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        class_counts[CLASS_NAMES[class_id]] += 1
        counts[split] = {
            "images": len([path for path in image_dir.iterdir() if path.is_file()]),
            **{class_name: class_counts[class_name] for class_name in CLASS_NAMES},
        }
    return counts


def save_metric_chart() -> None:
    metrics = {"mAP50": 87.7, "mAP50-95": 44.5, "Precision": 91.2, "Recall": 84.3}
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(metrics.keys(), metrics.values(), color=CLASS_COLORS + ["#b45309"], width=0.62)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Porcentaje (%)")
    axis.set_title("Métricas de evaluación del modelo - CCTV v4")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "metricas_modelo_cctv_v4.png", dpi=180)
    plt.close(figure)


def save_dataset_charts(counts: dict[str, dict[str, int]]) -> None:
    totals = {class_name: sum(counts[split][class_name] for split in SPLITS) for class_name in CLASS_NAMES}

    figure, axis = plt.subplots(figsize=(8, 5.2))
    bars = axis.bar(totals.keys(), totals.values(), color=CLASS_COLORS, width=0.62)
    axis.set_ylabel("Cantidad de objetos anotados")
    axis.set_title("Distribución total de clases - Dataset CCTV v4")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.annotate(f"{int(bar.get_height())}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "distribucion_total_clases_cctv_v4.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    bottom = [0] * len(SPLITS)
    for class_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        values = [counts[split][class_name] for split in SPLITS]
        axis.bar(SPLITS, values, bottom=bottom, label=class_name, color=color)
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Cantidad de objetos anotados")
    axis.set_title("Clases anotadas por partición del dataset")
    axis.legend(title="Clase")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "clases_por_particion_cctv_v4.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.2))
    image_counts = [counts[split]["images"] for split in SPLITS]
    bars = axis.bar(SPLITS, image_counts, color=["#2457a6", "#4f83cc", "#7aa5d8"], width=0.62)
    axis.set_ylabel("Número de imágenes")
    axis.set_title("Imágenes por partición - Dataset CCTV v4")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.annotate(f"{int(bar.get_height())}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "imagenes_por_particion_cctv_v4.png", dpi=180)
    plt.close(figure)


def save_summary(counts: dict[str, dict[str, int]]) -> None:
    total_images = sum(counts[split]["images"] for split in SPLITS)
    totals = {class_name: sum(counts[split][class_name] for split in SPLITS) for class_name in CLASS_NAMES}
    report = """# Resumen del dataset y resultados - CCTV v4

Los conteos se calcularon directamente desde todas las etiquetas YOLO del dataset.

## Métricas del modelo

| Métrica | Valor |
|---|---:|
| mAP50 | 87.7%% |
| mAP50-95 | 44.5%% |
| Precision | 91.2%% |
| Recall | 84.3%% |

## Dataset completo

- Imágenes totales: %d
- Objetos anotados totales: %d

| Clase | Total |
|---|---:|
""" % (total_images, sum(totals.values()))
    report += "".join(f"| {class_name} | {totals[class_name]} |\n" for class_name in CLASS_NAMES)
    report += """
## Imágenes y objetos por partición

| Partición | Imágenes | person | helmet | vest |
|---|---:|---:|---:|---:|
"""
    report += "".join(
        f"| {split} | {counts[split]['images']} | {counts[split]['person']} | "
        f"{counts[split]['helmet']} | {counts[split]['vest']} |\n"
        for split in SPLITS
    )
    (OUTPUT_DIR / "resumen_cctv_v4.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = read_dataset_counts()
    save_metric_chart()
    save_dataset_charts(counts)
    save_summary(counts)
    print(f"Archivos generados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
