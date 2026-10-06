"""Capa de consolidacion para analisis (Power BI).

Consolida el estado EPP por track_id en EPISODIOS: un episodio es el
periodo continuo en el que una persona mantiene el mismo estado_epp
(COMPLETO / PARCIAL / SIN EPP). Cada episodio genera UNA fila en el CSV
analitico, con timestamp_inicio, timestamp_fin y duracion — asi una
persona que aparece en muchos frames NO se cuenta como varias personas.

CSV analitico: runs/analytics/ppe_monitoring_YYYYMMDD.csv (acumulativo
por dia, se anexa si ya existe). El CSV de evidencias (alertas) de la
app no se toca.
"""
import csv
import os
from datetime import datetime
from pathlib import Path

ANALYTICS_COLUMNS = [
    "event_id",
    "timestamp_inicio",
    "timestamp_fin",
    "fecha",
    "hora",
    "camera_id",
    "area",
    "track_id",
    "worker_id",
    "casco",
    "chaleco",
    "estado_epp",
    "detalle",
    "confidence",
    "duracion_evento_seg",
]

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_row(track_id, worker_id, effective_status, require_helmet, require_vest, confidence, status_ok, status_no_helmet, status_no_vest, now):
    """Convierte el estado efectivo de la app a los campos del CSV analitico.

    Solo se marcan SI/NO los EPP que el modelo realmente detecta
    (require_helmet / require_vest); si una clase no esta activa, su
    columna queda vacia.
    """
    if effective_status == status_ok:
        casco = "SI" if require_helmet else ""
        chaleco = "SI" if require_vest else ""
        estado_epp = "COMPLETO"
        detalle = "USO ADECUADO"
    elif effective_status == status_no_helmet:
        casco = "NO" if require_helmet else ""
        chaleco = "SI" if require_vest else ""
        estado_epp = "PARCIAL"
        detalle = "FALTA CASCO"
    elif effective_status == status_no_vest:
        casco = "SI" if require_helmet else ""
        chaleco = "NO" if require_vest else ""
        estado_epp = "PARCIAL"
        detalle = "FALTA CHALECO"
    else:  # Sin EPP
        casco = "NO" if require_helmet else ""
        chaleco = "NO" if require_vest else ""
        estado_epp = "SIN EPP"
        detalle = "SIN CASCO Y CHALECO"

    return {
        "track_id": track_id,
        "worker_id": worker_id,
        "casco": casco,
        "chaleco": chaleco,
        "estado_epp": estado_epp,
        "detalle": detalle,
        "confidence": confidence,
        "now": now,
    }


class EpisodeTracker:
    """Mantiene el episodio activo por track_id y escribe el CSV analitico."""

    def __init__(self, out_dir="runs/analytics", camera_id="", area=""):
        self.out_dir = Path(out_dir)
        self.camera_id = camera_id or ""
        self.area = area or ""
        self.active = {}   # track_id -> dict del episodio abierto
        self.today_file = None
        self.next_id = 1

    def _ensure_file(self, now):
        day = now.strftime("%Y%m%d")
        path = self.out_dir / f"ppe_monitoring_{day}.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.reader(fh))
            self.next_id = len(rows)  # 1 por header + N filas
        else:
            self.next_id = 1
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                csv.writer(fh).writerow(ANALYTICS_COLUMNS)
        self.today_file = path
        return path

    def _write_row(self, episode):
        if self.today_file is None or episode["now"].strftime("%Y%m%d") != self.today_file.stem.split("_")[-1]:
            self._ensure_file(episode["now"])
        episode["duracion_seg"] = max(0, int(round((episode["fin"] - episode["inicio"]).total_seconds())))
        with open(self.today_file, "a", encoding="utf-8-sig", newline="") as fh:
            csv.writer(fh).writerow([
                f"EVT-{self.next_id:06d}",
                episode["inicio"].strftime(TIME_FORMAT),
                episode["fin"].strftime(TIME_FORMAT),
                episode["inicio"].strftime("%Y-%m-%d"),
                episode["inicio"].strftime("%H:%M"),
                self.camera_id,
                self.area,
                episode["track_id"],
                episode["worker_id"],
                episode["casco"],
                episode["chaleco"],
                episode["estado_epp"],
                episode["detalle"],
                f"{episode['confidence']:.2f}",
                episode["duracion_seg"],
            ])
        self.next_id += 1

    def update(self, track_id, worker_id, effective_status, require_helmet,
               require_vest, confidence, status_ok, status_no_helmet,
               status_no_vest, now):
        """Registra el estado efectivo de una persona en este frame.

        Si el estado cambia respecto al episodio abierto de ese track_id,
        se cierra el episodio anterior (escribiendo su fila) y se abre uno
        nuevo. Devuelve el episodio activo tras la actualizacion.
        """
        row = build_row(track_id, worker_id, effective_status, require_helmet,
                        require_vest, confidence, status_ok, status_no_helmet,
                        status_no_vest, now)

        episode = self.active.get(track_id)
        if episode is not None and episode["estado_epp"] == row["estado_epp"]:
            # mismo episodio: extender
            episode["fin"] = now
            episode["confidence"] = max(episode["confidence"], confidence)
            episode["worker_id"] = episode["worker_id"] or row["worker_id"]
            return episode

        # cerrar el episodio anterior si existe
        if episode is not None:
            self._write_row(episode)

        new_episode = {
            "track_id": track_id,
            "worker_id": row["worker_id"],
            "casco": row["casco"],
            "chaleco": row["chaleco"],
            "estado_epp": row["estado_epp"],
            "detalle": row["detalle"],
            "confidence": confidence,
            "inicio": now,
            "fin": now,
            "now": now,
        }
        self.active[track_id] = new_episode
        return new_episode

    def prune(self, active_ids):
        """Cierra los episodios de personas que ya no estan en escena."""
        for tid in list(self.active):
            if tid not in active_ids:
                episode = self.active.pop(tid)
                self._write_row(episode)

    def close_all(self):
        """Cierra todos los episodios abiertos (fin de la sesion de captura)."""
        for tid in list(self.active):
            episode = self.active.pop(tid)
            self._write_row(episode)

    def file_path(self):
        return str(self.today_file) if self.today_file else None
