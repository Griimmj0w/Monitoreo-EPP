import time


class EventManager:
    def __init__(self, cooldown_sec=5.0):
        self.cooldown_sec = cooldown_sec
        self.last_emit = {}

    def should_emit(self, person_id, event_type=None):
        """Return True if an event for person_id/event_type should be emitted.

        `event_type` is optional to allow simple usage with just a person id.
        """
        current_time = time.time()
        key = (person_id, event_type)
        last_time = self.last_emit.get(key, 0)

        if current_time - last_time >= self.cooldown_sec:
            self.last_emit[key] = current_time
            return True
        return False


class TemporalConfirmer:
    """Confirma un estado por persona solo si se repite en varios frames.

    Evita alertas falsas por parpadeo de deteccion: un estado (p. ej.
    FALTA CASCO) se considera confirmado unicamente cuando el estado mas
    reciente aparece en al menos `required` de los ultimos `window` frames
    procesados para esa persona. Devuelve None mientras no este confirmado,
    para que el llamador trate el estado como "OK" (sin alerta).
    """

    def __init__(self, window=7, required=5):
        self.window = max(1, int(window))
        self.required = min(self.window, max(1, int(required)))
        self.history = {}  # person_id -> lista de estados (mas reciente al final)

    def update(self, person_id, status):
        hist = self.history.setdefault(person_id, [])
        hist.append(status)
        if len(hist) > self.window:
            hist.pop(0)
        return self.confirmed(person_id)

    def confirmed(self, person_id):
        hist = self.history.get(person_id, [])
        if not hist:
            return None
        candidate = hist[-1]
        count = sum(1 for s in hist if s == candidate)
        return candidate if count >= self.required else None

    def prune(self, active_ids):
        """Elimina el historial de personas que ya no estan en escena."""
        for tid in list(self.history):
            if tid not in active_ids:
                del self.history[tid]


class StickyOkGate:
    """Mantiene el estado OK de una persona durante `hold_sec` tras un OK
    confirmado, para que el parpadeo de deteccion posterior no genere
    falsas violaciones (modo demo). Con hold_sec <= 0 queda desactivado.
    """

    def __init__(self, hold_sec=20.0):
        self.hold_sec = max(0.0, float(hold_sec))
        self.ok_since = {}  # person_id -> timestamp del ultimo OK confirmado

    def mark_ok(self, person_id, now):
        self.ok_since[person_id] = now

    def is_holding(self, person_id, now):
        if self.hold_sec <= 0:
            return False
        return (now - self.ok_since.get(person_id, -1e18)) <= self.hold_sec

    def prune(self, active_ids):
        """Elimina el estado de personas que ya no estan en escena."""
        for tid in list(self.ok_since):
            if tid not in active_ids:
                del self.ok_since[tid]


