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

