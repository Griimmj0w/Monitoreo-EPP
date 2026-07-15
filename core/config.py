from dataclasses import dataclass

@dataclass
class AppConfig:

    CLASS_PERSON: str = "person"
    CLASS_HELMET: str = "helmet"
    CLASS_VEST: str = "vest"
    CLASS_TAG: str = "tag"


    MIN_IOU_PERSON_ITEM: float = 0.05  # casco/chaleco cerca o dentro de persona
    MIN_IOU_TAG_PERSON: float = 0.2   # etiqueta con casco/persona

    DEFAULT_CONF: float = 0.25
    DEFAULT_IOU: float = 0.5

    EVENT_COOLDOWN_SEC: float = 5.0    #no spamear eventos por persona

config = AppConfig()

# Backwards-compatible names expected by app.py
Config = AppConfig
CONFIG = config
