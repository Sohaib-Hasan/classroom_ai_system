"""
logging_setup.py
-------------------
Pehle app.py mein har exception ko ek generic `except Exception:` pakad
kar sirf student ko ek friendly message dikhata tha — asal error kahin
log hi nahi hota tha. Matlab production mein jab kabhi masla aata, dev ko
kabhi pata nahi chalta *kya* toota, kyunke traceback kahin save hi nahi
hota tha.

Fix: ek simple rotating file logger. Koi external service nahi (zero
budget), bas `logs/error.log` mein likhta hai, aur size zyada badhne se
rok deta hai (RotatingFileHandler) taake disk bhar na jaye.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "classroom_ai"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured (Streamlit reruns script baar baar)

    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = RotatingFileHandler(
        log_dir / "error.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    # Console pe bhi info level dikhayein — Streamlit Cloud ke "Manage app"
    # logs tab mein ye seedha visible hota hai, alag se file kholne ki
    # zaroorat nahi.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    return logger
