"""Configure root logging once for the whole process.

All other modules call logging.getLogger(__name__) to inherit this setup.
Call configure_logging() once at process startup (e.g. in main.py) before
any pipeline step runs.
"""

import logging
import sys
from pathlib import Path


def configure_logging(*, log_level: str, log_file: Path) -> None:
    """Set up root logger with a console handler and a rotating file handler.

    Args:
        log_level: String level name — "DEBUG", "INFO", "WARNING", or "ERROR".
                   Invalid names fall back to INFO.
        log_file:  Path where log lines are appended
        (file is created if absent).
    """
    numeric_level = getattr(
        logging, (log_level or "INFO").upper(), logging.INFO
        )

    # Ensure the log directory exists before opening the FileHandler.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename=str(log_file), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[console_handler, file_handler],
        force=True,
    )
