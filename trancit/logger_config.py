import logging
import sys
from typing import Optional

LOG_LEVEL_DEFAULT = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAME = "trancit"


def setup_logging(
    name: str = LOGGER_NAME,
    level: int = LOG_LEVEL_DEFAULT,
    log_format: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configures logging for the 'trancit' package.

    This should be called once at application start or from __init__.py.

    Args:
        name (str): Logger name (usually use __name__ or a fixed package name).
        log_file (str): File to write logs to.
        level (int): Logging level.
        log_format (str): Log message format.
        date_format (str): Timestamp format.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.info("Logging configured for '%s'", name)
    return logger
