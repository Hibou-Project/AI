from src.settings import SETTINGS

import colorlog
import logging


def _get_log_level():
    """Convert LOG_LEVEL string to logging constant."""
    level = getattr(logging, str(SETTINGS.LOG_LEVEL).upper(), logging.INFO)
    return level


class CustomLogger:
    """Modular logger: use different names for different processes/files.
    Each logger writes to its own file and to the console.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

        # Avoid duplicate handlers when reusing the same logger name
        if self.logger.handlers:
            return

        level = _get_log_level()

        self.logger.setLevel(level)
        self.logger.propagate = False


        # Colored formatter for console
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s %(name)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Console handler: logs always visible in terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


def update_global_log_level() -> None:
    """Update the log level of all app loggers and their handlers from SETTINGS.LOG_LEVEL."""
    level = _get_log_level()
    suppressed = {
        # "ultralytics"
    }
    # Update root logger
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
    # Update all registered loggers (skip suppressed external libs)
    for name in logging.root.manager.loggerDict:
        if name in suppressed:
            continue
        logger = logging.getLogger(name)
        if not isinstance(logger, logging.Logger):
            continue
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def blank_line_module(log_level="DEBUG", how_many_lines=1):
    if SETTINGS.LOG_LEVEL == log_level:
        for _ in range(how_many_lines):
            print("")


# external logger
# logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
