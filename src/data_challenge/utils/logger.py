import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

_ANSI = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m" # bold red
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _ANSI.get(record.levelname, "")
        return f"{color}{super().format(record)}{_RESET}"


class DailyRotatingFileHandler(RotatingFileHandler):
    """Log handler that writes to daily files and rotates by size.

    File naming: YYYY-MM-DD.log, YYYY-MM-DD.1.log, YYYY-MM-DD.2.log (max backup_count).
    """

    def __init__(self, log_dir: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 3):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(log_dir, f"{self._current_date}.log")
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")

    def _refresh_date(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            new_path = os.path.abspath(os.path.join(self.log_dir, f"{today}.log"))
            self.baseFilename = new_path
            if self.stream:
                self.stream.close()
                self.stream = None
            self.stream = self._open()

    def emit(self, record):
        self._refresh_date()
        super().emit(record)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        base = self.baseFilename  # e.g. logs/2026-05-21.log
        stem = base[:-4] if base.endswith(".log") else base

        for i in range(self.backupCount - 1, 0, -1):
            src = f"{stem}.{i}.log"
            dst = f"{stem}.{i + 1}.log"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

        rollover_dest = f"{stem}.1.log"
        if os.path.exists(base):
            if os.path.exists(rollover_dest):
                os.remove(rollover_dest)
            os.rename(base, rollover_dest)

        self.stream = self._open()


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = DailyRotatingFileHandler(log_dir, max_bytes=max_bytes, backup_count=backup_count)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColorFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console_handler)
    return logger
