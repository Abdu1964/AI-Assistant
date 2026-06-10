import logging
import logging.handlers
import os
import sys


_LEVEL_COLORS = {
    logging.DEBUG:    "\033[36m",     # cyan
    logging.INFO:     "\033[32m",     # green
    logging.WARNING:  "\033[33m",     # yellow
    logging.ERROR:    "\033[31m",     # red
    logging.CRITICAL: "\033[1;31m",   # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    _FMT = "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{_RESET}"
        result = super().format(record)
        record.levelname = original_levelname
        return result


_PLAIN_FMT = "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_NOISY_LOGGERS = [
    "urllib3", "httpx", "httpcore", "httpcore.http11",
    "pymongo", "pymongo.command", "pymongo.serverSelection",
    "neo4j", "neo4j.io", "neo4j.pool",
    "qdrant_client", "qdrant_client.http",
    "sentence_transformers", "transformers",
    "biocypher", "biocypher._core",
    "werkzeug", "engineio", "socketio",
    "PIL", "openai", "google",
]


def setup_logging(log_dir: str = "logfiles", level: int = logging.INFO) -> None:
    """Configure root logger once at app startup. Safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(level)

    # Console — colored
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColorFormatter())
    root.addHandler(console)

    # Rotating file — plain text
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "assistant.log"),
        when="D",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=_PLAIN_FMT, datefmt=_DATEFMT))
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
