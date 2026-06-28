import logging
import json
import os
import time
from datetime import datetime, timezone
from app.config import get_settings

settings = get_settings()


class JSONLineHandler(logging.Handler):
    """Writes one JSON object per line to a log file."""

    def __init__(self, log_file: str):
        super().__init__()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self._file = open(log_file, "a", buffering=1, encoding="utf-8")

    # Standard attributes on every LogRecord — skip these when serializing extras
    _SKIP = frozenset(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message", "asctime"}

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Capture any extra fields attached by logger.info(..., extra={...})
        # or by manually setting attributes on the LogRecord.
        for k, v in record.__dict__.items():
            if k not in self._SKIP:
                entry[k] = v
        self._file.write(json.dumps(entry, default=str) + "\n")

    def close(self) -> None:
        self._file.close()
        super().close()


def get_logger(name: str = "rag_service") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Console handler (human-readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s"))
    logger.addHandler(console)

    # Structured JSON file handler
    try:
        logger.addHandler(JSONLineHandler(settings.log_file))
    except Exception:
        pass

    return logger


def log_request(
    logger: logging.Logger,
    tenant_id: str,
    query: str,
    latency_ms: float,
    tokens_used: int,
    cached: bool,
) -> None:
    record = logging.LogRecord(
        name="rag_service",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="query_handled",
        args=(),
        exc_info=None,
    )
    record.extra = {
        "tenant_id": tenant_id,
        "query_preview": query[:100],
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
        "cached": cached,
    }
    logger.handle(record)
