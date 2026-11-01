import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "csrf",
    "database_url",
    "password",
    "pin",
    "secret",
    "session",
    "token",
}
ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(password|passwd|pin|token|secret|authorization|cookie|database_url|credentials?)"
    r"\s*([=:])\s*([^\s,;]+)"
)
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
URL_CREDENTIALS_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^@\s/]+)@")


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if key.lower() in SENSITIVE_KEYS else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if not isinstance(value, str):
        return value

    redacted = URL_CREDENTIALS_PATTERN.sub(r"\1[REDACTED]@", value)
    redacted = ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED}", redacted)
    return BEARER_PATTERN.sub(f"Bearer {REDACTED}", redacted)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive(record.msg)
        if record.args:
            record.args = redact_sensitive(record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("error_type", "request_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(redact_sensitive(payload), separators=(",", ":"))


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactionFilter())
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
