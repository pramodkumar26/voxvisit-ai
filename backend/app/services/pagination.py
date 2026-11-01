"""Opaque keyset-pagination cursors.

A cursor carries the last returned sort position plus a fingerprint of the search that created
it, so it cannot be replayed against different filters or another patient's listing. It is not a
credential: every query still applies its own authorization filters.
"""

import base64
import binascii
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.errors import InvalidArgumentError

MAX_PAGE_SIZE = 20
MAX_CURSOR_LENGTH = 512
INVALID_CURSOR = "The pagination cursor is not valid for this search."


def fingerprint(**filters: Any) -> str:
    canonical = json.dumps(filters, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def encode_cursor(kind: str, search: str, position: dict[str, str]) -> str:
    payload = json.dumps({"k": kind, "f": search, "p": position}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str, kind: str, search: str) -> dict[str, str]:
    if len(cursor) > MAX_CURSOR_LENGTH:
        raise InvalidArgumentError(INVALID_CURSOR)
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
        position = payload["p"]
        valid = (
            payload["k"] == kind
            and payload["f"] == search
            and isinstance(position, dict)
            and all(
                isinstance(key, str) and isinstance(value, str) for key, value in position.items()
            )
        )
    except (binascii.Error, ValueError, KeyError, TypeError):
        raise InvalidArgumentError(INVALID_CURSOR) from None
    if not valid:
        raise InvalidArgumentError(INVALID_CURSOR)
    return position


def timestamp_position(starts_at: datetime, row_id: uuid.UUID) -> dict[str, str]:
    return {"t": starts_at.astimezone(UTC).isoformat(), "i": str(row_id)}


def parse_timestamp_position(position: dict[str, str]) -> tuple[datetime, uuid.UUID]:
    try:
        starts_at = datetime.fromisoformat(position["t"])
        if starts_at.tzinfo is None:
            raise ValueError
        return starts_at, uuid.UUID(position["i"])
    except (KeyError, ValueError):
        raise InvalidArgumentError(INVALID_CURSOR) from None
