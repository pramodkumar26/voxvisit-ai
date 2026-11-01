"""Idempotency for state-changing requests.

A request claims its (actor, operation, key) scope inside the same transaction that performs the
change and completes the record just before commit. A visible record is therefore always a
completed one, and a rolled-back attempt leaves nothing behind, so a failed request can be retried
with the same key. Concurrent retries are serialized by the unique scope index: the second insert
waits for the first transaction, then returns its stored result (or proceeds if it rolled back).

Stored response snapshots are the sanitized response models, validated, scanned for forbidden
fields, and size-bounded. They never hold credentials, tokens, or unrestricted patient data.
"""

import hashlib
import hmac
import json
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Connection, func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.errors import ConflictError
from app.db.models import IdempotencyRecord

SCOPE_CONSTRAINT = "uq_idempotency_records_scope_key"
MAX_SNAPSHOT_BYTES = 4096
FORBIDDEN_FRAGMENTS = ("token", "pin", "password", "secret", "hash", "csrf", "cookie", "credential")


def request_hash(**parts: Any) -> bytes:
    """A stable digest of the normalized request parameters."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).digest()


def sanitized_snapshot(model: BaseModel) -> dict[str, Any]:
    """The stored form of a response. Refuses anything that could carry a secret."""
    snapshot = model.model_dump(mode="json")
    _reject_forbidden_keys(snapshot)
    if len(json.dumps(snapshot).encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("response snapshot is too large")
    return snapshot


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
                raise ValueError("response snapshot contains a forbidden field")
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def claim(
    connection: Connection,
    *,
    patient_id: uuid.UUID,
    session_id: uuid.UUID,
    operation: str,
    key: str,
    digest: bytes,
) -> tuple[uuid.UUID | None, dict[str, Any] | None]:
    """Claim the key for this transaction, or return the stored result of an earlier request.

    Returns (idempotency_id, None) when this transaction owns the key and must do the work, or
    (None, snapshot) when an earlier request already completed with the same parameters.
    """
    claimed = connection.execute(
        insert(IdempotencyRecord)
        .values(
            actor_type="PATIENT",
            actor_id=patient_id,
            session_id=session_id,
            operation=operation,
            idempotency_key=key,
            request_hash=digest,
            status="IN_PROGRESS",
        )
        .on_conflict_do_nothing(constraint=SCOPE_CONSTRAINT)
        .returning(IdempotencyRecord.idempotency_id)
    ).scalar_one_or_none()
    if claimed is not None:
        return claimed, None

    existing = connection.execute(
        select(
            IdempotencyRecord.request_hash,
            IdempotencyRecord.status,
            IdempotencyRecord.response_snapshot,
        ).where(
            IdempotencyRecord.actor_type == "PATIENT",
            IdempotencyRecord.actor_id == patient_id,
            IdempotencyRecord.operation == operation,
            IdempotencyRecord.idempotency_key == key,
        )
    ).one()
    if not hmac.compare_digest(existing.request_hash, digest):
        raise ConflictError(
            "IDEMPOTENCY_KEY_REUSED",
            "That idempotency key was already used for a different request.",
        )
    if existing.status != "COMPLETED" or existing.response_snapshot is None:
        raise ConflictError("REQUEST_IN_PROGRESS", "That request is still being processed.")
    return None, existing.response_snapshot


def complete(
    connection: Connection,
    idempotency_id: uuid.UUID,
    *,
    result_type: str,
    result_id: uuid.UUID,
    snapshot: dict[str, Any],
) -> None:
    connection.execute(
        update(IdempotencyRecord)
        .where(IdempotencyRecord.idempotency_id == idempotency_id)
        .values(
            status="COMPLETED",
            response_code=200,
            result_type=result_type,
            result_id=result_id,
            response_snapshot=snapshot,
            completed_at=func.clock_timestamp(),
        )
    )
