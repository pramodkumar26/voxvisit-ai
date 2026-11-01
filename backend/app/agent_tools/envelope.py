"""The approved tool response envelope and the mapping from backend errors into it.

Errors are sanitized by construction: only an approved code, a fixed message, a retry flag, and
optionally a stable reason, the names of invalid fields, or a small list of choices. Raw exception
text, SQL, stack traces, and submitted values never appear.
"""

import re
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from app.core.errors import ApplicationError

RETRYABLE_CODES = frozenset({"DEPENDENCY_UNAVAILABLE", "RATE_LIMITED"})
MAX_REPORTED_FIELDS = 10
MAX_FIELD_NAME_LENGTH = 60
MAX_VALIDATION_CATEGORY_LENGTH = 60
UNION_TAGS = frozenset({"BOOK", "RESCHEDULE", "CANCEL"})
SAFE_FIELD_CHARACTERS = re.compile(r"[^A-Za-z0-9_.]")
SAFE_CATEGORY_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]")
KNOWN_FIELD_NAMES = frozenset(
    {
        "action",
        "actor_type",
        "browse",
        "clinic_id",
        "clinic_name",
        "confirm",
        "confirmed",
        "csrf_token",
        "cursor",
        "doctor_id",
        "doctor_name",
        "end_date",
        "idempotency_key",
        "limit",
        "new_slot_id",
        "patient_id",
        "pending_action_id",
        "role",
        "session_id",
        "session_token",
        "slot_id",
        "specialty",
        "start_date",
        "status",
        "time_preference",
    }
)
UNKNOWN_FIELD = "unknown_field"


class ValidationDiagnostic(BaseModel):
    field: str
    value_type: str
    category: str


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool
    reason: str | None = None
    fields: list[str] | None = None
    validation: list[ValidationDiagnostic] | None = None
    choices: list[dict[str, Any]] | None = None


class ToolResponse(BaseModel):
    ok: bool
    request_id: UUID
    data: dict[str, Any] | None
    error: ToolError | None

    def to_dict(self) -> dict[str, Any]:
        """The wire form: always `ok`, `request_id`, `data`, and `error`."""
        return {
            "ok": self.ok,
            "request_id": str(self.request_id),
            "data": self.data,
            "error": self.error.model_dump(exclude_none=True) if self.error else None,
        }


def success(request_id: UUID, data: dict[str, Any]) -> ToolResponse:
    return ToolResponse(ok=True, request_id=request_id, data=data, error=None)


def failure(request_id: UUID, error: ToolError) -> ToolResponse:
    return ToolResponse(ok=False, request_id=request_id, data=None, error=error)


def error_from_application(exc: ApplicationError) -> ToolError:
    return ToolError(
        code=exc.code,
        message=exc.message,
        retryable=exc.code in RETRYABLE_CODES,
        reason=exc.reason,
    )


def _field_name(location: tuple) -> str:
    """A field path without the action tag pydantic adds for a union branch (BOOK, CANCEL, ...)."""
    parts = []
    for part in (part for part in location if part not in UNION_TAGS):
        if isinstance(part, str) and part in KNOWN_FIELD_NAMES:
            parts.append(part)
        elif isinstance(part, int):
            parts.append("item")
        else:
            parts.append(UNKNOWN_FIELD)
    parts = parts or ["arguments"]
    return SAFE_FIELD_CHARACTERS.sub("?", ".".join(parts))[:MAX_FIELD_NAME_LENGTH]


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list | tuple):
        return "array"
    return "other"


def _submitted_value(raw: Mapping[str, Any] | None, location: tuple) -> tuple[bool, Any]:
    if raw is None:
        return False, None
    current: Any = raw
    for part in (part for part in location if part not in UNION_TAGS):
        in_mapping = isinstance(current, Mapping) and part in current
        in_list = isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current)
        if not (in_mapping or in_list):
            return False, None
        current = current[part]
    return True, current


def _validation_diagnostic(
    error: dict[str, Any], raw: Mapping[str, Any] | None
) -> ValidationDiagnostic:
    location = tuple(error.get("loc", ()))
    category = SAFE_CATEGORY_CHARACTERS.sub("?", str(error.get("type", "validation_error")))[
        :MAX_VALIDATION_CATEGORY_LENGTH
    ]

    if category in {"union_tag_invalid", "union_tag_not_found"} and raw is not None:
        discriminator = str(error.get("ctx", {}).get("discriminator", "")).strip("'\"")
        if discriminator in KNOWN_FIELD_NAMES and discriminator in raw:
            location = (discriminator,)

    field = _field_name(location)
    found, value = _submitted_value(raw, location)
    value_type = _json_value_type(value) if found else "missing" if raw is not None else "unknown"
    return ValidationDiagnostic(field=field, value_type=value_type, category=category)


def error_from_validation(exc: ValidationError, raw: Mapping[str, Any] | None = None) -> ToolError:
    """Describe invalid fields and value shapes without retaining or echoing submitted values."""
    diagnostics = [_validation_diagnostic(error, raw) for error in exc.errors()]
    unique = {(item.field, item.value_type, item.category): item for item in diagnostics}
    bounded = [unique[key] for key in sorted(unique)[:MAX_REPORTED_FIELDS]]
    names = {item.field for item in bounded}
    return ToolError(
        code="INVALID_ARGUMENT",
        message="The tool arguments were not valid.",
        retryable=False,
        fields=sorted(names)[:MAX_REPORTED_FIELDS],
        validation=bounded,
    )


def simple_error(code: str, message: str, *, reason: str | None = None) -> ToolError:
    return ToolError(code=code, message=message, retryable=code in RETRYABLE_CODES, reason=reason)


INTERNAL_ERROR = ToolError(
    code="INTERNAL_ERROR",
    message="The request could not be completed.",
    retryable=False,
)
