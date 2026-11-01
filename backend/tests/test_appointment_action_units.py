import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.core.errors import ApplicationError, ConflictError
from app.models.appointment_actions import ConfirmRequest, PrepareRequest
from app.services import idempotency, write_access
from app.services.pending_actions import PENDING_ACTION_MINUTES

PREPARE = TypeAdapter(PrepareRequest)
SLOT = "00000000-0000-4000-8000-000000000001"


def test_request_hashes_ignore_key_order_and_detect_any_change() -> None:
    assert idempotency.request_hash(a="1", b="2") == idempotency.request_hash(b="2", a="1")
    assert idempotency.request_hash(a="1", b="2") != idempotency.request_hash(a="1", b="3")
    assert len(idempotency.request_hash(a="1")) == 32


class Harmless(BaseModel):
    result: str = "BOOKED"
    nested: dict[str, str] = {"clinic": "Central Clinic"}


@pytest.mark.parametrize(
    "field",
    ["session_token", "pin", "password_hash", "csrf_token", "secret", "cookie", "Credential"],
)
def test_snapshots_refuse_any_field_that_could_carry_a_secret(field: str) -> None:
    model = type("Leaky", (BaseModel,), {"__annotations__": {field: str}, field: "x"})()

    with pytest.raises(ValueError, match="forbidden field"):
        idempotency.sanitized_snapshot(model)


def test_snapshots_refuse_secrets_nested_inside_lists_and_objects() -> None:
    class Nested(BaseModel):
        items: list[dict[str, str]] = [{"ok": "1"}, {"auth_token": "x"}]

    with pytest.raises(ValueError, match="forbidden field"):
        idempotency.sanitized_snapshot(Nested())


def test_snapshots_are_bounded_in_size() -> None:
    class Big(BaseModel):
        text: str = "x" * (idempotency.MAX_SNAPSHOT_BYTES + 1)

    with pytest.raises(ValueError, match="too large"):
        idempotency.sanitized_snapshot(Big())
    assert idempotency.sanitized_snapshot(Harmless())["result"] == "BOOKED"


def test_the_approved_timeouts_and_expiry_are_configured() -> None:
    assert write_access.WRITE_TIMEOUT_MS == 12000
    assert PENDING_ACTION_MINUTES == 10


def test_conflict_errors_carry_a_machine_readable_reason() -> None:
    error = ConflictError("CONFIRMATION_EXPIRED", "expired")

    assert isinstance(error, ApplicationError)
    assert (error.status_code, error.code) == (409, "CONFLICT")
    assert error.body() == {
        "code": "CONFLICT",
        "message": "expired",
        "reason": "CONFIRMATION_EXPIRED",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "BOOK", "slot_id": SLOT},
        {"action": "CANCEL", "appointment_id": SLOT},
        {"action": "RESCHEDULE", "appointment_id": SLOT, "new_slot_id": SLOT},
    ],
)
def test_the_three_documented_request_shapes_parse(payload) -> None:
    assert PREPARE.validate_python(payload).action == payload["action"]


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "BOOK", "slot_id": SLOT, "patient_id": SLOT},
        {"action": "BOOK", "slot_id": SLOT, "appointment_id": SLOT},
        {"action": "CANCEL", "appointment_id": SLOT, "new_slot_id": SLOT},
        {"action": "JOIN", "slot_id": SLOT},
        {"slot_id": SLOT},
    ],
)
def test_undocumented_fields_and_actions_are_rejected(payload) -> None:
    with pytest.raises(ValidationError):
        PREPARE.validate_python(payload)


@pytest.mark.parametrize(
    "payload", [{}, {"confirmed": False}, {"confirmed": 1}, {"confirmed": None}]
)
def test_confirmation_requires_a_literal_true(payload) -> None:
    with pytest.raises(ValidationError):
        ConfirmRequest.model_validate(payload)
    assert ConfirmRequest.model_validate({"confirmed": True}).confirmed is True
