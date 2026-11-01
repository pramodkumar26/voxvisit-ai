"""Preparing an appointment action (backs the prepare_appointment_action tool).

Preparation validates the request, stores a session-bound pending action with an exact summary and
a 10-minute expiry, and stops. It never creates, moves, cancels, holds, or reserves anything, and
it only reads appointments and slots. It imports no INSERT/UPDATE/DELETE constructs, no slot,
history, or outbox models, and nothing from appointment_transactions: committing is a separate
operation that only the authenticated UI can reach (a test enforces this boundary).

A newer preparation retires the session's other ACTIVE pending actions in the same transaction
(see pending_action_retirement), so a review screen the patient has moved on from cannot commit.

The creation time and expiry are both taken from the database clock in one statement, so neither
the application server nor the browser can influence them.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, bindparam, select, text
from sqlalchemy.dialects.postgresql import JSONB

from app.auth.service import Principal
from app.core.errors import (
    ConflictError,
    InvalidArgumentError,
    ResourceNotAvailableError,
    SelfServiceCutoffError,
    SlotUnavailableError,
)
from app.db.models import Appointment
from app.models.appointment_actions import (
    ActionSummary,
    PendingActionView,
    PrepareBook,
    PrepareCancel,
    PrepareRequest,
    PrepareReschedule,
)
from app.services import idempotency
from app.services.action_support import (
    PREPARE_OPERATION,
    appointment_time,
    describe,
    lock_patient_session,
    record_success,
    require_pending_action_owner,
    slot_details,
)
from app.services.availability import is_slot_bookable
from app.services.pending_action_retirement import retire_active_pending_actions
from app.services.write_access import write_transaction

__all__ = ["prepare_appointment_action"]

PENDING_ACTION_MINUTES = 10

_INSERT = text(
    """
    INSERT INTO pending_actions
        (session_id, action_type, target_type, target_id, summary, expected_versions,
         status, created_at, expires_at, request_id)
    SELECT :session_id, :action_type, :target_type, :target_id, :summary, :expected_versions,
           'ACTIVE', clock.now, clock.now + make_interval(mins => :minutes), :request_id
    FROM (SELECT clock_timestamp() AS now) AS clock
    RETURNING pending_action_id, created_at, expires_at
    """
).bindparams(
    bindparam("summary", type_=JSONB),
    bindparam("expected_versions", type_=JSONB),
)


def prepare_appointment_action(
    engine: Any,
    *,
    principal: Principal,
    request: PrepareRequest,
    idempotency_key: str,
    request_id: uuid.UUID,
    now: datetime,
) -> tuple[PendingActionView, bool]:
    """Returns the pending action and whether it is a replay of an earlier identical request."""
    digest = idempotency.request_hash(**request.model_dump(mode="json"))
    with write_transaction(engine) as connection:
        claimed, replay = idempotency.claim(
            connection,
            patient_id=principal.patient_id,
            session_id=principal.session_id,
            operation=PREPARE_OPERATION,
            key=idempotency_key,
            digest=digest,
        )
        lock_patient_session(connection, principal, now)
        if replay is not None:
            view = PendingActionView.model_validate(replay)
            require_pending_action_owner(connection, view.pending_action_id, principal)
            return view, True

        prepared = _validate(connection, principal, request, now)
        # Under the session lock taken above, and in this transaction: a confirmation of an
        # earlier review either commits before this point or finds it INVALIDATED.
        superseded = retire_active_pending_actions(connection, principal.session_id, now)
        created = connection.execute(
            _INSERT,
            {
                "session_id": principal.session_id,
                "action_type": request.action,
                "target_type": prepared["target_type"],
                "target_id": prepared["target_id"],
                "summary": prepared["summary"].model_dump(mode="json"),
                "expected_versions": prepared["expected_versions"],
                "minutes": PENDING_ACTION_MINUTES,
                "request_id": request_id,
            },
        ).one()
        view = PendingActionView(
            pending_action_id=created.pending_action_id,
            action=request.action,
            summary=prepared["summary"],
            created_at=created.created_at,
            expires_at=created.expires_at,
        )
        idempotency.complete(
            connection,
            claimed,
            result_type="PENDING_ACTION",
            result_id=created.pending_action_id,
            snapshot=idempotency.sanitized_snapshot(view),
        )
        record_success(
            connection,
            "APPOINTMENT_ACTION_PREPARED",
            principal=principal,
            now=now,
            request_id=request_id,
            resource_type="PENDING_ACTION",
            resource_id=created.pending_action_id,
            details={"action": request.action, "superseded": str(superseded)},
        )
        return view, False


def _validate(
    connection: Connection, principal: Principal, request: PrepareRequest, now: datetime
) -> dict[str, Any]:
    if isinstance(request, PrepareBook):
        return _prepare_book(connection, request, now)
    if isinstance(request, PrepareReschedule):
        return _prepare_reschedule(connection, principal, request, now)
    return _prepare_cancel(connection, principal, request, now)


def _bookable_slot(connection: Connection, slot_id: uuid.UUID, now: datetime) -> Any:
    details = slot_details(connection, slot_id)
    if details is None:
        raise ResourceNotAvailableError()
    if not is_slot_bookable(connection, slot_id, now):
        raise SlotUnavailableError()
    return details


def _owned_confirmed_appointment(
    connection: Connection, principal: Principal, appointment_id: uuid.UUID, now: datetime
) -> tuple[Any, Any]:
    """The caller's own CONFIRMED appointment and its slot, still before the self-service cutoff.

    An appointment that is absent or belongs to someone else is reported identically.
    """
    appointment = connection.execute(
        select(
            Appointment.appointment_id, Appointment.status, Appointment.version, Appointment.slot_id
        ).where(
            Appointment.appointment_id == appointment_id,
            Appointment.patient_id == principal.patient_id,
        )
    ).one_or_none()
    if appointment is None:
        raise ResourceNotAvailableError()
    if appointment.status != "CONFIRMED":
        raise ConflictError(
            "APPOINTMENT_NOT_ACTIVE",
            "That appointment is no longer active. Please review your current appointments.",
        )
    slot = slot_details(connection, appointment.slot_id)
    if slot.starts_at <= now:
        raise SelfServiceCutoffError()
    return appointment, slot


def _prepare_book(connection: Connection, request: PrepareBook, now: datetime) -> dict[str, Any]:
    slot = _bookable_slot(connection, request.slot_id, now)
    proposed = appointment_time(slot)
    return {
        "target_type": "SLOT",
        "target_id": slot.slot_id,
        "summary": ActionSummary(
            operation="BOOK",
            description=describe("Book an appointment", proposed),
            proposed=proposed,
        ),
        "expected_versions": {"slot_id": str(slot.slot_id), "slot_version": slot.version},
    }


def _prepare_reschedule(
    connection: Connection, principal: Principal, request: PrepareReschedule, now: datetime
) -> dict[str, Any]:
    appointment, current_slot = _owned_confirmed_appointment(
        connection, principal, request.appointment_id, now
    )
    if request.new_slot_id == appointment.slot_id:
        raise InvalidArgumentError("Choose a different time than the current appointment.")
    new_slot = _bookable_slot(connection, request.new_slot_id, now)
    current, proposed = appointment_time(current_slot), appointment_time(new_slot)
    return {
        "target_type": "APPOINTMENT",
        "target_id": appointment.appointment_id,
        "summary": ActionSummary(
            operation="RESCHEDULE",
            description=(
                f"{describe('Move your appointment', current)} "
                f"to {describe('an appointment', proposed)}"
            ),
            current=current,
            proposed=proposed,
        ),
        "expected_versions": {
            "appointment_id": str(appointment.appointment_id),
            "appointment_version": appointment.version,
            "old_slot_id": str(appointment.slot_id),
            "old_slot_version": current_slot.version,
            "new_slot_id": str(new_slot.slot_id),
            "new_slot_version": new_slot.version,
        },
    }


def _prepare_cancel(
    connection: Connection, principal: Principal, request: PrepareCancel, now: datetime
) -> dict[str, Any]:
    appointment, slot = _owned_confirmed_appointment(
        connection, principal, request.appointment_id, now
    )
    current = appointment_time(slot)
    return {
        "target_type": "APPOINTMENT",
        "target_id": appointment.appointment_id,
        "summary": ActionSummary(
            operation="CANCEL",
            description=describe("Cancel your appointment", current),
            current=current,
        ),
        "expected_versions": {
            "appointment_id": str(appointment.appointment_id),
            "appointment_version": appointment.version,
            "old_slot_id": str(appointment.slot_id),
            "old_slot_version": slot.version,
        },
    }
