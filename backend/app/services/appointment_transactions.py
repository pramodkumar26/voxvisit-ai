"""Committing a confirmed appointment action: BOOK, RESCHEDULE, or CANCEL.

Every action runs as one READ COMMITTED transaction with a 12-second timeout. Locks are taken in
the canonical order from docs/database-design.md, skipping the tables an action does not need:

    idempotency claim -> auth_sessions -> pending_actions -> appointments -> slots (ascending id)

Nothing a request read before its locks is trusted: after the locks are held the transaction
rechecks the session, the pending action's database-time expiry, ownership, record versions, the
shared bookable-slot rules, and the self-service cutoff. Any failure raises, which rolls back every
statement (appointment, slots, history, outbox, pending action, idempotency record, audit event).
Success is reported only after commit.

Business time (bookable slot start, self-service cutoff, session deadlines) comes from the
injected application clock, as in availability search. Pending-action expiry uses PostgreSQL time
and the stored creation timestamp exclusively.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, Engine, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from app.auth.service import Principal
from app.core.errors import (
    ConflictError,
    ResourceNotAvailableError,
    SelfServiceCutoffError,
    SlotUnavailableError,
)
from app.db.models import Appointment, AppointmentHistory, OutboxEvent, PendingAction, Slot
from app.models.appointment_actions import AppointmentActionResult
from app.services import idempotency
from app.services.action_support import (
    CONFIRM_OPERATION,
    appointment_time,
    lock_patient_session,
    record_success,
    require_pending_action_owner,
    slot_details,
)
from app.services.availability import is_slot_bookable
from app.services.write_access import write_transaction

__all__ = ["confirm_appointment_action"]

CONFIRMED_SLOT_INDEX = "uq_appointments_confirmed_slot"

_EXPIRED = text(
    """
    SELECT clock_timestamp() >= LEAST(expires_at, created_at + interval '10 minutes')
    FROM pending_actions WHERE pending_action_id = :id
    """
)


@dataclass(frozen=True)
class Context:
    principal: Principal
    now: datetime
    request_id: uuid.UUID


def confirm_appointment_action(
    engine: Engine,
    *,
    principal: Principal,
    pending_action_id: uuid.UUID,
    idempotency_key: str,
    request_id: uuid.UUID,
    now: datetime,
) -> tuple[AppointmentActionResult, bool]:
    """Commit a prepared action. Returns the result and whether it replays an earlier request."""
    digest = idempotency.request_hash(pending_action_id=str(pending_action_id))
    context = Context(principal, now, request_id)
    try:
        with write_transaction(engine) as connection:
            claimed, replay = idempotency.claim(
                connection,
                patient_id=principal.patient_id,
                session_id=principal.session_id,
                operation=CONFIRM_OPERATION,
                key=idempotency_key,
                digest=digest,
            )
            # Even a replay re-verifies the session and account, and that the pending action is
            # this session's own, before anything stored is returned.
            lock_patient_session(connection, principal, now)
            if replay is not None:
                require_pending_action_owner(connection, pending_action_id, principal)
                return AppointmentActionResult.model_validate(replay), True

            pending = _lock_pending_action(connection, pending_action_id, principal)
            result = _HANDLERS[pending.action_type](connection, context, pending)

            connection.execute(
                update(PendingAction)
                .where(PendingAction.pending_action_id == pending_action_id)
                .values(status="CONSUMED")
            )
            idempotency.complete(
                connection,
                claimed,
                result_type="APPOINTMENT",
                result_id=result.appointment_id,
                snapshot=idempotency.sanitized_snapshot(result),
            )
            record_success(
                connection,
                f"APPOINTMENT_{result.result}",
                principal=principal,
                now=now,
                request_id=request_id,
                resource_type="APPOINTMENT",
                resource_id=result.appointment_id,
                details={"pending_action_id": str(pending_action_id)},
            )
            return result, False
    except IntegrityError as exc:
        # The unique index on confirmed appointments is the final guard against a double booking.
        if getattr(getattr(exc.orig, "diag", None), "constraint_name", None) == (
            CONFIRMED_SLOT_INDEX
        ):
            raise SlotUnavailableError() from None
        raise


# --- locking and shared checks -----------------------------------------------------------------


def _stale() -> ConflictError:
    return ConflictError(
        "CONFIRMATION_STALE",
        "The details changed after this confirmation was prepared. "
        "Please review the current details and confirm again.",
    )


def _lock_pending_action(
    connection: Connection, pending_action_id: uuid.UUID, principal: Principal
):
    """Lock the caller's own pending action. Another session's action looks like a missing one."""
    pending = connection.execute(
        select(
            PendingAction.pending_action_id,
            PendingAction.action_type,
            PendingAction.target_id,
            PendingAction.expected_versions,
            PendingAction.status,
        )
        .where(
            PendingAction.pending_action_id == pending_action_id,
            PendingAction.session_id == principal.session_id,
        )
        .with_for_update()
    ).one_or_none()
    if pending is None or pending.action_type not in _HANDLERS:
        raise ResourceNotAvailableError()
    if pending.status == "CONSUMED":
        raise ConflictError(
            "CONFIRMATION_CONSUMED",
            "This confirmation was already used. Please review your current appointments.",
        )
    if pending.status == "INVALIDATED":
        raise ConflictError(
            "CONFIRMATION_INVALIDATED",
            "This confirmation is no longer valid. Please review the current details.",
        )
    if pending.status != "ACTIVE":
        raise _expired()
    return pending


def _expected(pending: Any, *names: str) -> dict[str, Any]:
    """The ids and versions recorded at preparation, parsed strictly.

    Anything missing or malformed fails closed as stale, including an action prepared before a
    version was recorded: it cannot commit, and the patient simply reviews and confirms again.
    """
    recorded = pending.expected_versions
    parsed: dict[str, Any] = {}
    try:
        for name in names:
            value = recorded[name]
            if name.endswith("_id"):
                value = uuid.UUID(value)
            elif not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(name)
            parsed[name] = value
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _stale() from None
    return parsed


def _expired() -> ConflictError:
    return ConflictError(
        "CONFIRMATION_EXPIRED",
        "This confirmation expired. Please review the current details and confirm again.",
    )


def _require_unexpired(connection: Connection, pending: Any) -> None:
    """Database time against the stored creation timestamp, evaluated after every lock is held."""
    if connection.execute(_EXPIRED, {"id": pending.pending_action_id}).scalar_one():
        raise _expired()


def _lock_appointment(connection: Connection, pending: Any, principal: Principal) -> Any:
    appointment = connection.execute(
        select(
            Appointment.appointment_id,
            Appointment.status,
            Appointment.version,
            Appointment.slot_id,
        )
        .where(
            Appointment.appointment_id == pending.target_id,
            Appointment.patient_id == principal.patient_id,
        )
        .with_for_update()
    ).one_or_none()
    if appointment is None:
        raise ResourceNotAvailableError()
    if appointment.status != "CONFIRMED":
        raise ConflictError(
            "APPOINTMENT_NOT_ACTIVE",
            "That appointment is no longer active. Please review your current appointments.",
        )
    return appointment


def _lock_slots(connection: Connection, *slot_ids: uuid.UUID) -> dict[uuid.UUID, Any]:
    """Lock slots one at a time in ascending id order, whatever order the caller names them in."""
    locked = {}
    for slot_id in sorted(set(slot_ids)):
        row = connection.execute(
            select(
                Slot.slot_id,
                Slot.clinic_id,
                Slot.doctor_id,
                Slot.starts_at,
                Slot.ends_at,
                Slot.status,
                Slot.version,
            )
            .where(Slot.slot_id == slot_id)
            .with_for_update()
        ).one_or_none()
        if row is None:
            raise ResourceNotAvailableError()
        locked[slot_id] = row
    return locked


def _change_slot(connection: Connection, slot: Any, status: str) -> int:
    """Version-checked slot transition. Returns the new version."""
    new_version = connection.execute(
        update(Slot)
        .where(Slot.slot_id == slot.slot_id, Slot.version == slot.version)
        .values(status=status, version=slot.version + 1)
        .returning(Slot.version)
    ).scalar_one_or_none()
    if new_version is None:
        raise _stale()
    return new_version


def _append_history(
    connection: Connection,
    context: Context,
    appointment_id: uuid.UUID,
    action: str,
    *,
    old_slot_id: uuid.UUID | None,
    new_slot_id: uuid.UUID | None,
) -> None:
    connection.execute(
        insert(AppointmentHistory).values(
            appointment_id=appointment_id,
            action=action,
            actor_id=context.principal.patient_id,
            actor_type="PATIENT",
            old_slot_id=old_slot_id,
            new_slot_id=new_slot_id,
            request_id=context.request_id,
        )
    )


def _emit_slot_released(
    connection: Connection, slot: Any, released_version: int, *, reason: str
) -> None:
    """Durable work for waitlist processing, written in the releasing transaction.

    The dedupe key includes the slot version produced by the release, so each actual release has
    exactly one event and a retry can never add a second.
    """
    connection.execute(
        insert(OutboxEvent).values(
            event_type="SLOT_RELEASED",
            aggregate_type="SLOT",
            aggregate_id=slot.slot_id,
            dedupe_key=f"SLOT_RELEASED:{slot.slot_id}:{released_version}",
            payload={
                "slot_id": str(slot.slot_id),
                "clinic_id": str(slot.clinic_id),
                "doctor_id": str(slot.doctor_id),
                "starts_at": slot.starts_at.isoformat(),
                "ends_at": slot.ends_at.isoformat(),
                "reason": reason,
            },
        )
    )


def _result(
    connection: Connection,
    result: str,
    appointment_id: uuid.UUID,
    status: str,
    slot_id: uuid.UUID,
    previous_slot_id: uuid.UUID | None = None,
) -> AppointmentActionResult:
    previous = None
    if previous_slot_id is not None:
        previous = appointment_time(slot_details(connection, previous_slot_id))
    return AppointmentActionResult(
        result=result,
        appointment_id=appointment_id,
        status=status,
        appointment=appointment_time(slot_details(connection, slot_id)),
        previous=previous,
        completed_at=connection.execute(select(func.clock_timestamp())).scalar_one(),
    )


# --- the three actions ------------------------------------------------------------------------


def _book(connection: Connection, context: Context, pending: Any) -> AppointmentActionResult:
    expected = _expected(pending, "slot_version")
    slot_id = pending.target_id
    slot = _lock_slots(connection, slot_id)[slot_id]
    _require_unexpired(connection, pending)
    if not is_slot_bookable(connection, slot_id, context.now):
        raise SlotUnavailableError()
    if slot.version != expected["slot_version"]:
        raise _stale()

    appointment_id = connection.execute(
        insert(Appointment)
        .values(patient_id=context.principal.patient_id, slot_id=slot_id, status="CONFIRMED")
        .returning(Appointment.appointment_id)
    ).scalar_one()
    _change_slot(connection, slot, "BOOKED")
    _append_history(
        connection, context, appointment_id, "BOOKED", old_slot_id=None, new_slot_id=slot_id
    )
    return _result(connection, "BOOKED", appointment_id, "CONFIRMED", slot_id)


def _reschedule(connection: Connection, context: Context, pending: Any) -> AppointmentActionResult:
    expected = _expected(
        pending,
        "old_slot_id",
        "old_slot_version",
        "new_slot_id",
        "new_slot_version",
        "appointment_version",
    )
    old_slot_id, new_slot_id = expected["old_slot_id"], expected["new_slot_id"]

    appointment = _lock_appointment(connection, pending, context.principal)
    if appointment.slot_id != old_slot_id or appointment.version != expected["appointment_version"]:
        raise _stale()
    slots = _lock_slots(connection, old_slot_id, new_slot_id)
    old_slot, new_slot = slots[old_slot_id], slots[new_slot_id]
    _require_unexpired(connection, pending)
    if old_slot.status != "BOOKED" or old_slot.version != expected["old_slot_version"]:
        raise _stale()
    if old_slot.starts_at <= context.now:
        raise SelfServiceCutoffError()
    if not is_slot_bookable(connection, new_slot_id, context.now):
        raise SlotUnavailableError()
    if new_slot.version != expected["new_slot_version"]:
        raise _stale()

    released_version = _change_slot(connection, old_slot, "AVAILABLE")
    _change_slot(connection, new_slot, "BOOKED")
    moved = connection.execute(
        update(Appointment)
        .where(
            Appointment.appointment_id == appointment.appointment_id,
            Appointment.version == appointment.version,
        )
        .values(slot_id=new_slot_id, version=appointment.version + 1)
    )
    if moved.rowcount != 1:
        raise _stale()
    _append_history(
        connection,
        context,
        appointment.appointment_id,
        "RESCHEDULED",
        old_slot_id=old_slot_id,
        new_slot_id=new_slot_id,
    )
    _emit_slot_released(connection, old_slot, released_version, reason="RESCHEDULED")
    return _result(
        connection,
        "RESCHEDULED",
        appointment.appointment_id,
        "CONFIRMED",
        new_slot_id,
        previous_slot_id=old_slot_id,
    )


def _cancel(connection: Connection, context: Context, pending: Any) -> AppointmentActionResult:
    expected = _expected(pending, "old_slot_id", "old_slot_version", "appointment_version")
    old_slot_id = expected["old_slot_id"]

    appointment = _lock_appointment(connection, pending, context.principal)
    if appointment.slot_id != old_slot_id or appointment.version != expected["appointment_version"]:
        raise _stale()
    slot = _lock_slots(connection, old_slot_id)[old_slot_id]
    _require_unexpired(connection, pending)
    if slot.status != "BOOKED" or slot.version != expected["old_slot_version"]:
        raise _stale()
    if slot.starts_at <= context.now:
        raise SelfServiceCutoffError()

    cancelled = connection.execute(
        update(Appointment)
        .where(
            Appointment.appointment_id == appointment.appointment_id,
            Appointment.version == appointment.version,
        )
        .values(status="CANCELLED", cancelled_at=func.now(), version=appointment.version + 1)
    )
    if cancelled.rowcount != 1:
        raise _stale()
    released_version = _change_slot(connection, slot, "AVAILABLE")
    _append_history(
        connection,
        context,
        appointment.appointment_id,
        "CANCELLED",
        old_slot_id=old_slot_id,
        new_slot_id=None,
    )
    _emit_slot_released(connection, slot, released_version, reason="CANCELLED")
    return _result(connection, "CANCELLED", appointment.appointment_id, "CANCELLED", old_slot_id)


_HANDLERS = {"BOOK": _book, "RESCHEDULE": _reschedule, "CANCEL": _cancel}
