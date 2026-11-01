"""Pieces shared by preparing and confirming appointment actions.

Neither the pending-action service nor the transaction service duplicates these: the session lock,
the patient-visible summaries, and the audit insert.
"""

import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Connection, exists, insert, select

from app.auth.service import Principal
from app.core.errors import (
    AuthenticationRequiredError,
    ResourceNotAvailableError,
    SessionExpiredError,
)
from app.db.models import (
    AuthSession,
    Clinic,
    Doctor,
    Patient,
    PatientCredential,
    PendingAction,
    SecurityAuditEvent,
    Slot,
)
from app.models.appointment_actions import AppointmentTime

PREPARE_OPERATION = "appointment_action.prepare"
CONFIRM_OPERATION = "appointment_action.confirm"


def lock_patient_session(
    connection: Connection, principal: Principal, now: datetime, *, lock: bool = True
) -> None:
    """Lock the caller's session row and recheck that it is still a usable patient session.

    With lock=False the same checks run without taking the row lock, for read-only transactions
    (the agent tool layer verifies a session before a read).

    The lock is FOR NO KEY UPDATE: it serializes against logout, expiry, and revocation (which
    update the row) without conflicting with the foreign-key share locks other requests take when
    they insert idempotency records, so concurrent requests of one session cannot deadlock here.
    """
    query = select(
        AuthSession.status,
        AuthSession.patient_id,
        AuthSession.idle_expires_at,
        AuthSession.absolute_expires_at,
    ).where(AuthSession.session_id == principal.session_id)
    if lock:
        query = query.with_for_update(key_share=True)
    session = connection.execute(query).one_or_none()
    if session is None or session.status != "ACTIVE" or session.patient_id != principal.patient_id:
        raise AuthenticationRequiredError()
    if now >= session.idle_expires_at or now >= session.absolute_expires_at:
        raise SessionExpiredError()
    # The same account checks authentication makes, repeated now that the locks are held: an
    # account deactivated or disabled after the request authenticated cannot commit a change.
    account = connection.execute(
        select(Patient.active, PatientCredential.disabled)
        .join(PatientCredential, PatientCredential.patient_id == Patient.patient_id)
        .where(Patient.patient_id == principal.patient_id)
    ).one_or_none()
    if account is None or not account.active or account.disabled:
        raise AuthenticationRequiredError()


def require_pending_action_owner(
    connection: Connection, pending_action_id: uuid.UUID, principal: Principal
) -> None:
    """A stored result may be replayed only to the session that owns the pending action.

    Pending actions are bound to the session that prepared them, so a replay gets exactly the
    answer a fresh request would: another session or another patient's request looks like a
    missing action, and nothing from the original result is disclosed.
    """
    owned = connection.execute(
        select(
            exists().where(
                PendingAction.pending_action_id == pending_action_id,
                PendingAction.session_id == principal.session_id,
            )
        )
    ).scalar_one()
    if not owned:
        raise ResourceNotAvailableError()


def slot_details(connection: Connection, slot_id: uuid.UUID) -> Any:
    return connection.execute(
        select(
            Slot.slot_id,
            Slot.starts_at,
            Slot.ends_at,
            Slot.status,
            Slot.version,
            Clinic.name.label("clinic_name"),
            Clinic.timezone,
            Doctor.display_name.label("doctor_name"),
            Doctor.specialty,
        )
        .join(Clinic, Clinic.clinic_id == Slot.clinic_id)
        .join(Doctor, Doctor.doctor_id == Slot.doctor_id)
        .where(Slot.slot_id == slot_id)
    ).one_or_none()


def appointment_time(details: Any) -> AppointmentTime:
    """Clinic-local display values plus unambiguous instants for one slot."""
    local = details.starts_at.astimezone(ZoneInfo(details.timezone))
    hour = local.hour % 12 or 12
    return AppointmentTime(
        clinic=details.clinic_name,
        doctor=details.doctor_name,
        specialty=details.specialty,
        date=local.date().isoformat(),
        time=f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}",
        timezone=details.timezone,
        starts_at_local=local,
        starts_at_utc=details.starts_at,
    )


def describe(verb: str, shown: AppointmentTime) -> str:
    return (
        f"{verb} with {shown.doctor} at {shown.clinic} on {shown.date} at {shown.time} "
        f"({shown.timezone})"
    )


def record_success(
    connection: Connection,
    event_type: str,
    *,
    principal: Principal,
    now: datetime,
    request_id: uuid.UUID,
    resource_type: str,
    resource_id: uuid.UUID,
    details: dict[str, str],
) -> None:
    """A sanitized audit event written in the same transaction as the change it records."""
    connection.execute(
        insert(SecurityAuditEvent).values(
            event_type=event_type,
            result="SUCCESS",
            actor_type=principal.actor_type,
            actor_id=principal.patient_id,
            session_id=principal.session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            occurred_at=now,
            details=details,
        )
    )
