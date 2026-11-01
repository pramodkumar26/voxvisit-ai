import itertools
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import Connection, insert
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Appointment,
    AppointmentHistory,
    AuthSession,
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    HandoffNote,
    HandoffTicket,
    IdempotencyRecord,
    NotificationEvent,
    OutboxEvent,
    Patient,
    PatientCredential,
    PendingAction,
    SecurityAuditEvent,
    Slot,
    SlotOffer,
    StaffCredential,
    StaffUser,
    WaitlistEntry,
)

BASE_TIME = datetime(2030, 1, 7, 15, 0, tzinfo=UTC)
FAR_FUTURE = BASE_TIME + timedelta(days=365)
_slot_hours = itertools.count()


def _token() -> str:
    return uuid.uuid4().hex[:10]


DEFAULTS: dict[type, Callable[[], dict[str, Any]]] = {
    Clinic: lambda: {
        "name": f"Clinic {_token()}",
        "timezone": "America/Denver",
        "address_line": "1 Synthetic Way",
        "city": "Fictional City",
        "region": "CO",
        "postal_code": "80000",
    },
    ClinicOperatingHours: lambda: {
        "day_of_week": 1,
        "opens_at": time(8),
        "closes_at": time(17),
    },
    ClinicHoliday: lambda: {"local_date": date(2030, 12, 25), "label": "Synthetic closure"},
    Doctor: lambda: {"display_name": f"Dr. {_token()}", "specialty": "Primary Care"},
    DoctorClinic: dict,
    DoctorAvailabilityRule: lambda: {
        "day_of_week": 1,
        "start_time": time(9),
        "end_time": time(12),
        "effective_start_date": date(2030, 1, 1),
    },
    DoctorLeave: lambda: {
        "starts_at": BASE_TIME,
        "ends_at": BASE_TIME + timedelta(days=1),
        "reason_code": "LEAVE",
    },
    Slot: lambda: {"starts_at": BASE_TIME, "ends_at": BASE_TIME + timedelta(minutes=30)},
    Patient: lambda: {"patient_code": f"P-{_token()}", "display_name": f"Patient {_token()}"},
    StaffUser: lambda: {"username": f"staff.{_token()}", "display_name": f"Staff {_token()}"},
    PatientCredential: lambda: {"pin_hash": "synthetic-argon2id-hash"},
    StaffCredential: lambda: {"password_hash": "synthetic-argon2id-hash"},
    AuthSession: lambda: {
        "token_hash": uuid.uuid4().bytes,
        "idle_expires_at": FAR_FUTURE,
        "absolute_expires_at": FAR_FUTURE,
    },
    PendingAction: lambda: {
        "action_type": "BOOK",
        "target_type": "SLOT",
        "summary": {"clinic": "Synthetic"},
        "expected_versions": {"slot": 1},
        "created_at": BASE_TIME,
        "expires_at": BASE_TIME + timedelta(minutes=10),
        "request_id": uuid.uuid4(),
    },
    Appointment: dict,
    AppointmentHistory: lambda: {
        "action": "BOOKED",
        "actor_type": "SYSTEM",
        "request_id": uuid.uuid4(),
    },
    WaitlistEntry: lambda: {
        "specialty": "Primary Care",
        "start_date": date(2030, 2, 1),
        "end_date": date(2030, 2, 28),
        "time_preference": "ANYTIME",
    },
    SlotOffer: lambda: {"expires_at": FAR_FUTURE, "request_id": uuid.uuid4()},
    HandoffTicket: lambda: {
        "category": "SCHEDULING",
        "redacted_summary": "Synthetic scheduling question.",
    },
    HandoffNote: lambda: {"note_text": "Synthetic note.", "request_id": uuid.uuid4()},
    NotificationEvent: lambda: {
        "notification_type": "OFFER_AVAILABLE",
        "request_id": uuid.uuid4(),
    },
    IdempotencyRecord: lambda: {
        "actor_type": "PATIENT",
        "operation": "book_appointment",
        "idempotency_key": _token(),
        "request_hash": uuid.uuid4().bytes,
        "status": "COMPLETED",
    },
    OutboxEvent: lambda: {
        "event_type": "SLOT_RELEASED",
        "aggregate_type": "SLOT",
        "dedupe_key": f"slot-released:{uuid.uuid4()}",
    },
    SecurityAuditEvent: lambda: {
        "event_type": "LOGIN",
        "result": "SUCCESS",
        "request_id": uuid.uuid4(),
    },
}


def make(conn: Connection, model: type, **overrides: Any) -> SimpleNamespace:
    values = {**DEFAULTS[model](), **overrides}
    statement = insert(model).values(**values).returning(*model.__table__.c)
    return SimpleNamespace(**conn.execute(statement).one()._mapping)


def reject(conn: Connection, constraint: str, statement: Any) -> None:
    with pytest.raises(IntegrityError) as exc_info, conn.begin_nested():
        conn.execute(statement)
    assert exc_info.value.orig.diag.constraint_name == constraint


def reject_insert(conn: Connection, model: type, constraint: str, **overrides: Any) -> None:
    reject(conn, constraint, insert(model).values(**{**DEFAULTS[model](), **overrides}))


def make_context(conn: Connection) -> SimpleNamespace:
    clinic = make(conn, Clinic)
    doctor = make(conn, Doctor)
    make(conn, DoctorClinic, doctor_id=doctor.doctor_id, clinic_id=clinic.clinic_id)
    return SimpleNamespace(clinic=clinic, doctor=doctor)


def make_slot(
    conn: Connection,
    context: SimpleNamespace | None = None,
    **overrides: Any,
) -> SimpleNamespace:
    context = context or make_context(conn)
    if "starts_at" not in overrides:
        overrides["starts_at"] = BASE_TIME + timedelta(hours=next(_slot_hours))
        overrides["ends_at"] = overrides["starts_at"] + timedelta(minutes=30)
    return make(
        conn,
        Slot,
        doctor_id=context.doctor.doctor_id,
        clinic_id=context.clinic.clinic_id,
        **overrides,
    )


def make_waitlist_entry(
    conn: Connection,
    patient: SimpleNamespace,
    clinic: SimpleNamespace,
    **overrides: Any,
) -> SimpleNamespace:
    return make(
        conn,
        WaitlistEntry,
        patient_id=patient.patient_id,
        clinic_id=clinic.clinic_id,
        **overrides,
    )


def make_patient_session(
    conn: Connection, patient: SimpleNamespace, **overrides: Any
) -> SimpleNamespace:
    return make(
        conn,
        AuthSession,
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=patient.patient_id,
        **overrides,
    )


def make_staff_session(
    conn: Connection, staff: SimpleNamespace, **overrides: Any
) -> SimpleNamespace:
    return make(
        conn,
        AuthSession,
        actor_type="STAFF",
        role="STAFF",
        staff_id=staff.staff_id,
        **overrides,
    )
