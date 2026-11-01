"""Read-only queries that turn authoritative rows into records.

Every function takes the AccessContext and applies its visibility rule in SQL or immediately
after, so no caller can forget it. Nothing here writes; the adapter also runs them inside a
READ ONLY transaction.
"""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import Connection, Date, String, and_, cast, exists, func, literal, or_, select

from app.auth.permissions import Permission
from app.core.errors import InvalidArgumentError, ResourceNotAvailableError
from app.db.models import Appointment, Clinic, Doctor, DoctorClinic, Patient, Slot
from app.fhir_like.access import AccessContext
from app.fhir_like.records import (
    AppointmentRecord,
    PatientRecord,
    PractitionerRecord,
    ScheduleRecord,
    SlotRecord,
)
from app.services.availability import bookable_slot_conditions, bookable_slot_select
from app.services.pagination import (
    INVALID_CURSOR,
    decode_cursor,
    encode_cursor,
    fingerprint,
    parse_timestamp_position,
    timestamp_position,
)

SCHEDULE_ORDER = func.concat(
    func.replace(cast(DoctorClinic.doctor_id, String), "-", ""),
    func.replace(cast(DoctorClinic.clinic_id, String), "-", ""),
)  # The same text as the Schedule id, so paging follows the ids.


def _can_see_everything(access: AccessContext) -> bool:
    return access.can(Permission.VIEW_CLINIC_SCHEDULES)


def _listed_doctor() -> Any:
    """A doctor the public directory lists: active, assigned to an active clinic."""
    return and_(
        Doctor.active,
        exists().where(
            DoctorClinic.doctor_id == Doctor.doctor_id,
            DoctorClinic.active,
            Clinic.clinic_id == DoctorClinic.clinic_id,
            Clinic.active,
        ),
    )


# --- Practitioner -----------------------------------------------------------------------------


def _practitioner(row: Any) -> PractitionerRecord:
    return PractitionerRecord(
        doctor_id=row.doctor_id,
        display_name=row.display_name,
        specialty=row.specialty,
        active=row.active,
        updated_at=row.updated_at,
    )


def get_practitioner(
    connection: Connection, access: AccessContext, doctor_id: uuid.UUID
) -> PractitionerRecord:
    row = connection.execute(
        select(
            Doctor.doctor_id,
            Doctor.display_name,
            Doctor.specialty,
            Doctor.active,
            Doctor.updated_at,
            _listed_doctor().label("listed"),
        ).where(Doctor.doctor_id == doctor_id)
    ).one_or_none()
    if row is None or not (row.listed or _can_see_everything(access)):
        raise ResourceNotAvailableError()
    return _practitioner(row)


def search_practitioners(
    connection: Connection,
    access: AccessContext,
    *,
    clinic_id: uuid.UUID | None,
    specialty: str | None,
    cursor: str | None,
    limit: int,
) -> tuple[list[PractitionerRecord], str | None]:
    search = fingerprint(clinic=clinic_id, specialty=specialty, all=_can_see_everything(access))
    conditions = [] if _can_see_everything(access) else [_listed_doctor()]
    if clinic_id:
        # Scoped to one clinic, the public sees a doctor only through an active assignment to
        # that very clinic and only while the clinic is active (not merely active elsewhere).
        # Staff may also see inactive assignments, which the active flag on Schedule reports.
        at_clinic = [
            DoctorClinic.doctor_id == Doctor.doctor_id,
            DoctorClinic.clinic_id == clinic_id,
        ]
        if not _can_see_everything(access):
            at_clinic += [
                DoctorClinic.active,
                Clinic.clinic_id == DoctorClinic.clinic_id,
                Clinic.active,
            ]
        conditions.append(exists().where(*at_clinic))
    if specialty:
        conditions.append(func.lower(Doctor.specialty) == specialty.strip().lower())
    if cursor:
        last = decode_cursor(cursor, "practitioner", search).get("i", "")
        try:
            conditions.append(Doctor.doctor_id > uuid.UUID(last))
        except ValueError:
            raise InvalidArgumentError(INVALID_CURSOR) from None
    rows = connection.execute(
        select(
            Doctor.doctor_id,
            Doctor.display_name,
            Doctor.specialty,
            Doctor.active,
            Doctor.updated_at,
        )
        .where(*conditions)
        .order_by(Doctor.doctor_id)
        .limit(limit + 1)
    ).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = encode_cursor("practitioner", search, {"i": str(page[-1].doctor_id)})
    return [_practitioner(row) for row in page], next_cursor


# --- Schedule ---------------------------------------------------------------------------------


def _schedule_select() -> Any:
    return (
        select(
            DoctorClinic.doctor_id,
            DoctorClinic.clinic_id,
            Doctor.display_name.label("doctor_name"),
            Doctor.specialty,
            Clinic.name.label("clinic_name"),
            Clinic.timezone.label("clinic_timezone"),
            and_(DoctorClinic.active, Doctor.active, Clinic.active).label("active"),
        )
        .join(Doctor, Doctor.doctor_id == DoctorClinic.doctor_id)
        .join(Clinic, Clinic.clinic_id == DoctorClinic.clinic_id)
    )


def _schedule(row: Any) -> ScheduleRecord:
    return ScheduleRecord(
        doctor_id=row.doctor_id,
        clinic_id=row.clinic_id,
        doctor_name=row.doctor_name,
        specialty=row.specialty,
        clinic_name=row.clinic_name,
        clinic_timezone=row.clinic_timezone,
        active=row.active,
    )


def get_schedule(
    connection: Connection, access: AccessContext, doctor_id: uuid.UUID, clinic_id: uuid.UUID
) -> ScheduleRecord:
    row = connection.execute(
        _schedule_select().where(
            DoctorClinic.doctor_id == doctor_id, DoctorClinic.clinic_id == clinic_id
        )
    ).one_or_none()
    if row is None or not (row.active or _can_see_everything(access)):
        raise ResourceNotAvailableError()
    return _schedule(row)


def search_schedules(
    connection: Connection,
    access: AccessContext,
    *,
    clinic_id: uuid.UUID | None,
    doctor_id: uuid.UUID | None,
    cursor: str | None,
    limit: int,
) -> tuple[list[ScheduleRecord], str | None]:
    search = fingerprint(clinic=clinic_id, doctor=doctor_id, all=_can_see_everything(access))
    conditions = []
    if not _can_see_everything(access):
        conditions.append(and_(DoctorClinic.active, Doctor.active, Clinic.active))
    if clinic_id:
        conditions.append(DoctorClinic.clinic_id == clinic_id)
    if doctor_id:
        conditions.append(DoctorClinic.doctor_id == doctor_id)
    if cursor:
        last = decode_cursor(cursor, "schedule", search).get("i", "")
        conditions.append(last < SCHEDULE_ORDER)
    rows = connection.execute(
        _schedule_select().where(*conditions).order_by(SCHEDULE_ORDER).limit(limit + 1)
    ).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last_row = page[-1]
        next_cursor = encode_cursor(
            "schedule", search, {"i": last_row.doctor_id.hex + last_row.clinic_id.hex}
        )
    return [_schedule(row) for row in page], next_cursor


# --- Slot -------------------------------------------------------------------------------------


def _slot_columns(now: datetime, access: AccessContext) -> list[Any]:
    owned = (
        exists().where(
            Appointment.slot_id == Slot.slot_id, Appointment.patient_id == access.patient_id
        )
        if access.patient_id
        else literal(False)
    )
    return [
        Slot.slot_id,
        Slot.doctor_id,
        Slot.clinic_id,
        Slot.starts_at,
        Slot.ends_at,
        Slot.status,
        Slot.removed_at,
        Slot.version,
        Slot.updated_at,
        Doctor.specialty,
        Clinic.timezone.label("clinic_timezone"),
        and_(*bookable_slot_conditions(now)).label("bookable"),
        owned.label("owned"),
    ]


def _slot(row: Any) -> SlotRecord:
    return SlotRecord(
        slot_id=row.slot_id,
        doctor_id=row.doctor_id,
        clinic_id=row.clinic_id,
        specialty=row.specialty,
        clinic_timezone=row.clinic_timezone,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        status=row.status,
        removed=row.removed_at is not None,
        bookable=row.bookable,
        version=row.version,
        updated_at=row.updated_at,
    )


def get_slot(
    connection: Connection, access: AccessContext, slot_id: uuid.UUID, now: datetime
) -> SlotRecord:
    row = connection.execute(
        bookable_slot_select(*_slot_columns(now, access)).where(Slot.slot_id == slot_id)
    ).one_or_none()
    if row is None or not (row.bookable or row.owned or _can_see_everything(access)):
        raise ResourceNotAvailableError()
    return _slot(row)


def slots_by_id(
    connection: Connection, access: AccessContext, slot_ids: list[uuid.UUID], now: datetime
) -> list[SlotRecord]:
    """The slots a search already found, in the order given, with versions for their meta."""
    rows = connection.execute(
        bookable_slot_select(*_slot_columns(now, access)).where(Slot.slot_id.in_(slot_ids))
    ).all()
    by_id = {row.slot_id: _slot(row) for row in rows}
    return [by_id[slot_id] for slot_id in slot_ids if slot_id in by_id]


# --- Patient ----------------------------------------------------------------------------------


def get_patient(
    connection: Connection, access: AccessContext, patient_id: uuid.UUID
) -> PatientRecord:
    permitted = access.can(Permission.VIEW_PATIENT_DIRECTORY) or (
        access.patient_id is not None and access.patient_id == patient_id
    )
    if not permitted:
        raise ResourceNotAvailableError()
    row = connection.execute(
        select(
            Patient.patient_id,
            Patient.patient_code,
            Patient.display_name,
            Patient.active,
            Patient.updated_at,
        ).where(Patient.patient_id == patient_id)
    ).one_or_none()
    if row is None:
        raise ResourceNotAvailableError()
    return PatientRecord(
        patient_id=row.patient_id,
        patient_code=row.patient_code,
        display_name=row.display_name,
        active=row.active,
        updated_at=row.updated_at,
    )


# --- Appointment ------------------------------------------------------------------------------


def _appointment_select() -> Any:
    return (
        select(
            Appointment.appointment_id,
            Appointment.patient_id,
            Appointment.slot_id,
            Appointment.status,
            Appointment.version,
            Appointment.cancelled_at,
            Appointment.updated_at,
            Slot.starts_at,
            Slot.ends_at,
            Slot.doctor_id,
            Slot.clinic_id,
            Doctor.display_name.label("doctor_name"),
            Doctor.specialty,
            Clinic.name.label("clinic_name"),
            Clinic.timezone.label("clinic_timezone"),
            Patient.display_name.label("patient_name"),
        )
        .join(Slot, Slot.slot_id == Appointment.slot_id)
        .join(Doctor, Doctor.doctor_id == Slot.doctor_id)
        .join(Clinic, Clinic.clinic_id == Slot.clinic_id)
        .join(Patient, Patient.patient_id == Appointment.patient_id)
    )


def _appointment(row: Any) -> AppointmentRecord:
    return AppointmentRecord(
        appointment_id=row.appointment_id,
        patient_id=row.patient_id,
        patient_name=row.patient_name,
        slot_id=row.slot_id,
        doctor_id=row.doctor_id,
        doctor_name=row.doctor_name,
        specialty=row.specialty,
        clinic_id=row.clinic_id,
        clinic_name=row.clinic_name,
        clinic_timezone=row.clinic_timezone,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        status=row.status,
        version=row.version,
        cancelled_at=row.cancelled_at,
        updated_at=row.updated_at,
    )


def _appointment_scope(access: AccessContext) -> list[Any]:
    """Patients see only their own appointments; staff see the schedule; nobody else sees any."""
    if _can_see_everything(access):
        return []
    if access.patient_id is not None:
        return [Appointment.patient_id == access.patient_id]
    raise ResourceNotAvailableError()


def get_appointment(
    connection: Connection, access: AccessContext, appointment_id: uuid.UUID
) -> AppointmentRecord:
    row = connection.execute(
        _appointment_select().where(
            Appointment.appointment_id == appointment_id, *_appointment_scope(access)
        )
    ).one_or_none()
    if row is None:
        raise ResourceNotAvailableError()
    return _appointment(row)


def appointments_by_id(
    connection: Connection, access: AccessContext, appointment_ids: list[uuid.UUID]
) -> list[AppointmentRecord]:
    """Records for ids a trusted listing returned, scoped again to the caller, in order."""
    rows = connection.execute(
        _appointment_select().where(
            Appointment.appointment_id.in_(appointment_ids), *_appointment_scope(access)
        )
    ).all()
    by_id = {row.appointment_id: _appointment(row) for row in rows}
    return [by_id[appointment_id] for appointment_id in appointment_ids if appointment_id in by_id]


def search_staff_appointments(
    connection: Connection,
    access: AccessContext,
    *,
    clinic_id: uuid.UUID | None,
    doctor_id: uuid.UUID | None,
    status: str | None,
    start_date: date | None,
    end_date: date | None,
    cursor: str | None,
    limit: int,
) -> tuple[list[AppointmentRecord], str | None]:
    if not _can_see_everything(access):
        raise ResourceNotAvailableError()
    search = fingerprint(
        clinic=clinic_id, doctor=doctor_id, status=status, start=start_date, end=end_date
    )
    local_date = cast(func.timezone(Clinic.timezone, Slot.starts_at), Date)
    conditions: list[Any] = []
    if clinic_id:
        conditions.append(Slot.clinic_id == clinic_id)
    if doctor_id:
        conditions.append(Slot.doctor_id == doctor_id)
    if status:
        conditions.append(Appointment.status == status)
    if start_date:
        conditions.append(local_date >= start_date)
    if end_date:
        conditions.append(local_date <= end_date)
    if cursor:
        last_start, last_id = parse_timestamp_position(decode_cursor(cursor, "staff-appt", search))
        conditions.append(
            or_(
                Slot.starts_at > last_start,
                and_(Slot.starts_at == last_start, Appointment.appointment_id > last_id),
            )
        )
    rows = connection.execute(
        _appointment_select()
        .where(*conditions)
        .order_by(Slot.starts_at, Appointment.appointment_id)
        .limit(limit + 1)
    ).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = encode_cursor(
            "staff-appt", search, timestamp_position(last.starts_at, last.appointment_id)
        )
    return [_appointment(row) for row in page], next_cursor
