"""A patient's own appointments (backs the get_my_appointments tool).

Identity is a parameter supplied by trusted code from the verified session. It is never read
from a request, and every query is filtered by it.
"""

import uuid
from datetime import date

from sqlalchemy import Connection, Date, and_, cast, func, or_, select

from app.core.errors import InvalidArgumentError
from app.db.models import Appointment, Clinic, Doctor, Slot
from app.models.scheduling import ClinicSummary, DoctorSummary, MyAppointment, MyAppointmentsPage
from app.services.pagination import (
    decode_cursor,
    encode_cursor,
    fingerprint,
    parse_timestamp_position,
    timestamp_position,
)
from app.services.timeutil import time_fields

CURSOR_KIND = "my-appointments"


def list_my_appointments(
    connection: Connection,
    *,
    patient_id: uuid.UUID,
    status: str | None,
    start_date: date | None,
    end_date: date | None,
    cursor: str | None,
    limit: int,
) -> MyAppointmentsPage:
    if start_date and end_date and end_date < start_date:
        raise InvalidArgumentError("The end date must not be before the start date.")
    search = fingerprint(patient=patient_id, status=status, start=start_date, end=end_date)

    conditions = [Appointment.patient_id == patient_id]
    if status:
        conditions.append(Appointment.status == status)
    local_date = cast(func.timezone(Clinic.timezone, Slot.starts_at), Date)
    if start_date:
        conditions.append(local_date >= start_date)
    if end_date:
        conditions.append(local_date <= end_date)
    if cursor:
        last_start, last_id = parse_timestamp_position(decode_cursor(cursor, CURSOR_KIND, search))
        conditions.append(
            or_(
                Slot.starts_at > last_start,
                and_(Slot.starts_at == last_start, Appointment.appointment_id > last_id),
            )
        )

    rows = connection.execute(
        select(
            Appointment.appointment_id,
            Appointment.status,
            Slot.starts_at,
            Slot.ends_at,
            Clinic.clinic_id,
            Clinic.name.label("clinic_name"),
            Clinic.timezone,
            Doctor.doctor_id,
            Doctor.display_name,
            Doctor.specialty,
        )
        .join(Slot, Slot.slot_id == Appointment.slot_id)
        .join(Clinic, Clinic.clinic_id == Slot.clinic_id)
        .join(Doctor, Doctor.doctor_id == Slot.doctor_id)
        .where(*conditions)
        .order_by(Slot.starts_at, Appointment.appointment_id)
        .limit(limit + 1)
    ).all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = encode_cursor(
            CURSOR_KIND, search, timestamp_position(last.starts_at, last.appointment_id)
        )
    return MyAppointmentsPage(
        appointments=[
            MyAppointment(
                appointment_id=row.appointment_id,
                status=row.status,
                clinic=ClinicSummary(
                    clinic_id=row.clinic_id, name=row.clinic_name, timezone=row.timezone
                ),
                doctor=DoctorSummary(
                    doctor_id=row.doctor_id, display_name=row.display_name, specialty=row.specialty
                ),
                **time_fields(row.starts_at, row.ends_at, row.timezone),
            )
            for row in page
        ],
        next_cursor=next_cursor,
    )
