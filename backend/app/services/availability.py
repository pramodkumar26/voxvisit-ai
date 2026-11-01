"""Appointment availability search (backs the search_appointment_availability tool).

A returned slot is a search result, never a reservation. Booking must recheck availability
inside its own write transaction, using the same bookable_slot_conditions defined here.
"""

import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Connection,
    Date,
    SmallInteger,
    Time,
    and_,
    cast,
    exists,
    func,
    not_,
    or_,
    select,
)

from app.core.errors import InvalidArgumentError, ResourceNotAvailableError
from app.db.models import (
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    Slot,
)
from app.models.scheduling import (
    AvailabilityPage,
    AvailableSlot,
    ClinicSummary,
    DoctorSummary,
    SearchWindow,
)
from app.services.pagination import (
    decode_cursor,
    encode_cursor,
    fingerprint,
    parse_timestamp_position,
    timestamp_position,
)
from app.services.search_text import normalize_specialty
from app.services.timeutil import AFTERNOON, MORNING, local_window_utc, time_fields

CURSOR_KIND = "availability"
MAX_WINDOW_DAYS = 31
TIME_PREFERENCES = {"MORNING": MORNING, "AFTERNOON": AFTERNOON}


def validate_window(start: date, end: date) -> None:
    if end < start:
        raise InvalidArgumentError("The end date must not be before the start date.")
    if (end - start).days + 1 > MAX_WINDOW_DAYS:
        raise InvalidArgumentError(f"The search window must be {MAX_WINDOW_DAYS} days or fewer.")


def slot_local_fields() -> tuple:
    """Clinic-local start time, end time, calendar date, and weekday (Monday = 0) of a slot."""
    local_start = func.timezone(Clinic.timezone, Slot.starts_at)
    local_end = func.timezone(Clinic.timezone, Slot.ends_at)
    return (
        cast(local_start, Time),
        cast(local_end, Time),
        cast(local_start, Date),
        cast(func.extract("isodow", local_start) - 1, SmallInteger),
    )


def bookable_slot_conditions(now: datetime) -> list:
    """The one definition of a genuinely bookable slot.

    Availability search, appointment-action preparation, and the booking and rescheduling
    transactions all use it, so a slot that is searchable is bookable by the same rules.
    It expects Slot joined to Clinic, Doctor, and the doctor's DoctorClinic assignment
    (see bookable_slot_select). Doctor leave applies only at the clinic where it was recorded.
    """
    start_time, end_time, local_date, weekday = slot_local_fields()
    return [
        Slot.status == "AVAILABLE",
        Slot.removed_at.is_(None),
        Slot.starts_at > now,
        Clinic.active,
        Doctor.active,
        DoctorClinic.active,
        exists().where(
            ClinicOperatingHours.clinic_id == Slot.clinic_id,
            ClinicOperatingHours.day_of_week == weekday,
            ClinicOperatingHours.opens_at <= start_time,
            ClinicOperatingHours.closes_at >= end_time,
        ),
        exists().where(
            DoctorAvailabilityRule.doctor_id == Slot.doctor_id,
            DoctorAvailabilityRule.clinic_id == Slot.clinic_id,
            DoctorAvailabilityRule.active,
            DoctorAvailabilityRule.day_of_week == weekday,
            DoctorAvailabilityRule.start_time <= start_time,
            DoctorAvailabilityRule.end_time >= end_time,
            DoctorAvailabilityRule.effective_start_date <= local_date,
            or_(
                DoctorAvailabilityRule.effective_end_date.is_(None),
                DoctorAvailabilityRule.effective_end_date >= local_date,
            ),
        ),
        not_(
            exists().where(
                ClinicHoliday.clinic_id == Slot.clinic_id,
                ClinicHoliday.local_date == local_date,
                or_(
                    ClinicHoliday.start_time.is_(None),
                    and_(ClinicHoliday.start_time < end_time, ClinicHoliday.end_time > start_time),
                ),
            )
        ),
        not_(
            exists().where(
                DoctorLeave.doctor_id == Slot.doctor_id,
                DoctorLeave.clinic_id == Slot.clinic_id,
                DoctorLeave.starts_at < Slot.ends_at,
                DoctorLeave.ends_at > Slot.starts_at,
            )
        ),
    ]


def bookable_slot_select(*columns):
    """A select over Slot with the joins that bookable_slot_conditions needs."""
    return (
        select(*columns)
        .select_from(Slot)
        .join(Clinic, Clinic.clinic_id == Slot.clinic_id)
        .join(Doctor, Doctor.doctor_id == Slot.doctor_id)
        .join(
            DoctorClinic,
            and_(
                DoctorClinic.doctor_id == Slot.doctor_id, DoctorClinic.clinic_id == Slot.clinic_id
            ),
        )
    )


def is_slot_bookable(connection: Connection, slot_id: uuid.UUID, now: datetime) -> bool:
    """Whether the slot satisfies every bookable rule at this moment."""
    return connection.execute(
        select(
            exists(
                bookable_slot_select(Slot.slot_id).where(
                    Slot.slot_id == slot_id, *bookable_slot_conditions(now)
                )
            )
        )
    ).scalar_one()


def search_availability(
    connection: Connection,
    *,
    clinic_id: uuid.UUID,
    start_date: date,
    end_date: date,
    doctor_id: uuid.UUID | None,
    specialty: str | None,
    time_preference: str,
    cursor: str | None,
    limit: int,
    now: datetime,
) -> AvailabilityPage:
    validate_window(start_date, end_date)
    clinic = connection.execute(
        select(Clinic.clinic_id, Clinic.name, Clinic.timezone).where(
            Clinic.clinic_id == clinic_id, Clinic.active
        )
    ).one_or_none()
    if clinic is None:
        raise ResourceNotAvailableError()
    if (
        doctor_id
        and not connection.execute(
            select(
                exists().where(
                    DoctorClinic.doctor_id == doctor_id,
                    DoctorClinic.clinic_id == clinic_id,
                    DoctorClinic.active,
                    Doctor.doctor_id == DoctorClinic.doctor_id,
                    Doctor.active,
                )
            )
        ).scalar_one()
    ):
        raise ResourceNotAvailableError()

    zone = ZoneInfo(clinic.timezone)
    lower, upper = local_window_utc(zone, start_date, end_date)
    wanted_specialty = normalize_specialty(specialty) if specialty else None
    search = fingerprint(
        clinic=clinic_id,
        start=start_date,
        end=end_date,
        doctor=doctor_id,
        specialty=wanted_specialty,
        preference=time_preference,
    )

    start_time, _, _, _ = slot_local_fields()
    conditions = [
        Slot.clinic_id == clinic_id,
        Slot.starts_at >= lower,
        Slot.starts_at < upper,
        *bookable_slot_conditions(now),
    ]
    if doctor_id:
        conditions.append(Slot.doctor_id == doctor_id)
    if wanted_specialty:
        conditions.append(func.lower(Doctor.specialty) == wanted_specialty)
    if time_preference in TIME_PREFERENCES:
        opens, closes = TIME_PREFERENCES[time_preference]
        conditions += [start_time >= opens, start_time < closes]
    if cursor:
        last_start, last_id = parse_timestamp_position(decode_cursor(cursor, CURSOR_KIND, search))
        conditions.append(
            or_(
                Slot.starts_at > last_start,
                and_(Slot.starts_at == last_start, Slot.slot_id > last_id),
            )
        )

    rows = connection.execute(
        bookable_slot_select(
            Slot.slot_id,
            Slot.starts_at,
            Slot.ends_at,
            Doctor.doctor_id,
            Doctor.display_name,
            Doctor.specialty,
        )
        .where(*conditions)
        .order_by(Slot.starts_at, Slot.slot_id)
        .limit(limit + 1)
    ).all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = encode_cursor(
            CURSOR_KIND, search, timestamp_position(last.starts_at, last.slot_id)
        )
    summary = ClinicSummary(clinic_id=clinic.clinic_id, name=clinic.name, timezone=clinic.timezone)
    return AvailabilityPage(
        window=SearchWindow(start_date=start_date, end_date=end_date, timezone=clinic.timezone),
        slots=[
            AvailableSlot(
                slot_id=row.slot_id,
                clinic=summary,
                doctor=DoctorSummary(
                    doctor_id=row.doctor_id,
                    display_name=row.display_name,
                    specialty=row.specialty,
                ),
                **time_fields(row.starts_at, row.ends_at, clinic.timezone),
            )
            for row in page
        ],
        next_cursor=next_cursor,
    )
