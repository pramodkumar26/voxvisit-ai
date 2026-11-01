"""Builders for precise scheduling scenarios. Every call uses fresh, uniquely named records, so
tests can share one database and still assert exact results by clinic or doctor."""

import uuid
from datetime import date, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from factories import make
from sqlalchemy import Engine

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
from app.seed.plan import localize

WEEKDAYS = range(5)
ALL_WEEK = range(7)
BUSINESS_HOURS = (time(8), time(17))
BUSINESS_RULE = ((time(9), time(12)), (time(13), time(16)))
ALL_DAY = (time(0), time(23, 59, 59))
MONDAY = date(2026, 12, 7)  # A Monday well after the shared test clock (2026-11-02).


def token() -> str:
    return uuid.uuid4().hex[:8]


def new_clinic(
    engine: Engine,
    *,
    timezone: str = "America/Denver",
    active: bool = True,
    hours: tuple[time, time] = BUSINESS_HOURS,
    open_days=WEEKDAYS,
    name: str | None = None,
) -> SimpleNamespace:
    with engine.begin() as connection:
        clinic = make(
            connection,
            Clinic,
            name=name or f"Clinic {token()}",
            timezone=timezone,
            active=active,
        )
        for weekday in open_days:
            make(
                connection,
                ClinicOperatingHours,
                clinic_id=clinic.clinic_id,
                day_of_week=weekday,
                opens_at=hours[0],
                closes_at=hours[1],
            )
    return SimpleNamespace(
        clinic_id=clinic.clinic_id, name=clinic.name, timezone=timezone, zone=ZoneInfo(timezone)
    )


def new_doctor(
    engine: Engine,
    clinic: SimpleNamespace,
    *,
    specialty: str = "Primary Care",
    name: str | None = None,
    active: bool = True,
    assigned: bool = True,
    assignment_active: bool = True,
    rule_days=WEEKDAYS,
    rule_windows=BUSINESS_RULE,
    rule_active: bool = True,
    effective_start: date = date(2026, 1, 1),
    effective_end: date | None = None,
) -> SimpleNamespace:
    with engine.begin() as connection:
        doctor = make(
            connection,
            Doctor,
            display_name=name or f"Dr. {token()}",
            specialty=specialty,
            active=active,
        )
        if assigned:
            make(
                connection,
                DoctorClinic,
                doctor_id=doctor.doctor_id,
                clinic_id=clinic.clinic_id,
                active=assignment_active,
            )
            for weekday in rule_days:
                for start, end in rule_windows:
                    make(
                        connection,
                        DoctorAvailabilityRule,
                        doctor_id=doctor.doctor_id,
                        clinic_id=clinic.clinic_id,
                        day_of_week=weekday,
                        start_time=start,
                        end_time=end,
                        effective_start_date=effective_start,
                        effective_end_date=effective_end,
                        active=rule_active,
                    )
    return SimpleNamespace(
        doctor_id=doctor.doctor_id, display_name=doctor.display_name, specialty=specialty
    )


def assign_doctor(engine: Engine, doctor, clinic) -> None:
    """Also assign an existing doctor to another clinic, available all day on every day."""
    with engine.begin() as connection:
        make(connection, DoctorClinic, doctor_id=doctor.doctor_id, clinic_id=clinic.clinic_id)
        for weekday in ALL_WEEK:
            make(
                connection,
                DoctorAvailabilityRule,
                doctor_id=doctor.doctor_id,
                clinic_id=clinic.clinic_id,
                day_of_week=weekday,
                start_time=ALL_DAY[0],
                end_time=ALL_DAY[1],
                effective_start_date=date(2026, 1, 1),
            )


def add_slots(
    engine: Engine,
    clinic: SimpleNamespace,
    doctor: SimpleNamespace,
    moments: list[tuple[date, time]],
    *,
    minutes: int = 30,
    **overrides,
) -> list[uuid.UUID]:
    ids = []
    with engine.begin() as connection:
        for day, start in moments:
            starts_at = localize(clinic.zone, day, start)
            slot = make(
                connection,
                Slot,
                doctor_id=doctor.doctor_id,
                clinic_id=clinic.clinic_id,
                starts_at=starts_at,
                ends_at=starts_at + timedelta(minutes=minutes),
                **overrides,
            )
            ids.append(slot.slot_id)
    return ids


def add_holiday(engine, clinic, day: date, start: time | None = None, end: time | None = None):
    with engine.begin() as connection:
        make(
            connection,
            ClinicHoliday,
            clinic_id=clinic.clinic_id,
            local_date=day,
            start_time=start,
            end_time=end,
            label="Synthetic closure",
        )


def add_leave(engine, clinic, doctor, starts: tuple[date, time], ends: tuple[date, time]):
    with engine.begin() as connection:
        make(
            connection,
            DoctorLeave,
            doctor_id=doctor.doctor_id,
            clinic_id=clinic.clinic_id,
            starts_at=localize(clinic.zone, *starts),
            ends_at=localize(clinic.zone, *ends),
            reason_code="LEAVE",
        )


def half_hours(day: date, first: time, count: int) -> list[tuple[date, time]]:
    start = first.hour * 60 + first.minute
    return [
        (day, time((start + 30 * index) // 60, (start + 30 * index) % 60)) for index in range(count)
    ]
