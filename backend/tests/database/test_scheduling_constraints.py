import uuid
from datetime import date, time, timedelta

import pytest
from factories import (
    BASE_TIME,
    make,
    make_context,
    make_slot,
    reject,
    reject_insert,
)
from sqlalchemy import Connection, delete, update

from app.db.models import (
    Appointment,
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    Patient,
    Slot,
)


def test_slot_requires_an_existing_doctor_clinic_assignment(conn: Connection) -> None:
    clinic = make(conn, Clinic)
    doctor = make(conn, Doctor)

    reject_insert(
        conn,
        Slot,
        "fk_slots_doctor_clinic",
        doctor_id=doctor.doctor_id,
        clinic_id=clinic.clinic_id,
    )


def test_foreign_keys_reject_unknown_parents(conn: Connection) -> None:
    slot = make_slot(conn)

    reject_insert(
        conn,
        Appointment,
        "fk_appointments_patient_id_patients",
        patient_id=uuid.uuid4(),
        slot_id=slot.slot_id,
    )
    reject_insert(
        conn,
        Appointment,
        "fk_appointments_slot_id_slots",
        patient_id=make(conn, Patient).patient_id,
        slot_id=uuid.uuid4(),
    )


@pytest.mark.parametrize("status", ["PENDING", "RESERVED", "available"])
def test_slot_status_is_limited_to_approved_values(conn: Connection, status: str) -> None:
    context = make_context(conn)

    reject_insert(
        conn,
        Slot,
        "ck_slots_status_valid",
        doctor_id=context.doctor.doctor_id,
        clinic_id=context.clinic.clinic_id,
        status=status,
    )


def test_slot_end_must_follow_start(conn: Connection) -> None:
    context = make_context(conn)

    reject_insert(
        conn,
        Slot,
        "ck_slots_period_valid",
        doctor_id=context.doctor.doctor_id,
        clinic_id=context.clinic.clinic_id,
        starts_at=BASE_TIME,
        ends_at=BASE_TIME,
    )


def test_doctor_cannot_have_two_slots_with_the_same_start(conn: Connection) -> None:
    context = make_context(conn)
    make_slot(conn, context, starts_at=BASE_TIME, ends_at=BASE_TIME + timedelta(minutes=30))

    reject_insert(
        conn,
        Slot,
        "uq_slots_doctor_id_starts_at",
        doctor_id=context.doctor.doctor_id,
        clinic_id=context.clinic.clinic_id,
    )


@pytest.mark.parametrize("status", ["HELD", "BOOKED"])
def test_removed_slot_cannot_become_held_or_booked(conn: Connection, status: str) -> None:
    slot = make_slot(conn, removed_at=BASE_TIME)

    reject(
        conn,
        "ck_slots_removal_requires_available",
        update(Slot).where(Slot.slot_id == slot.slot_id).values(status=status),
    )


@pytest.mark.parametrize("status", ["HELD", "BOOKED"])
def test_unavailable_slot_cannot_be_soft_removed(conn: Connection, status: str) -> None:
    slot = make_slot(conn, status=status)

    reject(
        conn,
        "ck_slots_removal_requires_available",
        update(Slot).where(Slot.slot_id == slot.slot_id).values(removed_at=BASE_TIME),
    )


def test_available_slot_can_be_soft_removed(conn: Connection) -> None:
    slot = make_slot(conn)

    result = conn.execute(
        update(Slot).where(Slot.slot_id == slot.slot_id).values(removed_at=BASE_TIME)
    )

    assert result.rowcount == 1


def test_referenced_scheduling_records_cannot_be_deleted(conn: Connection) -> None:
    context = make_context(conn)
    slot = make_slot(conn, context)
    make(conn, Appointment, patient_id=make(conn, Patient).patient_id, slot_id=slot.slot_id)

    reject(conn, "fk_appointments_slot_id_slots", delete(Slot).where(Slot.slot_id == slot.slot_id))
    reject(
        conn,
        "fk_slots_doctor_clinic",
        delete(DoctorClinic).where(
            DoctorClinic.doctor_id == context.doctor.doctor_id,
            DoctorClinic.clinic_id == context.clinic.clinic_id,
        ),
    )


def test_clinic_and_doctor_identity_constraints(conn: Connection) -> None:
    clinic = make(conn, Clinic)

    reject_insert(conn, Clinic, "uq_clinics_name", name=clinic.name)
    reject_insert(conn, Clinic, "ck_clinics_timezone_not_blank", timezone="  ")
    reject_insert(conn, Doctor, "ck_doctors_specialty_not_blank", specialty="")


def test_operating_hours_constraints(conn: Connection) -> None:
    clinic = make(conn, Clinic)
    make(conn, ClinicOperatingHours, clinic_id=clinic.clinic_id)

    reject_insert(
        conn,
        ClinicOperatingHours,
        "uq_clinic_operating_hours_window",
        clinic_id=clinic.clinic_id,
    )
    reject_insert(
        conn,
        ClinicOperatingHours,
        "ck_clinic_operating_hours_day_of_week_valid",
        clinic_id=clinic.clinic_id,
        day_of_week=7,
    )
    reject_insert(
        conn,
        ClinicOperatingHours,
        "ck_clinic_operating_hours_window_valid",
        clinic_id=clinic.clinic_id,
        day_of_week=2,
        opens_at=time(17),
        closes_at=time(8),
    )


def test_holiday_constraints(conn: Connection) -> None:
    clinic = make(conn, Clinic)
    make(conn, ClinicHoliday, clinic_id=clinic.clinic_id)

    reject_insert(
        conn,
        ClinicHoliday,
        "uq_clinic_holidays_clinic_id_local_date_start_time_end_time",
        clinic_id=clinic.clinic_id,
    )
    reject_insert(
        conn,
        ClinicHoliday,
        "ck_clinic_holidays_window_valid",
        clinic_id=clinic.clinic_id,
        local_date=date(2030, 7, 4),
        start_time=time(9),
    )
    reject_insert(
        conn,
        ClinicHoliday,
        "ck_clinic_holidays_window_valid",
        clinic_id=clinic.clinic_id,
        local_date=date(2030, 7, 4),
        start_time=time(12),
        end_time=time(9),
    )


def test_availability_rule_and_leave_require_an_assignment_and_valid_periods(
    conn: Connection,
) -> None:
    context = make_context(conn)
    ids = {"doctor_id": context.doctor.doctor_id, "clinic_id": context.clinic.clinic_id}
    unassigned = {"doctor_id": make(conn, Doctor).doctor_id, "clinic_id": context.clinic.clinic_id}

    reject_insert(
        conn, DoctorAvailabilityRule, "fk_doctor_availability_rules_doctor_clinic", **unassigned
    )
    reject_insert(conn, DoctorLeave, "fk_doctor_leave_doctor_clinic", **unassigned)
    reject_insert(
        conn,
        DoctorAvailabilityRule,
        "ck_doctor_availability_rules_window_valid",
        start_time=time(12),
        end_time=time(9),
        **ids,
    )
    reject_insert(
        conn,
        DoctorAvailabilityRule,
        "ck_doctor_availability_rules_effective_dates_valid",
        effective_end_date=date(2029, 12, 31),
        **ids,
    )
    reject_insert(
        conn,
        DoctorLeave,
        "ck_doctor_leave_period_valid",
        ends_at=BASE_TIME,
        starts_at=BASE_TIME,
        **ids,
    )
