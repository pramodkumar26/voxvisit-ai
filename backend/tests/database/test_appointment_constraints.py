import uuid

from factories import (
    BASE_TIME,
    make,
    make_slot,
    reject,
    reject_insert,
)
from sqlalchemy import Connection, delete, func, select, update

from app.db.models import Appointment, AppointmentHistory, Patient, Slot


def book(conn: Connection, patient, slot, **overrides):
    return make(conn, Appointment, patient_id=patient.patient_id, slot_id=slot.slot_id, **overrides)


def cancel(conn: Connection, appointment) -> None:
    conn.execute(
        update(Appointment)
        .where(Appointment.appointment_id == appointment.appointment_id)
        .values(status="CANCELLED", cancelled_at=BASE_TIME)
    )


def test_appointment_status_is_limited_to_confirmed_and_cancelled(conn: Connection) -> None:
    patient = make(conn, Patient)
    slot = make_slot(conn)

    for status in ("PENDING", "RESCHEDULED", "COMPLETED"):
        reject_insert(
            conn,
            Appointment,
            "ck_appointments_status_valid",
            patient_id=patient.patient_id,
            slot_id=slot.slot_id,
            status=status,
        )


def test_slot_cannot_have_two_confirmed_appointments(conn: Connection) -> None:
    slot = make_slot(conn)
    book(conn, make(conn, Patient), slot)

    reject_insert(
        conn,
        Appointment,
        "uq_appointments_confirmed_slot",
        patient_id=make(conn, Patient).patient_id,
        slot_id=slot.slot_id,
    )


def test_same_patient_cannot_confirm_the_same_slot_twice(conn: Connection) -> None:
    slot = make_slot(conn)
    patient = make(conn, Patient)
    book(conn, patient, slot)

    reject_insert(
        conn,
        Appointment,
        "uq_appointments_confirmed_slot",
        patient_id=patient.patient_id,
        slot_id=slot.slot_id,
    )


def test_cancelled_appointments_are_preserved_beside_a_new_confirmed_booking(
    conn: Connection,
) -> None:
    slot = make_slot(conn)
    first_patient = make(conn, Patient)
    first = book(conn, first_patient, slot)
    make(
        conn,
        AppointmentHistory,
        appointment_id=first.appointment_id,
        actor_type="PATIENT",
        actor_id=first_patient.patient_id,
        new_slot_id=slot.slot_id,
    )

    cancel(conn, first)
    second = book(conn, make(conn, Patient), slot)

    rows = conn.execute(
        select(Appointment.appointment_id, Appointment.status)
        .where(Appointment.slot_id == slot.slot_id)
        .order_by(Appointment.created_at, Appointment.appointment_id)
    ).all()
    assert {row.appointment_id: row.status for row in rows} == {
        first.appointment_id: "CANCELLED",
        second.appointment_id: "CONFIRMED",
    }
    history = conn.execute(
        select(func.count()).where(AppointmentHistory.appointment_id == first.appointment_id)
    ).scalar_one()
    assert history == 1


def test_many_cancelled_appointments_may_share_a_slot(conn: Connection) -> None:
    slot = make_slot(conn)

    for _ in range(3):
        book(conn, make(conn, Patient), slot, status="CANCELLED", cancelled_at=BASE_TIME)

    count = conn.execute(
        select(func.count()).where(Appointment.slot_id == slot.slot_id)
    ).scalar_one()
    assert count == 3


def test_cancellation_timestamp_must_match_status(conn: Connection) -> None:
    patient = make(conn, Patient)
    slot = make_slot(conn)
    appointment = book(conn, patient, slot)

    reject(
        conn,
        "ck_appointments_cancellation_matches_status",
        update(Appointment)
        .where(Appointment.appointment_id == appointment.appointment_id)
        .values(status="CANCELLED"),
    )
    reject_insert(
        conn,
        Appointment,
        "ck_appointments_cancellation_matches_status",
        patient_id=patient.patient_id,
        slot_id=make_slot(conn).slot_id,
        cancelled_at=BASE_TIME,
    )


def test_appointments_with_history_cannot_be_deleted(conn: Connection) -> None:
    patient = make(conn, Patient)
    slot = make_slot(conn)
    appointment = book(conn, patient, slot)
    make(
        conn,
        AppointmentHistory,
        appointment_id=appointment.appointment_id,
        actor_type="PATIENT",
        actor_id=patient.patient_id,
        new_slot_id=slot.slot_id,
    )

    reject(
        conn,
        "fk_appointment_history_appointment_id_appointments",
        delete(Appointment).where(Appointment.appointment_id == appointment.appointment_id),
    )


def test_patients_and_slots_referenced_by_appointments_cannot_be_deleted(
    conn: Connection,
) -> None:
    patient = make(conn, Patient)
    slot = make_slot(conn)
    book(conn, patient, slot, status="CANCELLED", cancelled_at=BASE_TIME)

    reject(
        conn,
        "fk_appointments_patient_id_patients",
        delete(Patient).where(Patient.patient_id == patient.patient_id),
    )
    reject(conn, "fk_appointments_slot_id_slots", delete(Slot).where(Slot.slot_id == slot.slot_id))


def test_history_slots_are_protected_from_deletion(conn: Connection) -> None:
    patient = make(conn, Patient)
    old_slot = make_slot(conn)
    new_slot = make_slot(conn)
    appointment = book(conn, patient, new_slot)
    make(
        conn,
        AppointmentHistory,
        appointment_id=appointment.appointment_id,
        action="RESCHEDULED",
        actor_type="PATIENT",
        actor_id=patient.patient_id,
        old_slot_id=old_slot.slot_id,
        new_slot_id=new_slot.slot_id,
    )

    reject(
        conn,
        "fk_appointment_history_old_slot_id_slots",
        delete(Slot).where(Slot.slot_id == old_slot.slot_id),
    )


def test_history_action_and_actor_values_are_approved(conn: Connection) -> None:
    patient = make(conn, Patient)
    appointment = book(conn, patient, make_slot(conn))
    base = {"appointment_id": appointment.appointment_id, "actor_id": patient.patient_id}

    reject_insert(
        conn,
        AppointmentHistory,
        "ck_appointment_history_action_valid",
        actor_type="PATIENT",
        action="MODIFIED",
        **base,
    )
    reject_insert(
        conn,
        AppointmentHistory,
        "ck_appointment_history_actor_type_valid",
        actor_type="ADMIN",
        **base,
    )


def test_history_requires_actor_id_unless_system(conn: Connection) -> None:
    appointment = book(conn, make(conn, Patient), make_slot(conn))

    reject_insert(
        conn,
        AppointmentHistory,
        "ck_appointment_history_actor_identified",
        appointment_id=appointment.appointment_id,
        actor_type="PATIENT",
    )
    make(conn, AppointmentHistory, appointment_id=appointment.appointment_id, actor_type="SYSTEM")


def test_history_request_id_is_unique_per_appointment(conn: Connection) -> None:
    appointment = book(conn, make(conn, Patient), make_slot(conn))
    request_id = uuid.uuid4()
    make(
        conn,
        AppointmentHistory,
        appointment_id=appointment.appointment_id,
        actor_type="SYSTEM",
        request_id=request_id,
    )

    reject_insert(
        conn,
        AppointmentHistory,
        "uq_appointment_history_appointment_id_request_id",
        appointment_id=appointment.appointment_id,
        actor_type="SYSTEM",
        request_id=request_id,
    )
