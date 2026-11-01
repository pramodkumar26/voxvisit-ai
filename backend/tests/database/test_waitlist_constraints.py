import uuid
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from factories import (
    BASE_TIME,
    make,
    make_context,
    make_slot,
    make_waitlist_entry,
    reject,
    reject_insert,
)
from sqlalchemy import Connection, update

from app.db.models import Patient, SlotOffer, WaitlistEntry

OVERLAP = "ex_waitlist_entries_active_overlap"


def make_offer(conn: Connection, entry, slot, **overrides) -> SimpleNamespace:
    return make(
        conn,
        SlotOffer,
        waitlist_id=entry.waitlist_id,
        patient_id=entry.patient_id,
        slot_id=slot.slot_id,
        **overrides,
    )


def waitlisted_patient(conn: Connection, clinic) -> SimpleNamespace:
    patient = make(conn, Patient)
    entry = make_waitlist_entry(conn, patient, clinic)
    return SimpleNamespace(patient=patient, entry=entry)


def test_waitlist_status_and_time_preference_are_limited_to_approved_values(
    conn: Connection,
) -> None:
    clinic = make_context(conn).clinic
    ids = {"patient_id": make(conn, Patient).patient_id, "clinic_id": clinic.clinic_id}

    reject_insert(conn, WaitlistEntry, "ck_waitlist_entries_status_valid", status="OFFERED", **ids)
    reject_insert(
        conn,
        WaitlistEntry,
        "ck_waitlist_entries_time_preference_valid",
        time_preference="EVENING",
        **ids,
    )
    reject_insert(
        conn,
        WaitlistEntry,
        "ck_waitlist_entries_dates_valid",
        start_date=date(2030, 3, 2),
        end_date=date(2030, 3, 1),
        **ids,
    )


def test_overlapping_active_waitlists_are_rejected(conn: Connection) -> None:
    clinic = make_context(conn).clinic
    patient = make(conn, Patient)
    make_waitlist_entry(
        conn, patient, clinic, start_date=date(2030, 2, 1), end_date=date(2030, 2, 10)
    )

    overlapping_ranges = [
        (date(2030, 2, 5), date(2030, 2, 6)),
        (date(2030, 1, 20), date(2030, 2, 1)),
        (date(2030, 2, 10), date(2030, 2, 20)),
        (date(2030, 1, 1), date(2030, 3, 1)),
    ]
    for start, end in overlapping_ranges:
        reject_insert(
            conn,
            WaitlistEntry,
            OVERLAP,
            patient_id=patient.patient_id,
            clinic_id=clinic.clinic_id,
            start_date=start,
            end_date=end,
        )


def test_overlap_is_rejected_regardless_of_time_preference(conn: Connection) -> None:
    clinic = make_context(conn).clinic
    patient = make(conn, Patient)
    make_waitlist_entry(conn, patient, clinic, time_preference="MORNING")

    reject_insert(
        conn,
        WaitlistEntry,
        OVERLAP,
        patient_id=patient.patient_id,
        clinic_id=clinic.clinic_id,
        time_preference="AFTERNOON",
    )


def test_non_overlapping_or_unrelated_active_waitlists_are_allowed(conn: Connection) -> None:
    clinic = make_context(conn).clinic
    patient = make(conn, Patient)
    make_waitlist_entry(
        conn, patient, clinic, start_date=date(2030, 2, 1), end_date=date(2030, 2, 10)
    )

    make_waitlist_entry(
        conn, patient, clinic, start_date=date(2030, 2, 11), end_date=date(2030, 2, 20)
    )
    make_waitlist_entry(conn, patient, clinic, specialty="Cardiology")
    make_waitlist_entry(conn, patient, make_context(conn).clinic)
    make_waitlist_entry(conn, make(conn, Patient), clinic)


def test_terminal_waitlist_entries_do_not_block_new_active_entries(conn: Connection) -> None:
    clinic = make_context(conn).clinic
    patient = make(conn, Patient)
    first = make_waitlist_entry(conn, patient, clinic)

    reject_insert(
        conn, WaitlistEntry, OVERLAP, patient_id=patient.patient_id, clinic_id=clinic.clinic_id
    )

    conn.execute(
        update(WaitlistEntry)
        .where(WaitlistEntry.waitlist_id == first.waitlist_id)
        .values(status="CANCELLED")
    )
    make_waitlist_entry(conn, patient, clinic)
    make_waitlist_entry(conn, patient, clinic, status="FULFILLED")


def test_reactivating_a_cancelled_entry_cannot_create_an_overlap(conn: Connection) -> None:
    clinic = make_context(conn).clinic
    patient = make(conn, Patient)
    cancelled = make_waitlist_entry(conn, patient, clinic, status="CANCELLED")
    make_waitlist_entry(conn, patient, clinic)

    reject(
        conn,
        OVERLAP,
        update(WaitlistEntry)
        .where(WaitlistEntry.waitlist_id == cancelled.waitlist_id)
        .values(status="ACTIVE"),
    )


def test_offer_status_is_limited_to_approved_values(conn: Connection) -> None:
    context = make_context(conn)
    waiting = waitlisted_patient(conn, context.clinic)

    reject_insert(
        conn,
        SlotOffer,
        "ck_slot_offers_status_valid",
        waitlist_id=waiting.entry.waitlist_id,
        patient_id=waiting.patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
        status="PENDING",
    )


def test_slot_cannot_have_two_active_offers(conn: Connection) -> None:
    context = make_context(conn)
    slot = make_slot(conn, context)
    first = waitlisted_patient(conn, context.clinic)
    second = waitlisted_patient(conn, context.clinic)
    make_offer(conn, first.entry, slot)

    reject_insert(
        conn,
        SlotOffer,
        "uq_slot_offers_active_slot",
        waitlist_id=second.entry.waitlist_id,
        patient_id=second.patient.patient_id,
        slot_id=slot.slot_id,
    )


def test_released_slot_can_be_offered_to_another_patient(conn: Connection) -> None:
    context = make_context(conn)
    slot = make_slot(conn, context)
    first = waitlisted_patient(conn, context.clinic)
    second = waitlisted_patient(conn, context.clinic)
    first_offer = make_offer(conn, first.entry, slot)

    conn.execute(
        update(SlotOffer)
        .where(SlotOffer.offer_id == first_offer.offer_id)
        .values(status="DECLINED")
    )

    make_offer(conn, second.entry, slot)


def test_patient_cannot_have_two_active_offers(conn: Connection) -> None:
    context = make_context(conn)
    waiting = waitlisted_patient(conn, context.clinic)
    make_offer(conn, waiting.entry, make_slot(conn, context))

    reject_insert(
        conn,
        SlotOffer,
        "uq_slot_offers_active_patient",
        waitlist_id=waiting.entry.waitlist_id,
        patient_id=waiting.patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
    )


@pytest.mark.parametrize("terminal_status", ["DECLINED", "EXPIRED", "ACCEPTED"])
def test_slot_is_never_reoffered_to_the_same_patient(
    conn: Connection, terminal_status: str
) -> None:
    context = make_context(conn)
    slot = make_slot(conn, context)
    waiting = waitlisted_patient(conn, context.clinic)
    make_offer(conn, waiting.entry, slot, status=terminal_status)

    reject_insert(
        conn,
        SlotOffer,
        "uq_slot_offers_patient_id_slot_id",
        waitlist_id=waiting.entry.waitlist_id,
        patient_id=waiting.patient.patient_id,
        slot_id=slot.slot_id,
    )


def test_terminal_offers_do_not_block_a_patients_next_offer(conn: Connection) -> None:
    context = make_context(conn)
    waiting = waitlisted_patient(conn, context.clinic)
    make_offer(conn, waiting.entry, make_slot(conn, context), status="EXPIRED")
    make_offer(conn, waiting.entry, make_slot(conn, context), status="DECLINED")

    make_offer(conn, waiting.entry, make_slot(conn, context))


def test_offer_patient_must_own_the_waitlist_entry(conn: Connection) -> None:
    context = make_context(conn)
    owner = waitlisted_patient(conn, context.clinic)
    other_patient = make(conn, Patient)

    reject_insert(
        conn,
        SlotOffer,
        "fk_slot_offers_waitlist_patient",
        waitlist_id=owner.entry.waitlist_id,
        patient_id=other_patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
    )


def test_offer_must_reference_existing_records(conn: Connection) -> None:
    context = make_context(conn)
    waiting = waitlisted_patient(conn, context.clinic)

    reject_insert(
        conn,
        SlotOffer,
        "fk_slot_offers_slot_id_slots",
        waitlist_id=waiting.entry.waitlist_id,
        patient_id=waiting.patient.patient_id,
        slot_id=uuid.uuid4(),
    )
    reject_insert(
        conn,
        SlotOffer,
        "fk_slot_offers_waitlist_patient",
        waitlist_id=uuid.uuid4(),
        patient_id=waiting.patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
    )


def test_offer_must_expire_after_database_creation_time(conn: Connection) -> None:
    context = make_context(conn)
    waiting = waitlisted_patient(conn, context.clinic)

    reject_insert(
        conn,
        SlotOffer,
        "ck_slot_offers_expiry_follows_creation",
        waitlist_id=waiting.entry.waitlist_id,
        patient_id=waiting.patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
        expires_at=BASE_TIME - timedelta(days=365 * 10),
    )
