import uuid
from datetime import timedelta

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
from sqlalchemy import JSON, Connection, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    DoctorClinic,
    HandoffTicket,
    IdempotencyRecord,
    NotificationEvent,
    Patient,
    Slot,
    SlotOffer,
)

OVERLAP = "ex_slots_doctor_no_overlap"
HOUR = timedelta(hours=1)


def window(start_minutes: int, end_minutes: int) -> dict:
    return {
        "starts_at": BASE_TIME + timedelta(minutes=start_minutes),
        "ends_at": BASE_TIME + timedelta(minutes=end_minutes),
    }


def slot_for(context, **overrides) -> dict:
    return {
        "doctor_id": context.doctor.doctor_id,
        "clinic_id": context.clinic.clinic_id,
        **overrides,
    }


# --- 1. Overlapping doctor slots -------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [(15, 45), (-15, 15), (5, 25), (-30, 60)],
    ids=["tail-overlap", "head-overlap", "contained", "containing"],
)
def test_overlapping_slots_for_the_same_doctor_are_rejected(
    conn: Connection, start: int, end: int
) -> None:
    context = make_context(conn)
    make_slot(conn, context, **window(0, 30))

    reject_insert(conn, Slot, OVERLAP, **slot_for(context, **window(start, end)))


@pytest.mark.parametrize("existing_status", ["AVAILABLE", "HELD", "BOOKED"])
@pytest.mark.parametrize("new_status", ["AVAILABLE", "HELD", "BOOKED"])
def test_overlap_is_rejected_regardless_of_slot_status(
    conn: Connection, existing_status: str, new_status: str
) -> None:
    context = make_context(conn)
    make_slot(conn, context, status=existing_status, **window(0, 30))

    reject_insert(conn, Slot, OVERLAP, **slot_for(context, status=new_status, **window(15, 45)))


def test_adjacent_slots_for_the_same_doctor_are_allowed(conn: Connection) -> None:
    context = make_context(conn)

    make_slot(conn, context, **window(30, 60))
    make_slot(conn, context, **window(0, 30))
    make_slot(conn, context, **window(60, 90))

    count = conn.execute(
        select(func.count()).where(Slot.doctor_id == context.doctor.doctor_id)
    ).scalar_one()
    assert count == 3


def test_different_doctors_may_have_identical_slot_times(conn: Connection) -> None:
    first = make_context(conn)
    second = make_context(conn)

    make_slot(conn, first, **window(0, 30))
    make_slot(conn, second, **window(0, 30))
    make_slot(conn, first, **window(30, 60))


def test_same_doctor_cannot_overlap_across_clinics(conn: Connection) -> None:
    context = make_context(conn)
    other_clinic = make_context(conn).clinic
    make(
        conn,
        DoctorClinic,
        doctor_id=context.doctor.doctor_id,
        clinic_id=other_clinic.clinic_id,
    )
    make_slot(conn, context, **window(0, 30))

    reject_insert(
        conn,
        Slot,
        OVERLAP,
        doctor_id=context.doctor.doctor_id,
        clinic_id=other_clinic.clinic_id,
        **window(15, 45),
    )


def test_soft_removed_slots_do_not_block_new_overlapping_slots(conn: Connection) -> None:
    context = make_context(conn)
    removed = make_slot(conn, context, removed_at=BASE_TIME, **window(0, 30))

    replacement = make_slot(conn, context, **window(15, 45))

    rows = (
        conn.execute(select(Slot.slot_id).where(Slot.doctor_id == context.doctor.doctor_id))
        .scalars()
        .all()
    )
    assert set(rows) == {removed.slot_id, replacement.slot_id}


def test_removing_a_slot_frees_its_time_and_preserves_the_row(conn: Connection) -> None:
    context = make_context(conn)
    first = make_slot(conn, context, **window(0, 30))
    reject_insert(conn, Slot, OVERLAP, **slot_for(context, **window(10, 40)))

    conn.execute(update(Slot).where(Slot.slot_id == first.slot_id).values(removed_at=BASE_TIME))
    make_slot(conn, context, **window(10, 40))

    preserved = conn.execute(
        select(Slot.removed_at).where(Slot.slot_id == first.slot_id)
    ).scalar_one()
    assert preserved == BASE_TIME


def test_restoring_a_removed_slot_cannot_create_an_overlap(conn: Connection) -> None:
    context = make_context(conn)
    removed = make_slot(conn, context, removed_at=BASE_TIME, **window(0, 30))
    make_slot(conn, context, **window(15, 45))

    reject(
        conn,
        OVERLAP,
        update(Slot).where(Slot.slot_id == removed.slot_id).values(removed_at=None),
    )


def test_moving_a_slot_into_another_cannot_create_an_overlap(conn: Connection) -> None:
    context = make_context(conn)
    make_slot(conn, context, **window(0, 30))
    later = make_slot(conn, context, **window(60, 90))

    reject(
        conn,
        OVERLAP,
        update(Slot).where(Slot.slot_id == later.slot_id).values(**window(20, 50)),
    )


def test_identical_start_time_is_still_rejected_by_the_unique_constraint(
    conn: Connection,
) -> None:
    context = make_context(conn)
    make_slot(conn, context, **window(0, 30))

    with pytest.raises(IntegrityError) as exc_info, conn.begin_nested():
        conn.execute(insert(Slot).values(**slot_for(context, **window(0, 30))))

    assert exc_info.value.orig.diag.constraint_name in {OVERLAP, "uq_slots_doctor_id_starts_at"}


# --- 2. Notification ownership ---------------------------------------------------------------


def make_offer(conn: Connection):
    context = make_context(conn)
    patient = make(conn, Patient)
    entry = make_waitlist_entry(conn, patient, context.clinic)
    offer = make(
        conn,
        SlotOffer,
        waitlist_id=entry.waitlist_id,
        patient_id=patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
    )
    return patient, offer


def test_notification_for_the_offer_owner_is_allowed(conn: Connection) -> None:
    patient, offer = make_offer(conn)

    notification = make(
        conn, NotificationEvent, patient_id=patient.patient_id, offer_id=offer.offer_id
    )

    assert notification.offer_id == offer.offer_id


def test_notification_for_another_patients_offer_is_rejected(conn: Connection) -> None:
    _, offer = make_offer(conn)
    other = make(conn, Patient)

    reject_insert(
        conn,
        NotificationEvent,
        "fk_notification_events_offer_patient",
        patient_id=other.patient_id,
        offer_id=offer.offer_id,
    )


def test_reassigning_a_notification_to_another_patient_is_rejected(conn: Connection) -> None:
    patient, offer = make_offer(conn)
    notification = make(
        conn, NotificationEvent, patient_id=patient.patient_id, offer_id=offer.offer_id
    )

    reject(
        conn,
        "fk_notification_events_offer_patient",
        update(NotificationEvent)
        .where(NotificationEvent.notification_id == notification.notification_id)
        .values(patient_id=make(conn, Patient).patient_id),
    )


def test_notification_without_an_offer_is_supported(conn: Connection) -> None:
    patient = make(conn, Patient)

    notification = make(conn, NotificationEvent, patient_id=patient.patient_id)

    assert notification.offer_id is None


# --- 3. Handoff categories -------------------------------------------------------------------

CATEGORIES = [
    "SCHEDULING",
    "CLINIC_INFORMATION",
    "ACCESSIBILITY",
    "TECHNICAL_SUPPORT",
    "OTHER_ADMINISTRATIVE",
]


@pytest.mark.parametrize("category", CATEGORIES)
def test_each_approved_handoff_category_is_accepted(conn: Connection, category: str) -> None:
    assert make(conn, HandoffTicket, category=category).category == category


@pytest.mark.parametrize("category", ["BILLING", "scheduling", "", "OTHER", "SCHEDULING "])
def test_unapproved_handoff_categories_are_rejected(conn: Connection, category: str) -> None:
    reject_insert(conn, HandoffTicket, "ck_handoff_tickets_category_valid", category=category)


def test_handoff_category_is_required_and_has_no_default(conn: Connection) -> None:
    with pytest.raises(IntegrityError) as exc_info, conn.begin_nested():
        conn.execute(insert(HandoffTicket).values(redacted_summary="Synthetic question."))

    assert exc_info.value.orig.diag.column_name == "category"
    default = conn.execute(
        text(
            "SELECT column_default FROM information_schema.columns"
            " WHERE table_name = 'handoff_tickets' AND column_name = 'category'"
        )
    ).scalar_one()
    assert default is None


# --- 4. Anonymous idempotency scopes ---------------------------------------------------------


def test_anonymous_actor_scope_is_accepted_without_a_session(conn: Connection) -> None:
    record = make(conn, IdempotencyRecord, actor_type="ANONYMOUS", actor_id=uuid.uuid4())

    assert record.actor_type == "ANONYMOUS"
    assert record.session_id is None


def test_anonymous_key_is_unique_within_its_scope_and_operation(conn: Connection) -> None:
    scope = uuid.uuid4()
    record = make(
        conn,
        IdempotencyRecord,
        actor_type="ANONYMOUS",
        actor_id=scope,
        operation="request_staff_handoff",
    )

    reject_insert(
        conn,
        IdempotencyRecord,
        "uq_idempotency_records_scope_key",
        actor_type="ANONYMOUS",
        actor_id=scope,
        operation="request_staff_handoff",
        idempotency_key=record.idempotency_key,
    )


def test_anonymous_scopes_are_isolated_from_each_other_and_from_accounts(
    conn: Connection,
) -> None:
    scope = uuid.uuid4()
    first = make(
        conn,
        IdempotencyRecord,
        actor_type="ANONYMOUS",
        actor_id=scope,
        operation="request_staff_handoff",
    )
    shared = {"idempotency_key": first.idempotency_key}

    make(
        conn,
        IdempotencyRecord,
        actor_type="ANONYMOUS",
        actor_id=uuid.uuid4(),
        operation="request_staff_handoff",
        **shared,
    )
    make(
        conn,
        IdempotencyRecord,
        actor_type="ANONYMOUS",
        actor_id=scope,
        operation="another_operation",
        **shared,
    )
    for actor_type in ("PATIENT", "STAFF"):
        make(
            conn,
            IdempotencyRecord,
            actor_type=actor_type,
            actor_id=scope,
            operation="request_staff_handoff",
            **shared,
        )


@pytest.mark.parametrize("actor_type", ["SYSTEM", "anonymous", "PUBLIC", " "])
def test_unapproved_idempotency_actor_types_are_rejected(conn: Connection, actor_type: str) -> None:
    reject_insert(
        conn,
        IdempotencyRecord,
        "ck_idempotency_records_actor_type_valid",
        actor_type=actor_type,
        actor_id=uuid.uuid4(),
    )


def test_idempotency_actor_id_remains_required(conn: Connection) -> None:
    with pytest.raises(IntegrityError) as exc_info, conn.begin_nested():
        conn.execute(
            insert(IdempotencyRecord).values(
                actor_type="ANONYMOUS",
                operation="request_staff_handoff",
                idempotency_key="key",
                request_hash=b"hash",
                status="COMPLETED",
            )
        )

    assert exc_info.value.orig.diag.column_name == "actor_id"


# --- 5. Idempotency response snapshots -------------------------------------------------------


def test_response_snapshot_round_trips_a_json_object(conn: Connection) -> None:
    snapshot = {"ticket_id": str(uuid.uuid4()), "status": "OPEN", "simulated": True}

    record = make(
        conn,
        IdempotencyRecord,
        actor_id=uuid.uuid4(),
        response_code=201,
        result_type="HANDOFF_TICKET",
        result_id=uuid.uuid4(),
        response_snapshot=snapshot,
    )

    assert record.response_snapshot == snapshot
    assert (record.response_code, record.result_type) == (201, "HANDOFF_TICKET")
    assert record.result_id is not None


def test_response_snapshot_may_be_omitted_or_sql_null(conn: Connection) -> None:
    omitted = make(conn, IdempotencyRecord, actor_id=uuid.uuid4())
    explicit_none = make(conn, IdempotencyRecord, actor_id=uuid.uuid4(), response_snapshot=None)

    stored_as_sql_null = conn.execute(
        select(func.count()).where(
            IdempotencyRecord.idempotency_id.in_(
                [omitted.idempotency_id, explicit_none.idempotency_id]
            ),
            IdempotencyRecord.response_snapshot.is_(None),
        )
    ).scalar_one()
    assert stored_as_sql_null == 2


def test_empty_object_snapshot_is_allowed(conn: Connection) -> None:
    assert (
        make(conn, IdempotencyRecord, actor_id=uuid.uuid4(), response_snapshot={}).response_snapshot
        == {}
    )


@pytest.mark.parametrize(
    "snapshot",
    [JSON.NULL, [], [{"status": "OPEN"}], "OPEN", 201, True],
    ids=["json-null", "empty-array", "array-of-objects", "string", "number", "boolean"],
)
def test_non_object_response_snapshots_are_rejected(conn: Connection, snapshot) -> None:
    reject_insert(
        conn,
        IdempotencyRecord,
        "ck_idempotency_records_response_snapshot_is_object",
        actor_id=uuid.uuid4(),
        response_snapshot=snapshot,
    )
