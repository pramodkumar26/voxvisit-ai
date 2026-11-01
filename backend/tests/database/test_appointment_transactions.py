import re
import time as wall_time
import uuid
from contextlib import contextmanager
from datetime import time, timedelta

import pytest
from action_support import (
    NEAR_CLOCK,
    age_pending_action,
    appointment_row,
    book_body,
    cancel_body,
    changed_tables,
    confirm,
    confirm_job,
    error_of,
    holding_row_lock,
    patient_session,
    prepare,
    prepared,
    reschedule_body,
    rows,
    scalar,
    seed_appointment,
    slot_row,
    snapshot,
    world,
)
from auth_support import call
from scheduling_support import MONDAY, add_holiday, add_leave
from sqlalchemy import event, text

from app.core.errors import DependencyUnavailableError
from app.services import appointment_transactions, write_access
from app.services.write_access import write_transaction

# --- helpers --------------------------------------------------------------------------------


def availability_ids(env, scenario, day=MONDAY) -> set[str]:
    response = call(
        env,
        "GET",
        "/availability",
        params={
            "clinic_id": str(scenario.clinic.clinic_id),
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        },
    )
    assert response.status_code == 200, response.text
    return {slot["slot_id"] for slot in response.json()["slots"]}


def my_appointments(env, session, **params) -> list[dict]:
    response = call(env, "GET", "/me/appointments", session, params=params)
    assert response.status_code == 200, response.text
    return response.json()["appointments"]


def history_of(engine, appointment_id) -> list[tuple]:
    return rows(
        engine,
        "SELECT action, actor_type, actor_id, old_slot_id, new_slot_id FROM appointment_history"
        " WHERE appointment_id = :id ORDER BY timestamp, event_id",
        id=appointment_id,
    )


def outbox_for(engine, slot_id) -> list[tuple]:
    return rows(
        engine,
        "SELECT event_type, aggregate_type, status, payload, dedupe_key FROM outbox_events"
        " WHERE aggregate_id = :id ORDER BY created_at",
        id=slot_id,
    )


def pending_status(engine, pending_id) -> str:
    return scalar(
        engine,
        "SELECT status FROM pending_actions WHERE pending_action_id = :id",
        id=pending_id,
    )


# --- successful transactions ----------------------------------------------------------------


def test_booking_commits_the_appointment_slot_history_and_idempotency_record(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, session, book_body(slot_id))
    key = "booking-key-" + uuid.uuid4().hex

    response = confirm(auth_env, session, pending_id, key=key)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["result"], body["status"]) == ("BOOKED", "CONFIRMED")
    assert body["previous"] is None
    assert body["appointment"]["time"] == "9:00 AM"
    appointment_id = body["appointment_id"]

    appointment = appointment_row(auth_env.engine, appointment_id)
    assert (appointment.status, appointment.version, appointment.slot_id) == (
        "CONFIRMED",
        1,
        slot_id,
    )
    assert appointment.cancelled_at is None
    patient_id = scalar(
        auth_env.engine, "SELECT patient_id FROM patients WHERE patient_code = 'PT-1001'"
    )
    assert appointment.patient_id == patient_id
    assert slot_row(auth_env.engine, slot_id).status == "BOOKED"
    assert slot_row(auth_env.engine, slot_id).version == 2
    assert history_of(auth_env.engine, appointment_id) == [
        ("BOOKED", "PATIENT", patient_id, None, slot_id)
    ]
    assert outbox_for(auth_env.engine, slot_id) == []  # Booking releases nothing.
    assert pending_status(auth_env.engine, pending_id) == "CONSUMED"
    record = rows(
        auth_env.engine,
        "SELECT status, response_code, result_type, result_id, completed_at IS NOT NULL"
        " FROM idempotency_records WHERE idempotency_key = :key",
        key=key,
    )
    assert record == [("COMPLETED", 200, "APPOINTMENT", uuid.UUID(appointment_id), True)]


def test_a_booked_appointment_is_retrievable_and_the_slot_leaves_search(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env, "PT-1004")
    assert str(slot_id) in availability_ids(auth_env, scenario)

    response = confirm(auth_env, session, prepared(auth_env, session, book_body(slot_id)))

    window = {"start_date": MONDAY.isoformat(), "end_date": MONDAY.isoformat()}
    listed = my_appointments(auth_env, session, **window)
    assert [item["appointment_id"] for item in listed] == [response.json()["appointment_id"]]
    assert listed[0]["status"] == "CONFIRMED"
    assert listed[0]["starts_at_local"] == "2026-12-07T09:00:00-07:00"
    assert str(slot_id) not in availability_ids(auth_env, scenario)
    other = patient_session(auth_env, "PT-1005")
    assert my_appointments(auth_env, other, **window) == []


def test_rescheduling_updates_the_same_appointment_and_releases_the_old_slot(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, _, new_slot = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    session = patient_session(auth_env, "PT-1001")
    before_old, before_new = (
        slot_row(auth_env.engine, old_slot),
        slot_row(auth_env.engine, new_slot),
    )
    pending_id = prepared(auth_env, session, reschedule_body(appointment_id, new_slot))

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["result"], body["status"]) == ("RESCHEDULED", "CONFIRMED")
    assert body["appointment_id"] == str(appointment_id)
    assert (body["previous"]["time"], body["appointment"]["time"]) == ("9:00 AM", "10:00 AM")

    appointment = appointment_row(auth_env.engine, appointment_id)
    assert (appointment.status, appointment.slot_id, appointment.version) == (
        "CONFIRMED",
        new_slot,
        2,
    )
    assert slot_row(auth_env.engine, old_slot).status == "AVAILABLE"
    assert slot_row(auth_env.engine, old_slot).version == before_old.version + 1
    assert slot_row(auth_env.engine, new_slot).status == "BOOKED"
    assert slot_row(auth_env.engine, new_slot).version == before_new.version + 1
    patient_id = appointment.patient_id
    assert history_of(auth_env.engine, appointment_id) == [
        ("RESCHEDULED", "PATIENT", patient_id, old_slot, new_slot)
    ]
    [event_row] = outbox_for(auth_env.engine, old_slot)
    event_type, aggregate_type, status, payload, dedupe_key = event_row
    assert (event_type, aggregate_type, status) == ("SLOT_RELEASED", "SLOT", "PENDING")
    assert payload["reason"] == "RESCHEDULED" and payload["slot_id"] == str(old_slot)
    assert dedupe_key == f"SLOT_RELEASED:{old_slot}:{slot_row(auth_env.engine, old_slot).version}"
    assert outbox_for(auth_env.engine, new_slot) == []
    assert pending_status(auth_env.engine, pending_id) == "CONSUMED"
    assert {str(old_slot), str(new_slot)} & availability_ids(auth_env, scenario) == {str(old_slot)}
    assert [item["starts_at_local"] for item in my_appointments(auth_env, session)][-1:] == [
        "2026-12-07T10:00:00-07:00"
    ]


def test_cancelling_preserves_the_appointment_and_releases_the_slot(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    appointment_id = seed_appointment(auth_env.engine, "PT-1006", slot_id)
    session = patient_session(auth_env, "PT-1006")
    before = slot_row(auth_env.engine, slot_id)
    pending_id = prepared(auth_env, session, cancel_body(appointment_id))

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["result"], body["status"]) == ("CANCELLED", "CANCELLED")
    appointment = appointment_row(auth_env.engine, appointment_id)
    assert (appointment.status, appointment.version, appointment.slot_id) == (
        "CANCELLED",
        2,
        slot_id,
    )
    assert appointment.cancelled_at is not None
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"
    assert slot_row(auth_env.engine, slot_id).version == before.version + 1
    assert history_of(auth_env.engine, appointment_id) == [
        ("CANCELLED", "PATIENT", appointment.patient_id, slot_id, None)
    ]
    [(event_type, _, status, payload, _)] = outbox_for(auth_env.engine, slot_id)
    assert (event_type, status, payload["reason"]) == ("SLOT_RELEASED", "PENDING", "CANCELLED")
    cancelled = my_appointments(auth_env, session, status="CANCELLED")
    assert [item["appointment_id"] for item in cancelled] == [str(appointment_id)]
    assert str(slot_id) in availability_ids(auth_env, scenario)


def test_a_cancelled_appointment_is_never_restored_only_rebooked_as_a_new_one(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    slot_id = scenario.slots[0]
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", slot_id)
    session = patient_session(auth_env)
    confirm(auth_env, session, prepared(auth_env, session, cancel_body(appointment_id)))

    for body in (cancel_body(appointment_id), reschedule_body(appointment_id, scenario.slots[1])):
        response = prepare(auth_env, session, body)
        assert error_of(response)["reason"] == "APPOINTMENT_NOT_ACTIVE"

    rebooked = confirm(auth_env, session, prepared(auth_env, session, book_body(slot_id)))
    assert rebooked.json()["appointment_id"] != str(appointment_id)
    assert appointment_row(auth_env.engine, appointment_id).status == "CANCELLED"
    assert appointment_row(auth_env.engine, rebooked.json()["appointment_id"]).status == (
        "CONFIRMED"
    )
    assert len(outbox_for(auth_env.engine, slot_id)) == 1


def test_history_and_responses_carry_no_credentials_or_conversation_content(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, "PT-1007")
    pending = prepare(auth_env, session, book_body(scenario.slots[0]))
    result = confirm(auth_env, session, pending.json()["pending_action_id"])

    for response in (pending, result):
        for secret in (auth_env.pin, session.token, session.csrf, "argon2", "PT-1007"):
            assert secret not in response.text
    columns = rows(
        auth_env.engine,
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_name = 'appointment_history'",
    )
    assert {name for (name,) in columns} == {
        "event_id",
        "appointment_id",
        "action",
        "actor_id",
        "actor_type",
        "old_slot_id",
        "new_slot_id",
        "timestamp",
        "request_id",
    }
    snapshot_text = str(
        rows(
            auth_env.engine,
            "SELECT response_snapshot FROM idempotency_records WHERE result_id = :id",
            id=result.json()["appointment_id"],
        )
    )
    for secret in (auth_env.pin, session.token, session.csrf, "PT-1007", "hash"):
        assert secret not in snapshot_text


# --- confirmation validity ------------------------------------------------------------------


def test_confirming_an_unknown_pending_action_changes_nothing(auth_env) -> None:
    session = patient_session(auth_env)
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, uuid.uuid4())

    assert response.status_code == 404 and error_of(response)["code"] == "RESOURCE_NOT_AVAILABLE"
    assert snapshot(auth_env.engine) == before


@pytest.mark.parametrize("minutes_old", [10.0, 11.0, 600.0])
def test_expired_confirmations_are_rejected_using_database_time(auth_env, minutes_old) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    age_pending_action(auth_env.engine, pending_id, minutes=minutes_old)
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409
    assert (error_of(response)["code"], error_of(response)["reason"]) == (
        "CONFLICT",
        "CONFIRMATION_EXPIRED",
    )
    assert snapshot(auth_env.engine) == before
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_a_confirmation_just_inside_ten_minutes_is_accepted(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    age_pending_action(auth_env.engine, pending_id, minutes=9.5)

    assert confirm(auth_env, session, pending_id).status_code == 200


def test_the_application_clock_cannot_extend_or_shorten_a_confirmation(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    auth_env.clock.advance(minutes=14)  # Past the ten minutes by the application clock only.

    assert confirm(auth_env, session, pending_id).status_code == 200


def test_a_consumed_confirmation_cannot_be_used_again(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    assert confirm(auth_env, session, pending_id).status_code == 200
    before = snapshot(auth_env.engine)

    again = confirm(auth_env, session, pending_id)  # A new idempotency key: a new attempt.

    assert again.status_code == 409 and error_of(again)["reason"] == "CONFIRMATION_CONSUMED"
    assert snapshot(auth_env.engine) == before


def test_an_invalidated_confirmation_is_rejected(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE pending_actions SET status = 'INVALIDATED' WHERE pending_action_id = :id"),
            {"id": pending_id},
        )

    response = confirm(auth_env, session, pending_id)

    assert error_of(response)["reason"] == "CONFIRMATION_INVALIDATED"
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


@pytest.mark.parametrize("status", ["EXPIRED"])
def test_a_confirmation_marked_expired_is_rejected(auth_env, status) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE pending_actions SET status = :s WHERE pending_action_id = :id"),
            {"s": status, "id": pending_id},
        )

    assert error_of(confirm(auth_env, session, pending_id))["reason"] == "CONFIRMATION_EXPIRED"


def test_logging_out_invalidates_pending_confirmations(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    assert call(auth_env, "POST", "/auth/logout", session).status_code in (200, 204)

    assert pending_status(auth_env.engine, pending_id) == "INVALIDATED"
    again = patient_session(auth_env)
    response = confirm(auth_env, again, pending_id)
    assert response.status_code == 404  # It belonged to a session that no longer exists.
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_another_patients_pending_action_cannot_be_confirmed(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    owner = patient_session(auth_env, "PT-1001")
    thief = patient_session(auth_env, "PT-1002")
    pending_id = prepared(auth_env, owner, book_body(slot_id))
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, thief, pending_id)

    assert response.status_code == 404 and error_of(response)["code"] == "RESOURCE_NOT_AVAILABLE"
    assert snapshot(auth_env.engine) == before
    assert confirm(auth_env, owner, pending_id).status_code == 200


def test_a_confirmation_is_bound_to_the_session_that_prepared_it(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    first = patient_session(auth_env, "PT-1001")
    second = patient_session(auth_env, "PT-1001")  # Same patient, another browser session.
    pending_id = prepared(auth_env, first, book_body(slot_id))

    assert confirm(auth_env, second, pending_id).status_code == 404
    assert confirm(auth_env, first, pending_id).status_code == 200


def test_the_session_expiring_before_confirmation_blocks_the_commit(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    auth_env.clock.advance(minutes=16)
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 401 and error_of(response)["code"] == "SESSION_EXPIRED"
    assert snapshot(auth_env.engine)["appointments"] == before["appointments"]
    assert changed_tables(before, snapshot(auth_env.engine)) <= {"pending"}  # Invalidated.
    fresh = patient_session(auth_env)
    assert confirm(auth_env, fresh, pending_id).status_code == 404


def test_the_confirm_body_must_be_an_explicit_true(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))

    for body in (
        {},
        {"confirmed": False},
        {"confirmed": "yes"},
        {"confirmed": 1},
        {"confirmed": True, "x": 1},
    ):
        response = confirm(auth_env, session, pending_id, body=body)
        assert response.status_code == 422, body
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"
    assert pending_status(auth_env.engine, pending_id) == "ACTIVE"


@pytest.mark.parametrize("fault", ["no_csrf", "wrong_csrf", "foreign_origin"])
def test_confirmation_requires_csrf_protection(auth_env, fault) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    options = {
        "no_csrf": {"csrf": None},
        "wrong_csrf": {"csrf": "forged"},
        "foreign_origin": {"origin": "https://evil.example"},
    }[fault]

    response = confirm(auth_env, session, pending_id, **options)

    assert response.status_code == 403 and error_of(response)["code"] == "CSRF_FAILED"
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_anonymous_callers_cannot_confirm(auth_env) -> None:
    response = confirm(auth_env, None, uuid.uuid4())

    assert response.status_code == 401 and error_of(response)["code"] == "AUTH_REQUIRED"


# --- state that changes between preparation and confirmation --------------------------------


def test_a_slot_taken_before_confirmation_fails_without_creating_an_appointment(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    early, late = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")
    early_pending = prepared(auth_env, early, book_body(slot_id))
    late_pending = prepared(auth_env, late, book_body(slot_id))  # Both see it as available.
    assert confirm(auth_env, early, early_pending).status_code == 200
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, late, late_pending)

    assert response.status_code == 409 and error_of(response)["code"] == "SLOT_UNAVAILABLE"
    assert snapshot(auth_env.engine) == before
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=slot_id)
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        "removed",
        "held",
        "inactive_doctor",
        "inactive_clinic",
        "inactive_assignment",
        "holiday",
        "leave",
        "rule_withdrawn",
        "hours_closed",
    ],
)
def test_availability_changes_before_confirmation_block_the_booking(auth_env, change) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    ids = {
        "slot": slot_id,
        "doctor": scenario.doctor.doctor_id,
        "clinic": scenario.clinic.clinic_id,
    }
    if change == "holiday":
        add_holiday(auth_env.engine, scenario.clinic, MONDAY)
    elif change == "leave":
        add_leave(
            auth_env.engine, scenario.clinic, scenario.doctor, (MONDAY, time(8)), (MONDAY, time(10))
        )
    else:
        statement = {
            "removed": "UPDATE slots SET removed_at = now() WHERE slot_id = :slot",
            "held": "UPDATE slots SET status = 'HELD', version = version + 1 WHERE slot_id = :slot",
            "inactive_doctor": "UPDATE doctors SET active = false WHERE doctor_id = :doctor",
            "inactive_clinic": "UPDATE clinics SET active = false WHERE clinic_id = :clinic",
            "inactive_assignment": "UPDATE doctor_clinics SET active = false"
            " WHERE doctor_id = :doctor AND clinic_id = :clinic",
            "rule_withdrawn": "UPDATE doctor_availability_rules SET active = false"
            " WHERE doctor_id = :doctor",
            "hours_closed": "DELETE FROM clinic_operating_hours WHERE clinic_id = :clinic",
        }[change]
        with auth_env.engine.begin() as connection:
            connection.execute(text(statement), ids)
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["code"] == "SLOT_UNAVAILABLE"
    assert snapshot(auth_env.engine) == before


def test_a_slot_that_was_booked_and_released_since_preparation_is_stale(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    with auth_env.engine.begin() as connection:  # Another patient booked and cancelled it.
        connection.execute(
            text("UPDATE slots SET version = version + 2 WHERE slot_id = :id"), {"id": slot_id}
        )

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_STALE"
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_rescheduling_fails_atomically_when_the_new_slot_was_taken(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, _, new_slot = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    session = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, session, reschedule_body(appointment_id, new_slot))
    seed_appointment(auth_env.engine, "PT-1002", new_slot)  # Someone else wins the new slot.
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["code"] == "SLOT_UNAVAILABLE"
    assert snapshot(auth_env.engine) == before
    appointment = appointment_row(auth_env.engine, appointment_id)
    assert (appointment.slot_id, appointment.status, appointment.version) == (
        old_slot,
        "CONFIRMED",
        1,
    )
    assert slot_row(auth_env.engine, old_slot).status == "BOOKED"
    assert outbox_for(auth_env.engine, old_slot) == []


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_changes_to_the_appointment_make_the_confirmation_stale(auth_env, action) -> None:
    scenario = world(auth_env.engine, count=3)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env)
    body = (
        reschedule_body(appointment_id, scenario.slots[2])
        if action == "RESCHEDULE"
        else cancel_body(appointment_id)
    )
    pending_id = prepared(auth_env, session, body)
    with auth_env.engine.begin() as connection:  # Rescheduled elsewhere in the meantime.
        connection.execute(
            text(
                "UPDATE appointments SET version = version + 1, slot_id = :new"
                " WHERE appointment_id = :id"
            ),
            {"new": scenario.slots[1], "id": appointment_id},
        )
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_STALE"
    assert snapshot(auth_env.engine) == before


def test_a_new_slot_whose_version_changed_makes_a_reschedule_stale(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, reschedule_body(appointment_id, scenario.slots[2]))
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET version = version + 1 WHERE slot_id = :id"),
            {"id": scenario.slots[2]},
        )

    response = confirm(auth_env, session, pending_id)

    assert error_of(response)["reason"] == "CONFIRMATION_STALE"
    assert appointment_row(auth_env.engine, appointment_id).slot_id == scenario.slots[0]


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_an_appointment_cancelled_since_preparation_is_not_changed_again(auth_env, action) -> None:
    scenario = world(auth_env.engine, count=3)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env)
    pending_id = prepared(
        auth_env,
        session,
        reschedule_body(appointment_id, scenario.slots[2])
        if action == "RESCHEDULE"
        else cancel_body(appointment_id),
    )
    other_cancel = prepared(auth_env, session, cancel_body(appointment_id))
    assert confirm(auth_env, session, other_cancel).status_code == 200
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409
    assert error_of(response)["reason"] == "APPOINTMENT_NOT_ACTIVE"
    assert snapshot(auth_env.engine) == before
    assert len(outbox_for(auth_env.engine, scenario.slots[0])) == 1  # No second release event.


# --- self-service cutoff at commit time -----------------------------------------------------


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_the_cutoff_is_rechecked_at_confirmation(auth_env, action) -> None:
    scenario = world(auth_env.engine, [(NEAR_CLOCK, time(8, 10)), (NEAR_CLOCK, time(9))])
    started, later = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", started)
    session = patient_session(auth_env)
    pending_id = prepared(
        auth_env,
        session,
        reschedule_body(appointment_id, later)
        if action == "RESCHEDULE"
        else cancel_body(appointment_id),
    )
    auth_env.clock.advance(minutes=10)  # 08:10 local: the appointment starts now.
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["code"] == "SELF_SERVICE_CUTOFF"
    assert "staff assistance" in error_of(response)["message"]
    assert snapshot(auth_env.engine) == before
    assert appointment_row(auth_env.engine, appointment_id).status == "CONFIRMED"


def test_the_cutoff_boundary_is_exact_in_a_non_utc_clinic_timezone(auth_env) -> None:
    # Both appointments start at 15:10 UTC (20:40 in Kolkata); the test clock starts at 15:00 UTC.
    moment = [(NEAR_CLOCK, time(20, 40))]
    first = world(auth_env.engine, moment, timezone="Asia/Kolkata")
    second = world(auth_env.engine, moment, timezone="Asia/Kolkata")
    session = patient_session(auth_env)
    one = seed_appointment(auth_env.engine, "PT-1001", first.slots[0])
    two = seed_appointment(auth_env.engine, "PT-1001", second.slots[0])
    pending_one = prepared(auth_env, session, cancel_body(one))
    pending_two = prepared(auth_env, session, cancel_body(two))

    auth_env.clock.advance(minutes=9, seconds=59)  # One second before the start.
    allowed = confirm(auth_env, session, pending_one)
    auth_env.clock.advance(seconds=1)  # Exactly the start.
    refused = confirm(auth_env, session, pending_two)

    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["appointment"]["starts_at_local"] == "2026-11-02T20:40:00+05:30"
    assert refused.status_code == 409 and error_of(refused)["code"] == "SELF_SERVICE_CUTOFF"
    assert appointment_row(auth_env.engine, two).status == "CONFIRMED"


# --- canonical lock order ------------------------------------------------------------------


LOCK_PATTERNS = {
    "idempotency": re.compile(r"INSERT INTO idempotency_records"),
    "session": re.compile(r"FROM auth_sessions.*FOR NO KEY UPDATE", re.S),
    "pending_action": re.compile(r"FROM pending_actions.*FOR UPDATE", re.S),
    "appointment": re.compile(r"FROM appointments.*FOR UPDATE", re.S),
    "slot": re.compile(r"FROM slots.*FOR UPDATE", re.S),
}


@contextmanager
def recorded_locks(engine):
    seen: list[tuple[str, object]] = []

    def record(connection, cursor, statement, parameters, context, executemany):
        for name, pattern in LOCK_PATTERNS.items():
            if pattern.search(statement):
                value = None
                if name == "slot":
                    value = next(iter(parameters.values()))
                seen.append((name, value))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


def test_booking_locks_follow_the_canonical_order(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))

    with recorded_locks(auth_env.app.state.db_engine) as seen:
        assert confirm(auth_env, session, pending_id).status_code == 200

    assert seen == [
        ("idempotency", None),
        ("session", None),
        ("pending_action", None),
        ("slot", slot_id),
    ]


def test_cancellation_locks_follow_the_canonical_order(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", slot_id)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, cancel_body(appointment_id))

    with recorded_locks(auth_env.app.state.db_engine) as seen:
        assert confirm(auth_env, session, pending_id).status_code == 200

    assert seen == [
        ("idempotency", None),
        ("session", None),
        ("pending_action", None),
        ("appointment", None),
        ("slot", slot_id),
    ]


@pytest.mark.parametrize("new_slot_has_lower_id", [True, False])
def test_rescheduling_locks_both_slots_in_ascending_id_order(
    auth_env, new_slot_has_lower_id
) -> None:
    scenario = world(auth_env.engine, count=2)
    low, high = sorted(scenario.slots)
    old_slot, new_slot = (high, low) if new_slot_has_lower_id else (low, high)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, reschedule_body(appointment_id, new_slot))

    with recorded_locks(auth_env.app.state.db_engine) as seen:
        assert confirm(auth_env, session, pending_id).status_code == 200

    assert seen == [
        ("idempotency", None),
        ("session", None),
        ("pending_action", None),
        ("appointment", None),
        ("slot", low),
        ("slot", high),
    ]


# --- no partial writes ----------------------------------------------------------------------

FAILURE_POINTS = {
    "BOOK": [
        ("appointments", "INSERT"),
        ("slots", "UPDATE"),
        ("appointment_history", "INSERT"),
        ("pending_actions", "UPDATE"),
        ("idempotency_records", "UPDATE"),
        ("security_audit_events", "INSERT"),
    ],
    "RESCHEDULE": [
        ("slots", "UPDATE"),
        ("appointments", "UPDATE"),
        ("appointment_history", "INSERT"),
        ("outbox_events", "INSERT"),
        ("pending_actions", "UPDATE"),
        ("idempotency_records", "UPDATE"),
        ("security_audit_events", "INSERT"),
    ],
    "CANCEL": [
        ("appointments", "UPDATE"),
        ("slots", "UPDATE"),
        ("appointment_history", "INSERT"),
        ("outbox_events", "INSERT"),
        ("pending_actions", "UPDATE"),
        ("idempotency_records", "UPDATE"),
        ("security_audit_events", "INSERT"),
    ],
}


@contextmanager
def injected_failure(engine, table: str, operation: str):
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE OR REPLACE FUNCTION inject_failure() RETURNS trigger LANGUAGE plpgsql AS"
                " $$ BEGIN RAISE EXCEPTION 'injected failure'; END $$"
            )
        )
        connection.execute(
            text(
                f"CREATE TRIGGER inject_failure BEFORE {operation} ON {table}"
                " FOR EACH ROW EXECUTE FUNCTION inject_failure()"
            )
        )
    try:
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(text(f"DROP TRIGGER IF EXISTS inject_failure ON {table}"))


@pytest.mark.parametrize(
    ("action", "table", "operation"),
    [(action, table, op) for action, points in FAILURE_POINTS.items() for table, op in points],
)
def test_a_database_failure_at_any_write_rolls_back_everything(
    auth_env, action, table, operation
) -> None:
    scenario = world(auth_env.engine, count=3)
    session = patient_session(auth_env, "PT-1001")
    if action == "BOOK":
        body = book_body(scenario.slots[0])
    else:
        appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
        body = (
            reschedule_body(appointment_id, scenario.slots[2])
            if action == "RESCHEDULE"
            else cancel_body(appointment_id)
        )
    pending_id = prepared(auth_env, session, body)
    key = "retry-key-" + uuid.uuid4().hex
    before = snapshot(auth_env.engine)

    with (
        injected_failure(auth_env.engine, table, operation),
        pytest.raises(Exception, match=r"injected failure|InternalError|RaiseException"),
    ):
        confirm(auth_env, session, pending_id, key=key)

    assert snapshot(auth_env.engine) == before  # No appointment, slot, history, outbox, or record.
    assert pending_status(auth_env.engine, pending_id) == "ACTIVE"
    retry = confirm(auth_env, session, pending_id, key=key)  # The same key is still usable.
    assert retry.status_code == 200, retry.text
    assert "Idempotent-Replay" not in retry.headers  # It really was a first execution.
    # Exactly one release event, however many attempts it took.
    releases = 0 if action == "BOOK" else 1
    assert len(outbox_for(auth_env.engine, scenario.slots[0])) == releases
    assert len(history_of(auth_env.engine, retry.json()["appointment_id"])) == 1


# --- timeouts -------------------------------------------------------------------------------


def test_write_transactions_use_read_committed_and_the_approved_timeout(auth_env) -> None:
    assert write_access.WRITE_TIMEOUT_MS == 12000
    with write_transaction(auth_env.app.state.db_engine) as connection:
        isolation = connection.execute(text("SHOW transaction_isolation")).scalar_one()
        timeout = connection.execute(text("SHOW statement_timeout")).scalar_one()
        whole = connection.execute(text("SHOW transaction_timeout")).scalar_one()

    assert (isolation, timeout, whole) == ("read committed", "12s", "12s")


def test_a_lock_wait_past_the_timeout_fails_safely_and_can_be_retried(
    auth_env, monkeypatch
) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    monkeypatch.setattr(
        appointment_transactions,
        "write_transaction",
        lambda engine: write_access.write_transaction(engine, timeout_ms=300),
    )
    before = snapshot(auth_env.engine)
    key = "timeout-key-" + uuid.uuid4().hex

    with holding_row_lock(auth_env.engine, "slots", "slot_id", slot_id):
        response = confirm(auth_env, session, pending_id, key=key)

    assert response.status_code == 503
    assert error_of(response)["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "statement timeout" not in response.text and "psycopg" not in response.text
    assert snapshot(auth_env.engine) == before
    monkeypatch.undo()
    assert confirm(auth_env, session, pending_id, key=key).status_code == 200


# --- audit ----------------------------------------------------------------------------------


def test_successful_and_rejected_attempts_leave_sanitized_audit_events(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    confirm(auth_env, session, pending_id)
    confirm(auth_env, session, pending_id)  # Rejected: already consumed.

    events = rows(
        auth_env.engine,
        "SELECT event_type, result, details FROM security_audit_events"
        " WHERE event_type LIKE 'APPOINTMENT_%' ORDER BY occurred_at, event_type",
    )

    by_type = {event_type: (result, details) for event_type, result, details in events}
    assert by_type["APPOINTMENT_ACTION_PREPARED"][1] == {"action": "BOOK"}
    assert by_type["APPOINTMENT_BOOKED"][0] == "SUCCESS"
    assert by_type["APPOINTMENT_ACTION_REJECTED"] == (
        "DENIED",
        {"operation": "confirm", "code": "CONFLICT", "reason": "CONFIRMATION_CONSUMED"},
    )
    for _, _, details in events:
        text_form = str(details)
        for secret in (auth_env.pin, session.token, session.csrf):
            assert secret not in text_form


def test_confirmation_timestamps_are_utc_instants_and_display_values_are_local(auth_env) -> None:
    slot_id = world(auth_env.engine, [(MONDAY + timedelta(days=1), time(14, 30))]).slots[0]
    session = patient_session(auth_env)

    result = confirm(auth_env, session, prepared(auth_env, session, book_body(slot_id))).json()

    assert result["appointment"]["starts_at_utc"] == "2026-12-08T21:30:00Z"
    assert result["appointment"]["starts_at_local"] == "2026-12-08T14:30:00-07:00"
    assert (result["appointment"]["date"], result["appointment"]["time"]) == (
        "2026-12-08",
        "2:30 PM",
    )
    assert result["completed_at"].endswith(("Z", "+00:00"))


# --- defense in depth -----------------------------------------------------------------------


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_ownership_is_rechecked_at_commit_even_if_a_pending_action_is_tampered_with(
    auth_env, action
) -> None:
    scenario = world(auth_env.engine, count=4)
    mine = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    theirs = seed_appointment(auth_env.engine, "PT-1002", scenario.slots[1])
    session = patient_session(auth_env, "PT-1001")
    body = reschedule_body(mine, scenario.slots[2]) if action == "RESCHEDULE" else cancel_body(mine)
    pending_id = prepared(auth_env, session, body)
    with auth_env.engine.begin() as connection:  # Point the pending action at someone else's.
        connection.execute(
            text(
                "UPDATE pending_actions SET target_id = :theirs,"
                " expected_versions = jsonb_set(jsonb_set(expected_versions,"
                " '{appointment_id}', to_jsonb(CAST(:theirs AS text))),"
                " '{old_slot_id}', to_jsonb(CAST(:slot AS text)))"
                " WHERE pending_action_id = :id"
            ),
            {"theirs": theirs, "slot": scenario.slots[1], "id": pending_id},
        )
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 404 and error_of(response)["code"] == "RESOURCE_NOT_AVAILABLE"
    assert snapshot(auth_env.engine) == before
    assert appointment_row(auth_env.engine, theirs).status == "CONFIRMED"


def test_every_release_has_its_own_event_even_when_a_slot_is_rebooked_repeatedly(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")

    for _ in range(3):
        booked = confirm(auth_env, session, prepared(auth_env, session, book_body(slot_id)))
        confirm(
            auth_env,
            session,
            prepared(auth_env, session, cancel_body(booked.json()["appointment_id"])),
        )

    events = outbox_for(auth_env.engine, slot_id)
    assert len(events) == 3
    assert len({dedupe_key for *_, dedupe_key in events}) == 3
    assert slot_row(auth_env.engine, slot_id).version == 1 + 3 * 2


# --- the timeout bounds the whole transaction -----------------------------------------------


def test_a_transaction_made_of_individually_fast_statements_still_times_out(auth_env) -> None:
    """Each statement is well under the limit; only a whole-transaction bound stops the chain."""
    started = wall_time.monotonic()
    with (
        pytest.raises(DependencyUnavailableError),
        write_transaction(auth_env.app.state.db_engine, timeout_ms=600) as connection,
    ):
        for _ in range(10):
            connection.execute(text("SELECT pg_sleep(0.2)"))

    assert wall_time.monotonic() - started < 3
    with write_transaction(auth_env.app.state.db_engine) as connection:  # The pool recovers.
        assert connection.execute(text("SELECT 1")).scalar_one() == 1


def test_chained_lock_waits_cannot_outlive_the_transaction_timeout(auth_env, monkeypatch) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(scenario.slots[0]))
    job = confirm_job(auth_env, session, pending_id)  # Authenticated before any lock is held.
    monkeypatch.setattr(
        appointment_transactions,
        "write_transaction",
        lambda engine: write_access.write_transaction(engine, timeout_ms=800),
    )
    started = wall_time.monotonic()

    # The pending action is held, so the lock wait alone exceeds the limit.
    with (
        holding_row_lock(auth_env.engine, "pending_actions", "pending_action_id", pending_id),
        pytest.raises(DependencyUnavailableError),
    ):
        job()

    assert wall_time.monotonic() - started < 5
    assert slot_row(auth_env.engine, scenario.slots[0]).status == "AVAILABLE"
    assert pending_status(auth_env.engine, pending_id) == "ACTIVE"


# --- clocks ---------------------------------------------------------------------------------


def test_moving_the_application_clock_backwards_cannot_revive_an_expired_confirmation(
    auth_env,
) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    age_pending_action(auth_env.engine, pending_id, minutes=11)  # Expired by database time.
    auth_env.clock.advance(minutes=-5)  # An application clock that is behind changes nothing.

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_EXPIRED"
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_an_application_clock_far_ahead_of_the_database_does_not_expire_a_fresh_confirmation(
    auth_env,
) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    auth_env.clock.advance(minutes=14)  # Session still valid (activity refreshes it).

    assert confirm(auth_env, session, pending_id).status_code == 200


def test_pending_action_lifetime_uses_only_database_timestamps(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))

    created, expires, now = rows(
        auth_env.engine,
        "SELECT created_at, expires_at, clock_timestamp() FROM pending_actions"
        " WHERE pending_action_id = :id",
        id=pending_id,
    )[0]

    assert expires - created == timedelta(minutes=10)
    assert timedelta(0) <= now - created < timedelta(seconds=30)  # Stamped by the database.


# --- every version the action depends on, including the old slot -----------------------------


def prepare_existing(auth_env, action: str):
    scenario = world(auth_env.engine, count=3)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env, "PT-1001")
    body = (
        reschedule_body(appointment_id, scenario.slots[2])
        if action == "RESCHEDULE"
        else cancel_body(appointment_id)
    )
    return scenario, appointment_id, session, prepared(auth_env, session, body)


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_preparation_records_the_old_slots_version(auth_env, action) -> None:
    scenario, _, _, pending_id = prepare_existing(auth_env, action)

    stored = rows(
        auth_env.engine,
        "SELECT expected_versions->>'old_slot_version' FROM pending_actions"
        " WHERE pending_action_id = :id",
        id=pending_id,
    )

    assert stored == [(str(slot_row(auth_env.engine, scenario.slots[0]).version),)]


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_a_change_to_the_old_slot_after_preparation_makes_the_confirmation_stale(
    auth_env, action
) -> None:
    scenario, appointment_id, session, pending_id = prepare_existing(auth_env, action)
    with auth_env.engine.begin() as connection:  # Only the slot changed; the appointment did not.
        connection.execute(
            text("UPDATE slots SET version = version + 1 WHERE slot_id = :id"),
            {"id": scenario.slots[0]},
        )
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_STALE"
    assert snapshot(auth_env.engine) == before
    assert appointment_row(auth_env.engine, appointment_id).status == "CONFIRMED"
    assert outbox_for(auth_env.engine, scenario.slots[0]) == []


@pytest.mark.parametrize("action", ["RESCHEDULE", "CANCEL"])
def test_an_unchanged_old_slot_still_confirms(auth_env, action) -> None:
    _, _, session, pending_id = prepare_existing(auth_env, action)

    assert confirm(auth_env, session, pending_id).status_code == 200


@pytest.mark.parametrize(
    ("action", "missing"),
    [
        ("BOOK", "slot_version"),
        ("RESCHEDULE", "old_slot_version"),
        ("RESCHEDULE", "appointment_version"),
        ("RESCHEDULE", "new_slot_version"),
        ("RESCHEDULE", "old_slot_id"),
        ("RESCHEDULE", "new_slot_id"),
        ("CANCEL", "old_slot_version"),
        ("CANCEL", "appointment_version"),
        ("CANCEL", "old_slot_id"),
    ],
)
def test_pending_actions_missing_an_expected_version_fail_closed_as_stale(
    auth_env, action, missing
) -> None:
    """Includes actions prepared before the old-slot version was recorded: they cannot commit,
    and they fail with a clear stale response instead of an internal error."""
    if action == "BOOK":
        slot_id = world(auth_env.engine, count=1).slots[0]
        session = patient_session(auth_env, "PT-1001")
        pending_id = prepared(auth_env, session, book_body(slot_id))
    else:
        _, _, session, pending_id = prepare_existing(auth_env, action)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pending_actions SET expected_versions = expected_versions - :key"
                " WHERE pending_action_id = :id"
            ),
            {"key": missing, "id": pending_id},
        )
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, pending_id)

    assert response.status_code == 409, response.text
    assert error_of(response)["reason"] == "CONFIRMATION_STALE"
    assert snapshot(auth_env.engine) == before
