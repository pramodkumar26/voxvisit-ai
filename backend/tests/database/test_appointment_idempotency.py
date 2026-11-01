import uuid
from datetime import time

from action_support import (
    appointment_row,
    book_body,
    cancel_body,
    changed_tables,
    confirm,
    error_of,
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
from scheduling_support import MONDAY
from sqlalchemy import text


def key() -> str:
    return "idem-" + uuid.uuid4().hex


# --- preparation ----------------------------------------------------------------------------


def test_repeating_a_preparation_returns_the_original_pending_action(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    shared_key = key()
    first = prepare(auth_env, session, book_body(slot_id), key=shared_key)
    before = snapshot(auth_env.engine)
    auth_env.clock.advance(minutes=5)

    second = prepare(auth_env, session, book_body(slot_id), key=shared_key)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()  # Same ID, same summary, and the same expiry.
    assert "Idempotent-Replay" not in first.headers
    assert second.headers["Idempotent-Replay"] == "true"
    assert snapshot(auth_env.engine) == before  # Not even a second pending action or audit event.


def test_a_different_key_prepares_a_separate_pending_action(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)

    first = prepare(auth_env, session, book_body(slot_id))
    second = prepare(auth_env, session, book_body(slot_id))

    assert first.json()["pending_action_id"] != second.json()["pending_action_id"]


def test_reusing_a_preparation_key_for_different_parameters_is_rejected(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    session = patient_session(auth_env)
    shared_key = key()
    assert (
        prepare(auth_env, session, book_body(scenario.slots[0]), key=shared_key).status_code == 200
    )
    before = snapshot(auth_env.engine)

    for different in (
        book_body(scenario.slots[1]),
        cancel_body("00000000-0000-4000-8000-000000000009"),
    ):
        response = prepare(auth_env, session, different, key=shared_key)
        assert response.status_code == 409
        assert (error_of(response)["code"], error_of(response)["reason"]) == (
            "CONFLICT",
            "IDEMPOTENCY_KEY_REUSED",
        )
    assert snapshot(auth_env.engine) == before


def test_a_rejected_preparation_stores_nothing_so_the_key_can_be_reused(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env)
    shared_key = key()
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET status = 'BOOKED' WHERE slot_id = :id"), {"id": slot_id}
        )
    before = snapshot(auth_env.engine)

    rejected = prepare(auth_env, session, book_body(slot_id), key=shared_key)

    assert rejected.status_code == 409
    assert snapshot(auth_env.engine) == before
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET status = 'AVAILABLE' WHERE slot_id = :id"), {"id": slot_id}
        )
    assert prepare(auth_env, session, book_body(slot_id), key=shared_key).status_code == 200


# --- confirmation ---------------------------------------------------------------------------


def test_retrying_a_booking_returns_the_original_result_without_a_second_appointment(
    auth_env,
) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    shared_key = key()
    first = confirm(auth_env, session, pending_id, key=shared_key)
    before = snapshot(auth_env.engine)

    second = confirm(auth_env, session, pending_id, key=shared_key)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert "Idempotent-Replay" not in first.headers
    assert second.headers["Idempotent-Replay"] == "true"
    assert snapshot(auth_env.engine) == before  # No new appointment, history, or audit row.
    assert slot_row(auth_env.engine, slot_id).version == 2


def test_retrying_a_cancellation_does_not_emit_another_release_event(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", slot_id)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, cancel_body(appointment_id))
    shared_key = key()
    first = confirm(auth_env, session, pending_id, key=shared_key)

    for _ in range(3):
        again = confirm(auth_env, session, pending_id, key=shared_key)
        assert again.json() == first.json()

    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM outbox_events WHERE aggregate_id = :id",
            id=slot_id,
        )
        == 1
    )
    assert appointment_row(auth_env.engine, appointment_id).version == 2


def test_retrying_a_reschedule_does_not_move_or_release_anything_again(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, _, new_slot = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, reschedule_body(appointment_id, new_slot))
    shared_key = key()
    first = confirm(auth_env, session, pending_id, key=shared_key)
    before = snapshot(auth_env.engine)

    second = confirm(auth_env, session, pending_id, key=shared_key)

    assert second.json() == first.json() and second.json()["result"] == "RESCHEDULED"
    assert snapshot(auth_env.engine) == before


def test_a_retry_returns_the_original_result_even_after_later_changes(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env)
    shared_key = key()
    pending_id = prepared(auth_env, session, book_body(slot_id))
    first = confirm(auth_env, session, pending_id, key=shared_key)
    appointment_id = first.json()["appointment_id"]
    confirm(auth_env, session, prepared(auth_env, session, cancel_body(appointment_id)))

    replay = confirm(auth_env, session, pending_id, key=shared_key)

    assert replay.json() == first.json() and replay.json()["result"] == "BOOKED"
    assert appointment_row(auth_env.engine, appointment_id).status == "CANCELLED"


def test_a_replay_is_served_only_to_the_session_that_owns_the_pending_action(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    original = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, original, book_body(slot_id))
    shared_key = key()
    first = confirm(auth_env, original, pending_id, key=shared_key)
    before = snapshot(auth_env.engine)

    again = confirm(auth_env, original, pending_id, key=shared_key)
    other_session = confirm(
        auth_env, patient_session(auth_env, "PT-1001"), pending_id, key=shared_key
    )
    other_patient = confirm(
        auth_env, patient_session(auth_env, "PT-1002"), pending_id, key=shared_key
    )

    assert again.json() == first.json() and again.headers["Idempotent-Replay"] == "true"
    for refused in (other_session, other_patient):
        assert refused.status_code == 404
        assert error_of(refused)["code"] == "RESOURCE_NOT_AVAILABLE"
        assert "Idempotent-Replay" not in refused.headers
        assert first.json()["appointment_id"] not in refused.text  # The result is not disclosed.
    assert snapshot(auth_env.engine) == before


def test_a_preparation_replay_is_served_only_to_the_session_that_owns_the_action(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    original = patient_session(auth_env, "PT-1001")
    shared_key = key()
    first = prepare(auth_env, original, book_body(slot_id), key=shared_key)
    before = snapshot(auth_env.engine)

    again = prepare(auth_env, original, book_body(slot_id), key=shared_key)
    other_session = prepare(
        auth_env, patient_session(auth_env, "PT-1001"), book_body(slot_id), key=shared_key
    )

    assert again.json() == first.json() and again.headers["Idempotent-Replay"] == "true"
    assert other_session.status_code == 404
    assert first.json()["pending_action_id"] not in other_session.text
    assert snapshot(auth_env.engine) == before
    # Another patient using the same key string is simply a separate request.
    assert (
        prepare(
            auth_env, patient_session(auth_env, "PT-1002"), book_body(slot_id), key=shared_key
        ).status_code
        == 200
    )


def test_reusing_a_confirmation_key_for_a_different_pending_action_is_rejected(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    session = patient_session(auth_env)
    first_pending = prepared(auth_env, session, book_body(scenario.slots[0]))
    shared_key = key()
    assert confirm(auth_env, session, first_pending, key=shared_key).status_code == 200
    second_pending = prepared(auth_env, session, book_body(scenario.slots[1]))
    before = snapshot(auth_env.engine)

    response = confirm(auth_env, session, second_pending, key=shared_key)

    assert response.status_code == 409
    assert error_of(response)["reason"] == "IDEMPOTENCY_KEY_REUSED"
    assert snapshot(auth_env.engine) == before
    assert slot_row(auth_env.engine, scenario.slots[1]).status == "AVAILABLE"


def test_idempotency_keys_are_scoped_to_the_patient_and_the_operation(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    shared_key = key()
    one, two = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")

    first = prepare(auth_env, one, book_body(scenario.slots[0]), key=shared_key)
    second = prepare(auth_env, two, book_body(scenario.slots[1]), key=shared_key)

    assert first.status_code == second.status_code == 200
    assert first.json()["pending_action_id"] != second.json()["pending_action_id"]
    # The same key string used for the other operation is a separate scope.
    result = confirm(auth_env, one, first.json()["pending_action_id"], key=shared_key)
    assert result.status_code == 200 and "Idempotent-Replay" not in result.headers


def test_a_rejected_confirmation_stores_nothing_so_the_key_can_be_retried(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    shared_key = key()
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET removed_at = now() WHERE slot_id = :id"), {"id": slot_id}
        )
    before = snapshot(auth_env.engine)

    rejected = confirm(auth_env, session, pending_id, key=shared_key)

    assert rejected.status_code == 409
    assert changed_tables(before, snapshot(auth_env.engine)) == set()
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM idempotency_records WHERE idempotency_key = :key",
            key=shared_key,
        )
        == 0
    )
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET removed_at = NULL WHERE slot_id = :id"), {"id": slot_id}
        )
    # The slot version is unchanged, so the same confirmation and key now succeed.
    assert confirm(auth_env, session, pending_id, key=shared_key).status_code == 200


def test_response_snapshots_hold_only_the_approved_display_fields(auth_env) -> None:
    scenario = world(auth_env.engine, [(MONDAY, time(9))])
    session = patient_session(auth_env, "PT-1008")
    pending = prepare(auth_env, session, book_body(scenario.slots[0]))
    result = confirm(auth_env, session, pending.json()["pending_action_id"])

    stored = rows(
        auth_env.engine,
        "SELECT operation, response_snapshot FROM idempotency_records"
        " WHERE result_id IN (:pending, :appointment) ORDER BY operation",
        pending=pending.json()["pending_action_id"],
        appointment=result.json()["appointment_id"],
    )

    assert [operation for operation, _ in stored] == [
        "appointment_action.confirm",
        "appointment_action.prepare",
    ]
    confirm_snapshot, prepare_snapshot = (snapshot_value for _, snapshot_value in stored)
    assert set(confirm_snapshot) == {
        "result",
        "appointment_id",
        "status",
        "appointment",
        "previous",
        "completed_at",
    }
    assert set(prepare_snapshot) == {
        "pending_action_id",
        "action",
        "stored_status",
        "ui_status",
        "summary",
        "created_at",
        "expires_at",
        "confirmation_required",
    }
    combined = str(stored).lower()
    for forbidden in (
        auth_env.pin,
        session.token.lower(),
        session.csrf.lower(),
        "pt-1008",
        "patient_id",
        "argon2",
        "password",
    ):
        assert forbidden not in combined
    assert all(len(str(value)) < 4096 for _, value in stored)
