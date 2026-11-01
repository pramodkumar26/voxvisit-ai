"""Races between real database transactions.

Each race holds a row lock until every contender is provably waiting, then releases it, so the
contention is deterministic and the assertions are about outcomes, not timing luck.
"""

import uuid

import pytest
from action_support import (
    appointment_row,
    assert_slots_consistent,
    book_body,
    cancel_body,
    confirm_job,
    holding_row_lock,
    patient_session,
    prepare_job,
    prepared,
    principal_of,
    race,
    reschedule_body,
    scalar,
    seed_appointment,
    slot_row,
    snapshot,
    world,
)
from sqlalchemy import text

from app.core.errors import (
    AuthenticationRequiredError,
    ConflictError,
    SessionExpiredError,
    SlotUnavailableError,
)
from app.models.appointment_actions import AppointmentActionResult, PendingActionView


def winners(results) -> list:
    return [item for item in results if not isinstance(item, Exception)]


def failures(results) -> list[Exception]:
    return [item for item in results if isinstance(item, Exception)]


def release_events(engine, slot_id) -> int:
    return scalar(
        engine,
        "SELECT count(*) FROM outbox_events WHERE aggregate_id = :id"
        " AND event_type = 'SLOT_RELEASED'",
        id=slot_id,
    )


def test_two_patients_racing_for_one_slot_produce_exactly_one_appointment(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    first, second = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")
    jobs = [
        confirm_job(auth_env, first, prepared(auth_env, first, book_body(slot_id))),
        confirm_job(auth_env, second, prepared(auth_env, second, book_body(slot_id))),
    ]
    version_before = slot_row(auth_env.engine, slot_id).version

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(auth_env.engine, "slots", "slot_id", slot_id),
    )

    [(winner, replayed)] = winners(results)
    [loser] = failures(results)
    assert isinstance(winner, AppointmentActionResult) and replayed is False
    assert isinstance(loser, SlotUnavailableError)
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=slot_id)
        == 1
    )
    assert slot_row(auth_env.engine, slot_id).version == version_before + 1
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM appointment_history WHERE new_slot_id = :id",
            id=slot_id,
        )
        == 1
    )
    assert_slots_consistent(auth_env.engine, [slot_id])


def test_many_patients_racing_without_any_lock_hold_still_yield_one_winner(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    codes = ["PT-1001", "PT-1002", "PT-1003", "PT-1004", "PT-1005", "PT-1006"]
    sessions = [patient_session(auth_env, code) for code in codes]
    jobs = [
        confirm_job(auth_env, session, prepared(auth_env, session, book_body(slot_id)))
        for session in sessions
    ]

    results = race(auth_env.engine, jobs)

    assert len(winners(results)) == 1
    assert all(isinstance(item, SlotUnavailableError) for item in failures(results))
    assert_slots_consistent(auth_env.engine, [slot_id])


def test_concurrent_retries_with_one_key_book_once_and_return_the_same_result(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    key = "concurrent-" + uuid.uuid4().hex
    jobs = [confirm_job(auth_env, session, pending_id, key=key) for _ in range(2)]

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(auth_env.engine, "slots", "slot_id", slot_id),
    )

    assert not failures(results), failures(results)
    (first, first_replay), (second, second_replay) = results
    assert sorted([first_replay, second_replay]) == [False, True]
    assert first.appointment_id == second.appointment_id
    assert first.model_dump() == second.model_dump()
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=slot_id)
        == 1
    )
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM appointment_history WHERE appointment_id = :id",
            id=first.appointment_id,
        )
        == 1
    )
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM idempotency_records WHERE idempotency_key = :key",
            key=key,
        )
        == 1
    )


def test_one_pending_action_confirmed_twice_concurrently_commits_once(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    jobs = [confirm_job(auth_env, session, pending_id) for _ in range(2)]  # Different keys.

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(auth_env.engine, "slots", "slot_id", slot_id),
    )

    assert len(winners(results)) == 1
    [loser] = failures(results)
    assert isinstance(loser, ConflictError) and loser.reason == "CONFIRMATION_CONSUMED"
    assert_slots_consistent(auth_env.engine, [slot_id])


def test_concurrent_cancellation_retries_release_the_slot_once(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", slot_id)
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, cancel_body(appointment_id))
    key = "cancel-retry-" + uuid.uuid4().hex
    version_before = slot_row(auth_env.engine, slot_id).version
    jobs = [confirm_job(auth_env, session, pending_id, key=key) for _ in range(3)]

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(
            auth_env.engine, "appointments", "appointment_id", appointment_id
        ),
        blocked=3,
    )

    assert not failures(results), failures(results)
    assert sorted(replayed for _, replayed in results) == [False, True, True]
    assert {result.appointment_id for result, _ in results} == {appointment_id}
    assert release_events(auth_env.engine, slot_id) == 1
    assert slot_row(auth_env.engine, slot_id).version == version_before + 1
    assert appointment_row(auth_env.engine, appointment_id).version == 2


@pytest.mark.parametrize("round_number", range(4))
def test_cancelling_and_rescheduling_one_appointment_have_one_winner(
    auth_env, round_number
) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, _, new_slot = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    cancel_session = patient_session(auth_env, "PT-1001")
    move_session = patient_session(auth_env, "PT-1001")  # Two browser tabs, two sessions.
    cancel_pending = prepared(auth_env, cancel_session, cancel_body(appointment_id))
    move_pending = prepared(auth_env, move_session, reschedule_body(appointment_id, new_slot))
    jobs = [
        confirm_job(auth_env, cancel_session, cancel_pending),
        confirm_job(auth_env, move_session, move_pending),
    ]
    if round_number % 2:
        jobs.reverse()

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(
            auth_env.engine, "appointments", "appointment_id", appointment_id
        ),
    )

    [(winner, _)] = winners(results)
    [loser] = failures(results)
    assert isinstance(loser, ConflictError)
    assert loser.reason in {"APPOINTMENT_NOT_ACTIVE", "CONFIRMATION_STALE"}
    appointment = appointment_row(auth_env.engine, appointment_id)
    if winner.result == "CANCELLED":
        assert appointment.status == "CANCELLED" and appointment.slot_id == old_slot
        assert slot_row(auth_env.engine, new_slot).status == "AVAILABLE"
    else:
        assert winner.result == "RESCHEDULED"
        assert appointment.status == "CONFIRMED" and appointment.slot_id == new_slot
        assert slot_row(auth_env.engine, new_slot).status == "BOOKED"
    assert slot_row(auth_env.engine, old_slot).status == "AVAILABLE"
    assert release_events(auth_env.engine, old_slot) == 1  # Never two release events.
    assert_slots_consistent(auth_env.engine, scenario.slots)


def test_two_reschedules_of_one_appointment_to_different_slots_have_one_winner(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, first_new, second_new = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    one, two = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1001")
    jobs = [
        confirm_job(
            auth_env, one, prepared(auth_env, one, reschedule_body(appointment_id, first_new))
        ),
        confirm_job(
            auth_env, two, prepared(auth_env, two, reschedule_body(appointment_id, second_new))
        ),
    ]

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(
            auth_env.engine, "appointments", "appointment_id", appointment_id
        ),
    )

    assert len(winners(results)) == 1
    [loser] = failures(results)
    assert isinstance(loser, ConflictError) and loser.reason == "CONFIRMATION_STALE"
    booked = [
        slot
        for slot in (first_new, second_new)
        if slot_row(auth_env.engine, slot).status == "BOOKED"
    ]
    assert len(booked) == 1
    assert appointment_row(auth_env.engine, appointment_id).slot_id == booked[0]
    assert release_events(auth_env.engine, old_slot) == 1
    assert_slots_consistent(auth_env.engine, scenario.slots)


def test_rescheduling_races_another_patients_booking_of_the_new_slot(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    old_slot, _, contested = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", old_slot)
    mover, buyer = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")
    jobs = [
        confirm_job(
            auth_env, mover, prepared(auth_env, mover, reschedule_body(appointment_id, contested))
        ),
        confirm_job(auth_env, buyer, prepared(auth_env, buyer, book_body(contested))),
    ]

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(auth_env.engine, "slots", "slot_id", contested),
    )

    [(winner, _)] = winners(results)
    [loser] = failures(results)
    assert isinstance(loser, SlotUnavailableError)
    assert (
        scalar(
            auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=contested
        )
        == 1
    )
    if winner.result == "BOOKED":  # The reschedule lost: the original appointment is untouched.
        assert appointment_row(auth_env.engine, appointment_id).slot_id == old_slot
        assert slot_row(auth_env.engine, old_slot).status == "BOOKED"
        assert release_events(auth_env.engine, old_slot) == 0
    else:
        assert appointment_row(auth_env.engine, appointment_id).slot_id == contested
        assert release_events(auth_env.engine, old_slot) == 1
    assert_slots_consistent(auth_env.engine, scenario.slots)


def test_many_reschedules_into_one_free_slot_finish_without_deadlocks(auth_env) -> None:
    scenario = world(auth_env.engine, count=7)
    *old_slots, target = scenario.slots
    codes = ["PT-1001", "PT-1002", "PT-1003", "PT-1004", "PT-1005", "PT-1006"]
    jobs = []
    for code, old_slot in zip(codes, old_slots, strict=True):
        appointment_id = seed_appointment(auth_env.engine, code, old_slot)
        session = patient_session(auth_env, code)
        pending_id = prepared(auth_env, session, reschedule_body(appointment_id, target))
        jobs.append(confirm_job(auth_env, session, pending_id))

    results = race(auth_env.engine, jobs)

    assert len(winners(results)) == 1
    assert all(isinstance(item, SlotUnavailableError) for item in failures(results)), results
    assert sum(release_events(auth_env.engine, slot) for slot in old_slots) == 1
    assert_slots_consistent(auth_env.engine, scenario.slots)


def test_concurrent_preparation_retries_create_one_pending_action(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    key = "prepare-race-" + uuid.uuid4().hex
    body = book_body(slot_id)
    jobs = [prepare_job(auth_env, session, body, key=key) for _ in range(3)]

    results = race(
        auth_env.engine,
        jobs,
        lock=lambda: holding_row_lock(
            auth_env.engine,
            "auth_sessions",
            "session_id",
            principal_of(auth_env, session).session_id,
            mode="NO KEY UPDATE",
        ),
        blocked=3,
    )

    assert not failures(results), failures(results)
    assert sorted(replayed for _, replayed in results) == [False, True, True]
    assert len({view.pending_action_id for view, _ in results}) == 1
    assert all(isinstance(view, PendingActionView) for view, _ in results)
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM idempotency_records WHERE idempotency_key = :key",
            key=key,
        )
        == 1
    )


def test_a_session_revoked_after_authentication_cannot_commit(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    job = confirm_job(auth_env, session, pending_id)  # Authenticated now...
    with auth_env.engine.begin() as connection:  # ...revoked before the transaction runs.
        connection.execute(
            text(
                "UPDATE auth_sessions SET status = 'REVOKED', terminated_at = now(),"
                " termination_reason = 'TEST' WHERE session_id = :id"
            ),
            {"id": principal_of(auth_env, session).session_id},
        )
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        job()

    assert snapshot(auth_env.engine) == before


def test_a_session_that_expires_before_the_locks_cannot_commit(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    job = confirm_job(auth_env, session, pending_id)
    auth_env.clock.advance(minutes=20)  # Past the idle deadline, with no request in between.
    before = snapshot(auth_env.engine)

    with pytest.raises(SessionExpiredError):
        job()

    assert snapshot(auth_env.engine) == before
    assert slot_row(auth_env.engine, slot_id).status == "AVAILABLE"


def test_logout_and_confirmation_serialize_on_the_session_row(auth_env) -> None:
    """Whichever of logout and confirm gets the session row first, the outcome is coherent."""
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    session_id = principal_of(auth_env, session).session_id

    def logout():
        with auth_env.engine.begin() as connection:
            connection.execute(
                text("SELECT 1 FROM auth_sessions WHERE session_id = :id FOR NO KEY UPDATE"),
                {"id": session_id},
            )
            connection.execute(
                text(
                    "UPDATE auth_sessions SET status = 'LOGGED_OUT', terminated_at = now(),"
                    " termination_reason = 'LOGOUT' WHERE session_id = :id"
                ),
                {"id": session_id},
            )
            connection.execute(
                text(
                    "UPDATE pending_actions SET status = 'INVALIDATED'"
                    " WHERE session_id = :id AND status = 'ACTIVE'"
                ),
                {"id": session_id},
            )

    results = race(
        auth_env.engine,
        [confirm_job(auth_env, session, pending_id), logout],
    )

    committed = scalar(
        auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=slot_id
    )
    confirm_result = results[0]
    if isinstance(confirm_result, Exception):
        assert committed == 0 and slot_row(auth_env.engine, slot_id).status == "AVAILABLE"
        assert isinstance(confirm_result, AuthenticationRequiredError | ConflictError)
    else:
        assert committed == 1 and slot_row(auth_env.engine, slot_id).status == "BOOKED"
    assert_slots_consistent(auth_env.engine, [slot_id])


def test_a_principal_that_does_not_match_its_session_row_cannot_commit(auth_env) -> None:
    from dataclasses import replace

    from action_support import new_key

    from app.services.appointment_transactions import confirm_appointment_action

    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, session, book_body(slot_id))
    other_patient = scalar(
        auth_env.engine, "SELECT patient_id FROM patients WHERE patient_code = 'PT-1002'"
    )
    forged = replace(principal_of(auth_env, session), patient_id=other_patient)
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        confirm_appointment_action(
            auth_env.app.state.db_engine,
            principal=forged,
            pending_action_id=uuid.UUID(pending_id),
            idempotency_key=new_key(),
            request_id=uuid.uuid4(),
            now=auth_env.clock(),
        )

    assert snapshot(auth_env.engine) == before


def test_the_unique_index_is_the_final_guard_against_a_double_booking(auth_env) -> None:
    """Even if a slot's status is wrong, a second confirmed appointment cannot be created."""
    from factories import make

    from app.db.models import Appointment

    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, session, book_body(slot_id))
    with auth_env.engine.begin() as connection:  # Corrupt state: confirmed, yet slot AVAILABLE.
        patient_id = connection.execute(
            text("SELECT patient_id FROM patients WHERE patient_code = 'PT-1002'")
        ).scalar_one()
        make(connection, Appointment, patient_id=patient_id, slot_id=slot_id, status="CONFIRMED")
    before = snapshot(auth_env.engine)

    with pytest.raises(SlotUnavailableError):
        confirm_job(auth_env, session, pending_id)()

    assert snapshot(auth_env.engine) == before
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :id", id=slot_id)
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE patients SET active = false WHERE patient_code = 'PT-1001'",
        "UPDATE patient_credentials SET disabled = true"
        " WHERE patient_id = (SELECT patient_id FROM patients WHERE patient_code = 'PT-1001')",
    ],
)
def test_an_account_deactivated_after_authentication_cannot_commit(auth_env, change) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")
    pending_id = prepared(auth_env, session, book_body(slot_id))
    job = confirm_job(auth_env, session, pending_id)  # Authenticated while still active...
    with auth_env.engine.begin() as connection:
        connection.execute(text(change))  # ...deactivated before the transaction's locks.
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        job()

    assert snapshot(auth_env.engine) == before


def test_a_deactivated_account_cannot_prepare(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")
    job = prepare_job(auth_env, session, book_body(slot_id))
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE patients SET active = false WHERE patient_code = 'PT-1001'")
        )
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        job()

    assert snapshot(auth_env.engine) == before


def completed_booking(auth_env, code="PT-1001"):
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, code)
    pending_id = prepared(auth_env, session, book_body(slot_id))
    key = "replay-" + uuid.uuid4().hex
    confirm_job(auth_env, session, pending_id, key=key)()
    return session, pending_id, key, slot_id


@pytest.mark.parametrize(
    "change",
    [
        "revoked",
        "logged_out",
        "account_deactivated",
        "credential_disabled",
    ],
)
def test_a_confirmation_replay_re_verifies_the_session_and_account(auth_env, change) -> None:
    session, pending_id, key, _ = completed_booking(auth_env)
    replay = confirm_job(auth_env, session, pending_id, key=key)  # Authenticated while valid...
    statements = {
        "revoked": "UPDATE auth_sessions SET status = 'REVOKED', terminated_at = now(),"
        " termination_reason = 'TEST' WHERE session_id = :id",
        "logged_out": "UPDATE auth_sessions SET status = 'LOGGED_OUT', terminated_at = now(),"
        " termination_reason = 'LOGOUT' WHERE session_id = :id",
        "account_deactivated": "UPDATE patients SET active = false WHERE patient_code = 'PT-1001'",
        "credential_disabled": "UPDATE patient_credentials SET disabled = true WHERE patient_id ="
        " (SELECT patient_id FROM patients WHERE patient_code = 'PT-1001')",
    }
    with auth_env.engine.begin() as connection:  # ...and invalidated before the transaction.
        connection.execute(
            text(statements[change]),
            {"id": principal_of(auth_env, session).session_id}
            if "id" in statements[change]
            else {},
        )
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        replay()

    assert snapshot(auth_env.engine) == before


def test_a_confirmation_replay_after_the_session_expired_is_refused(auth_env) -> None:
    session, pending_id, key, _ = completed_booking(auth_env)
    replay = confirm_job(auth_env, session, pending_id, key=key)
    auth_env.clock.advance(minutes=20)

    with pytest.raises(SessionExpiredError):
        replay()


def test_a_preparation_replay_re_verifies_the_session(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env, "PT-1001")
    key = "prep-replay-" + uuid.uuid4().hex
    prepare_job(auth_env, session, book_body(slot_id), key=key)()
    replay = prepare_job(auth_env, session, book_body(slot_id), key=key)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE auth_sessions SET status = 'REVOKED', terminated_at = now(),"
                " termination_reason = 'TEST' WHERE session_id = :id"
            ),
            {"id": principal_of(auth_env, session).session_id},
        )
    before = snapshot(auth_env.engine)

    with pytest.raises(AuthenticationRequiredError):
        replay()

    assert snapshot(auth_env.engine) == before


def test_a_legitimate_service_level_replay_still_returns_the_original_without_changes(
    auth_env,
) -> None:
    session, pending_id, key, _ = completed_booking(auth_env)
    before = snapshot(auth_env.engine)

    result, replayed = confirm_job(auth_env, session, pending_id, key=key)()

    assert replayed is True and result.result == "BOOKED"
    assert snapshot(auth_env.engine) == before
