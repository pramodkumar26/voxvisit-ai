"""A review screen the patient has moved on from cannot commit.

A newer preparation retires the session's other ACTIVE pending actions in the same transaction, and
a discard retires one named action. Both are racing against confirmation here with real
transactions; the invariant is that an appointment exists if and only if its pending action was
CONSUMED, and a retired action never commits.
"""

import uuid

import pytest
from action_support import (
    book_body,
    confirm,
    confirm_job,
    error_of,
    patient_session,
    prepare,
    prepare_job,
    prepared,
    principal_of,
    race,
    rows,
    scalar,
    seed_appointment,
    snapshot,
    staff_session,
    world,
)
from auth_support import call

from app.agent_tools import ToolRuntime
from app.core.errors import (
    AuthenticationRequiredError,
    ConflictError,
    ForbiddenError,
)
from app.services.pending_action_retirement import discard_pending_action


def status_of(engine, pending_id) -> str:
    return scalar(
        engine, "SELECT status FROM pending_actions WHERE pending_action_id = :id", id=pending_id
    )


def appointments_on(engine, slot_id) -> int:
    return scalar(
        engine,
        "SELECT count(*) FROM appointments WHERE slot_id = :id AND status = 'CONFIRMED'",
        id=slot_id,
    )


def discard(env, session, pending_id, *, principal=None):
    return discard_pending_action(
        env.app.state.db_engine,
        principal=principal or principal_of(env, session),
        pending_action_id=uuid.UUID(str(pending_id)),
        request_id=uuid.uuid4(),
        now=env.clock(),
    )


# --- a newer preparation retires the earlier one ------------------------------------------------


def test_a_newer_preparation_retires_the_earlier_review_and_only_the_newer_can_commit(
    auth_env,
) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    old = prepared(auth_env, session, book_body(first_slot))
    new = prepared(auth_env, session, book_body(second_slot))

    assert status_of(auth_env.engine, old) == "INVALIDATED"
    assert status_of(auth_env.engine, new) == "ACTIVE"
    outdated = confirm(auth_env, session, old)  # The screen the patient left open.
    assert (
        outdated.status_code == 409 and error_of(outdated)["reason"] == "CONFIRMATION_INVALIDATED"
    )
    assert appointments_on(auth_env.engine, first_slot) == 0
    assert confirm(auth_env, session, new).status_code == 200
    assert appointments_on(auth_env.engine, second_slot) == 1


def test_changing_back_to_the_same_request_works_with_the_newest_review(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    old = prepared(auth_env, session, book_body(slot_id))
    new = prepared(auth_env, session, book_body(slot_id))

    assert error_of(confirm(auth_env, session, old))["reason"] == "CONFIRMATION_INVALIDATED"
    assert confirm(auth_env, session, new).status_code == 200


def test_another_patients_review_is_untouched_by_a_preparation(auth_env) -> None:
    mine, theirs = world(auth_env.engine, count=2).slots
    owner, other = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")
    other_pending = prepared(auth_env, other, book_body(theirs))

    prepared(auth_env, owner, book_body(mine))

    assert status_of(auth_env.engine, other_pending) == "ACTIVE"
    assert confirm(auth_env, other, other_pending).status_code == 200


def test_a_committed_action_keeps_its_status_when_a_newer_one_is_prepared(auth_env) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    done = prepared(auth_env, session, book_body(first_slot))
    assert confirm(auth_env, session, done).status_code == 200

    prepared(auth_env, session, book_body(second_slot))

    assert status_of(auth_env.engine, done) == "CONSUMED"
    assert appointments_on(auth_env.engine, first_slot) == 1


def test_replaying_an_old_preparation_does_not_retire_the_newer_one(auth_env) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    first = prepare(auth_env, session, book_body(first_slot), key="replay-key-" + "a" * 20)
    new = prepared(auth_env, session, book_body(second_slot))

    again = prepare(auth_env, session, book_body(first_slot), key="replay-key-" + "a" * 20)

    assert again.status_code == 200 and again.headers.get("Idempotent-Replay") == "true"
    assert again.json()["pending_action_id"] == first.json()["pending_action_id"]
    assert status_of(auth_env.engine, new) == "ACTIVE"  # A replay changes nothing.
    assert status_of(auth_env.engine, first.json()["pending_action_id"]) == "INVALIDATED"


def test_a_preparation_that_fails_leaves_the_current_review_alone(auth_env) -> None:
    open_slot, taken_slot = world(auth_env.engine, count=2).slots
    seed_appointment(auth_env.engine, "PT-1002", taken_slot)
    session = patient_session(auth_env)
    current = prepared(auth_env, session, book_body(open_slot))
    before = snapshot(auth_env.engine)

    failed = prepare(auth_env, session, book_body(taken_slot))

    assert failed.status_code == 409
    assert status_of(auth_env.engine, current) == "ACTIVE"
    assert snapshot(auth_env.engine) == before


def test_the_preparation_audit_records_how_many_reviews_it_retired(auth_env) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    prepared(auth_env, session, book_body(first_slot))
    prepared(auth_env, session, book_body(second_slot))

    counts = [
        row[0]["superseded"]
        for row in rows(
            auth_env.engine,
            "SELECT details FROM security_audit_events"
            " WHERE event_type = 'APPOINTMENT_ACTION_PREPARED' ORDER BY occurred_at",
        )
    ]

    assert counts[-2:] == ["0", "1"]


# --- discarding one review -----------------------------------------------------------------------


def test_a_discarded_review_cannot_commit_and_changes_nothing_else(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending = prepared(auth_env, session, book_body(slot_id))

    assert discard(auth_env, session, pending) is True

    assert status_of(auth_env.engine, pending) == "INVALIDATED"
    response = confirm(auth_env, session, pending)
    assert (
        response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_INVALIDATED"
    )
    assert appointments_on(auth_env.engine, slot_id) == 0
    audit = rows(
        auth_env.engine,
        "SELECT result, resource_type FROM security_audit_events"
        " WHERE event_type = 'APPOINTMENT_ACTION_DISCARDED'",
    )
    assert audit == [("SUCCESS", "PENDING_ACTION")]


def test_discarding_twice_or_after_commit_is_a_quiet_no_op(auth_env) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    done = prepared(auth_env, session, book_body(first_slot))
    assert confirm(auth_env, session, done).status_code == 200
    retired = prepared(auth_env, session, book_body(second_slot))
    assert discard(auth_env, session, retired) is True

    assert discard(auth_env, session, retired) is False
    assert discard(auth_env, session, done) is False
    assert discard(auth_env, session, uuid.uuid4()) is False

    assert status_of(auth_env.engine, done) == "CONSUMED"
    assert appointments_on(auth_env.engine, first_slot) == 1
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM security_audit_events"
            " WHERE event_type = 'APPOINTMENT_ACTION_DISCARDED'",
        )
        == 1
    )


def test_one_session_cannot_discard_anothers_review(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    owner, other = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")
    pending = prepared(auth_env, owner, book_body(slot_id))

    assert discard(auth_env, other, pending) is False

    assert status_of(auth_env.engine, pending) == "ACTIVE"
    assert confirm(auth_env, owner, pending).status_code == 200


def test_a_discard_needs_a_live_patient_session(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending = prepared(auth_env, session, book_body(slot_id))
    principal = principal_of(auth_env, session)
    assert call(auth_env, "POST", "/auth/logout", session).status_code in (200, 204)

    with pytest.raises(AuthenticationRequiredError):
        discard(auth_env, session, pending, principal=principal)


def test_the_runtime_discards_only_for_patients_and_exposes_no_tool_for_it(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending = prepared(auth_env, session, book_body(slot_id))
    runtime = ToolRuntime(auth_env.app.state.db_engine, clock=auth_env.clock)

    assert "discard" not in " ".join(runtime.registry.names()).lower()
    assert runtime.discard_pending_action(principal_of(auth_env, session), pending) is True
    assert status_of(auth_env.engine, pending) == "INVALIDATED"
    with pytest.raises(ForbiddenError):
        runtime.discard_pending_action(principal_of(auth_env, staff_session(auth_env)), pending)


# --- races ---------------------------------------------------------------------------------------


def consistent(engine, slot_id, pending_id) -> str:
    """An appointment exists exactly when the action was consumed; a retired one never commits."""
    state, held = status_of(engine, pending_id), appointments_on(engine, slot_id)
    assert (state == "CONSUMED") == (held == 1), (state, held)
    assert state in ("CONSUMED", "INVALIDATED")
    return state


@pytest.mark.parametrize("attempt", range(12))
def test_confirming_races_a_newer_preparation_and_never_commits_a_retired_review(
    auth_env, attempt
) -> None:
    first_slot, second_slot = world(auth_env.engine, count=2).slots
    session = patient_session(auth_env)
    old = prepared(auth_env, session, book_body(first_slot))

    results = race(
        auth_env.engine,
        [
            confirm_job(auth_env, session, old),
            prepare_job(auth_env, session, book_body(second_slot)),
        ],
    )

    confirmation, preparation = results
    new = preparation[0].pending_action_id
    state = consistent(auth_env.engine, first_slot, old)
    assert status_of(auth_env.engine, new) == "ACTIVE"
    if state == "INVALIDATED":
        assert isinstance(confirmation, ConflictError)
        assert confirmation.reason == "CONFIRMATION_INVALIDATED"
    else:
        assert not isinstance(confirmation, Exception)  # It committed first, completely.


@pytest.mark.parametrize("attempt", range(12))
def test_confirming_races_a_discard_and_exactly_one_of_them_wins(auth_env, attempt) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    pending = prepared(auth_env, session, book_body(slot_id))

    results = race(
        auth_env.engine,
        [
            confirm_job(auth_env, session, pending),
            lambda: discard(auth_env, session, pending),
        ],
    )

    confirmation, discarded = results
    state = consistent(auth_env.engine, slot_id, pending)
    if state == "CONSUMED":
        assert discarded is False and not isinstance(confirmation, Exception)
    else:
        assert discarded is True and isinstance(confirmation, ConflictError)


@pytest.mark.parametrize("attempt", range(8))
def test_two_preparations_in_one_session_leave_exactly_one_live_review(auth_env, attempt) -> None:
    slots = world(auth_env.engine, count=3).slots
    session = patient_session(auth_env)
    old = prepared(auth_env, session, book_body(slots[0]))

    results = race(
        auth_env.engine,
        [
            prepare_job(auth_env, session, book_body(slots[1])),
            prepare_job(auth_env, session, book_body(slots[2])),
        ],
    )

    assert not [r for r in results if isinstance(r, Exception)]
    live = rows(
        auth_env.engine,
        "SELECT pending_action_id FROM pending_actions WHERE session_id = :s AND status = 'ACTIVE'",
        s=principal_of(auth_env, session).session_id,
    )
    assert len(live) == 1
    assert status_of(auth_env.engine, old) == "INVALIDATED"
    assert str(live[0][0]) in {str(r[0].pending_action_id) for r in results}


# --- several sessions of one patient (the policy allows three, each independent) ----------------


def test_a_patients_sessions_keep_independent_reviews_and_none_can_retire_anothers(
    auth_env,
) -> None:
    first_slot, second_slot, third_slot = world(auth_env.engine, count=3).slots
    laptop = patient_session(auth_env, "PT-1001")
    phone = patient_session(auth_env, "PT-1001")  # The same patient, a second live session.
    on_laptop = prepared(auth_env, laptop, book_body(first_slot))
    on_phone = prepared(auth_env, phone, book_body(second_slot))

    # Preparing in one session leaves the other session's review alone.
    newer_laptop = prepared(auth_env, laptop, book_body(third_slot))
    assert status_of(auth_env.engine, on_laptop) == "INVALIDATED"
    assert status_of(auth_env.engine, on_phone) == "ACTIVE"
    # A discard from one session cannot reach the other's review, even for the same patient.
    assert discard(auth_env, laptop, on_phone) is False
    assert status_of(auth_env.engine, on_phone) == "ACTIVE"
    # Nor can the other session commit a review it does not own, retired or not.
    for foreign in (on_laptop, newer_laptop):
        assert confirm(auth_env, phone, foreign).status_code == 404
    assert appointments_on(auth_env.engine, first_slot) == 0
    assert appointments_on(auth_env.engine, third_slot) == 0
    assert confirm(auth_env, phone, on_phone).status_code == 200


def test_a_session_revoked_by_the_session_limit_cannot_discard_or_commit(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    oldest = patient_session(auth_env, "PT-1001")
    pending = prepared(auth_env, oldest, book_body(slot_id))
    principal = principal_of(auth_env, oldest)
    for _ in range(3):  # The fourth live session revokes the oldest.
        auth_env.clock.advance(seconds=1)
        newest = patient_session(auth_env, "PT-1001")

    assert status_of(auth_env.engine, pending) == "INVALIDATED"
    with pytest.raises(AuthenticationRequiredError):
        discard(auth_env, oldest, pending, principal=principal)
    assert confirm(auth_env, oldest, pending).status_code == 401
    assert confirm(auth_env, newest, pending).status_code == 404  # Not the newest session's.
    assert appointments_on(auth_env.engine, slot_id) == 0
