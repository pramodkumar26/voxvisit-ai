"""The agent tool layer against PostgreSQL: the four tools, authorization, preparation with no side
effects, runtime-owned idempotency, confirmation-bypass attempts, and sanitized tracing."""

import json
import logging
import uuid
from datetime import time, timedelta

import pytest
from action_support import (
    NEAR_CLOCK,
    changed_tables,
    new_patient,
    patient_id_of,
    patient_session,
    principal_of,
    rows,
    scalar,
    seed_appointment,
    snapshot,
    staff_session,
    world,
)
from auth_support import call
from scheduling_support import MONDAY, half_hours, new_clinic, new_doctor
from sqlalchemy import create_engine, text

from app.agent_tools import ToolRuntime, build_default_registry
from app.agent_tools import tools as tool_module
from app.agent_tools.tracing import InMemoryTraceSink, LoggingTraceSink
from app.core.errors import DependencyUnavailableError
from app.core.logging import JsonFormatter
from app.services.pending_actions import prepare_appointment_action

FOUR = (
    "search_clinic_directory",
    "search_appointment_availability",
    "get_my_appointments",
    "prepare_appointment_action",
)


def token() -> str:
    return uuid.uuid4().hex[:8]


def agent(env, **options) -> tuple[ToolRuntime, InMemoryTraceSink]:
    sink = InMemoryTraceSink()
    return (
        ToolRuntime(
            env.app.state.db_engine,
            clock=env.clock,
            trace_sink=sink,
            sleep=lambda seconds: None,
            **options,
        ),
        sink,
    )


def use(env, tool: str, arguments: dict | None, session=None, *, runtime=None, call_=None) -> dict:
    """Invoke a tool as the agent would, and return the wire form of the envelope."""
    runtime = runtime or agent(env)[0]
    who = principal_of(env, session) if session else None
    response = runtime.invoke(tool, arguments, principal=who, call=call_).to_dict()
    json.dumps(response)  # The envelope is always serializable.
    assert set(response) == {"ok", "request_id", "data", "error"}
    return response


def book(slot_id) -> dict:
    return {"action": "BOOK", "slot_id": str(slot_id)}


def window(clinic_id, day=MONDAY, days=0) -> dict:
    return {
        "clinic_id": str(clinic_id),
        "start_date": day.isoformat(),
        "end_date": (day + timedelta(days=days)).isoformat(),
    }


# --- search_clinic_directory ------------------------------------------------------------------


def test_directory_search_returns_the_approved_envelope(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)

    response = use(auth_env, "search_clinic_directory", {"clinic_name": scenario.clinic.name})

    assert response["ok"] is True and response["error"] is None
    assert uuid.UUID(response["request_id"])
    [clinic] = response["data"]["clinics"]
    assert clinic["clinic_id"] == str(scenario.clinic.clinic_id)
    assert set(response["data"]) == {"clinics", "doctors", "next_cursor"}


def test_a_search_that_finds_nothing_is_no_results(auth_env) -> None:
    response = use(auth_env, "search_clinic_directory", {"clinic_name": f"Nowhere{token()}"})

    assert response["ok"] is False and response["data"] is None
    assert response["error"] == {
        "code": "NO_RESULTS",
        "message": "No matching clinic or doctor was found.",
        "retryable": False,
    }


def test_a_name_matching_several_clinics_is_ambiguous_with_a_small_list_of_choices(
    auth_env,
) -> None:
    stem = f"Zeta{token()}"
    north = new_clinic(auth_env.engine, name=f"{stem} North")
    south = new_clinic(auth_env.engine, name=f"{stem} South")

    response = use(auth_env, "search_clinic_directory", {"clinic_name": stem})

    assert response["ok"] is False and response["error"]["code"] == "AMBIGUOUS_MATCH"
    choices = response["error"]["choices"]
    assert {c["clinic_id"] for c in choices} == {str(north.clinic_id), str(south.clinic_id)}
    assert set(choices[0]) == {"clinic_id", "name", "city"}
    assert response["data"] is None


def test_choices_are_capped_at_five(auth_env) -> None:
    stem = f"Many{token()}"
    for index in range(8):
        new_clinic(auth_env.engine, name=f"{stem} {index}")

    response = use(auth_env, "search_clinic_directory", {"clinic_name": stem})

    assert response["error"]["code"] == "AMBIGUOUS_MATCH" and len(response["error"]["choices"]) == 5


def test_an_exact_name_is_not_ambiguous_even_if_a_longer_name_also_matches(auth_env) -> None:
    stem = f"Orbit{token()}"
    exact = new_clinic(auth_env.engine, name=stem)
    new_clinic(auth_env.engine, name=f"{stem} Annex")

    response = use(auth_env, "search_clinic_directory", {"clinic_name": stem})

    assert response["ok"] is True
    assert str(exact.clinic_id) in {c["clinic_id"] for c in response["data"]["clinics"]}


def test_a_doctor_name_matching_several_doctors_is_ambiguous(auth_env) -> None:
    stem = f"Quill{token()}"
    clinic = new_clinic(auth_env.engine)
    first = new_doctor(auth_env.engine, clinic, name=f"Dr. {stem} Adams", specialty="Cardiology")
    second = new_doctor(auth_env.engine, clinic, name=f"Dr. {stem} Baker", specialty="Neurology")

    ambiguous = use(auth_env, "search_clinic_directory", {"doctor_name": stem})
    exact = use(auth_env, "search_clinic_directory", {"doctor_name": f"{stem} Adams"})

    assert ambiguous["error"]["code"] == "AMBIGUOUS_MATCH"
    assert {c["doctor_id"] for c in ambiguous["error"]["choices"]} == {
        str(first.doctor_id),
        str(second.doctor_id),
    }
    assert set(ambiguous["error"]["choices"][0]) == {"doctor_id", "display_name", "specialty"}
    assert exact["ok"] is True and exact["data"]["doctors"][0]["doctor_id"] == str(first.doctor_id)


def test_a_specialty_search_with_many_matches_is_a_normal_result_not_ambiguity(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    specialty = f"Spec{token()}"
    for _ in range(3):
        new_doctor(auth_env.engine, clinic, specialty=specialty)

    response = use(auth_env, "search_clinic_directory", {"specialty": specialty})

    assert response["ok"] is True and len(response["data"]["doctors"]) == 3


def test_a_directory_search_needs_a_filter_or_browse(auth_env) -> None:
    response = use(auth_env, "search_clinic_directory", {})

    assert response["error"]["code"] == "INVALID_ARGUMENT"
    assert use(auth_env, "search_clinic_directory", {"browse": True})["ok"] is True


def test_directory_pages_are_capped_and_continue_with_the_cursor(auth_env) -> None:
    stem = f"Page{token()}"
    for index in range(23):
        new_clinic(auth_env.engine, name=f"{stem} {index:02d}")

    # Browsing is not a name search, so a long listing is a normal result.
    first = use(auth_env, "search_clinic_directory", {"browse": True})
    assert first["ok"] and len(first["data"]["clinics"]) + len(first["data"]["doctors"]) == 20
    second = use(
        auth_env,
        "search_clinic_directory",
        {"browse": True, "cursor": first["data"]["next_cursor"]},
    )
    assert second["ok"] and second["data"]["clinics"] != first["data"]["clinics"]


# --- search_appointment_availability ----------------------------------------------------------


def test_availability_matches_the_rest_api_and_uses_the_envelope(auth_env) -> None:
    scenario = world(auth_env.engine, count=4)
    expected = call(
        auth_env, "GET", "/availability", params=window(scenario.clinic.clinic_id)
    ).json()

    response = use(auth_env, "search_appointment_availability", window(scenario.clinic.clinic_id))

    assert response["ok"] is True
    assert response["data"] == expected
    assert [s["slot_id"] for s in response["data"]["slots"]] == [str(s) for s in scenario.slots]
    slot = response["data"]["slots"][0]
    assert slot["starts_at_utc"].endswith("Z") and slot["starts_at_local"].endswith("-07:00")


def test_availability_pages_hold_at_most_twenty_slots(auth_env) -> None:
    scenario = world(auth_env.engine, [(MONDAY, t) for _, t in half_hours(MONDAY, time(0), 25)])
    arguments = window(scenario.clinic.clinic_id)

    first = use(auth_env, "search_appointment_availability", arguments)
    second = use(
        auth_env,
        "search_appointment_availability",
        {**arguments, "cursor": first["data"]["next_cursor"]},
    )

    assert len(first["data"]["slots"]) == 20 and first["data"]["next_cursor"]
    assert len(second["data"]["slots"]) == 5 and second["data"]["next_cursor"] is None
    ids = [s["slot_id"] for s in first["data"]["slots"] + second["data"]["slots"]]
    assert ids == [str(s) for s in scenario.slots]


def test_the_agent_cannot_choose_the_page_size(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)

    response = use(
        auth_env,
        "search_appointment_availability",
        {**window(scenario.clinic.clinic_id), "limit": 500},
    )

    assert response["error"]["code"] == "INVALID_ARGUMENT" and response["error"]["fields"] == [
        "limit"
    ]


def test_no_matching_times_is_no_availability(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)

    response = use(
        auth_env,
        "search_appointment_availability",
        window(scenario.clinic.clinic_id, MONDAY + timedelta(days=2)),
    )

    assert response["error"] == {
        "code": "NO_AVAILABILITY",
        "message": "No matching times are currently available for that date range.",
        "retryable": False,
    }


def test_availability_validation_and_unknown_clinics_are_sanitized(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    clinic = scenario.clinic.clinic_id

    too_long = use(auth_env, "search_appointment_availability", window(clinic, days=40))
    backwards = use(
        auth_env,
        "search_appointment_availability",
        {**window(clinic), "end_date": (MONDAY - timedelta(days=1)).isoformat()},
    )
    unknown = use(auth_env, "search_appointment_availability", window(uuid.uuid4()))
    forged = use(
        auth_env, "search_appointment_availability", {**window(clinic), "cursor": "not-a-cursor"}
    )

    assert too_long["error"]["code"] == backwards["error"]["code"] == "INVALID_ARGUMENT"
    assert unknown["error"]["code"] == "RESOURCE_NOT_AVAILABLE"
    assert forged["error"]["code"] == "INVALID_ARGUMENT"


def test_availability_hides_everything_that_is_not_bookable(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE slots SET status = 'BOOKED' WHERE slot_id = :id"),
            {"id": scenario.slots[0]},
        )
        connection.execute(
            text("UPDATE slots SET removed_at = now() WHERE slot_id = :id"),
            {"id": scenario.slots[1]},
        )

    response = use(auth_env, "search_appointment_availability", window(scenario.clinic.clinic_id))

    assert [s["slot_id"] for s in response["data"]["slots"]] == [str(scenario.slots[2])]


# --- get_my_appointments ----------------------------------------------------------------------


def test_a_patient_sees_only_their_own_appointments(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    mine, theirs = new_patient(auth_env), new_patient(auth_env)
    own_id = seed_appointment(auth_env.engine, mine, scenario.slots[0])
    other_id = seed_appointment(auth_env.engine, theirs, scenario.slots[1])

    response = use(auth_env, "get_my_appointments", {}, patient_session(auth_env, mine))

    assert response["ok"] is True
    assert [a["appointment_id"] for a in response["data"]["appointments"]] == [str(own_id)]
    assert str(other_id) not in json.dumps(response)


def test_an_empty_appointment_list_is_a_successful_result(auth_env) -> None:
    session = patient_session(auth_env, new_patient(auth_env))

    response = use(auth_env, "get_my_appointments", {}, session)

    assert response["ok"] is True and response["data"]["appointments"] == []


def test_appointment_filters_work_and_patient_identity_cannot_be_supplied(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    code = new_patient(auth_env)
    seed_appointment(auth_env.engine, code, scenario.slots[0])
    session = patient_session(auth_env, code)

    cancelled = use(auth_env, "get_my_appointments", {"status": "CANCELLED"}, session)
    spoofed = use(
        auth_env,
        "get_my_appointments",
        {"patient_id": str(patient_id_of(auth_env.engine, "PT-1001"))},
        session,
    )

    assert cancelled["ok"] and cancelled["data"]["appointments"] == []
    assert spoofed["error"]["code"] == "INVALID_ARGUMENT" and spoofed["error"]["fields"] == [
        "patient_id"
    ]


def test_appointment_cursors_are_bound_to_the_patient(auth_env) -> None:
    scenario = world(auth_env.engine, [(MONDAY, t) for _, t in half_hours(MONDAY, time(0), 21)])
    mine = new_patient(auth_env)
    for slot in scenario.slots:
        seed_appointment(auth_env.engine, mine, slot)
    mine_session = patient_session(auth_env, mine)
    other_session = patient_session(auth_env, new_patient(auth_env))

    first = use(auth_env, "get_my_appointments", {}, mine_session)
    cursor = first["data"]["next_cursor"]
    second = use(auth_env, "get_my_appointments", {"cursor": cursor}, mine_session)
    stolen = use(auth_env, "get_my_appointments", {"cursor": cursor}, other_session)

    assert len(first["data"]["appointments"]) == 20 and len(second["data"]["appointments"]) == 1
    assert stolen["error"]["code"] == "INVALID_ARGUMENT"


# --- authorization ----------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["get_my_appointments", "prepare_appointment_action"])
def test_public_callers_cannot_use_patient_tools(auth_env, tool: str) -> None:
    scenario = world(auth_env.engine, count=1)
    before = snapshot(auth_env.engine)

    response = use(auth_env, tool, book(scenario.slots[0]) if tool.startswith("prepare") else {})

    assert response["error"]["code"] == "AUTH_REQUIRED" and response["data"] is None
    assert snapshot(auth_env.engine) == before


@pytest.mark.parametrize("tool", FOUR)
def test_staff_sessions_are_refused_by_every_tool(auth_env, tool: str) -> None:
    scenario = world(auth_env.engine, count=1)
    arguments = {
        "search_clinic_directory": {"browse": True},
        "search_appointment_availability": window(scenario.clinic.clinic_id),
        "get_my_appointments": {},
        "prepare_appointment_action": book(scenario.slots[0]),
    }[tool]
    before = snapshot(auth_env.engine)

    response = use(auth_env, tool, arguments, staff_session(auth_env))

    assert response["error"]["code"] == "FORBIDDEN" and response["error"]["retryable"] is False
    assert snapshot(auth_env.engine)["pending"] == before["pending"]


def test_public_tools_work_without_a_session_and_for_patients(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))

    for who in (None, session):
        assert use(
            auth_env, "search_appointment_availability", window(scenario.clinic.clinic_id), who
        )["ok"]


def test_an_expired_session_is_refused_by_patient_tools_but_not_public_ones(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    runtime, _ = agent(auth_env)
    stale = principal_of(auth_env, session)
    auth_env.clock.advance(minutes=20)

    mine = runtime.invoke("get_my_appointments", {}, principal=stale).to_dict()
    prepare = runtime.invoke(
        "prepare_appointment_action", book(scenario.slots[0]), principal=stale
    ).to_dict()
    public = runtime.invoke("search_clinic_directory", {"browse": True}, principal=stale).to_dict()

    assert mine["error"]["code"] == prepare["error"]["code"] == "SESSION_EXPIRED"
    assert public["ok"] is True


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE auth_sessions SET status = 'REVOKED', terminated_at = now(),"
        " termination_reason = 'TEST' WHERE session_id = :id",
        "UPDATE auth_sessions SET status = 'LOGGED_OUT', terminated_at = now(),"
        " termination_reason = 'LOGOUT' WHERE session_id = :id",
    ],
)
def test_a_session_ended_after_authentication_is_refused(auth_env, change) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    runtime, _ = agent(auth_env)
    who = principal_of(auth_env, session)
    with auth_env.engine.begin() as connection:
        connection.execute(text(change), {"id": who.session_id})
    before = snapshot(auth_env.engine)

    for tool, arguments in (
        ("get_my_appointments", {}),
        ("prepare_appointment_action", book(scenario.slots[0])),
    ):
        assert runtime.invoke(tool, arguments, principal=who).error.code == "AUTH_REQUIRED"
    assert snapshot(auth_env.engine) == before


def test_a_deactivated_account_is_refused(auth_env) -> None:
    code = new_patient(auth_env)
    session = patient_session(auth_env, code)
    runtime, _ = agent(auth_env)
    who = principal_of(auth_env, session)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE patients SET active = false WHERE patient_code = :c"), {"c": code}
        )

    assert runtime.invoke("get_my_appointments", {}, principal=who).error.code == "AUTH_REQUIRED"


# --- prepare_appointment_action ---------------------------------------------------------------


def test_preparing_a_booking_returns_a_pending_action_and_changes_nothing(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    before = snapshot(auth_env.engine)

    response = use(auth_env, "prepare_appointment_action", book(scenario.slots[0]), session)

    assert response["ok"] is True
    data = response["data"]
    assert data["action"] == "BOOK" and data["ui_status"] == "AWAITING_UI_CONFIRMATION"
    assert data["confirmation_required"] is True and data["stored_status"] == "ACTIVE"
    assert data["summary"]["proposed"]["clinic"] == scenario.clinic.name
    assert "version" not in json.dumps(data) and "expected" not in json.dumps(data)
    after = snapshot(auth_env.engine)
    assert changed_tables(before, after) == {"pending", "idempotency", "audit"}
    assert after["slots"] == before["slots"] and after["appointments"] == before["appointments"]


def test_reschedule_and_cancel_are_prepared_for_the_owner_only(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    owner, other = new_patient(auth_env), new_patient(auth_env)
    appointment_id = seed_appointment(auth_env.engine, owner, scenario.slots[0])
    mine, theirs = patient_session(auth_env, owner), patient_session(auth_env, other)
    move = {
        "action": "RESCHEDULE",
        "appointment_id": str(appointment_id),
        "new_slot_id": str(scenario.slots[2]),
    }
    cancel = {"action": "CANCEL", "appointment_id": str(appointment_id)}

    assert (
        use(auth_env, "prepare_appointment_action", move, mine)["data"]["summary"]["operation"]
        == "RESCHEDULE"
    )
    assert (
        use(auth_env, "prepare_appointment_action", cancel, mine)["data"]["summary"]["operation"]
        == "CANCEL"
    )
    for arguments in (move, cancel):
        denied = use(auth_env, "prepare_appointment_action", arguments, theirs)
        assert denied["error"]["code"] == "RESOURCE_NOT_AVAILABLE"
        assert str(appointment_id) not in json.dumps(denied)


def test_backend_refusals_arrive_as_approved_codes(auth_env) -> None:
    taken = world(auth_env.engine, count=1)
    seed_appointment(auth_env.engine, "PT-1002", taken.slots[0])
    started = world(auth_env.engine, [(NEAR_CLOCK, time(8, 10))])
    code = new_patient(auth_env)
    own = seed_appointment(auth_env.engine, code, started.slots[0])
    session = patient_session(auth_env, code)
    runtime, _ = agent(auth_env)
    who = principal_of(auth_env, session)
    auth_env.clock.advance(minutes=10)  # The appointment starts now.

    unavailable = runtime.invoke("prepare_appointment_action", book(taken.slots[0]), principal=who)
    cutoff = runtime.invoke(
        "prepare_appointment_action",
        {"action": "CANCEL", "appointment_id": str(own)},
        principal=who,
    )
    missing = runtime.invoke("prepare_appointment_action", book(uuid.uuid4()), principal=who)

    assert unavailable.error.code == "SLOT_UNAVAILABLE"
    assert cutoff.error.code == "SELF_SERVICE_CUTOFF"
    assert missing.error.code == "RESOURCE_NOT_AVAILABLE"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"action": "BOOK"},
        {"action": "TELEPORT", "slot_id": str(uuid.uuid4())},
        {"action": "BOOK", "slot_id": "not-a-uuid"},
        {"action": "BOOK", "slot_id": str(uuid.uuid4()), "patient_id": str(uuid.uuid4())},
        {"action": "CANCEL", "appointment_id": str(uuid.uuid4()), "new_slot_id": str(uuid.uuid4())},
    ],
)
def test_malformed_preparation_arguments_are_rejected_without_side_effects(
    auth_env, arguments
) -> None:
    session = patient_session(auth_env, new_patient(auth_env))
    before = snapshot(auth_env.engine)

    response = use(auth_env, "prepare_appointment_action", arguments, session)

    assert response["error"]["code"] == "INVALID_ARGUMENT" and response["error"]["fields"]
    assert snapshot(auth_env.engine) == before


# --- runtime-owned idempotency ----------------------------------------------------------------


def prepare_keys(env) -> list[str]:
    return [
        key
        for (key,) in rows(
            env.engine,
            "SELECT idempotency_key FROM idempotency_records"
            " WHERE operation = 'appointment_action.prepare' ORDER BY created_at",
        )
    ]


def test_each_decision_gets_its_own_key_and_the_model_never_supplies_one(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    existing = set(prepare_keys(auth_env))

    first = use(auth_env, "prepare_appointment_action", book(scenario.slots[0]), session)
    second = use(auth_env, "prepare_appointment_action", book(scenario.slots[0]), session)
    supplied = use(
        auth_env,
        "prepare_appointment_action",
        {**book(scenario.slots[0]), "idempotency_key": "model-chosen-key-123456"},
        session,
    )

    assert first["data"]["pending_action_id"] != second["data"]["pending_action_id"]
    assert supplied["error"]["fields"] == ["idempotency_key"]
    new_keys = [key for key in prepare_keys(auth_env) if key not in existing]
    assert len(new_keys) == 2 and len(set(new_keys)) == 2
    assert all(key.startswith("tool.") for key in new_keys)
    assert "model-chosen-key-123456" not in prepare_keys(auth_env)


def test_retrying_the_same_call_returns_the_original_pending_action(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    runtime, _ = agent(auth_env)
    who = principal_of(auth_env, session)
    call_ = runtime.new_call()
    before = snapshot(auth_env.engine)

    first = runtime.invoke(
        "prepare_appointment_action", book(scenario.slots[0]), principal=who, call=call_
    )
    again = runtime.invoke(
        "prepare_appointment_action", book(scenario.slots[0]), principal=who, call=call_
    )

    assert first.ok and again.ok and first.data == again.data
    after = snapshot(auth_env.engine)
    assert len(after["pending"]) == len(before["pending"]) + 1
    assert len(after["idempotency"]) == len(before["idempotency"]) + 1


def test_a_lost_response_after_commit_is_resolved_by_the_same_key(auth_env, monkeypatch) -> None:
    """The first attempt commits, then reports a dependency failure. The runtime's retry reuses
    its key and gets the stored result, so exactly one pending action exists."""
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    keys = []
    attempts = []

    def commit_then_lose_the_answer(*args, **kwargs):
        keys.append(kwargs["idempotency_key"])
        result = prepare_appointment_action(*args, **kwargs)
        attempts.append(1)
        if len(attempts) == 1:
            raise DependencyUnavailableError()
        return result

    monkeypatch.setattr(tool_module, "prepare_appointment_action", commit_then_lose_the_answer)
    runtime, sink = agent(auth_env)
    before = snapshot(auth_env.engine)

    response = runtime.invoke(
        "prepare_appointment_action",
        book(scenario.slots[0]),
        principal=principal_of(auth_env, session),
    )

    assert response.ok and len(keys) == 2 and keys[0] == keys[1]
    assert sink.traces[-1].attempts == 2
    assert len(snapshot(auth_env.engine)["pending"]) == len(before["pending"]) + 1


def test_a_retried_call_cannot_change_what_it_asks_for(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    session = patient_session(auth_env, new_patient(auth_env))
    runtime, _ = agent(auth_env)
    who = principal_of(auth_env, session)
    call_ = runtime.new_call()
    assert runtime.invoke(
        "prepare_appointment_action", book(scenario.slots[0]), principal=who, call=call_
    ).ok
    before = snapshot(auth_env.engine)

    changed = runtime.invoke(
        "prepare_appointment_action", book(scenario.slots[1]), principal=who, call=call_
    )

    assert changed.error.code == "INVALID_ARGUMENT" and changed.error.reason == "CALL_REUSED"
    assert snapshot(auth_env.engine) == before


# --- the agent cannot confirm or commit -------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "confirm_appointment_action",
        "confirm_appointment",
        "confirm",
        "commit_appointment_action",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
        "execute_pending_action",
        "approve_pending_action",
    ],
)
def test_confirmation_and_commit_tools_do_not_exist_for_the_agent(auth_env, tool: str) -> None:
    scenario = world(auth_env.engine, count=1)
    code = new_patient(auth_env)
    session = patient_session(auth_env, code)
    pending = use(auth_env, "prepare_appointment_action", book(scenario.slots[0]), session)
    before = snapshot(auth_env.engine)

    response = use(
        auth_env,
        tool,
        {"pending_action_id": pending["data"]["pending_action_id"], "confirmed": True},
        session,
    )

    assert response["error"]["code"] == "INVALID_ARGUMENT"
    assert response["error"]["reason"] == "UNKNOWN_TOOL"
    assert snapshot(auth_env.engine) == before  # Nothing was committed or consumed.


@pytest.mark.parametrize(
    "smuggled",
    [
        {"confirmed": True},
        {"confirm": True},
        {"pending_action_id": str(uuid.uuid4())},
        {"idempotency_key": "tool.abcdefabcdefabcdef"},
        {"csrf_token": "x"},
        {"session_token": "x"},
        {"patient_id": str(uuid.uuid4())},
    ],
)
def test_confirmation_cannot_be_smuggled_into_any_tool_call(auth_env, smuggled) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    before = snapshot(auth_env.engine)
    base = {
        "search_clinic_directory": {"browse": True},
        "search_appointment_availability": window(scenario.clinic.clinic_id),
        "get_my_appointments": {},
        "prepare_appointment_action": book(scenario.slots[0]),
    }

    for tool, arguments in base.items():
        response = use(auth_env, tool, {**arguments, **smuggled}, session)
        assert response["error"]["code"] == "INVALID_ARGUMENT", tool
        assert set(response["error"]["fields"]) == set(smuggled), tool
    assert snapshot(auth_env.engine) == before


def test_a_prepared_action_stays_unconfirmed_however_the_tools_are_used(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    code = new_patient(auth_env)
    session = patient_session(auth_env, code)
    pending = use(auth_env, "prepare_appointment_action", book(scenario.slots[0]), session)
    pending_id = pending["data"]["pending_action_id"]

    def pending_status() -> str:
        return scalar(
            auth_env.engine,
            "SELECT status FROM pending_actions WHERE pending_action_id = :i",
            i=pending_id,
        )

    for tool, arguments in (
        ("get_my_appointments", {}),
        ("search_appointment_availability", window(scenario.clinic.clinic_id)),
    ):
        use(auth_env, tool, arguments, session)
    assert pending_status() == "ACTIVE"  # Reading changes nothing about a review.

    for slot in scenario.slots:  # Preparing again retires the earlier review; it never commits it.
        use(auth_env, "prepare_appointment_action", book(slot), session)
    assert pending_status() == "INVALIDATED"
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM appointments a JOIN patients p USING (patient_id)"
            " WHERE p.patient_code = :c",
            c=code,
        )
        == 0
    )
    assert (
        scalar(auth_env.engine, "SELECT status FROM slots WHERE slot_id = :i", i=scenario.slots[0])
        == "AVAILABLE"
    )


def test_the_registered_handlers_cannot_reach_the_commit_service() -> None:
    assert not hasattr(tool_module, "confirm_appointment_action")
    assert "appointment_transactions" not in vars(tool_module)
    registry = build_default_registry()
    assert registry.names() == set(FOUR)
    for definition in registry.definitions():
        assert "confirm" not in definition["name"]


# --- sanitized errors and tracing -------------------------------------------------------------


def test_traces_record_each_call_without_arguments_or_results(auth_env) -> None:
    sentinel = f"Sentinel{token()}"
    scenario = world(auth_env.engine, count=1)
    code = new_patient(auth_env, f"{sentinel} Patient")
    session = patient_session(auth_env, code)
    runtime, sink = agent(auth_env)
    who = principal_of(auth_env, session)

    runtime.invoke("search_clinic_directory", {"clinic_name": sentinel}, principal=who)
    runtime.invoke(
        "search_appointment_availability", window(scenario.clinic.clinic_id), principal=who
    )
    runtime.invoke("prepare_appointment_action", book(scenario.slots[0]), principal=who)
    runtime.invoke("get_my_appointments", {"patient_id": str(who.patient_id)}, principal=who)
    runtime.invoke(f"leak-{sentinel}", {"session_token": who.csrf_token}, principal=who)

    assert [(t.tool, t.outcome, t.error_category) for t in sink.traces] == [
        ("search_clinic_directory", "error", "NO_RESULTS"),
        ("search_appointment_availability", "ok", None),
        ("prepare_appointment_action", "ok", None),
        ("get_my_appointments", "error", "INVALID_ARGUMENT"),
        ("unknown", "error", "INVALID_ARGUMENT"),
    ]
    assert all(t.duration_ms >= 0 for t in sink.traces)
    # Calls rejected before a handler ran (invalid arguments, unknown tool) made no attempt.
    assert [t.attempts for t in sink.traces] == [1, 1, 1, 0, 0]
    assert len({t.request_id for t in sink.traces}) == 5
    dumped = repr(sink.traces)
    for private in (sentinel, str(who.patient_id), who.csrf_token, code, str(scenario.slots[0])):
        assert private not in dumped


def test_logged_traces_are_sanitized_end_to_end(auth_env, caplog) -> None:
    sentinel = f"Sentinel{token()}"
    code = new_patient(auth_env, f"{sentinel} Patient")
    session = patient_session(auth_env, code)
    runtime = ToolRuntime(
        auth_env.app.state.db_engine, clock=auth_env.clock, trace_sink=LoggingTraceSink()
    )
    who = principal_of(auth_env, session)

    with caplog.at_level(logging.INFO, logger="app.agent_tools"):
        runtime.invoke("get_my_appointments", {}, principal=who)
        runtime.invoke("search_clinic_directory", {"clinic_name": sentinel}, principal=who)

    lines = [
        JsonFormatter().format(record)
        for record in caplog.records
        if record.name == "app.agent_tools"
    ]
    assert len(lines) == 2
    combined = "\n".join(lines)
    for private in (sentinel, str(who.patient_id), who.csrf_token, code, auth_env.pin):
        assert private not in combined
    assert json.loads(lines[1])["error_category"] == "NO_RESULTS"


def test_unexpected_failures_are_generic_and_leak_nothing(auth_env, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("postgresql://user:hunter2@host/db SELECT * FROM patient_credentials")

    monkeypatch.setattr(tool_module.directory, "search_directory", explode)
    runtime, sink = agent(auth_env)

    response = runtime.invoke("search_clinic_directory", {"browse": True}, principal=None).to_dict()

    assert response["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "The request could not be completed.",
        "retryable": False,
    }
    for leaked in ("hunter2", "SELECT", "patient_credentials", "postgresql://"):
        assert leaked not in json.dumps(response) + repr(sink.traces)
    assert sink.traces[-1].error_category == "INTERNAL_ERROR"


def test_an_unreachable_database_is_a_retryable_failure_after_one_retry(auth_env) -> None:
    dead = create_engine(
        "postgresql+psycopg://nobody@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    sink = InMemoryTraceSink()
    runtime = ToolRuntime(dead, clock=auth_env.clock, trace_sink=sink, sleep=lambda seconds: None)

    response = runtime.invoke("search_clinic_directory", {"browse": True}, principal=None).to_dict()

    assert response["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert response["error"]["retryable"] is True
    assert sink.traces[-1].attempts == 2
    for leaked in ("127.0.0.1", "nobody", "psycopg", "connect"):
        assert leaked not in json.dumps(response)
    dead.dispose()


def test_tools_leave_the_database_unchanged_except_for_preparation(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    code = new_patient(auth_env)
    seed_appointment(auth_env.engine, code, scenario.slots[0])
    session = patient_session(auth_env, code)
    before = snapshot(auth_env.engine)

    for tool, arguments in (
        ("search_clinic_directory", {"browse": True}),
        ("search_appointment_availability", window(scenario.clinic.clinic_id)),
        ("get_my_appointments", {}),
        ("get_my_appointments", {"status": "CANCELLED"}),
    ):
        use(auth_env, tool, arguments, session)

    assert snapshot(auth_env.engine) == before


def test_the_envelope_request_id_is_the_one_stored_with_the_pending_action(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    runtime, sink = agent(auth_env)

    response = runtime.invoke(
        "prepare_appointment_action",
        book(scenario.slots[0]),
        principal=principal_of(auth_env, session),
    )

    stored = scalar(
        auth_env.engine,
        "SELECT request_id FROM pending_actions WHERE pending_action_id = :id",
        id=response.data["pending_action_id"],
    )
    assert stored == response.request_id == sink.traces[-1].request_id
