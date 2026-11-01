import ast
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest
from action_support import (
    NEAR_CLOCK,
    book_body,
    cancel_body,
    changed_tables,
    confirm,
    error_of,
    patient_session,
    prepare,
    reschedule_body,
    rows,
    scalar,
    seed_appointment,
    snapshot,
    staff_session,
    world,
)
from auth_support import call
from scheduling_support import (
    MONDAY,
    add_holiday,
    add_leave,
    add_slots,
    assign_doctor,
    new_clinic,
    new_doctor,
)
from sqlalchemy import text

APP_DIR = Path(__file__).parents[2] / "app"


# --- preparation changes nothing but the pending action ------------------------------------


def test_preparing_a_booking_returns_an_exact_summary_and_reserves_nothing(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    slot_id = scenario.slots[0]
    session = patient_session(auth_env)
    before = snapshot(auth_env.engine)

    response = prepare(auth_env, session, book_body(slot_id))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action"] == "BOOK"
    assert body["stored_status"] == "ACTIVE"
    assert body["ui_status"] == "AWAITING_UI_CONFIRMATION"
    assert body["confirmation_required"] is True
    summary = body["summary"]
    assert summary["operation"] == "BOOK"
    assert summary["current"] is None
    proposed = summary["proposed"]
    assert proposed["clinic"] == scenario.clinic.name
    assert proposed["doctor"] == scenario.doctor.display_name
    assert (proposed["date"], proposed["time"], proposed["timezone"]) == (
        MONDAY.isoformat(),
        "9:00 AM",
        "America/Denver",
    )
    assert proposed["starts_at_local"] == "2026-12-07T09:00:00-07:00"
    assert proposed["starts_at_utc"] == "2026-12-07T16:00:00Z"
    assert summary["description"] == (
        f"Book an appointment with {scenario.doctor.display_name} at {scenario.clinic.name}"
        " on 2026-12-07 at 9:00 AM (America/Denver)"
    )

    after = snapshot(auth_env.engine)
    assert changed_tables(before, after) == {"pending", "idempotency", "audit"}
    assert after["slots"] == before["slots"]
    assert after["appointments"] == before["appointments"]
    assert after["history"] == before["history"]
    assert after["outbox"] == before["outbox"]


def test_expected_versions_stay_server_side(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    response = prepare(auth_env, patient_session(auth_env), book_body(slot_id))

    assert "version" not in response.text and "expected" not in response.text
    stored = rows(
        auth_env.engine,
        "SELECT expected_versions->>'slot_version', target_type FROM pending_actions"
        " WHERE pending_action_id = :id",
        id=response.json()["pending_action_id"],
    )
    assert stored == [("1", "SLOT")]


def test_pending_action_expiry_comes_from_database_time_not_the_application_clock(
    auth_env,
) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    auth_env.clock.advance(minutes=3)  # The application clock is not the database clock.

    response = prepare(auth_env, session, book_body(slot_id))
    body = response.json()

    created = scalar(
        auth_env.engine,
        "SELECT created_at FROM pending_actions WHERE pending_action_id = :id",
        id=body["pending_action_id"],
    )
    database_now = scalar(auth_env.engine, "SELECT clock_timestamp()")
    assert timedelta(0) <= database_now - created < timedelta(seconds=30)
    assert datetime.fromisoformat(body["created_at"]) == created
    assert datetime.fromisoformat(body["expires_at"]) == created + timedelta(minutes=10)
    assert scalar(
        auth_env.engine,
        "SELECT expires_at - created_at FROM pending_actions WHERE pending_action_id = :id",
        id=body["pending_action_id"],
    ) == timedelta(minutes=10)


def test_preparing_a_reschedule_shows_current_and_proposed_and_changes_nothing(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env)
    before = snapshot(auth_env.engine)

    response = prepare(auth_env, session, reschedule_body(appointment_id, scenario.slots[2]))

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["operation"] == "RESCHEDULE"
    assert (summary["current"]["time"], summary["proposed"]["time"]) == ("9:00 AM", "10:00 AM")
    assert "to an appointment with" in summary["description"]
    after = snapshot(auth_env.engine)
    assert changed_tables(before, after) == {"pending", "idempotency", "audit"}


def test_preparing_a_cancellation_shows_the_appointment_and_changes_nothing(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    session = patient_session(auth_env)
    before = snapshot(auth_env.engine)

    response = prepare(auth_env, session, cancel_body(appointment_id))

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["operation"] == "CANCEL" and summary["proposed"] is None
    assert summary["current"]["doctor"] == scenario.doctor.display_name
    assert changed_tables(before, snapshot(auth_env.engine)) == {"pending", "idempotency", "audit"}


def test_summaries_use_clinic_local_time_in_another_timezone(auth_env) -> None:
    scenario = world(auth_env.engine, [(date(2026, 12, 8), time(15, 30))], timezone="Asia/Kolkata")

    response = prepare(auth_env, patient_session(auth_env), book_body(scenario.slots[0]))

    proposed = response.json()["summary"]["proposed"]
    assert (proposed["date"], proposed["time"]) == ("2026-12-08", "3:30 PM")
    assert proposed["starts_at_local"] == "2026-12-08T15:30:00+05:30"
    assert proposed["starts_at_utc"] == "2026-12-08T10:00:00Z"


# --- validation -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spoil",
    ["booked", "held", "removed", "inactive_doctor", "inactive_clinic", "inactive_assignment"],
)
def test_slots_that_are_not_bookable_cannot_be_prepared(auth_env, spoil: str) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    statements = {
        "booked": "UPDATE slots SET status = 'BOOKED' WHERE slot_id = :slot",
        "held": "UPDATE slots SET status = 'HELD' WHERE slot_id = :slot",
        "removed": "UPDATE slots SET removed_at = now() WHERE slot_id = :slot",
        "inactive_doctor": "UPDATE doctors SET active = false WHERE doctor_id = :doctor",
        "inactive_clinic": "UPDATE clinics SET active = false WHERE clinic_id = :clinic",
        "inactive_assignment": "UPDATE doctor_clinics SET active = false WHERE doctor_id = :doctor",
    }
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(statements[spoil]),
            {
                "slot": slot_id,
                "doctor": scenario.doctor.doctor_id,
                "clinic": scenario.clinic.clinic_id,
            },
        )
    before = snapshot(auth_env.engine)

    response = prepare(auth_env, patient_session(auth_env), book_body(slot_id))

    assert response.status_code == 409 and error_of(response)["code"] == "SLOT_UNAVAILABLE"
    assert snapshot(auth_env.engine)["pending"] == before["pending"]


def test_a_past_slot_cannot_be_prepared(auth_env) -> None:
    # 07:00 local on the clock's own day is an hour before the 08:00 start of the test clock.
    slot_id = world(auth_env.engine, [(NEAR_CLOCK, time(7))]).slots[0]

    response = prepare(auth_env, patient_session(auth_env), book_body(slot_id))

    assert error_of(response)["code"] == "SLOT_UNAVAILABLE"


def test_holidays_and_leave_block_preparation_with_the_same_rules_as_search(auth_env) -> None:
    scenario = world(auth_env.engine, count=4)
    add_holiday(auth_env.engine, scenario.clinic, MONDAY, time(9), time(9, 30))  # first slot only
    add_leave(
        auth_env.engine, scenario.clinic, scenario.doctor, (MONDAY, time(9, 30)), (MONDAY, time(10))
    )
    session = patient_session(auth_env)

    codes = {
        index: prepare(auth_env, session, book_body(slot)).status_code
        for index, slot in enumerate(scenario.slots)
    }

    # 09:00 is inside the partial closure, 09:30 inside the leave, 10:00 and 10:30 are open.
    assert codes == {0: 409, 1: 409, 2: 200, 3: 200}


def test_a_whole_day_closure_blocks_preparation(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    add_holiday(auth_env.engine, scenario.clinic, MONDAY)

    response = prepare(auth_env, patient_session(auth_env), book_body(scenario.slots[0]))

    assert error_of(response)["code"] == "SLOT_UNAVAILABLE"


def test_a_slot_outside_operating_hours_cannot_be_prepared(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)  # Open 08:00-17:00 on weekdays.
    doctor = new_doctor(auth_env.engine, clinic, rule_windows=((time(0), time(23, 59)),))
    [early, saturday] = [
        add_slots(auth_env.engine, clinic, doctor, [moment])[0]
        for moment in ((MONDAY, time(6)), (MONDAY + timedelta(days=5), time(10)))
    ]
    session = patient_session(auth_env)

    assert error_of(prepare(auth_env, session, book_body(early)))["code"] == "SLOT_UNAVAILABLE"
    assert error_of(prepare(auth_env, session, book_body(saturday)))["code"] == "SLOT_UNAVAILABLE"


def test_leave_at_one_clinic_does_not_block_the_same_doctor_at_another(auth_env) -> None:
    clinic_a = new_clinic(auth_env.engine, hours=(time(0), time(23, 59, 59)), open_days=range(7))
    clinic_b = new_clinic(auth_env.engine, hours=(time(0), time(23, 59, 59)), open_days=range(7))
    doctor = new_doctor(
        auth_env.engine,
        clinic_a,
        rule_days=range(7),
        rule_windows=((time(0), time(23, 59, 59)),),
    )
    assign_doctor(auth_env.engine, doctor, clinic_b)
    [at_a] = add_slots(auth_env.engine, clinic_a, doctor, [(MONDAY, time(9))])
    [at_b] = add_slots(auth_env.engine, clinic_b, doctor, [(MONDAY, time(10))])
    add_leave(auth_env.engine, clinic_a, doctor, (MONDAY, time(8)), (MONDAY, time(12)))
    session = patient_session(auth_env)

    assert error_of(prepare(auth_env, session, book_body(at_a)))["code"] == "SLOT_UNAVAILABLE"
    assert prepare(auth_env, session, book_body(at_b)).status_code == 200


def test_unknown_slot_and_unknown_appointment_are_reported_as_unavailable(auth_env) -> None:
    session = patient_session(auth_env)

    unknown_slot = prepare(auth_env, session, book_body("00000000-0000-4000-8000-000000000001"))
    unknown_appointment = prepare(
        auth_env, session, cancel_body("00000000-0000-4000-8000-000000000002")
    )

    for response in (unknown_slot, unknown_appointment):
        assert response.status_code == 404
        assert error_of(response)["code"] == "RESOURCE_NOT_AVAILABLE"


def test_another_patients_appointment_looks_exactly_like_a_missing_one(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    others = seed_appointment(auth_env.engine, "PT-1002", scenario.slots[0])
    session = patient_session(auth_env, "PT-1001")

    cancel = prepare(auth_env, session, cancel_body(others))
    move = prepare(auth_env, session, reschedule_body(others, scenario.slots[1]))
    missing = prepare(auth_env, session, cancel_body("00000000-0000-4000-8000-000000000003"))

    for response in (cancel, move):
        assert response.status_code == 404
        assert response.json() == missing.json()
        assert "PT-1002" not in response.text


def test_a_cancelled_appointment_cannot_be_prepared_again(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE appointments SET status = 'CANCELLED', cancelled_at = now()"
                " WHERE appointment_id = :id"
            ),
            {"id": appointment_id},
        )
    session = patient_session(auth_env)

    for body in (cancel_body(appointment_id), reschedule_body(appointment_id, scenario.slots[1])):
        response = prepare(auth_env, session, body)
        assert response.status_code == 409
        assert error_of(response)["code"] == "CONFLICT"
        assert error_of(response)["reason"] == "APPOINTMENT_NOT_ACTIVE"


def test_rescheduling_to_the_current_slot_is_rejected(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])

    response = prepare(
        auth_env, patient_session(auth_env), reschedule_body(appointment_id, scenario.slots[0])
    )

    assert response.status_code == 422 and error_of(response)["code"] == "INVALID_ARGUMENT"


def test_the_self_service_cutoff_blocks_preparation_at_and_after_the_start(auth_env) -> None:
    scenario = world(auth_env.engine, [(NEAR_CLOCK, time(8, 10)), (NEAR_CLOCK, time(9))])
    started, later = scenario.slots
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", started)
    other_id = seed_appointment(auth_env.engine, "PT-1001", later)
    session = patient_session(auth_env)
    assert prepare(auth_env, session, cancel_body(appointment_id)).status_code == 200

    auth_env.clock.advance(minutes=10)  # 08:10 local: exactly the start time.
    at_start = prepare(auth_env, session, cancel_body(appointment_id))
    reschedule = prepare(auth_env, session, reschedule_body(appointment_id, later))

    assert at_start.status_code == 409 and error_of(at_start)["code"] == "SELF_SERVICE_CUTOFF"
    assert error_of(reschedule)["code"] == "SELF_SERVICE_CUTOFF"
    assert prepare(auth_env, session, cancel_body(other_id)).status_code == 200


def test_the_cutoff_compares_instants_not_wall_clock_times(auth_env) -> None:
    # 08:00 in Honolulu is 18:00 UTC, three hours after the 15:00 UTC test clock, even though its
    # local clock time reads "earlier" than 15:00. A wall-clock comparison would call it past.
    scenario = world(
        auth_env.engine, [(NEAR_CLOCK, time(8)), (NEAR_CLOCK, time(4))], timezone="Pacific/Honolulu"
    )
    future, past = scenario.slots  # 04:00 HST is 14:00 UTC, an hour before the clock.
    session = patient_session(auth_env)

    assert prepare(auth_env, session, book_body(future)).status_code == 200
    assert error_of(prepare(auth_env, session, book_body(past)))["code"] == "SLOT_UNAVAILABLE"
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", future)
    assert prepare(auth_env, session, cancel_body(appointment_id)).status_code == 200


# --- request validation and authorization ---------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"action": "BOOK"},
        {"action": "TELEPORT", "slot_id": "00000000-0000-4000-8000-000000000001"},
        {"action": "BOOK", "slot_id": "not-a-uuid"},
        {"action": "CANCEL"},
        {"action": "RESCHEDULE", "appointment_id": "00000000-0000-4000-8000-000000000001"},
        {"action": "BOOK", "slot_id": "00000000-0000-4000-8000-000000000001", "patient_id": "x"},
        {
            "action": "CANCEL",
            "appointment_id": "00000000-0000-4000-8000-000000000001",
            "role": "STAFF",
        },
        {"action": "BOOK", "slot_id": "00000000-0000-4000-8000-000000000001", "version": 1},
        {"action": "BOOK", "slot_id": "00000000-0000-4000-8000-000000000001", "new_slot_id": "x"},
    ],
)
def test_malformed_or_unsupported_fields_are_rejected(auth_env, body) -> None:
    response = prepare(auth_env, patient_session(auth_env), body)

    assert response.status_code == 422 and error_of(response)["code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize(
    "key", ["short", "has spaces in the key!!!", "x" * 129, "slashes/and;semicolons-0123456789"]
)
def test_idempotency_keys_must_be_well_formed(auth_env, key: str) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]

    response = prepare(auth_env, patient_session(auth_env), book_body(slot_id), key=key)

    assert response.status_code == 422 and error_of(response)["code"] == "INVALID_ARGUMENT"


def test_an_idempotency_key_is_required(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)

    response = call(auth_env, "POST", "/me/appointment-actions", session, json=book_body(slot_id))

    assert response.status_code == 422
    assert "Idempotency-Key" in response.json()["error"]["fields"]


def test_anonymous_callers_cannot_prepare(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]

    response = prepare(auth_env, None, book_body(slot_id))

    assert response.status_code == 401 and error_of(response)["code"] == "AUTH_REQUIRED"


def test_staff_cannot_prepare_or_confirm_patient_actions(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    patient = patient_session(auth_env)
    pending_id = prepare(auth_env, patient, book_body(slot_id)).json()["pending_action_id"]
    staff = staff_session(auth_env)
    before = snapshot(auth_env.engine)

    attempts = [prepare(auth_env, staff, book_body(slot_id)), confirm(auth_env, staff, pending_id)]

    for response in attempts:
        assert response.status_code == 403 and error_of(response)["code"] == "FORBIDDEN"
    assert snapshot(auth_env.engine) == before


@pytest.mark.parametrize("fault", ["no_csrf", "wrong_csrf", "no_origin", "foreign_origin"])
def test_mutations_require_csrf_and_an_allowed_origin(auth_env, fault: str) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    options = {
        "no_csrf": {"csrf": None},
        "wrong_csrf": {"csrf": "forged"},
        "no_origin": {"origin": None},
        "foreign_origin": {"origin": "https://evil.example"},
    }[fault]
    before = snapshot(auth_env.engine)

    response = prepare(auth_env, session, book_body(slot_id), **options)

    assert response.status_code == 403 and error_of(response)["code"] == "CSRF_FAILED"
    assert snapshot(auth_env.engine) == before


def test_an_expired_session_cannot_prepare(auth_env) -> None:
    slot_id = world(auth_env.engine, count=1).slots[0]
    session = patient_session(auth_env)
    auth_env.clock.advance(minutes=16)

    response = prepare(auth_env, session, book_body(slot_id))

    assert response.status_code == 401 and error_of(response)["code"] == "SESSION_EXPIRED"


# --- the agent can prepare but has no route to commit ---------------------------------------


def test_confirmation_is_a_separate_ui_only_router_from_preparation(auth_env) -> None:
    paths = auth_env.app.openapi()["paths"]
    tags = {
        path: {tag for operation in item.values() for tag in operation["tags"]}
        for path, item in paths.items()
        if "appointment-actions" in path
    }

    assert tags == {
        "/me/appointment-actions": {"appointment-actions"},
        "/me/appointment-actions/{pending_action_id}/confirm": {"ui-confirmation"},
    }


def _module_names(module: str) -> tuple[set[str], str]:
    source = (APP_DIR / "services" / f"{module}.py").read_text()
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names, source


def test_the_preparation_service_cannot_commit_anything() -> None:
    imported, source = _module_names("pending_actions")

    # No path to the commit service, and no write constructs or write models for the tables a
    # confirmed action changes.
    assert not {"appointment_transactions", "confirm_appointment_action"} & imported
    assert not {"insert", "update", "delete"} & imported
    assert not {"Slot", "AppointmentHistory", "OutboxEvent", "PendingAction"} & imported
    statements = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("INSERT", "UPDATE"))
    ]
    assert statements == ["INSERT INTO pending_actions"]


def test_only_the_confirm_route_imports_the_commit_service() -> None:
    importers = []
    for path in APP_DIR.rglob("*.py"):
        if path.name == "appointment_transactions.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = {alias.name for alias in getattr(node, "names", [])}
            module = getattr(node, "module", None) or ""
            if isinstance(node, ast.ImportFrom | ast.Import) and (
                "appointment_transactions" in names | {module}
                or "confirm_appointment_action" in names
            ):
                importers.append(path.relative_to(APP_DIR).as_posix())

    assert sorted(set(importers)) == ["api/routes/appointment_actions.py"]
