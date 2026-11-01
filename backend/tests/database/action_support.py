"""Helpers for the appointment-action tests: scenarios, requests, state snapshots, and races."""

import threading
import time as clock_time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, time
from types import SimpleNamespace
from typing import Any

from auth_support import call, sign_in
from factories import make
from scheduling_support import (
    ALL_DAY,
    ALL_WEEK,
    MONDAY,
    add_slots,
    half_hours,
    new_clinic,
    new_doctor,
)
from sqlalchemy import Engine, text

from app.db.models import Appointment

PREPARE_PATH = "/me/appointment-actions"
NEAR_CLOCK = date(2026, 11, 2)  # The shared test clock starts at 08:00 local Denver on this day.


def new_key() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex[:8]


def world(
    engine: Engine,
    moments: list[tuple[date, time]] | None = None,
    *,
    count: int = 4,
    timezone: str = "America/Denver",
) -> SimpleNamespace:
    """A fresh clinic and doctor that are open and available all day, every day, with slots."""
    clinic = new_clinic(engine, timezone=timezone, hours=ALL_DAY, open_days=ALL_WEEK)
    doctor = new_doctor(engine, clinic, rule_days=ALL_WEEK, rule_windows=(ALL_DAY,))
    slot_ids = add_slots(engine, clinic, doctor, moments or half_hours(MONDAY, time(9), count))
    return SimpleNamespace(clinic=clinic, doctor=doctor, slots=slot_ids)


def patient_id_of(engine: Engine, code: str) -> uuid.UUID:
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT patient_id FROM patients WHERE patient_code = :code"), {"code": code}
        ).scalar_one()


def patient_session(env, code: str = "PT-1001"):
    return sign_in(env, "patient", code, env.pin)


def staff_session(env):
    return sign_in(env, "staff", "staff.morgan", env.password)


def seed_appointment(engine: Engine, code: str, slot_id: uuid.UUID) -> uuid.UUID:
    """A CONFIRMED appointment created directly in the database (not through the API)."""
    with engine.begin() as connection:
        patient_id = connection.execute(
            text("SELECT patient_id FROM patients WHERE patient_code = :code"), {"code": code}
        ).scalar_one()
        connection.execute(
            text("UPDATE slots SET status = 'BOOKED', version = version + 1 WHERE slot_id = :id"),
            {"id": slot_id},
        )
        appointment = make(
            connection, Appointment, patient_id=patient_id, slot_id=slot_id, status="CONFIRMED"
        )
    return appointment.appointment_id


# --- requests -------------------------------------------------------------------------------


def prepare(env, session, body: dict[str, Any], *, key: str | None = None, **kwargs):
    headers = {"Idempotency-Key": key or new_key(), **kwargs.pop("headers", {})}
    return call(env, "POST", PREPARE_PATH, session, json=body, headers=headers, **kwargs)


def confirm(
    env,
    session,
    pending_action_id: str | uuid.UUID,
    *,
    key: str | None = None,
    body: dict[str, Any] | None = None,
    **kwargs,
):
    headers = {"Idempotency-Key": key or new_key(), **kwargs.pop("headers", {})}
    return call(
        env,
        "POST",
        f"{PREPARE_PATH}/{pending_action_id}/confirm",
        session,
        json={"confirmed": True} if body is None else body,
        headers=headers,
        **kwargs,
    )


def prepared(env, session, body: dict[str, Any]) -> str:
    response = prepare(env, session, body)
    assert response.status_code == 200, response.text
    return response.json()["pending_action_id"]


def book_body(slot_id) -> dict[str, str]:
    return {"action": "BOOK", "slot_id": str(slot_id)}


def reschedule_body(appointment_id, new_slot_id) -> dict[str, str]:
    return {
        "action": "RESCHEDULE",
        "appointment_id": str(appointment_id),
        "new_slot_id": str(new_slot_id),
    }


def cancel_body(appointment_id) -> dict[str, str]:
    return {"action": "CANCEL", "appointment_id": str(appointment_id)}


def error_of(response) -> dict[str, str]:
    return response.json()["error"]


# --- database state -------------------------------------------------------------------------


def snapshot(engine: Engine) -> dict[str, Any]:
    """Every table an appointment action may touch, for exact before/after comparison."""
    queries = {
        "slots": "SELECT slot_id, status, version, removed_at FROM slots ORDER BY slot_id",
        "appointments": (
            "SELECT appointment_id, patient_id, slot_id, status, version, cancelled_at"
            " FROM appointments ORDER BY appointment_id"
        ),
        "history": (
            "SELECT event_id, appointment_id, action FROM appointment_history ORDER BY event_id"
        ),
        "outbox": "SELECT outbox_event_id, dedupe_key FROM outbox_events ORDER BY outbox_event_id",
        "pending": "SELECT pending_action_id, status FROM pending_actions ORDER BY 1",
        "idempotency": (
            "SELECT idempotency_id, operation, status FROM idempotency_records ORDER BY 1"
        ),
        "audit": (
            "SELECT audit_event_id, event_type FROM security_audit_events"
            " WHERE event_type LIKE 'APPOINTMENT_%' AND event_type <> 'APPOINTMENT_ACTION_REJECTED'"
            " ORDER BY 1"
        ),
    }
    with engine.connect() as connection:
        return {
            name: [tuple(row) for row in connection.execute(text(sql)).all()]
            for name, sql in queries.items()
        }


def changed_tables(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {name for name in before if before[name] != after[name]}


def slot_row(engine: Engine, slot_id) -> SimpleNamespace:
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT status, version FROM slots WHERE slot_id = :id"), {"id": slot_id}
        ).one()
    return SimpleNamespace(status=row.status, version=row.version)


def appointment_row(engine: Engine, appointment_id) -> SimpleNamespace:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT patient_id, slot_id, status, version, cancelled_at"
                " FROM appointments WHERE appointment_id = :id"
            ),
            {"id": appointment_id},
        ).one()
    return SimpleNamespace(**row._mapping)


def scalar(engine: Engine, sql: str, **params: Any) -> Any:
    with engine.connect() as connection:
        return connection.execute(text(sql), params).scalar_one()


def rows(engine: Engine, sql: str, **params: Any) -> list[tuple]:
    with engine.connect() as connection:
        return [tuple(row) for row in connection.execute(text(sql), params).all()]


def age_pending_action(engine: Engine, pending_action_id, *, minutes: float) -> None:
    """Make a pending action look `minutes` old using only database timestamps."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pending_actions"
                " SET created_at = clock_timestamp() - make_interval(secs => :seconds),"
                "     expires_at = clock_timestamp() - make_interval(secs => :seconds)"
                "                  + interval '10 minutes'"
                " WHERE pending_action_id = :id"
            ),
            {"id": pending_action_id, "seconds": minutes * 60},
        )


# --- races ----------------------------------------------------------------------------------


@contextmanager
def holding_row_lock(
    engine: Engine, table: str, column: str, value, mode: str = "UPDATE"
) -> Iterator[None]:
    """Hold a row lock, so racing transactions queue behind it deterministically."""
    connection = engine.connect()
    transaction = connection.begin()
    connection.execute(
        text(f"SELECT 1 FROM {table} WHERE {column} = :value FOR {mode}"), {"value": value}
    )
    try:
        yield
    finally:
        transaction.commit()
        connection.close()


def wait_until_blocked(engine: Engine, waiting: int, timeout: float = 15.0) -> None:
    """Block until `waiting` other backends are waiting on a lock."""
    deadline = clock_time.monotonic() + timeout
    while clock_time.monotonic() < deadline:
        blocked = scalar(
            engine,
            "SELECT count(*) FROM pg_stat_activity"
            " WHERE datname = current_database() AND wait_event_type = 'Lock'",
        )
        if blocked >= waiting:
            return
        clock_time.sleep(0.02)
    raise AssertionError(f"expected {waiting} blocked transactions")


def race(
    engine: Engine,
    jobs: list[Callable[[], Any]],
    *,
    lock: Callable[[], Any] | None = None,
    blocked: int | None = None,
) -> list[Any]:
    """Run jobs concurrently. Each result is the return value or the raised exception.

    With a lock factory, the jobs are started while it is held and released only once `blocked`
    of them are provably waiting, so the race is deterministic instead of timing-dependent.
    """

    def guarded(job: Callable[[], Any]) -> Any:
        try:
            return job()
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        if lock is None:
            barrier = threading.Barrier(len(jobs))
            futures = [pool.submit(guarded, _after(barrier, job)) for job in jobs]
        else:
            with lock():
                futures = [pool.submit(guarded, job) for job in jobs]
                wait_until_blocked(engine, blocked if blocked is not None else len(jobs))
        return [future.result(timeout=30) for future in futures]


def _after(barrier: threading.Barrier, job: Callable[[], Any]) -> Callable[[], Any]:
    def run() -> Any:
        barrier.wait()
        return job()

    return run


# --- service-level calls for races ----------------------------------------------------------


def principal_of(env, session):
    return env.service.authenticate(session.token)


def confirm_job(env, session, pending_id, *, key: str | None = None):
    """A callable that commits a prepared action through the service, for use in a race."""
    from app.services.appointment_transactions import confirm_appointment_action

    principal = principal_of(env, session)
    idempotency_key = key or new_key()

    def job():
        return confirm_appointment_action(
            env.app.state.db_engine,
            principal=principal,
            pending_action_id=uuid.UUID(str(pending_id)),
            idempotency_key=idempotency_key,
            request_id=uuid.uuid4(),
            now=env.clock(),
        )

    return job


def prepare_job(env, session, body: dict[str, Any], *, key: str | None = None):
    from pydantic import TypeAdapter

    from app.models.appointment_actions import PrepareRequest
    from app.services.pending_actions import prepare_appointment_action

    principal = principal_of(env, session)
    request = TypeAdapter(PrepareRequest).validate_python(body)
    idempotency_key = key or new_key()

    def job():
        return prepare_appointment_action(
            env.app.state.db_engine,
            principal=principal,
            request=request,
            idempotency_key=idempotency_key,
            request_id=uuid.uuid4(),
            now=env.clock(),
        )

    return job


def assert_slots_consistent(engine: Engine, slot_ids) -> None:
    """A BOOKED slot has exactly one CONFIRMED appointment, and no other slot has any."""
    for slot_id in slot_ids:
        status = slot_row(engine, slot_id).status
        confirmed = scalar(
            engine,
            "SELECT count(*) FROM appointments WHERE slot_id = :id AND status = 'CONFIRMED'",
            id=slot_id,
        )
        assert confirmed == (1 if status == "BOOKED" else 0), (slot_id, status, confirmed)


def new_patient(env, name: str | None = None) -> str:
    """A fresh fictional patient who can sign in with the shared test PIN. Returns the code."""
    from database_support import FAST_HASHER
    from factories import make

    from app.db.models import Patient, PatientCredential

    code = "PT-" + uuid.uuid4().hex[:8].upper()
    with env.engine.begin() as connection:
        patient = make(connection, Patient, patient_code=code, display_name=name or f"Test {code}")
        make(
            connection,
            PatientCredential,
            patient_id=patient.patient_id,
            pin_hash=FAST_HASHER.hash(env.pin),
        )
    return code
