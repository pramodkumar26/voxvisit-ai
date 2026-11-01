from collections import defaultdict
from contextlib import contextmanager
from datetime import date, timedelta
from itertools import pairwise
from types import SimpleNamespace
from zoneinfo import ZoneInfo, available_timezones

import pytest
from argon2 import PasswordHasher
from database_support import FAST_HASHER, disposable_database, random_credentials, run_migration
from factories import make
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import IntegrityError

from app.core.config import get_settings
from app.db.models import (
    Appointment,
    AppointmentHistory,
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    HandoffNote,
    HandoffTicket,
    IdempotencyRecord,
    Patient,
    PatientCredential,
    Slot,
    StaffCredential,
    StaffUser,
    WaitlistEntry,
)
from app.seed.cli import main as seed_cli
from app.seed.credentials import SeedCredentialError
from app.seed.loader import seed_database
from app.seed.plan import build_plan

ANCHOR = date(2026, 10, 28)
SEEDED_TABLES = [
    Clinic,
    ClinicOperatingHours,
    ClinicHoliday,
    Doctor,
    DoctorClinic,
    DoctorAvailabilityRule,
    DoctorLeave,
    Slot,
    Patient,
    StaffUser,
    PatientCredential,
    StaffCredential,
    Appointment,
    AppointmentHistory,
    WaitlistEntry,
    HandoffTicket,
    HandoffNote,
]


class ForbiddenHasher:
    def hash(self, secret: str) -> str:
        raise AssertionError("hashing should not happen when every credential exists")


class FailingHasher:
    def hash(self, secret: str) -> str:
        raise RuntimeError("hasher failure")


def seed(engine: Engine, credentials: SimpleNamespace, **overrides):
    options = {"anchor_date": ANCHOR, "hasher": FAST_HASHER, **overrides}
    return seed_database(
        engine, patient_pin=credentials.pin, staff_password=credentials.password, **options
    )


@contextmanager
def migrated_database():
    with disposable_database() as engine:
        run_migration(engine, "upgrade", "head")
        yield engine


@pytest.fixture(scope="module")
def credentials() -> SimpleNamespace:
    return random_credentials()


@pytest.fixture(scope="module")
def seeded(credentials: SimpleNamespace):
    with migrated_database() as engine:
        result = seed(engine, credentials)
        yield SimpleNamespace(engine=engine, result=result, credentials=credentials)


@pytest.fixture
def empty_database():
    with migrated_database() as engine:
        yield engine


def count(engine: Engine, model: type) -> int:
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(model)).scalar_one()


def rows(engine: Engine, sql: str, **params) -> list:
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(text(sql), params).mappings()]


def snapshot(engine: Engine) -> dict[str, list]:
    state = {}
    with engine.connect() as connection:
        for model in SEEDED_TABLES:
            table = model.__table__
            statement = select(table).order_by(*table.primary_key.columns)
            state[table.name] = [tuple(row) for row in connection.execute(statement)]
    return state


def assert_state_machine_consistency(engine: Engine) -> None:
    checks = {
        "held slots without offers": "SELECT count(*) FROM slots WHERE status = 'HELD'",
        "confirmed on non-booked slot": """
            SELECT count(*) FROM appointments a JOIN slots s USING (slot_id)
            WHERE a.status = 'CONFIRMED' AND s.status <> 'BOOKED'""",
        "booked slot without exactly one confirmed": """
            SELECT count(*) FROM slots s WHERE s.status = 'BOOKED' AND
            (SELECT count(*) FROM appointments a
             WHERE a.slot_id = s.slot_id AND a.status = 'CONFIRMED') <> 1""",
        "available slot with confirmed": """
            SELECT count(*) FROM slots s WHERE s.status = 'AVAILABLE' AND EXISTS
            (SELECT 1 FROM appointments a
             WHERE a.slot_id = s.slot_id AND a.status = 'CONFIRMED')""",
        "cancelled appointment on non-available slot": """
            SELECT count(*) FROM appointments a JOIN slots s USING (slot_id)
            WHERE a.status = 'CANCELLED' AND s.status <> 'AVAILABLE'""",
        "confirmed without single booked event": """
            SELECT count(*) FROM appointments a WHERE a.status = 'CONFIRMED' AND
            (SELECT count(*) FROM appointment_history h
             WHERE h.appointment_id = a.appointment_id AND h.action = 'BOOKED') <> 1""",
        "confirmed with cancel event": """
            SELECT count(*) FROM appointments a WHERE a.status = 'CONFIRMED' AND EXISTS
            (SELECT 1 FROM appointment_history h
             WHERE h.appointment_id = a.appointment_id AND h.action = 'CANCELLED')""",
        "cancelled without booked and cancelled events": """
            SELECT count(*) FROM appointments a WHERE a.status = 'CANCELLED' AND
            (SELECT count(DISTINCT h.action) FROM appointment_history h
             WHERE h.appointment_id = a.appointment_id
               AND h.action IN ('BOOKED', 'CANCELLED')) <> 2""",
    }
    for description, sql in checks.items():
        assert rows(engine, sql)[0]["count"] == 0, description


# --- clean seed ------------------------------------------------------------------------------


def test_seed_populates_a_clean_migrated_database(seeded) -> None:
    plan = build_plan(ANCHOR)
    engine = seeded.engine

    assert seeded.result.committed
    assert seeded.result.skipped_appointments == []
    assert count(engine, Clinic) == 3
    assert count(engine, Doctor) == 8
    assert count(engine, Patient) == 14
    assert count(engine, StaffUser) == 2
    assert count(engine, PatientCredential) == 14
    assert count(engine, StaffCredential) == 2
    assert count(engine, Slot) == len(plan.slots) > 800
    assert count(engine, Appointment) == 11
    assert count(engine, AppointmentHistory) == 8 + 3 * 2
    assert count(engine, WaitlistEntry) == 3
    assert count(engine, HandoffTicket) == 4
    assert count(engine, HandoffNote) == 2
    assert count(engine, DoctorLeave) == len(plan.doctor_leave) == 3
    assert count(engine, ClinicHoliday) == len(plan.clinic_holidays) == 3
    for table, counts in seeded.result.tables.items():
        assert counts.existing == 0, table


def test_seed_has_confirmed_cancelled_booked_and_available_records(seeded) -> None:
    statuses = {
        row["status"]: row["count"]
        for row in rows(seeded.engine, "SELECT status, count(*) FROM appointments GROUP BY 1")
    }
    slots = {
        row["status"]: row["count"]
        for row in rows(seeded.engine, "SELECT status, count(*) FROM slots GROUP BY 1")
    }

    assert statuses == {"CONFIRMED": 8, "CANCELLED": 3}
    assert slots["BOOKED"] == 8
    assert slots["AVAILABLE"] > 800
    assert set(slots) == {"AVAILABLE", "BOOKED"}


def test_appointment_slot_and_history_state_is_consistent(seeded) -> None:
    assert_state_machine_consistency(seeded.engine)
    history_owner_mismatches = rows(
        seeded.engine,
        """SELECT count(*) FROM appointment_history h JOIN appointments a USING (appointment_id)
           WHERE h.actor_type <> 'PATIENT' OR h.actor_id <> a.patient_id
              OR COALESCE(h.new_slot_id, h.old_slot_id) <> a.slot_id""",
    )
    assert history_owner_mismatches[0]["count"] == 0


def test_seeded_schedule_respects_hours_availability_holidays_leave_and_timezones(seeded) -> None:
    engine = seeded.engine
    clinics = {row["clinic_id"]: row for row in rows(engine, "SELECT * FROM clinics")}
    hours = {
        (row["clinic_id"], row["day_of_week"]): row
        for row in rows(engine, "SELECT * FROM clinic_operating_hours")
    }
    rules = defaultdict(list)
    for row in rows(engine, "SELECT * FROM doctor_availability_rules WHERE active"):
        rules[(row["doctor_id"], row["clinic_id"])].append(row)
    holidays = defaultdict(list)
    for row in rows(engine, "SELECT * FROM clinic_holidays"):
        holidays[(row["clinic_id"], row["local_date"])].append(row)
    leaves = defaultdict(list)
    for row in rows(engine, "SELECT * FROM doctor_leave"):
        leaves[row["doctor_id"]].append(row)
    slots = rows(
        engine, "SELECT * FROM slots WHERE removed_at IS NULL ORDER BY doctor_id, starts_at"
    )

    assert all(clinic["timezone"] in available_timezones() for clinic in clinics.values())
    for slot in slots:
        zone = ZoneInfo(clinics[slot["clinic_id"]]["timezone"])
        start, end = slot["starts_at"].astimezone(zone), slot["ends_at"].astimezone(zone)
        day, weekday = start.date(), start.weekday()
        assert 1 <= (day - ANCHOR).days <= 21
        opening = hours[(slot["clinic_id"], weekday)]
        assert opening["opens_at"] <= start.time() and end.time() <= opening["closes_at"]
        assert any(
            rule["day_of_week"] == weekday
            and rule["start_time"] <= start.time()
            and end.time() <= rule["end_time"]
            and rule["effective_start_date"] <= day
            and (rule["effective_end_date"] is None or day <= rule["effective_end_date"])
            for rule in rules[(slot["doctor_id"], slot["clinic_id"])]
        )
        for closure in holidays[(slot["clinic_id"], day)]:
            full_day = closure["start_time"] is None
            assert not full_day
            assert end.time() <= closure["start_time"] or start.time() >= closure["end_time"]
        for leave in leaves[slot["doctor_id"]]:
            assert slot["ends_at"] <= leave["starts_at"] or slot["starts_at"] >= leave["ends_at"]
    for earlier, later in pairwise(slots):
        if earlier["doctor_id"] == later["doctor_id"]:
            assert earlier["ends_at"] <= later["starts_at"]


def test_confirmed_appointments_sit_inside_doctor_availability(seeded) -> None:
    booked = rows(
        seeded.engine,
        """SELECT s.starts_at, s.doctor_id, s.clinic_id, c.timezone
           FROM appointments a JOIN slots s USING (slot_id) JOIN clinics c USING (clinic_id)
           WHERE a.status = 'CONFIRMED'""",
    )
    rules = rows(seeded.engine, "SELECT * FROM doctor_availability_rules")

    assert len(booked) == 8
    for appointment in booked:
        local = appointment["starts_at"].astimezone(ZoneInfo(appointment["timezone"]))
        assert any(
            rule["doctor_id"] == appointment["doctor_id"]
            and rule["clinic_id"] == appointment["clinic_id"]
            and rule["day_of_week"] == local.weekday()
            and rule["start_time"] <= local.time() < rule["end_time"]
            for rule in rules
        )


def test_waitlist_and_handoff_records_are_valid(seeded) -> None:
    waitlist = rows(
        seeded.engine,
        """SELECT w.specialty, w.status FROM waitlist_entries w
           WHERE EXISTS (SELECT 1 FROM doctor_clinics dc JOIN doctors d USING (doctor_id)
                         WHERE dc.clinic_id = w.clinic_id AND d.specialty = w.specialty)""",
    )
    tickets = rows(seeded.engine, "SELECT category, status, patient_id FROM handoff_tickets")

    assert len(waitlist) == 3 and {row["status"] for row in waitlist} == {"ACTIVE"}
    assert {row["category"] for row in tickets} <= {
        "SCHEDULING",
        "CLINIC_INFORMATION",
        "ACCESSIBILITY",
        "TECHNICAL_SUPPORT",
        "OTHER_ADMINISTRATIVE",
    }
    assert any(row["patient_id"] is None for row in tickets)
    assert any(row["patient_id"] is not None for row in tickets)


def test_seed_leaves_unimplemented_service_tables_empty(seeded) -> None:
    for table in ("slot_offers", "pending_actions", "auth_sessions", "outbox_events"):
        assert rows(seeded.engine, f"SELECT count(*) FROM {table}")[0]["count"] == 0


# --- credentials -----------------------------------------------------------------------------


def test_all_accounts_have_verifiable_argon2id_hashes(seeded) -> None:
    patient_hashes = [
        r["pin_hash"] for r in rows(seeded.engine, "SELECT pin_hash FROM patient_credentials")
    ]
    staff_hashes = [
        r["password_hash"]
        for r in rows(seeded.engine, "SELECT password_hash FROM staff_credentials")
    ]
    verifier = PasswordHasher()

    assert len(patient_hashes) == 14 and len(staff_hashes) == 2
    for digest in patient_hashes:
        assert digest.startswith("$argon2id$v=19$")
        assert verifier.verify(digest, seeded.credentials.pin)
    for digest in staff_hashes:
        assert digest.startswith("$argon2id$v=19$")
        assert verifier.verify(digest, seeded.credentials.password)
    assert len(set(patient_hashes + staff_hashes)) == 16


def test_plaintext_credentials_are_not_stored(seeded) -> None:
    for model in SEEDED_TABLES:
        table = model.__tablename__
        hits = rows(
            seeded.engine,
            f"SELECT count(*) FROM {table} t WHERE t::text LIKE '%{seeded.credentials.password}%'",
        )
        assert hits[0]["count"] == 0, table


def test_invalid_credentials_stop_the_seed_before_any_write(empty_database: Engine) -> None:
    good = random_credentials()

    with pytest.raises(SeedCredentialError):
        seed(empty_database, SimpleNamespace(pin="12345", password=good.password))
    with pytest.raises(SeedCredentialError):
        seed(empty_database, SimpleNamespace(pin=good.pin, password="short"))

    assert count(empty_database, Clinic) == 0


# --- idempotency and safety ------------------------------------------------------------------


def test_repeating_the_seed_changes_nothing_and_does_not_rehash(empty_database: Engine) -> None:
    credentials = random_credentials()
    seed(empty_database, credentials)
    before = snapshot(empty_database)

    result = seed(empty_database, credentials, hasher=ForbiddenHasher())

    assert result.committed
    assert result.inserted_total == 0
    assert all(counts.inserted == 0 for counts in result.tables.values())
    assert snapshot(empty_database) == before


def test_repeating_the_seed_does_not_overwrite_changed_appointment_state(
    empty_database: Engine,
) -> None:
    credentials = random_credentials()
    seed(empty_database, credentials)
    with empty_database.begin() as connection:
        victim = connection.execute(
            text(
                """SELECT a.appointment_id, a.slot_id FROM appointments a
                   WHERE a.status = 'CONFIRMED' ORDER BY a.created_at LIMIT 1"""
            )
        ).one()
        connection.execute(
            text(
                "UPDATE appointments SET status = 'CANCELLED', cancelled_at = now(), version = 9"
                " WHERE appointment_id = :id"
            ),
            {"id": victim.appointment_id},
        )
        connection.execute(
            text("UPDATE slots SET status = 'AVAILABLE', version = 9 WHERE slot_id = :id"),
            {"id": victim.slot_id},
        )
    before = snapshot(empty_database)

    result = seed(empty_database, credentials, hasher=ForbiddenHasher())

    assert result.inserted_total == 0
    assert snapshot(empty_database) == before
    stored = rows(
        empty_database,
        "SELECT status, version FROM appointments WHERE appointment_id = :id",
        id=victim.appointment_id,
    )
    assert stored == [{"status": "CANCELLED", "version": 9}]


def test_seeding_with_a_later_anchor_adds_schedule_but_never_duplicates_bookings(
    empty_database: Engine,
) -> None:
    credentials = random_credentials()
    seed(empty_database, credentials)
    before = snapshot(empty_database)

    result = seed(empty_database, credentials, anchor_date=ANCHOR + timedelta(days=7))

    assert result.tables["slots"].inserted > 0
    assert result.tables["appointments"].inserted == 0
    assert result.tables["appointments"].existing == 11
    assert result.skipped_appointments == []
    after = snapshot(empty_database)
    for table in ("appointments", "appointment_history", "waitlist_entries", "handoff_tickets"):
        assert after[table] == before[table]
    assert count(empty_database, Clinic) == 3
    assert count(empty_database, Doctor) == 8
    assert count(empty_database, Patient) == 14
    assert_state_machine_consistency(empty_database)


def test_unrelated_records_are_left_unchanged(empty_database: Engine) -> None:
    credentials = random_credentials()
    with empty_database.begin() as connection:
        clinic = make(connection, Clinic)
        doctor = make(connection, Doctor)
        make(connection, DoctorClinic, doctor_id=doctor.doctor_id, clinic_id=clinic.clinic_id)
        slot = make(connection, Slot, doctor_id=doctor.doctor_id, clinic_id=clinic.clinic_id)
        patient = make(connection, Patient)
        appointment = make(
            connection, Appointment, patient_id=patient.patient_id, slot_id=slot.slot_id
        )
        ticket = make(connection, HandoffTicket)
        make(connection, IdempotencyRecord, actor_id=patient.patient_id)

    queries = {
        "SELECT * FROM clinics WHERE clinic_id = :id": clinic.clinic_id,
        "SELECT * FROM doctors WHERE doctor_id = :id": doctor.doctor_id,
        "SELECT * FROM slots WHERE slot_id = :id": slot.slot_id,
        "SELECT * FROM patients WHERE patient_id = :id": patient.patient_id,
        "SELECT * FROM appointments WHERE appointment_id = :id": appointment.appointment_id,
        "SELECT * FROM handoff_tickets WHERE ticket_id = :id": ticket.ticket_id,
        "SELECT * FROM idempotency_records WHERE actor_id = :id": patient.patient_id,
    }
    before = {sql: rows(empty_database, sql, id=value) for sql, value in queries.items()}
    assert all(len(found) == 1 for found in before.values())

    seed(empty_database, credentials)

    assert {sql: rows(empty_database, sql, id=value) for sql, value in queries.items()} == before
    assert count(empty_database, Clinic) == 4
    assert count(empty_database, Doctor) == 9
    assert count(empty_database, Appointment) == 12
    assert count(empty_database, HandoffTicket) == 5


def test_an_appointment_whose_slot_was_taken_is_skipped_not_forced(empty_database: Engine) -> None:
    credentials = random_credentials()
    seed(empty_database, credentials)
    item = next(entry for entry in build_plan(ANCHOR).appointments if entry.key == "confirmed-01")
    with empty_database.begin() as connection:
        params = {"appointment": item.appointment_id}
        connection.execute(
            text("DELETE FROM appointment_history WHERE appointment_id = :appointment"), params
        )
        connection.execute(
            text("DELETE FROM appointments WHERE appointment_id = :appointment"), params
        )
        other = make(connection, Patient)
        taken = make(connection, Appointment, patient_id=other.patient_id, slot_id=item.slot_id)
        make(
            connection,
            AppointmentHistory,
            appointment_id=taken.appointment_id,
            actor_type="PATIENT",
            actor_id=other.patient_id,
            new_slot_id=item.slot_id,
        )

    result = seed(empty_database, credentials, hasher=ForbiddenHasher())

    assert result.skipped_appointments == ["confirmed-01"]
    on_slot = rows(
        empty_database,
        "SELECT appointment_id FROM appointments WHERE slot_id = :id",
        id=item.slot_id,
    )
    assert on_slot == [{"appointment_id": taken.appointment_id}]
    assert_state_machine_consistency(empty_database)


def test_dry_run_reports_counts_and_saves_nothing(empty_database: Engine) -> None:
    result = seed(empty_database, random_credentials(), dry_run=True)

    assert not result.committed
    assert result.tables["slots"].inserted > 800
    for model in SEEDED_TABLES:
        assert count(empty_database, model) == 0, model.__tablename__


@pytest.mark.parametrize("failure", ["unrelated-conflict", "hasher-error"])
def test_a_failed_seed_rolls_back_completely(empty_database: Engine, failure: str) -> None:
    credentials = random_credentials()
    unrelated_clinics = 0
    overrides = {}
    if failure == "unrelated-conflict":
        with empty_database.begin() as connection:
            make(connection, Clinic, name="Lakeside Community Clinic")
        unrelated_clinics = 1
        expected = IntegrityError
    else:
        overrides["hasher"] = FailingHasher()
        expected = RuntimeError

    with pytest.raises(expected):
        seed(empty_database, credentials, **overrides)

    for model in SEEDED_TABLES:
        expected_count = unrelated_clinics if model is Clinic else 0
        assert count(empty_database, model) == expected_count, model.__tablename__


# --- command line ----------------------------------------------------------------------------


@pytest.fixture
def cli_environment(empty_database: Engine, monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", empty_database.url.render_as_string(hide_password=False))
    monkeypatch.setenv("APP_ENV", "test")
    credentials = random_credentials()
    monkeypatch.setenv("SEED_PATIENT_PIN", credentials.pin)
    monkeypatch.setenv("SEED_STAFF_PASSWORD", credentials.password)
    get_settings.cache_clear()
    yield SimpleNamespace(engine=empty_database, credentials=credentials)
    get_settings.cache_clear()


def test_cli_seeds_repeats_safely_and_never_prints_credentials(cli_environment, capsys) -> None:
    arguments = ["--anchor-date", "2026-10-28"]

    assert seed_cli(arguments) == 0
    first = capsys.readouterr().out
    assert seed_cli(arguments) == 0
    second = capsys.readouterr().out

    assert "Seed committed." in first and "Seed committed." in second
    assert count(cli_environment.engine, Clinic) == 3
    assert count(cli_environment.engine, Slot) > 800
    for line in second.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit():
            assert parts[1] == "0", line
    for output in (first, second):
        assert cli_environment.credentials.pin not in output
        assert cli_environment.credentials.password not in output


def test_cli_dry_run_saves_nothing(cli_environment, capsys) -> None:
    assert seed_cli(["--anchor-date", "2026-10-28", "--dry-run"]) == 0

    assert "Dry run: nothing was saved." in capsys.readouterr().out
    assert count(cli_environment.engine, Clinic) == 0


def test_cli_requires_demonstration_credentials(
    cli_environment, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.delenv("SEED_PATIENT_PIN")
    monkeypatch.delenv("SEED_STAFF_PASSWORD")

    assert seed_cli([]) == 2

    assert "SEED_PATIENT_PIN" in capsys.readouterr().out
    assert count(cli_environment.engine, Clinic) == 0


def test_cli_rejects_a_malformed_pin_without_echoing_it(
    cli_environment, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SEED_PATIENT_PIN", "abc123x")

    assert seed_cli([]) == 2

    assert "abc123x" not in capsys.readouterr().out
    assert count(cli_environment.engine, Clinic) == 0


def test_cli_refuses_to_seed_production(
    cli_environment, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_SECRET_KEY", "production-shaped-secret-key-0123456789")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.test")
    get_settings.cache_clear()

    assert seed_cli([]) == 2

    assert "production" in capsys.readouterr().out
    assert count(cli_environment.engine, Clinic) == 0


def test_cli_reports_conflicts_and_rolls_back(cli_environment, capsys) -> None:
    with cli_environment.engine.begin() as connection:
        make(connection, Clinic, name="Harborview Medical Associates")

    assert seed_cli(["--anchor-date", "2026-10-28"]) == 1

    output = capsys.readouterr().out
    assert "rolled back" in output
    assert cli_environment.credentials.password not in output
    assert count(cli_environment.engine, Doctor) == 0
