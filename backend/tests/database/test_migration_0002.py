import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import Connection, Engine, inspect, text

Migrate = Callable[[Engine, str, str], None]

NEXT_REVISION_ERROR = "Migration 0002 stopped"


def scalar(connection: Connection, sql: str, **params):
    return connection.execute(text(sql), params).scalar_one()


def seed_schedule(connection: Connection) -> dict:
    ids = {
        "clinic": scalar(
            connection,
            "INSERT INTO clinics (name, timezone, address_line, city, region, postal_code)"
            " VALUES ('Seed Clinic', 'America/Denver', '1 Way', 'City', 'CO', '80000')"
            " RETURNING clinic_id",
        ),
        "doctor": scalar(
            connection,
            "INSERT INTO doctors (display_name, specialty) VALUES ('Dr. Seed', 'Primary Care')"
            " RETURNING doctor_id",
        ),
    }
    connection.execute(
        text("INSERT INTO doctor_clinics (doctor_id, clinic_id) VALUES (:doctor, :clinic)"),
        ids,
    )
    return ids


def seed_slot(connection: Connection, ids: dict, start: str, end: str):
    return scalar(
        connection,
        "INSERT INTO slots (doctor_id, clinic_id, starts_at, ends_at)"
        " VALUES (:doctor, :clinic, :start, :end) RETURNING slot_id",
        start=start,
        end=end,
        **ids,
    )


def seed_patient(connection: Connection, code: str):
    return scalar(
        connection,
        "INSERT INTO patients (patient_code, display_name) VALUES (:code, :code)"
        " RETURNING patient_id",
        code=code,
    )


def columns(engine: Engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def current_revision(engine: Engine) -> str:
    with engine.connect() as connection:
        return scalar(connection, "SELECT version_num FROM alembic_version")


def test_upgrade_preserves_existing_rows_and_backfills_handoff_categories(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "0001")
    with fresh_database.begin() as connection:
        ids = seed_schedule(connection)
        seed_slot(connection, ids, "2030-01-07 15:00+00", "2030-01-07 15:30+00")
        seed_slot(connection, ids, "2030-01-07 15:30+00", "2030-01-07 16:00+00")
        for summary in ("First legacy ticket.", "Second legacy ticket."):
            connection.execute(
                text("INSERT INTO handoff_tickets (redacted_summary) VALUES (:summary)"),
                {"summary": summary},
            )
        connection.execute(
            text(
                "INSERT INTO idempotency_records"
                " (actor_type, actor_id, operation, idempotency_key, request_hash, status)"
                " VALUES ('PATIENT', gen_random_uuid(), 'book', 'k', '\\x01', 'COMPLETED')"
            )
        )

    migrate(fresh_database, "upgrade", "0002")

    with fresh_database.connect() as connection:
        categories = connection.execute(
            text("SELECT redacted_summary, category FROM handoff_tickets ORDER BY redacted_summary")
        ).all()
        assert [tuple(row) for row in categories] == [
            ("First legacy ticket.", "OTHER_ADMINISTRATIVE"),
            ("Second legacy ticket.", "OTHER_ADMINISTRATIVE"),
        ]
        assert scalar(connection, "SELECT count(*) FROM slots") == 2
        assert scalar(connection, "SELECT count(*) FROM idempotency_records") == 1
        assert scalar(connection, "SELECT response_snapshot IS NULL FROM idempotency_records")
    assert current_revision(fresh_database) == "0002"


def test_downgrade_restores_previous_schema_without_losing_other_rows(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "head")
    with fresh_database.begin() as connection:
        ids = seed_schedule(connection)
        seed_slot(connection, ids, "2030-01-07 15:00+00", "2030-01-07 15:30+00")
        connection.execute(
            text(
                "INSERT INTO handoff_tickets (category, redacted_summary)"
                " VALUES ('ACCESSIBILITY', 'Needs a ramp.')"
            )
        )

    migrate(fresh_database, "downgrade", "0001")

    assert current_revision(fresh_database) == "0001"
    assert "category" not in columns(fresh_database, "handoff_tickets")
    assert "response_snapshot" not in columns(fresh_database, "idempotency_records")
    with fresh_database.connect() as connection:
        assert scalar(connection, "SELECT count(*) FROM slots") == 1
        assert scalar(connection, "SELECT count(*) FROM handoff_tickets") == 1
        assert (
            scalar(
                connection,
                "SELECT count(*) FROM pg_constraint WHERE conname = 'ex_slots_doctor_no_overlap'",
            )
            == 0
        )

    migrate(fresh_database, "upgrade", "head")
    with fresh_database.connect() as connection:
        assert scalar(connection, "SELECT category FROM handoff_tickets") == "OTHER_ADMINISTRATIVE"


def test_upgrade_stops_on_overlapping_slots_without_changing_data(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "0001")
    with fresh_database.begin() as connection:
        ids = seed_schedule(connection)
        seed_slot(connection, ids, "2030-01-07 15:00+00", "2030-01-07 15:30+00")
        seed_slot(connection, ids, "2030-01-07 15:15+00", "2030-01-07 15:45+00")
        connection.execute(
            text("INSERT INTO handoff_tickets (redacted_summary) VALUES ('Legacy.')")
        )

    with pytest.raises(RuntimeError, match=NEXT_REVISION_ERROR) as exc_info:
        migrate(fresh_database, "upgrade", "head")

    assert "slot overlap" in str(exc_info.value)
    assert current_revision(fresh_database) == "0001"
    assert "category" not in columns(fresh_database, "handoff_tickets")
    with fresh_database.connect() as connection:
        assert scalar(connection, "SELECT count(*) FROM slots") == 2
        assert scalar(connection, "SELECT count(*) FROM handoff_tickets") == 1


def test_soft_removed_overlapping_slots_do_not_block_the_upgrade(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "0001")
    with fresh_database.begin() as connection:
        ids = seed_schedule(connection)
        seed_slot(connection, ids, "2030-01-07 15:00+00", "2030-01-07 15:30+00")
        removed = seed_slot(connection, ids, "2030-01-07 15:15+00", "2030-01-07 15:45+00")
        connection.execute(
            text("UPDATE slots SET removed_at = now() WHERE slot_id = :slot"), {"slot": removed}
        )

    migrate(fresh_database, "upgrade", "0002")

    assert current_revision(fresh_database) == "0002"


def test_upgrade_stops_on_mismatched_notification_ownership(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "0001")
    with fresh_database.begin() as connection:
        ids = seed_schedule(connection)
        slot = seed_slot(connection, ids, "2030-01-07 15:00+00", "2030-01-07 15:30+00")
        owner = seed_patient(connection, "OWNER")
        other = seed_patient(connection, "OTHER")
        waitlist = scalar(
            connection,
            "INSERT INTO waitlist_entries"
            " (patient_id, clinic_id, specialty, start_date, end_date, time_preference)"
            " VALUES (:patient, :clinic, 'Primary Care', '2030-02-01', '2030-02-28', 'ANYTIME')"
            " RETURNING waitlist_id",
            patient=owner,
            clinic=ids["clinic"],
        )
        offer = scalar(
            connection,
            "INSERT INTO slot_offers (waitlist_id, slot_id, patient_id, expires_at, request_id)"
            " VALUES (:waitlist, :slot, :patient, '2031-01-01', :request) RETURNING offer_id",
            waitlist=waitlist,
            slot=slot,
            patient=owner,
            request=uuid.uuid4(),
        )
        connection.execute(
            text(
                "INSERT INTO notification_events"
                " (patient_id, offer_id, notification_type, request_id)"
                " VALUES (:patient, :offer, 'OFFER_AVAILABLE', :request)"
            ),
            {"patient": other, "offer": offer, "request": uuid.uuid4()},
        )

    with pytest.raises(RuntimeError, match=NEXT_REVISION_ERROR) as exc_info:
        migrate(fresh_database, "upgrade", "head")

    assert "notification ownership" in str(exc_info.value)
    assert current_revision(fresh_database) == "0001"
    with fresh_database.connect() as connection:
        assert scalar(connection, "SELECT count(*) FROM notification_events") == 1


def test_upgrade_stops_on_unapproved_idempotency_actor_types(
    fresh_database: Engine, migrate: Migrate
) -> None:
    migrate(fresh_database, "upgrade", "0001")
    with fresh_database.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO idempotency_records"
                " (actor_type, actor_id, operation, idempotency_key, request_hash, status)"
                " VALUES ('SYSTEM', gen_random_uuid(), 'op', 'k', '\\x01', 'COMPLETED')"
            )
        )

    with pytest.raises(RuntimeError, match=NEXT_REVISION_ERROR) as exc_info:
        migrate(fresh_database, "upgrade", "head")

    assert "actor types" in str(exc_info.value)
    assert current_revision(fresh_database) == "0001"
    with fresh_database.connect() as connection:
        assert scalar(connection, "SELECT count(*) FROM idempotency_records") == 1
