from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, text
from sqlalchemy.engine import Connection

from app.db.base import Base

EXPECTED_TABLES = {
    "appointment_history",
    "appointments",
    "auth_sessions",
    "clinic_holidays",
    "clinic_operating_hours",
    "clinics",
    "doctor_availability_rules",
    "doctor_clinics",
    "doctor_leave",
    "doctors",
    "handoff_notes",
    "handoff_tickets",
    "idempotency_records",
    "notification_events",
    "outbox_events",
    "patient_credentials",
    "patients",
    "pending_actions",
    "security_audit_events",
    "slot_offers",
    "slots",
    "staff_credentials",
    "staff_users",
    "waitlist_entries",
}

REQUIRED_INDEXES = {
    "uq_appointments_confirmed_slot",
    "uq_slot_offers_active_slot",
    "uq_slot_offers_active_patient",
    "uq_auth_sessions_staff_active",
    "uq_staff_users_username_normalized",
    "ix_slots_bookable_starts_at",
    "ix_waitlist_entries_matching",
    "ix_slot_offers_active_expires_at",
    "ix_outbox_events_status_available_at",
    "ix_security_audit_events_network_occurred_at",
}

REQUIRED_CONSTRAINTS = {
    "ck_slots_removal_requires_available": "c",
    "ex_waitlist_entries_active_overlap": "x",
    "fk_slot_offers_waitlist_patient": "f",
    "fk_slots_doctor_clinic": "f",
    "uq_slot_offers_patient_id_slot_id": "u",
    "uq_waitlist_entries_waitlist_id_patient_id": "u",
    "pk_doctor_clinics": "p",
    "ex_slots_doctor_no_overlap": "x",
    "uq_slots_doctor_id_starts_at": "u",
    "uq_slot_offers_offer_id_patient_id": "u",
    "fk_notification_events_offer_patient": "f",
    "ck_handoff_tickets_category_valid": "c",
    "ck_idempotency_records_actor_type_valid": "c",
    "ck_idempotency_records_response_snapshot_is_object": "c",
}


def table_names(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


def test_upgrade_creates_every_approved_table(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        assert table_names(connection) == EXPECTED_TABLES | {"alembic_version"}


def test_upgrade_installs_btree_gist(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        installed = connection.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'btree_gist'")
        ).scalar_one()
    assert installed == 1


def test_database_is_at_the_single_migration_head(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        versions = (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        )
    assert versions == ["0003"]


def test_required_indexes_exist(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        existing = set(
            connection.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))
            .scalars()
            .all()
        )
    assert existing >= REQUIRED_INDEXES


def test_required_constraints_exist_with_expected_type(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        existing = dict(
            connection.execute(
                text(
                    "SELECT conname, contype::text FROM pg_constraint"
                    " WHERE connamespace = 'public'::regnamespace"
                )
            ).all()
        )
    assert {name: existing.get(name) for name in REQUIRED_CONSTRAINTS} == REQUIRED_CONSTRAINTS


def test_schema_has_no_triggers(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        triggers = connection.execute(
            text("SELECT count(*) FROM pg_trigger WHERE NOT tgisinternal")
        ).scalar_one()
    assert triggers == 0


def test_migrated_schema_matches_models(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_downgrade_removes_schema_and_upgrade_can_repeat(fresh_database: Engine, migrate) -> None:
    migrate(fresh_database, "upgrade", "head")
    migrate(fresh_database, "downgrade", "base")

    with fresh_database.connect() as connection:
        assert table_names(connection) == {"alembic_version"}
        extensions = connection.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'btree_gist'")
        ).scalar_one()
    assert extensions == 0

    migrate(fresh_database, "upgrade", "head")
    with fresh_database.connect() as connection:
        assert table_names(connection) == EXPECTED_TABLES | {"alembic_version"}


def test_constraint_and_index_names_match_models(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        constraints = set(
            connection.execute(
                text(
                    "SELECT conname FROM pg_constraint"
                    " WHERE connamespace = 'public'::regnamespace AND contype <> 'n'"
                )
            )
            .scalars()
            .all()
        )
        indexes = set(
            connection.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))
            .scalars()
            .all()
        )

    tables = Base.metadata.tables.values()
    model_constraints = {c.name for table in tables for c in table.constraints}
    model_indexes = {index.name for table in tables for index in table.indexes}

    assert constraints - {"alembic_version_pkc"} == model_constraints
    assert indexes - constraints == model_indexes
