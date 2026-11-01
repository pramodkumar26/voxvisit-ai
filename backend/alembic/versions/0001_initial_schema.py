"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-31 17:46:32.977590
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table(
        "clinics",
        sa.Column(
            "clinic_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("address_line", sa.String(length=200), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("region", sa.String(length=100), nullable=False),
        sa.Column("postal_code", sa.String(length=20), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("length(btrim(name)) > 0", name=op.f("ck_clinics_name_not_blank")),
        sa.CheckConstraint(
            "length(btrim(timezone)) > 0", name=op.f("ck_clinics_timezone_not_blank")
        ),
        sa.PrimaryKeyConstraint("clinic_id", name=op.f("pk_clinics")),
        sa.UniqueConstraint("name", name=op.f("uq_clinics_name")),
    )
    op.create_index("ix_clinics_active", "clinics", ["active"], unique=False)
    op.create_index(
        "ix_clinics_name_normalized", "clinics", [sa.literal_column("lower(name)")], unique=False
    )
    op.create_table(
        "doctors",
        sa.Column(
            "doctor_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("specialty", sa.String(length=80), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(display_name)) > 0", name=op.f("ck_doctors_display_name_not_blank")
        ),
        sa.CheckConstraint(
            "length(btrim(specialty)) > 0", name=op.f("ck_doctors_specialty_not_blank")
        ),
        sa.PrimaryKeyConstraint("doctor_id", name=op.f("pk_doctors")),
    )
    op.create_index(
        "ix_doctors_display_name_normalized",
        "doctors",
        [sa.literal_column("lower(display_name)")],
        unique=False,
    )
    op.create_index("ix_doctors_specialty_active", "doctors", ["specialty", "active"], unique=False)
    op.create_table(
        "outbox_events",
        sa.Column(
            "outbox_event_id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=60), nullable=False),
        sa.Column("aggregate_type", sa.String(length=60), nullable=False),
        sa.Column("aggregate_id", sa.Uuid(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), server_default="PENDING", nullable=False),
        sa.Column("dedupe_key", sa.String(length=200), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "(status = 'PROCESSED') = (processed_at IS NOT NULL)",
            name=op.f("ck_outbox_events_processed_matches_status"),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'PROCESSED')",
            name=op.f("ck_outbox_events_status_valid"),
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name=op.f("ck_outbox_events_attempt_count_non_negative")
        ),
        sa.CheckConstraint(
            "length(btrim(aggregate_type)) > 0",
            name=op.f("ck_outbox_events_aggregate_type_not_blank"),
        ),
        sa.CheckConstraint(
            "length(btrim(event_type)) > 0", name=op.f("ck_outbox_events_event_type_not_blank")
        ),
        sa.PrimaryKeyConstraint("outbox_event_id", name=op.f("pk_outbox_events")),
        sa.UniqueConstraint("dedupe_key", name=op.f("uq_outbox_events_dedupe_key")),
    )
    op.create_index(
        "ix_outbox_events_aggregate",
        "outbox_events",
        ["aggregate_type", "aggregate_id"],
        unique=False,
    )
    op.create_index("ix_outbox_events_created_at", "outbox_events", ["created_at"], unique=False)
    op.create_index(
        "ix_outbox_events_processed_at", "outbox_events", ["processed_at"], unique=False
    )
    op.create_index(
        "ix_outbox_events_status_available_at",
        "outbox_events",
        ["status", "available_at"],
        unique=False,
    )
    op.create_table(
        "patients",
        sa.Column(
            "patient_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("patient_code", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(display_name)) > 0", name=op.f("ck_patients_display_name_not_blank")
        ),
        sa.CheckConstraint(
            "length(btrim(patient_code)) > 0", name=op.f("ck_patients_patient_code_not_blank")
        ),
        sa.PrimaryKeyConstraint("patient_id", name=op.f("pk_patients")),
        sa.UniqueConstraint("patient_code", name=op.f("uq_patients_patient_code")),
    )
    op.create_index(
        "ix_patients_display_name_normalized",
        "patients",
        [sa.literal_column("lower(display_name)")],
        unique=False,
    )
    op.create_table(
        "staff_users",
        sa.Column(
            "staff_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(display_name)) > 0", name=op.f("ck_staff_users_display_name_not_blank")
        ),
        sa.CheckConstraint(
            "length(btrim(username)) > 0", name=op.f("ck_staff_users_username_not_blank")
        ),
        sa.PrimaryKeyConstraint("staff_id", name=op.f("pk_staff_users")),
    )
    op.create_index("ix_staff_users_active", "staff_users", ["active"], unique=False)
    op.create_index(
        "uq_staff_users_username_normalized",
        "staff_users",
        [sa.literal_column("lower(username)")],
        unique=True,
    )
    op.create_table(
        "auth_sessions",
        sa.Column(
            "session_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("staff_id", sa.Uuid(), nullable=True),
        sa.Column("token_hash", sa.LargeBinary(), nullable=False),
        sa.Column("status", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("idle_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("absolute_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("terminated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("termination_reason", sa.String(length=40), nullable=True),
        sa.CheckConstraint(
            "(actor_type = 'PATIENT' AND patient_id IS NOT NULL AND staff_id IS NULL) OR (actor_type = 'STAFF' AND staff_id IS NOT NULL AND patient_id IS NULL)",
            name=op.f("ck_auth_sessions_exactly_one_actor"),
        ),
        sa.CheckConstraint(
            "(status = 'ACTIVE') = (terminated_at IS NULL)",
            name=op.f("ck_auth_sessions_termination_matches_status"),
        ),
        sa.CheckConstraint(
            "actor_type IN ('PATIENT', 'STAFF')", name=op.f("ck_auth_sessions_actor_type_valid")
        ),
        sa.CheckConstraint(
            "role IN ('PATIENT', 'STAFF')", name=op.f("ck_auth_sessions_role_valid")
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'EXPIRED', 'LOGGED_OUT', 'REVOKED')",
            name=op.f("ck_auth_sessions_status_valid"),
        ),
        sa.CheckConstraint(
            "last_activity_at >= created_at AND idle_expires_at > created_at AND absolute_expires_at > created_at",
            name=op.f("ck_auth_sessions_deadlines_follow_creation"),
        ),
        sa.CheckConstraint("role = actor_type", name=op.f("ck_auth_sessions_role_matches_actor")),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_auth_sessions_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["staff_id"],
            ["staff_users.staff_id"],
            name=op.f("fk_auth_sessions_staff_id_staff_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("session_id", name=op.f("pk_auth_sessions")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_auth_sessions_token_hash")),
    )
    op.create_index(
        "ix_auth_sessions_absolute_expires_at",
        "auth_sessions",
        ["absolute_expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_auth_sessions_idle_expires_at", "auth_sessions", ["idle_expires_at"], unique=False
    )
    op.create_index(
        "ix_auth_sessions_patient_active",
        "auth_sessions",
        ["patient_id", "created_at"],
        unique=False,
        postgresql_where=sa.text("status = 'ACTIVE' AND patient_id IS NOT NULL"),
    )
    op.create_index(
        "uq_auth_sessions_staff_active",
        "auth_sessions",
        ["staff_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE' AND staff_id IS NOT NULL"),
    )
    op.create_table(
        "clinic_holidays",
        sa.Column(
            "holiday_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=True),
        sa.Column("end_time", sa.Time(), nullable=True),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.CheckConstraint(
            "(start_time IS NULL AND end_time IS NULL) OR (start_time IS NOT NULL AND end_time IS NOT NULL AND start_time < end_time)",
            name=op.f("ck_clinic_holidays_window_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_clinic_holidays_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("holiday_id", name=op.f("pk_clinic_holidays")),
        sa.UniqueConstraint(
            "clinic_id",
            "local_date",
            "start_time",
            "end_time",
            name=op.f("uq_clinic_holidays_clinic_id_local_date_start_time_end_time"),
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_table(
        "clinic_operating_hours",
        sa.Column(
            "hours_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("day_of_week", sa.SmallInteger(), nullable=False),
        sa.Column("opens_at", sa.Time(), nullable=False),
        sa.Column("closes_at", sa.Time(), nullable=False),
        sa.CheckConstraint(
            "day_of_week BETWEEN 0 AND 6", name=op.f("ck_clinic_operating_hours_day_of_week_valid")
        ),
        sa.CheckConstraint(
            "opens_at < closes_at", name=op.f("ck_clinic_operating_hours_window_valid")
        ),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_clinic_operating_hours_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("hours_id", name=op.f("pk_clinic_operating_hours")),
        sa.UniqueConstraint(
            "clinic_id",
            "day_of_week",
            "opens_at",
            "closes_at",
            name=op.f("uq_clinic_operating_hours_window"),
        ),
    )
    op.create_table(
        "doctor_clinics",
        sa.Column("doctor_id", sa.Uuid(), nullable=False),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_doctor_clinics_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id"],
            ["doctors.doctor_id"],
            name=op.f("fk_doctor_clinics_doctor_id_doctors"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("doctor_id", "clinic_id", name=op.f("pk_doctor_clinics")),
    )
    op.create_index(
        "ix_doctor_clinics_clinic_doctor",
        "doctor_clinics",
        ["clinic_id", "doctor_id"],
        unique=False,
    )
    op.create_table(
        "patient_credentials",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("pin_hash", sa.Text(), nullable=False),
        sa.Column(
            "failed_attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("failed_attempt_window_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disabled", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "failed_attempt_count >= 0",
            name=op.f("ck_patient_credentials_failed_attempt_count_non_negative"),
        ),
        sa.CheckConstraint(
            "length(btrim(pin_hash)) > 0", name=op.f("ck_patient_credentials_pin_hash_not_blank")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_patient_credentials_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("patient_id", name=op.f("pk_patient_credentials")),
    )
    op.create_table(
        "staff_credentials",
        sa.Column("staff_id", sa.Uuid(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "failed_attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("failed_attempt_window_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reset_required", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "failed_attempt_count >= 0",
            name=op.f("ck_staff_credentials_failed_attempt_count_non_negative"),
        ),
        sa.CheckConstraint(
            "length(btrim(password_hash)) > 0",
            name=op.f("ck_staff_credentials_password_hash_not_blank"),
        ),
        sa.ForeignKeyConstraint(
            ["staff_id"],
            ["staff_users.staff_id"],
            name=op.f("fk_staff_credentials_staff_id_staff_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("staff_id", name=op.f("pk_staff_credentials")),
    )
    op.create_table(
        "waitlist_entries",
        sa.Column(
            "waitlist_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("specialty", sa.String(length=80), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("time_preference", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        postgresql.ExcludeConstraint(
            (sa.column("patient_id"), "="),
            (sa.column("clinic_id"), "="),
            (sa.column("specialty"), "="),
            (sa.text("daterange(start_date, end_date, '[]')"), "&&"),
            where=sa.text("status = 'ACTIVE'"),
            using="gist",
            name="ex_waitlist_entries_active_overlap",
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'FULFILLED', 'CANCELLED')",
            name=op.f("ck_waitlist_entries_status_valid"),
        ),
        sa.CheckConstraint(
            "time_preference IN ('ANYTIME', 'MORNING', 'AFTERNOON')",
            name=op.f("ck_waitlist_entries_time_preference_valid"),
        ),
        sa.CheckConstraint("end_date >= start_date", name=op.f("ck_waitlist_entries_dates_valid")),
        sa.CheckConstraint("version >= 0", name=op.f("ck_waitlist_entries_version_non_negative")),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_waitlist_entries_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_waitlist_entries_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("waitlist_id", name=op.f("pk_waitlist_entries")),
        sa.UniqueConstraint(
            "waitlist_id", "patient_id", name=op.f("uq_waitlist_entries_waitlist_id_patient_id")
        ),
    )
    op.create_index(
        "ix_waitlist_entries_matching",
        "waitlist_entries",
        ["status", "clinic_id", "specialty", "created_at", "waitlist_id", "start_date", "end_date"],
        unique=False,
    )
    op.create_index(
        "ix_waitlist_entries_patient_status",
        "waitlist_entries",
        ["patient_id", "status"],
        unique=False,
    )
    op.create_table(
        "doctor_availability_rules",
        sa.Column(
            "availability_id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("doctor_id", sa.Uuid(), nullable=False),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("day_of_week", sa.SmallInteger(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
        sa.Column("effective_start_date", sa.Date(), nullable=False),
        sa.Column("effective_end_date", sa.Date(), nullable=True),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.CheckConstraint(
            "day_of_week BETWEEN 0 AND 6",
            name=op.f("ck_doctor_availability_rules_day_of_week_valid"),
        ),
        sa.CheckConstraint(
            "effective_end_date IS NULL OR effective_end_date >= effective_start_date",
            name=op.f("ck_doctor_availability_rules_effective_dates_valid"),
        ),
        sa.CheckConstraint(
            "start_time < end_time", name=op.f("ck_doctor_availability_rules_window_valid")
        ),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_doctor_availability_rules_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_doctor_availability_rules_doctor_clinic",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id"],
            ["doctors.doctor_id"],
            name=op.f("fk_doctor_availability_rules_doctor_id_doctors"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("availability_id", name=op.f("pk_doctor_availability_rules")),
    )
    op.create_index(
        "ix_doctor_availability_rules_lookup",
        "doctor_availability_rules",
        ["doctor_id", "clinic_id", "day_of_week", "effective_start_date", "effective_end_date"],
        unique=False,
    )
    op.create_table(
        "doctor_leave",
        sa.Column(
            "leave_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("doctor_id", sa.Uuid(), nullable=False),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason_code", sa.String(length=40), nullable=False),
        sa.CheckConstraint("ends_at > starts_at", name=op.f("ck_doctor_leave_period_valid")),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_doctor_leave_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_doctor_leave_doctor_clinic",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id"],
            ["doctors.doctor_id"],
            name=op.f("fk_doctor_leave_doctor_id_doctors"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("leave_id", name=op.f("pk_doctor_leave")),
    )
    op.create_index(
        "ix_doctor_leave_clinic_period",
        "doctor_leave",
        ["clinic_id", "starts_at", "ends_at"],
        unique=False,
    )
    op.create_index(
        "ix_doctor_leave_doctor_period",
        "doctor_leave",
        ["doctor_id", "starts_at", "ends_at"],
        unique=False,
    )
    op.create_table(
        "handoff_tickets",
        sa.Column(
            "ticket_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("created_by_session_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.Text(), server_default="OPEN", nullable=False),
        sa.Column("redacted_summary", sa.Text(), nullable=False),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'RESOLVED') = (resolved_at IS NOT NULL)",
            name=op.f("ck_handoff_tickets_resolution_matches_status"),
        ),
        sa.CheckConstraint(
            "status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED')",
            name=op.f("ck_handoff_tickets_status_valid"),
        ),
        sa.CheckConstraint(
            "char_length(redacted_summary) <= 500",
            name=op.f("ck_handoff_tickets_redacted_summary_length"),
        ),
        sa.CheckConstraint(
            "length(btrim(redacted_summary)) > 0",
            name=op.f("ck_handoff_tickets_redacted_summary_not_blank"),
        ),
        sa.CheckConstraint("version >= 0", name=op.f("ck_handoff_tickets_version_non_negative")),
        sa.ForeignKeyConstraint(
            ["created_by_session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_handoff_tickets_created_by_session_id_auth_sessions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_handoff_tickets_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("ticket_id", name=op.f("pk_handoff_tickets")),
    )
    op.create_index(
        "ix_handoff_tickets_patient_created_at",
        "handoff_tickets",
        ["patient_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_handoff_tickets_resolved_at", "handoff_tickets", ["resolved_at"], unique=False
    )
    op.create_index(
        "ix_handoff_tickets_status_created_at",
        "handoff_tickets",
        ["status", "created_at"],
        unique=False,
    )
    op.create_table(
        "idempotency_records",
        sa.Column(
            "idempotency_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("actor_type", sa.String(length=20), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("operation", sa.String(length=80), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.LargeBinary(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("response_code", sa.SmallInteger(), nullable=True),
        sa.Column("result_type", sa.String(length=40), nullable=True),
        sa.Column("result_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= created_at",
            name=op.f("ck_idempotency_records_completion_follows_creation"),
        ),
        sa.CheckConstraint(
            "length(btrim(actor_type)) > 0",
            name=op.f("ck_idempotency_records_actor_type_not_blank"),
        ),
        sa.CheckConstraint(
            "length(btrim(idempotency_key)) > 0",
            name=op.f("ck_idempotency_records_idempotency_key_not_blank"),
        ),
        sa.CheckConstraint(
            "length(btrim(operation)) > 0", name=op.f("ck_idempotency_records_operation_not_blank")
        ),
        sa.CheckConstraint(
            "length(btrim(status)) > 0", name=op.f("ck_idempotency_records_status_not_blank")
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_idempotency_records_session_id_auth_sessions"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("idempotency_id", name=op.f("pk_idempotency_records")),
        sa.UniqueConstraint(
            "actor_type",
            "actor_id",
            "operation",
            "idempotency_key",
            name=op.f("uq_idempotency_records_scope_key"),
        ),
    )
    op.create_index(
        "ix_idempotency_records_completed_at", "idempotency_records", ["completed_at"], unique=False
    )
    op.create_table(
        "pending_actions",
        sa.Column(
            "pending_action_id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("target_type", sa.String(length=40), nullable=False),
        sa.Column("target_id", sa.Uuid(), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expected_versions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action_type IN ('BOOK', 'RESCHEDULE', 'CANCEL', 'JOIN', 'ACCEPT', 'DECLINE')",
            name=op.f("ck_pending_actions_action_type_valid"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at AND expires_at <= created_at + interval '10 minutes'",
            name=op.f("ck_pending_actions_expiry_window_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'CONSUMED', 'INVALIDATED', 'EXPIRED')",
            name=op.f("ck_pending_actions_status_valid"),
        ),
        sa.CheckConstraint(
            "length(btrim(target_type)) > 0", name=op.f("ck_pending_actions_target_type_not_blank")
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_pending_actions_session_id_auth_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("pending_action_id", name=op.f("pk_pending_actions")),
        sa.UniqueConstraint("request_id", name=op.f("uq_pending_actions_request_id")),
    )
    op.create_index(
        "ix_pending_actions_expires_at", "pending_actions", ["expires_at"], unique=False
    )
    op.create_index(
        "ix_pending_actions_session_status",
        "pending_actions",
        ["session_id", "status"],
        unique=False,
    )
    op.create_table(
        "security_audit_events",
        sa.Column(
            "audit_event_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("actor_type", sa.String(length=20), nullable=True),
        sa.Column("actor_id", sa.Uuid(), nullable=True),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("resource_type", sa.String(length=40), nullable=True),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("result", sa.String(length=20), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("network_source", sa.String(length=64), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(event_type)) > 0",
            name=op.f("ck_security_audit_events_event_type_not_blank"),
        ),
        sa.CheckConstraint(
            "length(btrim(result)) > 0", name=op.f("ck_security_audit_events_result_not_blank")
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_security_audit_events_session_id_auth_sessions"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("audit_event_id", name=op.f("pk_security_audit_events")),
    )
    op.create_index(
        "ix_security_audit_events_actor_occurred_at",
        "security_audit_events",
        ["actor_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_security_audit_events_occurred_at",
        "security_audit_events",
        ["occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_security_audit_events_request_id", "security_audit_events", ["request_id"], unique=False
    )
    op.create_index(
        "ix_security_audit_events_type_occurred_at",
        "security_audit_events",
        ["event_type", "occurred_at"],
        unique=False,
    )
    op.create_table(
        "slots",
        sa.Column(
            "slot_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("doctor_id", sa.Uuid(), nullable=False),
        sa.Column("clinic_id", sa.Uuid(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="AVAILABLE", nullable=False),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "removed_at IS NULL OR status = 'AVAILABLE'",
            name=op.f("ck_slots_removal_requires_available"),
        ),
        sa.CheckConstraint(
            "status IN ('AVAILABLE', 'HELD', 'BOOKED')", name=op.f("ck_slots_status_valid")
        ),
        sa.CheckConstraint("ends_at > starts_at", name=op.f("ck_slots_period_valid")),
        sa.CheckConstraint("version >= 0", name=op.f("ck_slots_version_non_negative")),
        sa.ForeignKeyConstraint(
            ["clinic_id"],
            ["clinics.clinic_id"],
            name=op.f("fk_slots_clinic_id_clinics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_slots_doctor_clinic",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["doctor_id"],
            ["doctors.doctor_id"],
            name=op.f("fk_slots_doctor_id_doctors"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("slot_id", name=op.f("pk_slots")),
        sa.UniqueConstraint("doctor_id", "starts_at", name=op.f("uq_slots_doctor_id_starts_at")),
    )
    op.create_index(
        "ix_slots_bookable_starts_at",
        "slots",
        ["starts_at"],
        unique=False,
        postgresql_where=sa.text("status = 'AVAILABLE' AND removed_at IS NULL"),
    )
    op.create_index("ix_slots_clinic_starts_at", "slots", ["clinic_id", "starts_at"], unique=False)
    op.create_index("ix_slots_status_starts_at", "slots", ["status", "starts_at"], unique=False)
    op.create_table(
        "appointments",
        sa.Column(
            "appointment_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("slot_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.Text(), server_default="CONFIRMED", nullable=False),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(status = 'CANCELLED') = (cancelled_at IS NOT NULL)",
            name=op.f("ck_appointments_cancellation_matches_status"),
        ),
        sa.CheckConstraint(
            "status IN ('CONFIRMED', 'CANCELLED')", name=op.f("ck_appointments_status_valid")
        ),
        sa.CheckConstraint("version >= 0", name=op.f("ck_appointments_version_non_negative")),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_appointments_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["slot_id"],
            ["slots.slot_id"],
            name=op.f("fk_appointments_slot_id_slots"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("appointment_id", name=op.f("pk_appointments")),
    )
    op.create_index(
        "ix_appointments_patient_status", "appointments", ["patient_id", "status"], unique=False
    )
    op.create_index("ix_appointments_slot_id", "appointments", ["slot_id"], unique=False)
    op.create_index(
        "uq_appointments_confirmed_slot",
        "appointments",
        ["slot_id"],
        unique=True,
        postgresql_where=sa.text("status = 'CONFIRMED'"),
    )
    op.create_table(
        "handoff_notes",
        sa.Column(
            "note_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("staff_id", sa.Uuid(), nullable=False),
        sa.Column("note_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "length(btrim(note_text)) > 0", name=op.f("ck_handoff_notes_note_text_not_blank")
        ),
        sa.ForeignKeyConstraint(
            ["staff_id"],
            ["staff_users.staff_id"],
            name=op.f("fk_handoff_notes_staff_id_staff_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["ticket_id"],
            ["handoff_tickets.ticket_id"],
            name=op.f("fk_handoff_notes_ticket_id_handoff_tickets"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("note_id", name=op.f("pk_handoff_notes")),
        sa.UniqueConstraint("request_id", name=op.f("uq_handoff_notes_request_id")),
    )
    op.create_index(
        "ix_handoff_notes_ticket_created_at",
        "handoff_notes",
        ["ticket_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "slot_offers",
        sa.Column(
            "offer_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("waitlist_id", sa.Uuid(), nullable=False),
        sa.Column("slot_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'ACCEPTED', 'DECLINED', 'EXPIRED')",
            name=op.f("ck_slot_offers_status_valid"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at", name=op.f("ck_slot_offers_expiry_follows_creation")
        ),
        sa.CheckConstraint("version >= 0", name=op.f("ck_slot_offers_version_non_negative")),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_slot_offers_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["slot_id"],
            ["slots.slot_id"],
            name=op.f("fk_slot_offers_slot_id_slots"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["waitlist_id", "patient_id"],
            ["waitlist_entries.waitlist_id", "waitlist_entries.patient_id"],
            name="fk_slot_offers_waitlist_patient",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("offer_id", name=op.f("pk_slot_offers")),
        sa.UniqueConstraint(
            "patient_id", "slot_id", name=op.f("uq_slot_offers_patient_id_slot_id")
        ),
    )
    op.create_index(
        "ix_slot_offers_active_expires_at",
        "slot_offers",
        ["expires_at"],
        unique=False,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    op.create_index(
        "ix_slot_offers_patient_status", "slot_offers", ["patient_id", "status"], unique=False
    )
    op.create_index(
        "ix_slot_offers_slot_status", "slot_offers", ["slot_id", "status"], unique=False
    )
    op.create_index(
        "ix_slot_offers_waitlist_created_at",
        "slot_offers",
        ["waitlist_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "uq_slot_offers_active_patient",
        "slot_offers",
        ["patient_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    op.create_index(
        "uq_slot_offers_active_slot",
        "slot_offers",
        ["slot_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    op.create_table(
        "appointment_history",
        sa.Column(
            "event_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("appointment_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=True),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("old_slot_id", sa.Uuid(), nullable=True),
        sa.Column("new_slot_id", sa.Uuid(), nullable=True),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "action IN ('BOOKED', 'RESCHEDULED', 'CANCELLED')",
            name=op.f("ck_appointment_history_action_valid"),
        ),
        sa.CheckConstraint(
            "actor_type = 'SYSTEM' OR actor_id IS NOT NULL",
            name=op.f("ck_appointment_history_actor_identified"),
        ),
        sa.CheckConstraint(
            "actor_type IN ('PATIENT', 'STAFF', 'SYSTEM')",
            name=op.f("ck_appointment_history_actor_type_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["appointment_id"],
            ["appointments.appointment_id"],
            name=op.f("fk_appointment_history_appointment_id_appointments"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["new_slot_id"],
            ["slots.slot_id"],
            name=op.f("fk_appointment_history_new_slot_id_slots"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["old_slot_id"],
            ["slots.slot_id"],
            name=op.f("fk_appointment_history_old_slot_id_slots"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_appointment_history")),
        sa.UniqueConstraint(
            "appointment_id",
            "request_id",
            name=op.f("uq_appointment_history_appointment_id_request_id"),
        ),
    )
    op.create_index(
        "ix_appointment_history_actor_timestamp",
        "appointment_history",
        ["actor_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_appointment_history_appointment_timestamp",
        "appointment_history",
        ["appointment_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_appointment_history_request_id", "appointment_history", ["request_id"], unique=False
    )
    op.create_table(
        "notification_events",
        sa.Column(
            "notification_id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("offer_id", sa.Uuid(), nullable=True),
        sa.Column("notification_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.Text(), server_default="UNREAD", nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "(status = 'READ') = (read_at IS NOT NULL)",
            name=op.f("ck_notification_events_read_matches_status"),
        ),
        sa.CheckConstraint(
            "status IN ('UNREAD', 'READ')", name=op.f("ck_notification_events_status_valid")
        ),
        sa.CheckConstraint(
            "length(btrim(notification_type)) > 0",
            name=op.f("ck_notification_events_notification_type_not_blank"),
        ),
        sa.ForeignKeyConstraint(
            ["offer_id"],
            ["slot_offers.offer_id"],
            name=op.f("fk_notification_events_offer_id_slot_offers"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_notification_events_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("notification_id", name=op.f("pk_notification_events")),
        sa.UniqueConstraint("request_id", name=op.f("uq_notification_events_request_id")),
    )
    op.create_index(
        "ix_notification_events_created_at", "notification_events", ["created_at"], unique=False
    )
    op.create_index(
        "ix_notification_events_offer_id", "notification_events", ["offer_id"], unique=False
    )
    op.create_index(
        "ix_notification_events_patient_status_created_at",
        "notification_events",
        ["patient_id", "status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    for table in (
        "notification_events",
        "appointment_history",
        "slot_offers",
        "handoff_notes",
        "appointments",
        "slots",
        "security_audit_events",
        "pending_actions",
        "idempotency_records",
        "handoff_tickets",
        "doctor_leave",
        "doctor_availability_rules",
        "waitlist_entries",
        "staff_credentials",
        "patient_credentials",
        "doctor_clinics",
        "clinic_operating_hours",
        "clinic_holidays",
        "auth_sessions",
        "staff_users",
        "patients",
        "outbox_events",
        "doctors",
        "clinics",
    ):
        op.drop_table(table)

    op.execute("DROP EXTENSION IF EXISTS btree_gist")
