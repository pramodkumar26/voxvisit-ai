"""database hardening

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-31 19:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HANDOFF_CATEGORIES = (
    "SCHEDULING",
    "CLINIC_INFORMATION",
    "ACCESSIBILITY",
    "TECHNICAL_SUPPORT",
    "OTHER_ADMINISTRATIVE",
)
IDEMPOTENCY_ACTOR_TYPES = ("PATIENT", "STAFF", "ANONYMOUS")


def quoted(values: Sequence[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def fail_on_conflicts(description: str, query: str) -> None:
    rows = op.get_bind().execute(sa.text(query)).all()
    if rows:
        listed = "; ".join(str(tuple(row)) for row in rows[:10])
        raise RuntimeError(
            f"Migration 0002 stopped: {len(rows)} existing row(s) conflict with {description}. "
            f"No data was changed. First conflicts: {listed}"
        )


def check_existing_data() -> None:
    fail_on_conflicts(
        "the new per-doctor slot overlap constraint (slot_id_a, slot_id_b, doctor_id)",
        """
        SELECT a.slot_id, b.slot_id, a.doctor_id
        FROM slots a
        JOIN slots b
          ON a.doctor_id = b.doctor_id
         AND a.slot_id < b.slot_id
         AND tstzrange(a.starts_at, a.ends_at, '[)') && tstzrange(b.starts_at, b.ends_at, '[)')
        WHERE a.removed_at IS NULL AND b.removed_at IS NULL
        ORDER BY a.doctor_id, a.starts_at
        """,
    )
    fail_on_conflicts(
        "notification ownership (notification_id, notification patient, offer patient)",
        """
        SELECT n.notification_id, n.patient_id, o.patient_id
        FROM notification_events n
        JOIN slot_offers o ON o.offer_id = n.offer_id
        WHERE n.patient_id <> o.patient_id
        """,
    )
    fail_on_conflicts(
        "the approved idempotency actor types (idempotency_id, actor_type)",
        f"""
        SELECT idempotency_id, actor_type
        FROM idempotency_records
        WHERE actor_type NOT IN ({quoted(IDEMPOTENCY_ACTOR_TYPES)})
        """,
    )


def upgrade() -> None:
    check_existing_data()

    op.execute(
        """
        ALTER TABLE slots ADD CONSTRAINT ex_slots_doctor_no_overlap
        EXCLUDE USING gist (doctor_id WITH =, tstzrange(starts_at, ends_at, '[)') WITH &&)
        WHERE (removed_at IS NULL)
        """
    )

    op.create_unique_constraint(
        op.f("uq_slot_offers_offer_id_patient_id"), "slot_offers", ["offer_id", "patient_id"]
    )
    op.drop_constraint(
        op.f("fk_notification_events_offer_id_slot_offers"),
        "notification_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_notification_events_offer_patient",
        "notification_events",
        "slot_offers",
        ["offer_id", "patient_id"],
        ["offer_id", "patient_id"],
        ondelete="RESTRICT",
    )

    # One-time backfill: tickets created before categories existed were never classified.
    op.add_column("handoff_tickets", sa.Column("category", sa.Text(), nullable=True))
    op.execute("UPDATE handoff_tickets SET category = 'OTHER_ADMINISTRATIVE'")
    op.alter_column("handoff_tickets", "category", nullable=False)
    op.create_check_constraint(
        op.f("ck_handoff_tickets_category_valid"),
        "handoff_tickets",
        f"category IN ({quoted(HANDOFF_CATEGORIES)})",
    )

    op.drop_constraint(
        op.f("ck_idempotency_records_actor_type_not_blank"), "idempotency_records", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_idempotency_records_actor_type_valid"),
        "idempotency_records",
        f"actor_type IN ({quoted(IDEMPOTENCY_ACTOR_TYPES)})",
    )
    op.add_column(
        "idempotency_records",
        sa.Column("response_snapshot", postgresql.JSONB(none_as_null=True), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_idempotency_records_response_snapshot_is_object"),
        "idempotency_records",
        "response_snapshot IS NULL OR jsonb_typeof(response_snapshot) = 'object'",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_idempotency_records_response_snapshot_is_object"),
        "idempotency_records",
        type_="check",
    )
    op.drop_column("idempotency_records", "response_snapshot")
    op.drop_constraint(
        op.f("ck_idempotency_records_actor_type_valid"), "idempotency_records", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_idempotency_records_actor_type_not_blank"),
        "idempotency_records",
        "length(btrim(actor_type)) > 0",
    )

    op.drop_constraint(op.f("ck_handoff_tickets_category_valid"), "handoff_tickets", type_="check")
    op.drop_column("handoff_tickets", "category")

    op.drop_constraint(
        "fk_notification_events_offer_patient", "notification_events", type_="foreignkey"
    )
    op.create_foreign_key(
        op.f("fk_notification_events_offer_id_slot_offers"),
        "notification_events",
        "slot_offers",
        ["offer_id"],
        ["offer_id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint(op.f("uq_slot_offers_offer_id_patient_id"), "slot_offers", type_="unique")

    op.execute("ALTER TABLE slots DROP CONSTRAINT ex_slots_doctor_no_overlap")
