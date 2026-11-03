"""review handoffs

Adds the single-use tickets that move a review from a voice session to the patient's browser.
Only the SHA-256 of a ticket's code is stored. No existing object is changed.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-03 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "review_handoffs",
        sa.Column(
            "handoff_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("code_hash", sa.LargeBinary(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("voice_session_id", sa.Uuid(), nullable=False),
        sa.Column("voice_pending_action_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_session_id", sa.Uuid(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("browser_pending_action_id", sa.Uuid(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(claimed_session_id IS NULL) = (claimed_at IS NULL)",
            name=op.f("ck_review_handoffs_claim_is_complete"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at AND expires_at <= created_at + interval '15 minutes'",
            name=op.f("ck_review_handoffs_expiry_window_valid"),
        ),
        sa.CheckConstraint(
            "browser_pending_action_id IS NULL OR claimed_at IS NOT NULL",
            name=op.f("ck_review_handoffs_review_follows_claim"),
        ),
        sa.ForeignKeyConstraint(
            ["browser_pending_action_id"],
            ["pending_actions.pending_action_id"],
            name=op.f("fk_review_handoffs_browser_pending_action_id_pending_actions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["claimed_session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_review_handoffs_claimed_session_id_auth_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name=op.f("fk_review_handoffs_patient_id_patients"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["voice_pending_action_id"],
            ["pending_actions.pending_action_id"],
            name=op.f("fk_review_handoffs_voice_pending_action_id_pending_actions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["voice_session_id"],
            ["auth_sessions.session_id"],
            name=op.f("fk_review_handoffs_voice_session_id_auth_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("handoff_id", name=op.f("pk_review_handoffs")),
        sa.UniqueConstraint("code_hash", name=op.f("uq_review_handoffs_code_hash")),
    )
    op.create_index(
        "ix_review_handoffs_claimed_session_id",
        "review_handoffs",
        ["claimed_session_id"],
        unique=False,
    )
    op.create_index(
        "ix_review_handoffs_voice_session_created_at",
        "review_handoffs",
        ["voice_session_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_review_handoffs_voice_session_created_at", table_name="review_handoffs")
    op.drop_index("ix_review_handoffs_claimed_session_id", table_name="review_handoffs")
    op.drop_table("review_handoffs")
