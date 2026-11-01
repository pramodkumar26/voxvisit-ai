"""authentication support

Adds the index that keeps per-network-source login rate limiting cheap on the unauthenticated
login endpoints. No existing object is changed.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31 20:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_security_audit_events_network_occurred_at",
        "security_audit_events",
        ["network_source", "occurred_at"],
        postgresql_where=sa.text("network_source IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_security_audit_events_network_occurred_at",
        table_name="security_audit_events",
        postgresql_where=sa.text("network_source IS NOT NULL"),
    )
