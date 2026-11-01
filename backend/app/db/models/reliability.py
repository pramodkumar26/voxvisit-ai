import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, not_blank, one_of, uuid_pk
from app.db.enums import IdempotencyActorType, OutboxStatus


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"

    idempotency_id: Mapped[uuid_pk]
    actor_type: Mapped[str] = mapped_column(String(20))
    actor_id: Mapped[uuid.UUID]
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("auth_sessions.session_id", ondelete="SET NULL")
    )
    operation: Mapped[str] = mapped_column(String(80))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[bytes] = mapped_column(LargeBinary)
    status: Mapped[str] = mapped_column(String(20))
    response_code: Mapped[int | None] = mapped_column(SmallInteger)
    result_type: Mapped[str | None] = mapped_column(String(40))
    result_id: Mapped[uuid.UUID | None]
    response_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[created_at]
    completed_at: Mapped[datetime | None]

    __table_args__ = (
        UniqueConstraint(
            "actor_type",
            "actor_id",
            "operation",
            "idempotency_key",
            name="uq_idempotency_records_scope_key",
        ),
        one_of("actor_type", IdempotencyActorType),
        not_blank("operation"),
        not_blank("idempotency_key"),
        not_blank("status"),
        CheckConstraint(
            "response_snapshot IS NULL OR jsonb_typeof(response_snapshot) = 'object'",
            name="response_snapshot_is_object",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= created_at", name="completion_follows_creation"
        ),
        Index("ix_idempotency_records_completed_at", "completed_at"),
    )


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    outbox_event_id: Mapped[uuid_pk]
    event_type: Mapped[str] = mapped_column(String(60))
    aggregate_type: Mapped[str] = mapped_column(String(60))
    aggregate_id: Mapped[uuid.UUID]
    payload: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(Text, server_default=OutboxStatus.PENDING.value)
    dedupe_key: Mapped[str] = mapped_column(String(200))
    attempt_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    available_at: Mapped[datetime] = mapped_column(server_default=text("now()"))
    created_at: Mapped[created_at]
    processed_at: Mapped[datetime | None]
    last_attempt_at: Mapped[datetime | None]
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("dedupe_key"),
        one_of("status", OutboxStatus),
        not_blank("event_type"),
        not_blank("aggregate_type"),
        CheckConstraint("attempt_count >= 0", name="attempt_count_non_negative"),
        CheckConstraint(
            "(status = 'PROCESSED') = (processed_at IS NOT NULL)", name="processed_matches_status"
        ),
        Index("ix_outbox_events_status_available_at", "status", "available_at"),
        Index("ix_outbox_events_aggregate", "aggregate_type", "aggregate_id"),
        Index("ix_outbox_events_created_at", "created_at"),
        Index("ix_outbox_events_processed_at", "processed_at"),
    )


class SecurityAuditEvent(Base):
    __tablename__ = "security_audit_events"

    audit_event_id: Mapped[uuid_pk]
    event_type: Mapped[str] = mapped_column(String(80))
    actor_type: Mapped[str | None] = mapped_column(String(20))
    actor_id: Mapped[uuid.UUID | None]
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("auth_sessions.session_id", ondelete="SET NULL")
    )
    resource_type: Mapped[str | None] = mapped_column(String(40))
    resource_id: Mapped[uuid.UUID | None]
    result: Mapped[str] = mapped_column(String(20))
    occurred_at: Mapped[datetime] = mapped_column(server_default=text("now()"))
    request_id: Mapped[uuid.UUID]
    network_source: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))

    __table_args__ = (
        not_blank("event_type"),
        not_blank("result"),
        Index("ix_security_audit_events_actor_occurred_at", "actor_id", "occurred_at"),
        Index("ix_security_audit_events_type_occurred_at", "event_type", "occurred_at"),
        Index("ix_security_audit_events_request_id", "request_id"),
        Index("ix_security_audit_events_occurred_at", "occurred_at"),
        Index(
            "ix_security_audit_events_network_occurred_at",
            "network_source",
            "occurred_at",
            postgresql_where=text("network_source IS NOT NULL"),
        ),
    )
