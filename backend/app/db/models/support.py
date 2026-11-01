import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, not_blank, one_of, row_version, updated_at, uuid_pk
from app.db.enums import HandoffCategory, HandoffStatus, NotificationStatus


class HandoffTicket(Base):
    __tablename__ = "handoff_tickets"

    ticket_id: Mapped[uuid_pk]
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    created_by_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("auth_sessions.session_id", ondelete="SET NULL")
    )
    category: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default=HandoffStatus.OPEN.value)
    redacted_summary: Mapped[str] = mapped_column(Text)
    version: Mapped[row_version]
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]
    resolved_at: Mapped[datetime | None]

    __table_args__ = (
        one_of("category", HandoffCategory),
        one_of("status", HandoffStatus),
        not_blank("redacted_summary"),
        CheckConstraint("char_length(redacted_summary) <= 500", name="redacted_summary_length"),
        CheckConstraint(
            "(status = 'RESOLVED') = (resolved_at IS NOT NULL)", name="resolution_matches_status"
        ),
        CheckConstraint("version >= 0", name="version_non_negative"),
        Index("ix_handoff_tickets_status_created_at", "status", "created_at"),
        Index("ix_handoff_tickets_patient_created_at", "patient_id", "created_at"),
        Index("ix_handoff_tickets_resolved_at", "resolved_at"),
    )


class HandoffNote(Base):
    __tablename__ = "handoff_notes"

    note_id: Mapped[uuid_pk]
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("handoff_tickets.ticket_id", ondelete="RESTRICT")
    )
    staff_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("staff_users.staff_id", ondelete="RESTRICT")
    )
    note_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at]
    request_id: Mapped[uuid.UUID]

    __table_args__ = (
        UniqueConstraint("request_id"),
        not_blank("note_text"),
        Index("ix_handoff_notes_ticket_created_at", "ticket_id", "created_at"),
    )


class NotificationEvent(Base):
    __tablename__ = "notification_events"

    notification_id: Mapped[uuid_pk]
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    offer_id: Mapped[uuid.UUID | None]
    notification_type: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(Text, server_default=NotificationStatus.UNREAD.value)
    payload: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    created_at: Mapped[created_at]
    read_at: Mapped[datetime | None]
    request_id: Mapped[uuid.UUID]

    __table_args__ = (
        ForeignKeyConstraint(
            ["offer_id", "patient_id"],
            ["slot_offers.offer_id", "slot_offers.patient_id"],
            name="fk_notification_events_offer_patient",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("request_id"),
        one_of("status", NotificationStatus),
        not_blank("notification_type"),
        CheckConstraint("(status = 'READ') = (read_at IS NOT NULL)", name="read_matches_status"),
        Index(
            "ix_notification_events_patient_status_created_at", "patient_id", "status", "created_at"
        ),
        Index("ix_notification_events_offer_id", "offer_id"),
        Index("ix_notification_events_created_at", "created_at"),
    )
