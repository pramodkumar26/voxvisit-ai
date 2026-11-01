import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, one_of, row_version, updated_at, uuid_pk
from app.db.enums import AppointmentAction, AppointmentStatus, HistoryActorType


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id: Mapped[uuid_pk]
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    slot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("slots.slot_id", ondelete="RESTRICT"))
    status: Mapped[str] = mapped_column(Text, server_default=AppointmentStatus.CONFIRMED.value)
    version: Mapped[row_version]
    cancelled_at: Mapped[datetime | None]
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        one_of("status", AppointmentStatus),
        CheckConstraint(
            "(status = 'CANCELLED') = (cancelled_at IS NOT NULL)",
            name="cancellation_matches_status",
        ),
        CheckConstraint("version >= 0", name="version_non_negative"),
        Index("ix_appointments_patient_status", "patient_id", "status"),
        Index("ix_appointments_slot_id", "slot_id"),
        Index(
            "uq_appointments_confirmed_slot",
            "slot_id",
            unique=True,
            postgresql_where=text("status = 'CONFIRMED'"),
        ),
    )


class AppointmentHistory(Base):
    __tablename__ = "appointment_history"

    event_id: Mapped[uuid_pk]
    appointment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("appointments.appointment_id", ondelete="RESTRICT")
    )
    action: Mapped[str] = mapped_column(Text)
    actor_id: Mapped[uuid.UUID | None]
    actor_type: Mapped[str] = mapped_column(Text)
    old_slot_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("slots.slot_id", ondelete="RESTRICT")
    )
    new_slot_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("slots.slot_id", ondelete="RESTRICT")
    )
    timestamp: Mapped[datetime] = mapped_column(server_default=text("now()"))
    request_id: Mapped[uuid.UUID]

    __table_args__ = (
        UniqueConstraint("appointment_id", "request_id"),
        one_of("action", AppointmentAction),
        one_of("actor_type", HistoryActorType),
        CheckConstraint("actor_type = 'SYSTEM' OR actor_id IS NOT NULL", name="actor_identified"),
        Index("ix_appointment_history_appointment_timestamp", "appointment_id", "timestamp"),
        Index("ix_appointment_history_request_id", "request_id"),
        Index("ix_appointment_history_actor_timestamp", "actor_id", "timestamp"),
    )
