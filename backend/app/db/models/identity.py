import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, not_blank, one_of, updated_at, uuid_pk
from app.db.enums import ActorType, PendingActionStatus, PendingActionType, SessionStatus


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[uuid_pk]
    patient_code: Mapped[str] = mapped_column(String(32))
    display_name: Mapped[str] = mapped_column(String(120))
    active: Mapped[bool] = mapped_column(server_default=text("true"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        UniqueConstraint("patient_code"),
        not_blank("patient_code"),
        not_blank("display_name"),
        Index("ix_patients_display_name_normalized", text("lower(display_name)")),
    )


class StaffUser(Base):
    __tablename__ = "staff_users"

    staff_id: Mapped[uuid_pk]
    username: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(120))
    active: Mapped[bool] = mapped_column(server_default=text("true"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        not_blank("username"),
        not_blank("display_name"),
        Index("uq_staff_users_username_normalized", text("lower(username)"), unique=True),
        Index("ix_staff_users_active", "active"),
    )


class PatientCredential(Base):
    __tablename__ = "patient_credentials"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT"), primary_key=True
    )
    pin_hash: Mapped[str] = mapped_column(Text)
    failed_attempt_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    failed_attempt_window_started_at: Mapped[datetime | None]
    locked_until: Mapped[datetime | None]
    disabled: Mapped[bool] = mapped_column(server_default=text("false"))
    updated_at: Mapped[updated_at]

    __table_args__ = (
        not_blank("pin_hash"),
        CheckConstraint("failed_attempt_count >= 0", name="failed_attempt_count_non_negative"),
    )


class StaffCredential(Base):
    __tablename__ = "staff_credentials"

    staff_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("staff_users.staff_id", ondelete="RESTRICT"), primary_key=True
    )
    password_hash: Mapped[str] = mapped_column(Text)
    failed_attempt_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    failed_attempt_window_started_at: Mapped[datetime | None]
    locked_until: Mapped[datetime | None]
    reset_required: Mapped[bool] = mapped_column(server_default=text("false"))
    updated_at: Mapped[updated_at]

    __table_args__ = (
        not_blank("password_hash"),
        CheckConstraint("failed_attempt_count >= 0", name="failed_attempt_count_non_negative"),
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    session_id: Mapped[uuid_pk]
    actor_type: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    staff_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("staff_users.staff_id", ondelete="RESTRICT")
    )
    token_hash: Mapped[bytes] = mapped_column(LargeBinary)
    status: Mapped[str] = mapped_column(Text, server_default=SessionStatus.ACTIVE.value)
    created_at: Mapped[created_at]
    last_activity_at: Mapped[datetime] = mapped_column(server_default=text("now()"))
    idle_expires_at: Mapped[datetime]
    absolute_expires_at: Mapped[datetime]
    terminated_at: Mapped[datetime | None]
    termination_reason: Mapped[str | None] = mapped_column(String(40))

    __table_args__ = (
        UniqueConstraint("token_hash"),
        one_of("actor_type", ActorType),
        one_of("role", ActorType),
        one_of("status", SessionStatus),
        CheckConstraint("role = actor_type", name="role_matches_actor"),
        CheckConstraint(
            "(actor_type = 'PATIENT' AND patient_id IS NOT NULL AND staff_id IS NULL)"
            " OR (actor_type = 'STAFF' AND staff_id IS NOT NULL AND patient_id IS NULL)",
            name="exactly_one_actor",
        ),
        CheckConstraint(
            "last_activity_at >= created_at"
            " AND idle_expires_at > created_at"
            " AND absolute_expires_at > created_at",
            name="deadlines_follow_creation",
        ),
        CheckConstraint(
            "(status = 'ACTIVE') = (terminated_at IS NULL)", name="termination_matches_status"
        ),
        Index(
            "ix_auth_sessions_patient_active",
            "patient_id",
            "created_at",
            postgresql_where=text("status = 'ACTIVE' AND patient_id IS NOT NULL"),
        ),
        Index(
            "uq_auth_sessions_staff_active",
            "staff_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE' AND staff_id IS NOT NULL"),
        ),
        Index("ix_auth_sessions_idle_expires_at", "idle_expires_at"),
        Index("ix_auth_sessions_absolute_expires_at", "absolute_expires_at"),
    )


class PendingAction(Base):
    __tablename__ = "pending_actions"

    pending_action_id: Mapped[uuid_pk]
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("auth_sessions.session_id", ondelete="CASCADE")
    )
    action_type: Mapped[str] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[uuid.UUID | None]
    summary: Mapped[dict[str, Any]]
    expected_versions: Mapped[dict[str, Any]]
    status: Mapped[str] = mapped_column(Text, server_default=PendingActionStatus.ACTIVE.value)
    expires_at: Mapped[datetime]
    request_id: Mapped[uuid.UUID]
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        UniqueConstraint("request_id"),
        one_of("action_type", PendingActionType),
        one_of("status", PendingActionStatus),
        not_blank("target_type"),
        CheckConstraint(
            "expires_at > created_at AND expires_at <= created_at + interval '10 minutes'",
            name="expiry_window_valid",
        ),
        Index("ix_pending_actions_session_status", "session_id", "status"),
        Index("ix_pending_actions_expires_at", "expires_at"),
    )
