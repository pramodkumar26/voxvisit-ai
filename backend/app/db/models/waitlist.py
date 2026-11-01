import uuid
from datetime import date, datetime

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
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, one_of, row_version, updated_at, uuid_pk
from app.db.enums import OfferStatus, TimePreference, WaitlistStatus


class WaitlistEntry(Base):
    __tablename__ = "waitlist_entries"

    waitlist_id: Mapped[uuid_pk]
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    specialty: Mapped[str] = mapped_column(String(80))
    start_date: Mapped[date]
    end_date: Mapped[date]
    time_preference: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default=WaitlistStatus.ACTIVE.value)
    version: Mapped[row_version]
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        UniqueConstraint("waitlist_id", "patient_id"),
        ExcludeConstraint(
            ("patient_id", "="),
            ("clinic_id", "="),
            ("specialty", "="),
            (text("daterange(start_date, end_date, '[]')"), "&&"),
            using="gist",
            where=text("status = 'ACTIVE'"),
            name="ex_waitlist_entries_active_overlap",
        ),
        one_of("status", WaitlistStatus),
        one_of("time_preference", TimePreference),
        CheckConstraint("end_date >= start_date", name="dates_valid"),
        CheckConstraint("version >= 0", name="version_non_negative"),
        Index(
            "ix_waitlist_entries_matching",
            "status",
            "clinic_id",
            "specialty",
            "created_at",
            "waitlist_id",
            "start_date",
            "end_date",
        ),
        Index("ix_waitlist_entries_patient_status", "patient_id", "status"),
    )


class SlotOffer(Base):
    __tablename__ = "slot_offers"

    offer_id: Mapped[uuid_pk]
    waitlist_id: Mapped[uuid.UUID]
    slot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("slots.slot_id", ondelete="RESTRICT"))
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.patient_id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(Text, server_default=OfferStatus.ACTIVE.value)
    created_at: Mapped[datetime] = mapped_column(server_default=text("clock_timestamp()"))
    expires_at: Mapped[datetime]
    responded_at: Mapped[datetime | None]
    version: Mapped[row_version]
    request_id: Mapped[uuid.UUID]

    __table_args__ = (
        ForeignKeyConstraint(
            ["waitlist_id", "patient_id"],
            ["waitlist_entries.waitlist_id", "waitlist_entries.patient_id"],
            name="fk_slot_offers_waitlist_patient",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("patient_id", "slot_id"),
        UniqueConstraint("offer_id", "patient_id"),
        one_of("status", OfferStatus),
        CheckConstraint("expires_at > created_at", name="expiry_follows_creation"),
        CheckConstraint("version >= 0", name="version_non_negative"),
        Index(
            "uq_slot_offers_active_slot",
            "slot_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        Index(
            "uq_slot_offers_active_patient",
            "patient_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        Index(
            "ix_slot_offers_active_expires_at",
            "expires_at",
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        Index("ix_slot_offers_waitlist_created_at", "waitlist_id", "created_at"),
        Index("ix_slot_offers_patient_status", "patient_id", "status"),
        Index("ix_slot_offers_slot_status", "slot_id", "status"),
    )
