import uuid
from datetime import date, datetime, time

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at, not_blank, one_of, row_version, updated_at, uuid_pk
from app.db.enums import SlotStatus


class Clinic(Base):
    __tablename__ = "clinics"

    clinic_id: Mapped[uuid_pk]
    name: Mapped[str] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64))
    address_line: Mapped[str] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(100))
    postal_code: Mapped[str] = mapped_column(String(20))
    active: Mapped[bool] = mapped_column(server_default=text("true"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        UniqueConstraint("name"),
        not_blank("name"),
        not_blank("timezone"),
        Index("ix_clinics_active", "active"),
        Index("ix_clinics_name_normalized", text("lower(name)")),
    )


class ClinicOperatingHours(Base):
    __tablename__ = "clinic_operating_hours"

    hours_id: Mapped[uuid_pk]
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    day_of_week: Mapped[int] = mapped_column(SmallInteger)
    opens_at: Mapped[time]
    closes_at: Mapped[time]

    __table_args__ = (
        UniqueConstraint(
            "clinic_id",
            "day_of_week",
            "opens_at",
            "closes_at",
            name="uq_clinic_operating_hours_window",
        ),
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="day_of_week_valid"),
        CheckConstraint("opens_at < closes_at", name="window_valid"),
    )


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id: Mapped[uuid_pk]
    display_name: Mapped[str] = mapped_column(String(120))
    specialty: Mapped[str] = mapped_column(String(80))
    active: Mapped[bool] = mapped_column(server_default=text("true"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        not_blank("display_name"),
        not_blank("specialty"),
        Index("ix_doctors_specialty_active", "specialty", "active"),
        Index("ix_doctors_display_name_normalized", text("lower(display_name)")),
    )


class DoctorClinic(Base):
    __tablename__ = "doctor_clinics"

    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.doctor_id", ondelete="RESTRICT"), primary_key=True
    )
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT"), primary_key=True
    )
    active: Mapped[bool] = mapped_column(server_default=text("true"))

    __table_args__ = (Index("ix_doctor_clinics_clinic_doctor", "clinic_id", "doctor_id"),)


class DoctorAvailabilityRule(Base):
    __tablename__ = "doctor_availability_rules"

    availability_id: Mapped[uuid_pk]
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.doctor_id", ondelete="RESTRICT")
    )
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    day_of_week: Mapped[int] = mapped_column(SmallInteger)
    start_time: Mapped[time]
    end_time: Mapped[time]
    effective_start_date: Mapped[date]
    effective_end_date: Mapped[date | None]
    active: Mapped[bool] = mapped_column(server_default=text("true"))

    __table_args__ = (
        ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_doctor_availability_rules_doctor_clinic",
            ondelete="RESTRICT",
        ),
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="day_of_week_valid"),
        CheckConstraint("start_time < end_time", name="window_valid"),
        CheckConstraint(
            "effective_end_date IS NULL OR effective_end_date >= effective_start_date",
            name="effective_dates_valid",
        ),
        Index(
            "ix_doctor_availability_rules_lookup",
            "doctor_id",
            "clinic_id",
            "day_of_week",
            "effective_start_date",
            "effective_end_date",
        ),
    )


class ClinicHoliday(Base):
    __tablename__ = "clinic_holidays"

    holiday_id: Mapped[uuid_pk]
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    local_date: Mapped[date]
    start_time: Mapped[time | None]
    end_time: Mapped[time | None]
    label: Mapped[str] = mapped_column(String(120))

    __table_args__ = (
        UniqueConstraint(
            "clinic_id",
            "local_date",
            "start_time",
            "end_time",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "(start_time IS NULL AND end_time IS NULL)"
            " OR (start_time IS NOT NULL AND end_time IS NOT NULL AND start_time < end_time)",
            name="window_valid",
        ),
    )


class DoctorLeave(Base):
    __tablename__ = "doctor_leave"

    leave_id: Mapped[uuid_pk]
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.doctor_id", ondelete="RESTRICT")
    )
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    starts_at: Mapped[datetime]
    ends_at: Mapped[datetime]
    reason_code: Mapped[str] = mapped_column(String(40))

    __table_args__ = (
        ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_doctor_leave_doctor_clinic",
            ondelete="RESTRICT",
        ),
        CheckConstraint("ends_at > starts_at", name="period_valid"),
        Index("ix_doctor_leave_doctor_period", "doctor_id", "starts_at", "ends_at"),
        Index("ix_doctor_leave_clinic_period", "clinic_id", "starts_at", "ends_at"),
    )


class Slot(Base):
    __tablename__ = "slots"

    slot_id: Mapped[uuid_pk]
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.doctor_id", ondelete="RESTRICT")
    )
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinics.clinic_id", ondelete="RESTRICT")
    )
    starts_at: Mapped[datetime]
    ends_at: Mapped[datetime]
    status: Mapped[str] = mapped_column(Text, server_default=SlotStatus.AVAILABLE.value)
    version: Mapped[row_version]
    removed_at: Mapped[datetime | None]
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    __table_args__ = (
        ForeignKeyConstraint(
            ["doctor_id", "clinic_id"],
            ["doctor_clinics.doctor_id", "doctor_clinics.clinic_id"],
            name="fk_slots_doctor_clinic",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("doctor_id", "starts_at"),
        ExcludeConstraint(
            ("doctor_id", "="),
            (text("tstzrange(starts_at, ends_at, '[)')"), "&&"),
            using="gist",
            where=text("removed_at IS NULL"),
            name="ex_slots_doctor_no_overlap",
        ),
        one_of("status", SlotStatus),
        CheckConstraint("ends_at > starts_at", name="period_valid"),
        CheckConstraint(
            "removed_at IS NULL OR status = 'AVAILABLE'", name="removal_requires_available"
        ),
        CheckConstraint("version >= 0", name="version_non_negative"),
        Index("ix_slots_clinic_starts_at", "clinic_id", "starts_at"),
        Index("ix_slots_status_starts_at", "status", "starts_at"),
        Index(
            "ix_slots_bookable_starts_at",
            "starts_at",
            postgresql_where=text("status = 'AVAILABLE' AND removed_at IS NULL"),
        ),
    )
