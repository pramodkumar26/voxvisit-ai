import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from argon2 import PasswordHasher
from sqlalchemy import Connection, Engine, exists, insert, select, tuple_, update

from app.db.models import (
    Appointment,
    AppointmentHistory,
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    HandoffNote,
    HandoffTicket,
    Patient,
    PatientCredential,
    Slot,
    StaffCredential,
    StaffUser,
    WaitlistEntry,
)
from app.seed.credentials import (
    create_hasher,
    validate_patient_pin,
    validate_staff_password,
)
from app.seed.plan import AppointmentPlan, SeedPlan, build_plan

logger = logging.getLogger(__name__)
CHUNK_SIZE = 500


@dataclass
class TableResult:
    inserted: int = 0
    existing: int = 0


@dataclass
class SeedResult:
    tables: dict[str, TableResult] = field(default_factory=dict)
    skipped_appointments: list[str] = field(default_factory=list)
    committed: bool = False

    def record(self, table: str, inserted: int, existing: int) -> None:
        self.tables[table] = TableResult(inserted, existing)

    @property
    def inserted_total(self) -> int:
        return sum(table.inserted for table in self.tables.values())


def seed_database(
    engine: Engine,
    *,
    patient_pin: str,
    staff_password: str,
    anchor_date: date | None = None,
    now: datetime | None = None,
    hasher: PasswordHasher | None = None,
    dry_run: bool = False,
) -> SeedResult:
    """Insert any missing seed records in one transaction. Existing rows are never modified."""
    validate_patient_pin(patient_pin)
    validate_staff_password(staff_password)
    plan = build_plan(anchor_date, now)
    result = SeedResult()

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            _load(connection, plan, result, patient_pin, staff_password, hasher or create_hasher())
            if dry_run:
                transaction.rollback()
            else:
                transaction.commit()
                result.committed = True
        except BaseException:
            if transaction.is_active:
                transaction.rollback()
            raise
    return result


def _load(
    connection: Connection,
    plan: SeedPlan,
    result: SeedResult,
    patient_pin: str,
    staff_password: str,
    hasher: PasswordHasher,
) -> None:
    for model, rows, key in (
        (Clinic, plan.clinics, "clinic_id"),
        (ClinicOperatingHours, plan.clinic_hours, "hours_id"),
        (ClinicHoliday, plan.clinic_holidays, "holiday_id"),
        (Doctor, plan.doctors, "doctor_id"),
    ):
        _insert_missing(connection, result, model, rows, (key,))
    _insert_missing(
        connection, result, DoctorClinic, plan.doctor_clinics, ("doctor_id", "clinic_id")
    )
    _insert_missing(
        connection, result, DoctorAvailabilityRule, plan.availability_rules, ("availability_id",)
    )
    _insert_missing(connection, result, DoctorLeave, plan.doctor_leave, ("leave_id",))
    _insert_missing(connection, result, Slot, plan.slot_rows(), ("slot_id",))
    _insert_missing(connection, result, Patient, plan.patients, ("patient_id",))
    _insert_missing(connection, result, StaffUser, plan.staff_users, ("staff_id",))
    _insert_missing(connection, result, WaitlistEntry, plan.waitlist_entries, ("waitlist_id",))
    _insert_missing(connection, result, HandoffTicket, plan.handoff_tickets, ("ticket_id",))
    _insert_missing(connection, result, HandoffNote, plan.handoff_notes, ("note_id",))
    _load_appointments(connection, plan, result)
    _load_credentials(
        connection,
        result,
        PatientCredential,
        key="patient_id",
        accounts=plan.patients,
        hash_column="pin_hash",
        secret=patient_pin,
        hasher=hasher,
    )
    _load_credentials(
        connection,
        result,
        StaffCredential,
        key="staff_id",
        accounts=plan.staff_users,
        hash_column="password_hash",
        secret=staff_password,
        hasher=hasher,
    )


def _chunks(items: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), CHUNK_SIZE):
        yield items[start : start + CHUNK_SIZE]


def _existing_keys(
    connection: Connection, model: type, rows: Sequence[dict[str, Any]], key: tuple[str, ...]
) -> set[tuple[Any, ...]]:
    columns = [model.__table__.c[name] for name in key]
    existing: set[tuple[Any, ...]] = set()
    for chunk in _chunks(rows):
        wanted = [tuple(row[name] for name in key) for row in chunk]
        condition = (
            columns[0].in_([values[0] for values in wanted])
            if len(columns) == 1
            else tuple_(*columns).in_(wanted)
        )
        existing.update(
            tuple(found) for found in connection.execute(select(*columns).where(condition))
        )
    return existing


def _insert_missing(
    connection: Connection,
    result: SeedResult,
    model: type,
    rows: Sequence[dict[str, Any]],
    key: tuple[str, ...],
) -> None:
    existing = _existing_keys(connection, model, rows, key)
    missing = [row for row in rows if tuple(row[name] for name in key) not in existing]
    if missing:
        connection.execute(insert(model), missing)
    result.record(model.__tablename__, len(missing), len(existing))


def _load_credentials(
    connection: Connection,
    result: SeedResult,
    model: type,
    *,
    key: str,
    accounts: Sequence[dict[str, Any]],
    hash_column: str,
    secret: str,
    hasher: PasswordHasher,
) -> None:
    existing = _existing_keys(connection, model, accounts, (key,))
    rows = [
        {key: account[key], hash_column: hasher.hash(secret)}
        for account in accounts
        if (account[key],) not in existing
    ]
    if rows:
        connection.execute(insert(model), rows)
    result.record(model.__tablename__, len(rows), len(existing))


def _load_appointments(connection: Connection, plan: SeedPlan, result: SeedResult) -> None:
    appointment_rows = [{"appointment_id": item.appointment_id} for item in plan.appointments]
    existing = {
        found[0]
        for found in _existing_keys(connection, Appointment, appointment_rows, ("appointment_id",))
    }
    history_keys = [
        {"event_id": row["event_id"]} for item in plan.appointments for row in item.history_rows()
    ]
    existing_history = _existing_keys(connection, AppointmentHistory, history_keys, ("event_id",))
    inserted_appointments = inserted_history = 0

    for item in plan.appointments:
        if item.appointment_id in existing:
            continue
        if not _slot_is_free(connection, item):
            result.skipped_appointments.append(item.key)
            logger.warning("skipped seed appointment %s: its slot is no longer available", item.key)
            continue
        booked_versions = 2 if item.cancelled else 1
        connection.execute(
            update(Slot)
            .where(Slot.slot_id == item.slot_id)
            .values(
                status="AVAILABLE" if item.cancelled else "BOOKED",
                version=Slot.version + booked_versions,
                updated_at=item.cancelled_at or item.booked_at,
            )
        )
        connection.execute(insert(Appointment), [item.appointment_row()])
        history = item.history_rows()
        connection.execute(insert(AppointmentHistory), _uniform_rows(history))
        inserted_appointments += 1
        inserted_history += len(history)

    result.record("appointments", inserted_appointments, len(existing))
    result.record("appointment_history", inserted_history, len(existing_history))


def _slot_is_free(connection: Connection, item: AppointmentPlan) -> bool:
    slot = connection.execute(
        select(Slot.status, Slot.removed_at).where(Slot.slot_id == item.slot_id).with_for_update()
    ).one_or_none()
    if slot is None or slot.status != "AVAILABLE" or slot.removed_at is not None:
        return False
    return not connection.execute(
        select(
            exists().where(Appointment.slot_id == item.slot_id, Appointment.status == "CONFIRMED")
        )
    ).scalar_one()


def _uniform_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = {name for row in rows for name in row}
    return [{name: row.get(name) for name in columns} for row in rows]
