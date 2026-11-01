"""Public clinic and doctor directory (backs the search_clinic_directory tool).

Only active clinics, active doctors, and active assignments are visible. Results contain public
administrative details only.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, Connection, and_, func, or_, select

from app.core.errors import InvalidArgumentError
from app.db.models import Clinic, ClinicOperatingHours, Doctor, DoctorClinic
from app.models.scheduling import (
    ClinicSummary,
    DirectoryClinic,
    DirectoryDoctor,
    DirectoryPage,
    OperatingHours,
)
from app.services.pagination import decode_cursor, encode_cursor, fingerprint
from app.services.search_text import contains_all, normalize_specialty, search_tokens
from app.services.timeutil import WEEKDAYS

CURSOR_KIND = "directory"
CLINIC, DOCTOR = "clinic", "doctor"


@dataclass(frozen=True)
class _Filters:
    clinic_id: uuid.UUID | None
    clinic_tokens: list[str]
    doctor_tokens: list[str]
    specialty: str | None

    @property
    def narrows_doctors(self) -> bool:
        return bool(self.doctor_tokens or self.specialty)

    def clinic_conditions(self) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = [Clinic.active]
        if self.clinic_id:
            conditions.append(Clinic.clinic_id == self.clinic_id)
        conditions += contains_all(Clinic.name, self.clinic_tokens)
        return conditions

    def doctor_conditions(self) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = [Doctor.active]
        conditions += contains_all(Doctor.display_name, self.doctor_tokens)
        if self.specialty:
            conditions.append(func.lower(Doctor.specialty) == self.specialty)
        return conditions


def search_directory(
    connection: Connection,
    *,
    clinic_name: str | None,
    doctor_name: str | None,
    specialty: str | None,
    clinic_id: uuid.UUID | None,
    browse: bool,
    cursor: str | None,
    limit: int,
) -> DirectoryPage:
    if not (clinic_name or doctor_name or specialty or clinic_id or browse):
        raise InvalidArgumentError(
            "Provide at least one search filter or request a directory browse."
        )
    filters = _Filters(
        clinic_id=clinic_id,
        clinic_tokens=search_tokens(clinic_name) if clinic_name else [],
        doctor_tokens=search_tokens(doctor_name, drop_titles=True) if doctor_name else [],
        specialty=normalize_specialty(specialty) if specialty else None,
    )
    search = fingerprint(
        clinic_id=clinic_id,
        clinic=filters.clinic_tokens,
        doctor=filters.doctor_tokens,
        specialty=filters.specialty,
    )
    position = decode_cursor(cursor, CURSOR_KIND, search) if cursor else None

    assignments = (
        select(DoctorClinic.clinic_id, DoctorClinic.doctor_id)
        .join(Doctor, Doctor.doctor_id == DoctorClinic.doctor_id)
        .join(Clinic, Clinic.clinic_id == DoctorClinic.clinic_id)
        .where(DoctorClinic.active, *filters.clinic_conditions(), *filters.doctor_conditions())
        .subquery()
    )
    wanted = limit + 1
    clinics: list = []
    doctors: list = []
    if position is None or position.get("k") == CLINIC:
        clinics = _clinic_rows(connection, filters, assignments, position, wanted)
    if len(clinics) < wanted:
        doctor_position = position if position and position.get("k") == DOCTOR else None
        doctors = _doctor_rows(
            connection, filters, assignments, doctor_position, wanted - len(clinics)
        )

    ordered = [(CLINIC, row) for row in clinics] + [(DOCTOR, row) for row in doctors]
    page = ordered[:limit]
    next_cursor = None
    if len(ordered) > limit:
        kind, last = page[-1]
        next_cursor = encode_cursor(
            CURSOR_KIND, search, {"k": kind, "n": last.sort_key, "i": str(last.id)}
        )
    page_clinics = [row for kind, row in page if kind == CLINIC]
    page_doctors = [row for kind, row in page if kind == DOCTOR]
    return DirectoryPage(
        clinics=_clinic_models(connection, page_clinics),
        doctors=_doctor_models(connection, page_doctors),
        next_cursor=next_cursor,
    )


def _after(sort_key, id_column, position: dict[str, str] | None) -> list[ColumnElement[bool]]:
    if not position:
        return []
    try:
        last_id = uuid.UUID(position["i"])
        last_name = position["n"]
    except (KeyError, ValueError):
        raise InvalidArgumentError("The pagination cursor is not valid for this search.") from None
    return [or_(sort_key > last_name, and_(sort_key == last_name, id_column > last_id))]


def _clinic_rows(connection, filters, assignments, position, count):
    sort_key = func.lower(Clinic.name)
    statement = (
        select(
            Clinic.clinic_id,
            Clinic.name,
            Clinic.timezone,
            Clinic.address_line,
            Clinic.city,
            Clinic.region,
            Clinic.postal_code,
            sort_key.label("sort_key"),
            Clinic.clinic_id.label("id"),
        )
        .where(*filters.clinic_conditions(), *_after(sort_key, Clinic.clinic_id, position))
        .order_by(sort_key, Clinic.clinic_id)
        .limit(count)
    )
    if filters.narrows_doctors:
        statement = statement.where(Clinic.clinic_id.in_(select(assignments.c.clinic_id)))
    return connection.execute(statement).all()


def _doctor_rows(connection, filters, assignments, position, count):
    sort_key = func.lower(Doctor.display_name)
    statement = (
        select(
            Doctor.doctor_id,
            Doctor.display_name,
            Doctor.specialty,
            sort_key.label("sort_key"),
            Doctor.doctor_id.label("id"),
        )
        .where(
            *filters.doctor_conditions(),
            Doctor.doctor_id.in_(select(assignments.c.doctor_id)),
            *_after(sort_key, Doctor.doctor_id, position),
        )
        .order_by(sort_key, Doctor.doctor_id)
        .limit(count)
    )
    return connection.execute(statement).all()


def _clinic_models(connection: Connection, rows: list) -> list[DirectoryClinic]:
    if not rows:
        return []
    ids = [row.id for row in rows]
    hours: dict[uuid.UUID, list[OperatingHours]] = {clinic_id: [] for clinic_id in ids}
    for entry in connection.execute(
        select(ClinicOperatingHours)
        .where(ClinicOperatingHours.clinic_id.in_(ids))
        .order_by(
            ClinicOperatingHours.clinic_id,
            ClinicOperatingHours.day_of_week,
            ClinicOperatingHours.opens_at,
        )
    ):
        hours[entry.clinic_id].append(
            OperatingHours(
                day_of_week=entry.day_of_week,
                weekday=WEEKDAYS[entry.day_of_week],
                opens_at=entry.opens_at.strftime("%H:%M"),
                closes_at=entry.closes_at.strftime("%H:%M"),
            )
        )
    specialties: dict[uuid.UUID, list[str]] = {clinic_id: [] for clinic_id in ids}
    for clinic_id, specialty in connection.execute(
        select(DoctorClinic.clinic_id, Doctor.specialty)
        .join(Doctor, Doctor.doctor_id == DoctorClinic.doctor_id)
        .where(DoctorClinic.clinic_id.in_(ids), DoctorClinic.active, Doctor.active)
        .distinct()
        .order_by(DoctorClinic.clinic_id, Doctor.specialty)
    ):
        specialties[clinic_id].append(specialty)
    return [
        DirectoryClinic(
            clinic_id=row.clinic_id,
            name=row.name,
            timezone=row.timezone,
            address_line=row.address_line,
            city=row.city,
            region=row.region,
            postal_code=row.postal_code,
            hours=hours[row.id],
            specialties=specialties[row.id],
        )
        for row in rows
    ]


def _doctor_models(connection: Connection, rows: list) -> list[DirectoryDoctor]:
    if not rows:
        return []
    ids = [row.id for row in rows]
    clinics: dict[uuid.UUID, list[ClinicSummary]] = {doctor_id: [] for doctor_id in ids}
    for entry in connection.execute(
        select(DoctorClinic.doctor_id, Clinic.clinic_id, Clinic.name, Clinic.timezone)
        .join(Clinic, Clinic.clinic_id == DoctorClinic.clinic_id)
        .where(DoctorClinic.doctor_id.in_(ids), DoctorClinic.active, Clinic.active)
        .order_by(func.lower(Clinic.name), Clinic.clinic_id)
    ):
        clinics[entry.doctor_id].append(
            ClinicSummary(clinic_id=entry.clinic_id, name=entry.name, timezone=entry.timezone)
        )
    return [
        DirectoryDoctor(
            doctor_id=row.doctor_id,
            display_name=row.display_name,
            specialty=row.specialty,
            clinics=clinics[row.id],
        )
        for row in rows
    ]
