"""Synthetic fixtures for scenarios.

Patients, credentials, and sessions come from the existing deterministic seeder. Scheduling data is
created per scenario with unique names, so scenarios never interfere with each other or with the
seeded schedule. Everything here is fictional.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from argon2 import PasswordHasher
from sqlalchemy import Engine, insert, select, update

from app.db.models import (
    Appointment,
    Clinic,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    Patient,
    PatientCredential,
    Slot,
)

ALL_DAY = (time(0), time(23, 59, 59))
# Cheap Argon2 parameters: these are throwaway synthetic credentials in a throwaway database.
EVAL_HASHER = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)


@dataclass(frozen=True)
class ClinicFixture:
    clinic_id: uuid.UUID
    name: str
    timezone: str
    doctor_ids: list[uuid.UUID]
    doctor_names: list[str]
    slot_ids: list[list[uuid.UUID]]  # Per doctor, in start order.


PLACE_FIRST = (
    "Maple",
    "Cedar",
    "Willow",
    "Harbor",
    "Summit",
    "Meadow",
    "Aspen",
    "Birch",
    "Copper",
    "Juniper",
    "Granite",
    "Lantern",
    "Orchard",
    "Prairie",
    "Quarry",
    "Sable",
    "Timber",
    "Valley",
    "Walnut",
    "Yarrow",
)
PLACE_SECOND = (
    "Grove",
    "Ridge",
    "Heights",
    "Crossing",
    "Hollow",
    "Landing",
    "Park",
    "Point",
    "Terrace",
    "Bluff",
    "Commons",
    "Gate",
    "Fields",
    "Square",
    "Run",
    "Bend",
    "Rise",
    "Glen",
    "Shore",
    "Vista",
)


def _matches_by_search(query: str, name: str) -> bool:
    """Whether directory search for `query` would return `name`: every word of the query is found
    inside the name, as a plain substring (the same rule as app.services.search_text)."""
    haystack = name.casefold()
    return all(word in haystack for word in query.casefold().split())


def unique_place_name(engine: Engine, prefix: str, suffix: str) -> str:
    """A natural-looking clinic name that no other clinic has and that searching for it (or for
    any other generated name) cannot confuse with another, such as "Evaluation Clinic Maple Grove".

    A name made of a hex token invites a model to mistake it for an identifier, so the per-run
    variation is two ordinary words. Names also must not match each other under the directory's
    substring search ("row" is inside "yarrow"), or a search for one clinic would find two. The
    choice starts from the run's suffix and moves on until a name is free of both problems.
    """
    combinations = [(a, b) for a in PLACE_FIRST for b in PLACE_SECOND]
    start = int(suffix, 16) % len(combinations)
    with engine.connect() as connection:
        taken = list(
            connection.execute(select(Clinic.name).where(Clinic.name.like(f"{prefix} %"))).scalars()
        )
    for offset in range(len(combinations)):
        first, second = combinations[(start + offset) % len(combinations)]
        name = f"{prefix} {first} {second}"
        if not any(
            _matches_by_search(name, other) or _matches_by_search(other, name) for other in taken
        ):
            return name
    return f"{prefix} {suffix}"  # Every combination is used: fall back to a unique token.


def local_start(
    now: datetime, timezone: str, *, days_ahead: int = 14, hour: int = 9, minute: int = 0
) -> datetime:
    """A clinic-local time some days from `now`, as a UTC instant."""
    zone = ZoneInfo(timezone)
    day: date = (now.astimezone(zone) + timedelta(days=days_ahead)).date()
    return datetime.combine(day, time(hour, minute), tzinfo=zone).astimezone(UTC)


def create_clinic(
    engine: Engine,
    *,
    name: str,
    timezone: str,
    doctors: list[tuple[str, str]],
    slot_starts: list[datetime] | None = None,
    doctor_slot_starts: list[list[datetime]] | None = None,
    minutes: int = 30,
) -> ClinicFixture:
    """An active clinic, open and available all day, every day, with the given doctors and slots."""
    if doctor_slot_starts is not None and len(doctor_slot_starts) != len(doctors):
        raise ValueError("doctor_slot_starts must have one list per doctor")
    starts_by_doctor = doctor_slot_starts or [list(slot_starts or []) for _ in doctors]
    with engine.begin() as connection:
        clinic_id = connection.execute(
            insert(Clinic)
            .values(
                name=name,
                timezone=timezone,
                address_line="1 Synthetic Way",
                city="Fictional City",
                region="CO",
                postal_code="80000",
            )
            .returning(Clinic.clinic_id)
        ).scalar_one()
        for weekday in range(7):
            connection.execute(
                insert(ClinicOperatingHours).values(
                    clinic_id=clinic_id,
                    day_of_week=weekday,
                    opens_at=ALL_DAY[0],
                    closes_at=ALL_DAY[1],
                )
            )
        doctor_ids, slot_ids = [], []
        for index, (display_name, specialty) in enumerate(doctors):
            doctor_id = connection.execute(
                insert(Doctor)
                .values(display_name=display_name, specialty=specialty)
                .returning(Doctor.doctor_id)
            ).scalar_one()
            connection.execute(
                insert(DoctorClinic).values(doctor_id=doctor_id, clinic_id=clinic_id)
            )
            for weekday in range(7):
                connection.execute(
                    insert(DoctorAvailabilityRule).values(
                        doctor_id=doctor_id,
                        clinic_id=clinic_id,
                        day_of_week=weekday,
                        start_time=ALL_DAY[0],
                        end_time=ALL_DAY[1],
                        effective_start_date=date(2020, 1, 1),
                    )
                )
            doctor_ids.append(doctor_id)
            slot_ids.append(
                [
                    connection.execute(
                        insert(Slot)
                        .values(
                            doctor_id=doctor_id,
                            clinic_id=clinic_id,
                            starts_at=start,
                            ends_at=start + timedelta(minutes=minutes),
                        )
                        .returning(Slot.slot_id)
                    ).scalar_one()
                    for start in starts_by_doctor[index]
                ]
            )
    return ClinicFixture(
        clinic_id, name, timezone, doctor_ids, [name for name, _ in doctors], slot_ids
    )


def display_time(instant: datetime, timezone: str) -> str:
    """A time as the patient is shown it ("9:00 AM"), independent of the system under test."""
    local = instant.astimezone(ZoneInfo(timezone))
    return f"{local.hour % 12 or 12}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def create_patient(engine: Engine, suffix: str, pin: str, role: str = "A") -> str:
    """A fresh fictional patient who can sign in with the demo PIN. Returns the patient code.

    Every scenario run gets its own patients, so scenarios share no sessions, appointments, or
    pending actions, and the order they run in cannot change an outcome.
    """
    code = f"EVP-{suffix.upper()}-{role}"
    with engine.begin() as connection:
        patient_id = connection.execute(
            insert(Patient)
            .values(patient_code=code, display_name=f"Evaluation Patient {suffix} {role}")
            .returning(Patient.patient_id)
        ).scalar_one()
        connection.execute(
            insert(PatientCredential).values(patient_id=patient_id, pin_hash=EVAL_HASHER.hash(pin))
        )
    return code


def patient_id_of(engine: Engine, code: str) -> uuid.UUID:
    with engine.connect() as connection:
        return connection.execute(
            select(Patient.patient_id).where(Patient.patient_code == code)
        ).scalar_one()


def patient_identity(engine: Engine, code: str) -> list[str]:
    """The text that identifies a patient (code and display name), to keep out of any output."""
    with engine.connect() as connection:
        name = connection.execute(
            select(Patient.display_name).where(Patient.patient_code == code)
        ).scalar_one()
    return [code, name]


def book_slot_for(engine: Engine, patient_code: str, slot_id: uuid.UUID) -> tuple[uuid.UUID, int]:
    """Give the patient a CONFIRMED appointment on the slot, directly in the database.

    This is world setup (an earlier booking, or another patient's), never an agent action.
    Returns the appointment id and the slot's resulting version.
    """
    with engine.begin() as connection:
        patient_id = connection.execute(
            select(Patient.patient_id).where(Patient.patient_code == patient_code)
        ).scalar_one()
        version = connection.execute(
            update(Slot)
            .where(Slot.slot_id == slot_id)
            .values(status="BOOKED", version=Slot.version + 1)
            .returning(Slot.version)
        ).scalar_one()
        appointment_id = connection.execute(
            insert(Appointment)
            .values(patient_id=patient_id, slot_id=slot_id, status="CONFIRMED")
            .returning(Appointment.appointment_id)
        ).scalar_one()
    return appointment_id, version
