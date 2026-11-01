"""Plain records the queries produce and the mappers consume. They keep the mappers free of
database types, so the mapping rules are testable without a database."""

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PatientRecord:
    patient_id: uuid.UUID
    patient_code: str
    display_name: str
    active: bool
    updated_at: datetime


@dataclass(frozen=True)
class PractitionerRecord:
    doctor_id: uuid.UUID
    display_name: str
    specialty: str
    active: bool
    updated_at: datetime


@dataclass(frozen=True)
class ScheduleRecord:
    doctor_id: uuid.UUID
    clinic_id: uuid.UUID
    doctor_name: str
    specialty: str
    clinic_name: str
    clinic_timezone: str
    active: bool  # The assignment, the doctor, and the clinic are all active.


@dataclass(frozen=True)
class SlotRecord:
    slot_id: uuid.UUID
    doctor_id: uuid.UUID
    clinic_id: uuid.UUID
    specialty: str
    clinic_timezone: str
    starts_at: datetime
    ends_at: datetime
    status: str  # The stored slot status: AVAILABLE, HELD, or BOOKED.
    removed: bool
    bookable: bool  # Genuinely bookable now, by the same rules as availability search.
    version: int
    updated_at: datetime


@dataclass(frozen=True)
class AppointmentRecord:
    appointment_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    slot_id: uuid.UUID
    doctor_id: uuid.UUID
    doctor_name: str
    specialty: str
    clinic_id: uuid.UUID
    clinic_name: str
    clinic_timezone: str
    starts_at: datetime
    ends_at: datetime
    status: str  # The stored appointment status: CONFIRMED or CANCELLED.
    version: int
    cancelled_at: datetime | None
    updated_at: datetime
