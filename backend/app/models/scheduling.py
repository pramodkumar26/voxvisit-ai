"""Query and response schemas for the read-only scheduling endpoints.

Responses list their fields explicitly; nothing here mirrors a database table, and none of it
carries patient, credential, session, or version data beyond what the contract approves.
"""

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.services.pagination import MAX_PAGE_SIZE

Name = Field(default=None, min_length=1, max_length=80)
Cursor = Field(default=None, min_length=1, max_length=512)


class StrictQuery(BaseModel):
    """Unknown query parameters (for example a patient_id) are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


class DirectoryQuery(StrictQuery):
    clinic_name: str | None = Name
    doctor_name: str | None = Name
    specialty: str | None = Name
    clinic_id: UUID | None = None
    browse: bool = False
    cursor: str | None = Cursor
    limit: int = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class AvailabilityQuery(StrictQuery):
    clinic_id: UUID
    start_date: date
    end_date: date
    doctor_id: UUID | None = None
    specialty: str | None = Name
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    cursor: str | None = Cursor
    limit: int = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class MyAppointmentsQuery(StrictQuery):
    status: Literal["CONFIRMED", "CANCELLED"] | None = None
    start_date: date | None = None
    end_date: date | None = None
    cursor: str | None = Cursor
    limit: int = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class ClinicSummary(BaseModel):
    clinic_id: UUID
    name: str
    timezone: str


class DoctorSummary(BaseModel):
    doctor_id: UUID
    display_name: str
    specialty: str


class OperatingHours(BaseModel):
    day_of_week: int = Field(description="Monday=0 through Sunday=6")
    weekday: str
    opens_at: str = Field(description="Clinic-local HH:MM")
    closes_at: str = Field(description="Clinic-local HH:MM")


class DirectoryClinic(ClinicSummary):
    address_line: str
    city: str
    region: str
    postal_code: str
    hours: list[OperatingHours]
    specialties: list[str]


class DirectoryDoctor(DoctorSummary):
    clinics: list[ClinicSummary]


class DirectoryPage(BaseModel):
    clinics: list[DirectoryClinic]
    doctors: list[DirectoryDoctor]
    next_cursor: str | None


class TimeRange(BaseModel):
    """UTC instants plus the same moments in the clinic's own timezone, with their offsets."""

    starts_at_utc: datetime
    ends_at_utc: datetime
    starts_at_local: datetime
    ends_at_local: datetime
    clinic_timezone: str


class AvailableSlot(TimeRange):
    slot_id: UUID
    clinic: ClinicSummary
    doctor: DoctorSummary


class SearchWindow(BaseModel):
    start_date: date
    end_date: date
    timezone: str


class AvailabilityPage(BaseModel):
    window: SearchWindow
    slots: list[AvailableSlot]
    next_cursor: str | None


class MyAppointment(TimeRange):
    appointment_id: UUID
    status: Literal["CONFIRMED", "CANCELLED"]
    clinic: ClinicSummary
    doctor: DoctorSummary


class MyAppointmentsPage(BaseModel):
    appointments: list[MyAppointment]
    next_cursor: str | None
