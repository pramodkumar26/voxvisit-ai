"""Search parameters for the adapter. Unknown parameters (for example a patient id) are rejected.

These are this project's simplified parameters, not FHIR search parameters.
"""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.models.scheduling import Cursor, Name, StrictQuery
from app.services.pagination import MAX_PAGE_SIZE


class Paged(StrictQuery):
    cursor: str | None = Cursor
    limit: int = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class PractitionerSearch(Paged):
    clinic: UUID | None = None
    specialty: str | None = Name


class ScheduleSearch(Paged):
    clinic: UUID | None = None
    practitioner: UUID | None = None


class SlotSearch(Paged):
    clinic: UUID
    start_date: date
    end_date: date
    practitioner: UUID | None = None
    specialty: str | None = Name


class AppointmentSearch(Paged):
    status: Literal["booked", "cancelled"] | None = None
    start_date: date | None = None
    end_date: date | None = None
    clinic: UUID | None = None
    practitioner: UUID | None = None
