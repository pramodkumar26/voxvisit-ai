"""Model-visible arguments for each tool.

Only documented fields exist, and unknown fields are rejected, so the model cannot supply a patient
id, role, session token, idempotency key, page size, or any other trusted value. Pagination is
fixed by the runtime at the 20-item page limit.
"""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.appointment_actions import PrepareRequest
from app.models.scheduling import Cursor, Name

__all__ = [
    "AvailabilityArgs",
    "DirectoryArgs",
    "MyAppointmentsArgs",
    "PrepareRequest",
]


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DirectoryArgs(ToolArgs):
    clinic_name: str | None = Name
    doctor_name: str | None = Name
    specialty: str | None = Name
    clinic_id: UUID | None = None
    browse: bool = False
    cursor: str | None = Cursor


class AvailabilityArgs(ToolArgs):
    clinic_id: UUID
    start_date: date
    end_date: date
    doctor_id: UUID | None = None
    specialty: str | None = Name
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    cursor: str | None = Cursor


class MyAppointmentsArgs(ToolArgs):
    status: Literal["CONFIRMED", "CANCELLED"] | None = None
    start_date: date | None = None
    end_date: date | None = None
    cursor: str | None = Cursor
