"""Request and response schemas for preparing and confirming appointment actions.

Requests accept only the documented fields. A patient ID, role, or version is never accepted;
identity comes from the session and expected versions are stored server-side when an action is
prepared. Responses list their fields explicitly and are safe to store as idempotency snapshots.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrepareBook(StrictBody):
    action: Literal["BOOK"]
    slot_id: UUID


class PrepareReschedule(StrictBody):
    action: Literal["RESCHEDULE"]
    appointment_id: UUID
    new_slot_id: UUID


class PrepareCancel(StrictBody):
    action: Literal["CANCEL"]
    appointment_id: UUID


PrepareRequest = Annotated[
    PrepareBook | PrepareReschedule | PrepareCancel, Field(discriminator="action")
]


class ConfirmRequest(StrictBody):
    """The explicit confirmation: only the JSON value true. 1, "true", and "yes" are rejected."""

    confirmed: StrictBool

    @field_validator("confirmed")
    @classmethod
    def must_be_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirmation must be true")
        return value


class AppointmentTime(BaseModel):
    """A clinic, doctor, and time exactly as the patient is shown them."""

    clinic: str
    doctor: str
    specialty: str
    date: str
    time: str
    timezone: str
    starts_at_local: datetime
    starts_at_utc: datetime


class ActionSummary(BaseModel):
    operation: Literal["BOOK", "RESCHEDULE", "CANCEL"]
    description: str
    current: AppointmentTime | None = None
    proposed: AppointmentTime | None = None


class PendingActionView(BaseModel):
    pending_action_id: UUID
    action: Literal["BOOK", "RESCHEDULE", "CANCEL"]
    stored_status: Literal["ACTIVE"] = "ACTIVE"
    ui_status: Literal["AWAITING_UI_CONFIRMATION"] = "AWAITING_UI_CONFIRMATION"
    summary: ActionSummary
    created_at: datetime
    expires_at: datetime
    confirmation_required: Literal[True] = True


class AppointmentActionResult(BaseModel):
    """The committed outcome. Reported to the patient only after the transaction commits."""

    result: Literal["BOOKED", "RESCHEDULED", "CANCELLED"]
    appointment_id: UUID
    status: Literal["CONFIRMED", "CANCELLED"]
    appointment: AppointmentTime
    previous: AppointmentTime | None = None
    completed_at: datetime
