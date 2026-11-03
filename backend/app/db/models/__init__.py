"""SQLAlchemy models for the approved PostgreSQL schema."""

from app.db.models.appointments import Appointment, AppointmentHistory
from app.db.models.identity import (
    AuthSession,
    Patient,
    PatientCredential,
    PendingAction,
    ReviewHandoff,
    StaffCredential,
    StaffUser,
)
from app.db.models.reliability import IdempotencyRecord, OutboxEvent, SecurityAuditEvent
from app.db.models.scheduling import (
    Clinic,
    ClinicHoliday,
    ClinicOperatingHours,
    Doctor,
    DoctorAvailabilityRule,
    DoctorClinic,
    DoctorLeave,
    Slot,
)
from app.db.models.support import HandoffNote, HandoffTicket, NotificationEvent
from app.db.models.waitlist import SlotOffer, WaitlistEntry

__all__ = [
    "Appointment",
    "AppointmentHistory",
    "AuthSession",
    "Clinic",
    "ClinicHoliday",
    "ClinicOperatingHours",
    "Doctor",
    "DoctorAvailabilityRule",
    "DoctorClinic",
    "DoctorLeave",
    "HandoffNote",
    "HandoffTicket",
    "IdempotencyRecord",
    "NotificationEvent",
    "OutboxEvent",
    "Patient",
    "PatientCredential",
    "PendingAction",
    "ReviewHandoff",
    "SecurityAuditEvent",
    "Slot",
    "SlotOffer",
    "StaffCredential",
    "StaffUser",
    "WaitlistEntry",
]
