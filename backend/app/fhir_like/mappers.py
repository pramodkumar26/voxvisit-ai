"""Deterministic mappings from authoritative records to FHIR-like resources.

Pure functions: no database, no clock, no authorization. Who may see a record is decided before a
record reaches these functions (see access.py and adapter.py). Anything the mappings do not know
how to translate raises MappingError instead of being passed through, so an unexpected stored
state can never leak out as an invented FHIR status.
"""

import re
import uuid
from datetime import UTC
from zoneinfo import ZoneInfo

from app.fhir_like import resources as r
from app.fhir_like.records import (
    AppointmentRecord,
    PatientRecord,
    PractitionerRecord,
    ScheduleRecord,
    SlotRecord,
)

NAMESPACE = "urn:voxvisit:synthetic"
CLINIC_ID_SYSTEM = f"{NAMESPACE}:clinic-id"
PATIENT_CODE_SYSTEM = f"{NAMESPACE}:patient-code"
TIMEZONE_URL = f"{NAMESPACE}:clinic-timezone"
LOCAL_START_URL = f"{NAMESPACE}:clinic-local-start"
LOCAL_END_URL = f"{NAMESPACE}:clinic-local-end"
CANCELLED_AT_URL = f"{NAMESPACE}:cancelled-at"

RESOURCE_TYPES = ("Patient", "Practitioner", "Schedule", "Slot", "Appointment")
SCHEDULE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

# The approved state meanings (docs/slot-state-machine.md, docs/appointment-state-machine.md).
APPOINTMENT_STATUS = {"CONFIRMED": "booked", "CANCELLED": "cancelled"}
STORED_APPOINTMENT_STATUS = {fhir: stored for stored, fhir in APPOINTMENT_STATUS.items()}


class MappingError(ValueError):
    """A record cannot be represented faithfully."""


class InvalidReferenceError(ValueError):
    """A reference or resource id is malformed or names an unsupported type."""


def schedule_id(doctor_id: uuid.UUID, clinic_id: uuid.UUID) -> str:
    """A Schedule is a doctor at one clinic, which has no row of its own. Its id is the two UUIDs
    in hex, concatenated: 64 characters (the FHIR id limit), stable, and reversible."""
    return doctor_id.hex + clinic_id.hex


def parse_schedule_id(value: str) -> tuple[uuid.UUID, uuid.UUID]:
    if not SCHEDULE_ID_PATTERN.fullmatch(value):
        raise InvalidReferenceError("Not a valid Schedule id.")
    return uuid.UUID(value[:32]), uuid.UUID(value[32:])


def parse_reference(value: str) -> tuple[str, str]:
    """Split "Type/id" and validate both halves, without looking anything up."""
    resource_type, separator, resource_id = value.partition("/")
    if not separator or resource_type not in RESOURCE_TYPES or "/" in resource_id:
        raise InvalidReferenceError("Not a valid reference.")
    try:
        if resource_type == "Schedule":
            parse_schedule_id(resource_id)
        else:
            if str(uuid.UUID(resource_id)) != resource_id:
                raise ValueError
    except ValueError:
        raise InvalidReferenceError("Not a valid reference.") from None
    return resource_type, resource_id


def slot_status(stored_status: str, *, removed: bool, bookable: bool) -> str:
    """AVAILABLE -> free only when the slot is genuinely bookable now; an AVAILABLE slot that is
    removed, past, on a holiday or leave, or at an inactive clinic is busy-unavailable."""
    if removed:
        return "busy-unavailable"
    if stored_status == "BOOKED":
        return "busy"
    if stored_status == "HELD":
        return "busy-tentative"
    if stored_status == "AVAILABLE":
        return "free" if bookable else "busy-unavailable"
    raise MappingError(f"Unsupported slot status: {stored_status!r}")


def appointment_status(stored_status: str) -> str:
    try:
        return APPOINTMENT_STATUS[stored_status]
    except KeyError:
        raise MappingError(f"Unsupported appointment status: {stored_status!r}") from None


def _require(record: object, *names: str) -> None:
    missing = [name for name in names if getattr(record, name, None) in (None, "")]
    if missing:
        raise MappingError(f"{type(record).__name__} is missing: {', '.join(missing)}")


def _meta(*, version: int | None = None, updated_at=None) -> r.Meta:
    return r.Meta(
        versionId=None if version is None else str(version),
        lastUpdated=None if updated_at is None else updated_at.astimezone(UTC),
        tag=[r.Coding(system=r.MODEL_SYSTEM, code=r.MODEL_CODE, display=r.MODEL_DISPLAY)],
    )


def bundle_meta() -> r.Meta:
    return _meta()


def _reference(resource_type: str, resource_id: object, display: str | None = None) -> r.Reference:
    return r.Reference(reference=f"{resource_type}/{resource_id}", display=display)


def _clinic_reference(clinic_id: uuid.UUID, name: str) -> r.Reference:
    """The clinic is a logical reference: this adapter serves no Location resource."""
    return r.Reference(
        type="Location",
        identifier=r.Identifier(system=CLINIC_ID_SYSTEM, value=str(clinic_id)),
        display=name,
    )


def _local_extensions(timezone: str, starts_at, ends_at) -> list[r.Extension]:
    zone = ZoneInfo(timezone)
    return [
        r.Extension(url=TIMEZONE_URL, valueCode=timezone),
        r.Extension(url=LOCAL_START_URL, valueDateTime=starts_at.astimezone(zone)),
        r.Extension(url=LOCAL_END_URL, valueDateTime=ends_at.astimezone(zone)),
    ]


def to_patient(record: PatientRecord) -> r.Patient:
    _require(record, "patient_id", "patient_code", "display_name")
    return r.Patient(
        id=str(record.patient_id),
        meta=_meta(updated_at=record.updated_at),
        identifier=[r.Identifier(system=PATIENT_CODE_SYSTEM, value=record.patient_code)],
        active=record.active,
        name=[r.HumanName(text=record.display_name)],
    )


def to_practitioner(record: PractitionerRecord) -> r.Practitioner:
    _require(record, "doctor_id", "display_name", "specialty")
    return r.Practitioner(
        id=str(record.doctor_id),
        meta=_meta(updated_at=record.updated_at),
        active=record.active,
        name=[r.HumanName(text=record.display_name)],
        qualification=[{"code": r.Text(text=record.specialty)}],
    )


def to_schedule(record: ScheduleRecord) -> r.Schedule:
    _require(record, "doctor_id", "clinic_id", "doctor_name", "clinic_name", "clinic_timezone")
    return r.Schedule(
        id=schedule_id(record.doctor_id, record.clinic_id),
        meta=_meta(),
        active=record.active,
        actor=[
            _reference("Practitioner", record.doctor_id, record.doctor_name),
            _clinic_reference(record.clinic_id, record.clinic_name),
        ],
        specialty=[r.Text(text=record.specialty)],
        extension=[r.Extension(url=TIMEZONE_URL, valueCode=record.clinic_timezone)],
    )


def to_slot(record: SlotRecord) -> r.Slot:
    _require(record, "slot_id", "doctor_id", "clinic_id", "starts_at", "ends_at", "clinic_timezone")
    return r.Slot(
        id=str(record.slot_id),
        meta=_meta(version=record.version, updated_at=record.updated_at),
        status=slot_status(record.status, removed=record.removed, bookable=record.bookable),
        schedule=_reference("Schedule", schedule_id(record.doctor_id, record.clinic_id)),
        specialty=[r.Text(text=record.specialty)],
        start=record.starts_at.astimezone(UTC),
        end=record.ends_at.astimezone(UTC),
        extension=_local_extensions(record.clinic_timezone, record.starts_at, record.ends_at),
    )


def to_appointment(record: AppointmentRecord) -> r.Appointment:
    _require(
        record,
        "appointment_id",
        "patient_id",
        "slot_id",
        "doctor_id",
        "clinic_id",
        "starts_at",
        "ends_at",
        "clinic_timezone",
    )
    extensions = _local_extensions(record.clinic_timezone, record.starts_at, record.ends_at)
    if record.cancelled_at is not None:
        extensions.append(r.Extension(url=CANCELLED_AT_URL, valueDateTime=record.cancelled_at))
    return r.Appointment(
        id=str(record.appointment_id),
        meta=_meta(version=record.version, updated_at=record.updated_at),
        status=appointment_status(record.status),
        slot=[_reference("Slot", record.slot_id)],
        specialty=[r.Text(text=record.specialty)],
        start=record.starts_at.astimezone(UTC),
        end=record.ends_at.astimezone(UTC),
        participant=[
            r.AppointmentParticipant(
                actor=_reference("Patient", record.patient_id, record.patient_name)
            ),
            r.AppointmentParticipant(
                actor=_reference("Practitioner", record.doctor_id, record.doctor_name)
            ),
            r.AppointmentParticipant(actor=_clinic_reference(record.clinic_id, record.clinic_name)),
        ],
        extension=extensions,
    )
