"""Mapping rules, reference parsing, and access rules, with no database."""

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.auth.permissions import Permission
from app.core.errors import AuthenticationRequiredError, ForbiddenError
from app.fhir_like import mappers
from app.fhir_like import resources as r
from app.fhir_like.access import AccessContext
from app.fhir_like.adapter import FhirLikeAdapter
from app.fhir_like.records import (
    AppointmentRecord,
    PatientRecord,
    PractitionerRecord,
    ScheduleRecord,
    SlotRecord,
)

DOCTOR = uuid.UUID("11111111-1111-4111-8111-111111111111")
CLINIC = uuid.UUID("22222222-2222-4222-8222-222222222222")
SLOT = uuid.UUID("33333333-3333-4333-8333-333333333333")
PATIENT = uuid.UUID("44444444-4444-4444-8444-444444444444")
APPOINTMENT = uuid.UUID("55555555-5555-4555-8555-555555555555")
UPDATED = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
FORBIDDEN_WORDS = ("pin", "password", "hash", "token", "csrf", "cookie", "secret", "credential")


def slot_record(**overrides) -> SlotRecord:
    values = {
        "slot_id": SLOT,
        "doctor_id": DOCTOR,
        "clinic_id": CLINIC,
        "specialty": "Cardiology",
        "clinic_timezone": "America/Denver",
        "starts_at": datetime(2026, 11, 5, 21, 0, tzinfo=UTC),
        "ends_at": datetime(2026, 11, 5, 21, 30, tzinfo=UTC),
        "status": "AVAILABLE",
        "removed": False,
        "bookable": True,
        "version": 3,
        "updated_at": UPDATED,
    }
    return SlotRecord(**{**values, **overrides})


def appointment_record(**overrides) -> AppointmentRecord:
    values = {
        "appointment_id": APPOINTMENT,
        "patient_id": PATIENT,
        "patient_name": "Avery Calloway",
        "slot_id": SLOT,
        "doctor_id": DOCTOR,
        "doctor_name": "Dr. Rivera",
        "specialty": "Cardiology",
        "clinic_id": CLINIC,
        "clinic_name": "Central Clinic",
        "clinic_timezone": "America/Denver",
        "starts_at": datetime(2026, 11, 5, 21, 0, tzinfo=UTC),
        "ends_at": datetime(2026, 11, 5, 21, 30, tzinfo=UTC),
        "status": "CONFIRMED",
        "version": 2,
        "cancelled_at": None,
        "updated_at": UPDATED,
    }
    return AppointmentRecord(**{**values, **overrides})


def dump(resource) -> dict:
    return json.loads(resource.model_dump_json(exclude_none=True))


# --- identifiers and references ---------------------------------------------------------------


def test_schedule_ids_are_stable_64_character_hex_and_reversible() -> None:
    schedule = mappers.schedule_id(DOCTOR, CLINIC)

    assert schedule == DOCTOR.hex + CLINIC.hex
    assert len(schedule) == 64 and schedule == mappers.schedule_id(DOCTOR, CLINIC)
    assert mappers.parse_schedule_id(schedule) == (DOCTOR, CLINIC)


@pytest.mark.parametrize(
    "value", ["", "abc", "g" * 64, "A" * 64, "a" * 63, "a" * 65, DOCTOR.hex, str(DOCTOR)]
)
def test_malformed_schedule_ids_are_rejected(value: str) -> None:
    with pytest.raises(mappers.InvalidReferenceError):
        mappers.parse_schedule_id(value)


@pytest.mark.parametrize(
    "reference",
    [
        f"Patient/{PATIENT}",
        f"Practitioner/{DOCTOR}",
        f"Slot/{SLOT}",
        f"Appointment/{APPOINTMENT}",
        f"Schedule/{DOCTOR.hex}{CLINIC.hex}",
    ],
)
def test_well_formed_references_parse(reference: str) -> None:
    resource_type, resource_id = mappers.parse_reference(reference)

    assert reference == f"{resource_type}/{resource_id}"


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "Slot",
        "Slot/",
        "/abc",
        "Slot/not-a-uuid",
        f"Location/{CLINIC}",
        f"slot/{SLOT}",
        f"Slot/{SLOT}/extra",
        "Slot/AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        f"Slot/{SLOT.hex}",
        f"Schedule/{DOCTOR}",
        f"Organization/{CLINIC}",
        f"../Slot/{SLOT}",
    ],
)
def test_invalid_references_are_rejected(reference: str) -> None:
    with pytest.raises(mappers.InvalidReferenceError):
        mappers.parse_reference(reference)


# --- status translation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "removed", "bookable", "expected"),
    [
        ("AVAILABLE", False, True, "free"),
        ("AVAILABLE", False, False, "busy-unavailable"),  # holiday, leave, past, inactive clinic
        ("AVAILABLE", True, False, "busy-unavailable"),
        ("AVAILABLE", True, True, "busy-unavailable"),
        ("BOOKED", False, False, "busy"),
        ("HELD", False, False, "busy-tentative"),
    ],
)
def test_slot_statuses_translate_by_the_approved_state_meanings(
    stored, removed, bookable, expected
) -> None:
    assert mappers.slot_status(stored, removed=removed, bookable=bookable) == expected
    assert mappers.to_slot(
        slot_record(status=stored, removed=removed, bookable=bookable)
    ).status == (expected)


@pytest.mark.parametrize("stored", ["", "FREE", "available", "CANCELLED", "COMPLETED"])
def test_unknown_slot_statuses_are_refused_not_passed_through(stored: str) -> None:
    with pytest.raises(mappers.MappingError):
        mappers.slot_status(stored, removed=False, bookable=True)


@pytest.mark.parametrize(
    ("stored", "expected"), [("CONFIRMED", "booked"), ("CANCELLED", "cancelled")]
)
def test_appointment_statuses_translate(stored: str, expected: str) -> None:
    assert mappers.to_appointment(appointment_record(status=stored)).status == expected


@pytest.mark.parametrize("stored", ["COMPLETED", "NO_SHOW", "PENDING", "booked", ""])
def test_unsupported_appointment_statuses_are_refused(stored: str) -> None:
    with pytest.raises(mappers.MappingError):
        mappers.to_appointment(appointment_record(status=stored))


# --- resources, references, and times ---------------------------------------------------------


def test_a_slot_references_its_schedule_and_carries_utc_and_clinic_local_times() -> None:
    slot = dump(mappers.to_slot(slot_record()))

    assert slot["resourceType"] == "Slot" and slot["id"] == str(SLOT)
    assert slot["schedule"] == {"reference": f"Schedule/{DOCTOR.hex}{CLINIC.hex}"}
    assert (slot["start"], slot["end"]) == ("2026-11-05T21:00:00Z", "2026-11-05T21:30:00Z")
    extensions = {item["url"]: item for item in slot["extension"]}
    assert extensions[mappers.TIMEZONE_URL]["valueCode"] == "America/Denver"
    assert extensions[mappers.LOCAL_START_URL]["valueDateTime"] == "2026-11-05T14:00:00-07:00"
    assert extensions[mappers.LOCAL_END_URL]["valueDateTime"] == "2026-11-05T14:30:00-07:00"
    assert (
        slot["meta"]["versionId"] == "3" and slot["meta"]["lastUpdated"] == "2026-11-01T12:00:00Z"
    )


@pytest.mark.parametrize(
    ("timezone", "start", "local"),
    [
        ("America/Denver", datetime(2026, 3, 8, 8, 30, tzinfo=UTC), "2026-03-08T01:30:00-07:00"),
        ("America/Denver", datetime(2026, 3, 8, 9, 30, tzinfo=UTC), "2026-03-08T03:30:00-06:00"),
        ("America/Denver", datetime(2026, 11, 1, 7, 30, tzinfo=UTC), "2026-11-01T01:30:00-06:00"),
        ("America/Denver", datetime(2026, 11, 1, 8, 30, tzinfo=UTC), "2026-11-01T01:30:00-07:00"),
        ("Asia/Kolkata", datetime(2026, 12, 7, 10, 0, tzinfo=UTC), "2026-12-07T15:30:00+05:30"),
        ("Pacific/Auckland", datetime(2026, 12, 7, 10, 0, tzinfo=UTC), "2026-12-07T23:00:00+13:00"),
        ("Europe/London", datetime(2026, 7, 1, 8, 0, tzinfo=UTC), "2026-07-01T09:00:00+01:00"),
    ],
)
def test_clinic_local_times_follow_the_timezone_rules_including_daylight_saving(
    timezone, start, local
) -> None:
    slot = dump(
        mappers.to_slot(
            slot_record(clinic_timezone=timezone, starts_at=start, ends_at=start.replace(minute=59))
        )
    )

    extensions = {item["url"]: item for item in slot["extension"]}
    assert extensions[mappers.LOCAL_START_URL]["valueDateTime"] == local
    assert slot["start"] == start.strftime("%Y-%m-%dT%H:%M:%SZ")  # The instant is always UTC.


def test_an_appointment_links_its_slot_patient_doctor_and_clinic() -> None:
    appointment = dump(mappers.to_appointment(appointment_record()))

    assert appointment["status"] == "booked"
    assert appointment["slot"] == [{"reference": f"Slot/{SLOT}"}]
    assert appointment["participant"] == [
        {
            "actor": {"reference": f"Patient/{PATIENT}", "display": "Avery Calloway"},
            "status": "accepted",
        },
        {
            "actor": {"reference": f"Practitioner/{DOCTOR}", "display": "Dr. Rivera"},
            "status": "accepted",
        },
        {
            "actor": {
                "type": "Location",
                "identifier": {"system": mappers.CLINIC_ID_SYSTEM, "value": str(CLINIC)},
                "display": "Central Clinic",
            },
            "status": "accepted",
        },
    ]
    assert appointment["meta"]["versionId"] == "2"
    assert mappers.CANCELLED_AT_URL not in {item["url"] for item in appointment["extension"]}


def test_a_cancelled_appointment_records_when_it_was_cancelled() -> None:
    cancelled_at = datetime(2026, 11, 2, 18, 0, tzinfo=UTC)

    appointment = dump(
        mappers.to_appointment(appointment_record(status="CANCELLED", cancelled_at=cancelled_at))
    )

    assert appointment["status"] == "cancelled"
    cancelled = next(e for e in appointment["extension"] if e["url"] == mappers.CANCELLED_AT_URL)
    assert cancelled["valueDateTime"] == "2026-11-02T18:00:00Z"


def test_a_schedule_links_the_practitioner_and_names_the_clinic_logically() -> None:
    schedule = dump(
        mappers.to_schedule(
            ScheduleRecord(
                doctor_id=DOCTOR,
                clinic_id=CLINIC,
                doctor_name="Dr. Rivera",
                specialty="Cardiology",
                clinic_name="Central Clinic",
                clinic_timezone="America/Denver",
                active=True,
            )
        )
    )

    assert schedule["id"] == DOCTOR.hex + CLINIC.hex and schedule["active"] is True
    assert schedule["actor"][0] == {"reference": f"Practitioner/{DOCTOR}", "display": "Dr. Rivera"}
    assert schedule["actor"][1]["identifier"]["value"] == str(CLINIC)
    assert "reference" not in schedule["actor"][1]  # No Location resource is served.


def test_patient_and_practitioner_expose_only_the_approved_fields() -> None:
    patient = dump(
        mappers.to_patient(PatientRecord(PATIENT, "PT-1001", "Avery Calloway", True, UPDATED))
    )
    practitioner = dump(
        mappers.to_practitioner(
            PractitionerRecord(DOCTOR, "Dr. Rivera", "Cardiology", True, UPDATED)
        )
    )

    assert set(patient) == {"resourceType", "id", "meta", "identifier", "active", "name"}
    assert patient["identifier"] == [{"system": mappers.PATIENT_CODE_SYSTEM, "value": "PT-1001"}]
    assert patient["name"] == [{"text": "Avery Calloway"}]
    assert set(practitioner) == {"resourceType", "id", "meta", "active", "name", "qualification"}
    assert practitioner["qualification"] == [{"code": {"text": "Cardiology"}}]


def test_every_resource_is_labelled_as_synthetic_and_not_conformant() -> None:
    resources = [
        mappers.to_patient(PatientRecord(PATIENT, "PT-1001", "Avery", True, UPDATED)),
        mappers.to_practitioner(PractitionerRecord(DOCTOR, "Dr. R", "Cardiology", True, UPDATED)),
        mappers.to_slot(slot_record()),
        mappers.to_appointment(appointment_record()),
        mappers.to_schedule(
            ScheduleRecord(DOCTOR, CLINIC, "Dr. R", "Cardiology", "Clinic", "UTC", True)
        ),
    ]

    for resource in resources:
        [tag] = resource.meta.tag
        assert (tag.system, tag.code) == (r.MODEL_SYSTEM, "synthetic-fhir-like")
        assert "not HL7 FHIR conformant" in tag.display
    assert mappers.bundle_meta().tag[0].code == "synthetic-fhir-like"


def test_serialized_resources_hold_no_credential_like_fields() -> None:
    text = json.dumps(
        [
            dump(mappers.to_patient(PatientRecord(PATIENT, "PT-1001", "Avery", True, UPDATED))),
            dump(mappers.to_slot(slot_record())),
            dump(mappers.to_appointment(appointment_record())),
        ]
    ).lower()

    for word in FORBIDDEN_WORDS:
        assert word not in text, word


# --- missing or invalid data ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "field"),
    [
        (lambda: mappers.to_slot(slot_record(clinic_timezone="")), "clinic_timezone"),
        (lambda: mappers.to_slot(slot_record(starts_at=None)), "starts_at"),
        (lambda: mappers.to_slot(slot_record(doctor_id=None)), "doctor_id"),
        (lambda: mappers.to_appointment(appointment_record(slot_id=None)), "slot_id"),
        (lambda: mappers.to_appointment(appointment_record(patient_id=None)), "patient_id"),
        (
            lambda: mappers.to_appointment(appointment_record(clinic_timezone=None)),
            "clinic_timezone",
        ),
        (
            lambda: mappers.to_patient(PatientRecord(PATIENT, "", "Avery", True, UPDATED)),
            "patient_code",
        ),
        (
            lambda: mappers.to_practitioner(
                PractitionerRecord(DOCTOR, "", "Cardiology", True, UPDATED)
            ),
            "display_name",
        ),
    ],
)
def test_records_missing_required_values_are_refused(make, field: str) -> None:
    with pytest.raises(mappers.MappingError, match=field):
        make()


def test_resources_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        r.Patient(
            id="x",
            meta=mappers.bundle_meta(),
            identifier=[],
            active=True,
            name=[],
            birthDate="1990-01-01",
        )
    with pytest.raises(ValidationError):
        r.Slot(
            id="x",
            meta=mappers.bundle_meta(),
            status="cancelled",  # Not a slot status.
            schedule=r.Reference(reference="Schedule/x"),
            specialty=[],
            start=UPDATED,
            end=UPDATED,
            extension=[],
        )


# --- access rules (no database is touched before these raise) ---------------------------------


def adapter_for(access: AccessContext) -> FhirLikeAdapter:
    return FhirLikeAdapter(None, access, UPDATED)  # type: ignore[arg-type]


def test_access_contexts_come_only_from_verified_identities() -> None:
    assert AccessContext.from_principal(None).role == "PUBLIC"
    assert AccessContext.public().permissions == frozenset()
    with pytest.raises(ValueError, match="patient id"):
        AccessContext("PATIENT", None, frozenset())
    with pytest.raises(ValueError, match="patient id"):
        AccessContext("STAFF", PATIENT, frozenset())


def test_public_callers_must_authenticate_for_protected_resources() -> None:
    adapter = adapter_for(AccessContext.public())

    with pytest.raises(AuthenticationRequiredError):
        adapter.patient(PATIENT)
    with pytest.raises(AuthenticationRequiredError):
        adapter.appointment(APPOINTMENT)
    with pytest.raises(AuthenticationRequiredError):
        adapter.search_appointments()


def test_a_staff_context_without_the_permission_is_forbidden() -> None:
    adapter = adapter_for(
        AccessContext("STAFF", None, frozenset({Permission.ACCESS_STAFF_DASHBOARD}))
    )

    with pytest.raises(ForbiddenError):
        adapter.patient(PATIENT)
    with pytest.raises(ForbiddenError):
        adapter.appointment(APPOINTMENT)


def test_the_adapter_exposes_no_write_or_confirmation_operations() -> None:
    public = {name for name in dir(FhirLikeAdapter) if not name.startswith("_")}

    assert public == {
        "practitioner",
        "schedule",
        "slot",
        "patient",
        "appointment",
        "resolve",
        "search_practitioners",
        "search_schedules",
        "search_slots",
        "search_appointments",
    }
