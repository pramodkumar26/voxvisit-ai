"""The synthetic FHIR-like adapter against real PostgreSQL: mappings, references, access rules,
and the guarantee that reading changes nothing."""

import json
import re
import uuid
from datetime import date, time, timedelta

import pytest
from action_support import (
    NEAR_CLOCK,
    book_body,
    cancel_body,
    confirm,
    error_of,
    new_patient,
    patient_id_of,
    patient_session,
    prepared,
    reschedule_body,
    rows,
    scalar,
    seed_appointment,
    staff_session,
    world,
)
from auth_support import call
from scheduling_support import MONDAY, add_holiday, add_leave, new_clinic, new_doctor
from sqlalchemy import text

from app.core.errors import (
    AuthenticationRequiredError,
    InvalidArgumentError,
    ResourceNotAvailableError,
)
from app.fhir_like import mappers
from app.fhir_like.access import AccessContext
from app.fhir_like.adapter import open_adapter

BASE = "/integration/fhir-like"
FORBIDDEN_WORDS = (
    "pin_hash",
    "password",
    "argon2",
    "token",
    "csrf",
    "cookie",
    "secret",
    "credential",
)


def fhir(env, path: str, session=None, **params):
    return call(env, "GET", BASE + path, session, params=params)


def found(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def entries(bundle: dict) -> list[dict]:
    return [entry["resource"] for entry in bundle["entry"]]


def ids(bundle: dict) -> list[str]:
    return [resource["id"] for resource in entries(bundle)]


def extension(resource: dict, suffix: str) -> dict:
    return next(item for item in resource["extension"] if item["url"].endswith(suffix))


def references(value) -> list[str]:
    """Every literal "Type/id" reference inside a resource."""
    found_references: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            found_references.append(value["reference"])
        for item in value.values():
            found_references.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            found_references.extend(references(item))
    return found_references


def adapter(env, access: AccessContext):
    return open_adapter(env.app.state.db_engine, access, env.clock())


# --- Practitioner and Schedule ----------------------------------------------------------------


def test_a_doctor_maps_to_a_practitioner_for_anyone(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)

    response = fhir(auth_env, f"/Practitioner/{scenario.doctor.doctor_id}")

    body = found(response)
    assert body["resourceType"] == "Practitioner" and body["id"] == str(scenario.doctor.doctor_id)
    assert body["name"] == [{"text": scenario.doctor.display_name}]
    assert body["qualification"] == [{"code": {"text": "Primary Care"}}]
    assert body["active"] is True
    assert response.headers["X-Resource-Model"] == "synthetic-fhir-like"
    assert (
        response.headers["content-type"] == "application/json"
    )  # Never claims application/fhir+json.
    assert response.headers["cache-control"] == "no-store"
    assert body["meta"]["tag"][0]["code"] == "synthetic-fhir-like"


def test_unlisted_doctors_are_hidden_from_the_public_but_visible_to_staff(auth_env) -> None:
    inactive = world(auth_env.engine, count=1)
    unassigned = world(auth_env.engine, count=1)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE doctors SET active = false WHERE doctor_id = :id"),
            {"id": inactive.doctor.doctor_id},
        )
        connection.execute(
            text("UPDATE doctor_clinics SET active = false WHERE doctor_id = :id"),
            {"id": unassigned.doctor.doctor_id},
        )
    staff = staff_session(auth_env)
    missing = fhir(auth_env, f"/Practitioner/{uuid.uuid4()}")

    for scenario in (inactive, unassigned):
        public = fhir(auth_env, f"/Practitioner/{scenario.doctor.doctor_id}")
        assert public.status_code == 404 and public.json() == missing.json()
        assert found(fhir(auth_env, f"/Practitioner/{scenario.doctor.doctor_id}", staff))["id"] == (
            str(scenario.doctor.doctor_id)
        )
    assert (
        found(fhir(auth_env, f"/Practitioner/{inactive.doctor.doctor_id}", staff))["active"]
        is False
    )


def test_practitioner_search_filters_pages_and_never_repeats(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctors = [new_doctor(auth_env.engine, clinic, specialty="Neurology") for _ in range(5)]
    first = found(fhir(auth_env, "/Practitioner", clinic=str(clinic.clinic_id), limit=2))
    assert first["resourceType"] == "Bundle" and first["type"] == "searchset"
    assert first["meta"]["tag"][0]["code"] == "synthetic-fhir-like"
    assert len(first["entry"]) == 2 and first["link"][0]["relation"] == "next"

    collected, bundle = ids(first), first
    while bundle["link"]:
        url = bundle["link"][0]["url"]
        assert url.startswith(f"{BASE}/Practitioner?") and "limit=2" in url
        bundle = found(call(auth_env, "GET", url))
        collected += ids(bundle)

    assert sorted(collected) == collected  # Deterministic order.
    assert set(collected) == {str(doctor.doctor_id) for doctor in doctors}
    assert len(collected) == len(set(collected)) == 5
    assert all(
        entry["fullUrl"] == f"Practitioner/{entry['resource']['id']}" for entry in first["entry"]
    )
    none = found(
        fhir(auth_env, "/Practitioner", clinic=str(clinic.clinic_id), specialty="Cardiology")
    )
    assert none["entry"] == []


def test_a_schedule_joins_one_doctor_to_one_clinic(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    schedule_id = scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex

    body = found(fhir(auth_env, f"/Schedule/{schedule_id}"))

    assert body["resourceType"] == "Schedule" and body["id"] == schedule_id
    assert body["active"] is True
    practitioner, clinic = body["actor"]
    assert practitioner == {
        "reference": f"Practitioner/{scenario.doctor.doctor_id}",
        "display": scenario.doctor.display_name,
    }
    assert clinic == {
        "type": "Location",
        "identifier": {
            "system": mappers.CLINIC_ID_SYSTEM,
            "value": str(scenario.clinic.clinic_id),
        },
        "display": scenario.clinic.name,
    }
    assert extension(body, "clinic-timezone")["valueCode"] == "America/Denver"
    listed = found(fhir(auth_env, "/Schedule", clinic=str(scenario.clinic.clinic_id)))
    assert ids(listed) == [schedule_id]
    by_doctor = found(fhir(auth_env, "/Schedule", practitioner=str(scenario.doctor.doctor_id)))
    assert ids(by_doctor) == [schedule_id]


def test_schedules_follow_the_doctor_clinic_relationships(auth_env) -> None:
    from scheduling_support import assign_doctor

    one, two = world(auth_env.engine, count=1), world(auth_env.engine, count=1)
    assign_doctor(auth_env.engine, one.doctor, two.clinic)  # One doctor, two clinics.

    first = found(fhir(auth_env, "/Schedule", practitioner=str(one.doctor.doctor_id)))
    at_two = found(fhir(auth_env, "/Schedule", clinic=str(two.clinic.clinic_id)))

    assert set(ids(first)) == {
        one.doctor.doctor_id.hex + one.clinic.clinic_id.hex,
        one.doctor.doctor_id.hex + two.clinic.clinic_id.hex,
    }
    assert {resource["actor"][0]["reference"].split("/")[1] for resource in entries(at_two)} == {
        str(one.doctor.doctor_id),
        str(two.doctor.doctor_id),
    }
    assert {resource["actor"][1]["identifier"]["value"] for resource in entries(first)} == {
        str(one.clinic.clinic_id),
        str(two.clinic.clinic_id),
    }


def test_inactive_schedules_are_hidden_from_the_public(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    schedule_id = scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE doctor_clinics SET active = false WHERE doctor_id = :id"),
            {"id": scenario.doctor.doctor_id},
        )
    staff = staff_session(auth_env)

    assert fhir(auth_env, f"/Schedule/{schedule_id}").status_code == 404
    assert ids(found(fhir(auth_env, "/Schedule", clinic=str(scenario.clinic.clinic_id)))) == []
    assert found(fhir(auth_env, f"/Schedule/{schedule_id}", staff))["active"] is False


@pytest.mark.parametrize("bad", ["abc", "g" * 64, "A" * 64, "a" * 63, str(uuid.uuid4())])
def test_malformed_schedule_ids_are_rejected(auth_env, bad: str) -> None:
    response = fhir(auth_env, f"/Schedule/{bad}")

    assert response.status_code == 422 and error_of(response)["code"] == "INVALID_ARGUMENT"


def test_a_well_formed_schedule_id_for_an_unassigned_pair_is_not_found(auth_env) -> None:
    one, two = world(auth_env.engine, count=1), world(auth_env.engine, count=1)

    crossed = fhir(auth_env, f"/Schedule/{one.doctor.doctor_id.hex + two.clinic.clinic_id.hex}")

    assert crossed.status_code == 404 and error_of(crossed)["code"] == "RESOURCE_NOT_AVAILABLE"


# --- Slot -------------------------------------------------------------------------------------


def test_a_free_slot_references_its_schedule_with_utc_and_local_times(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    slot_id = scenario.slots[0]

    body = found(fhir(auth_env, f"/Slot/{slot_id}"))

    assert body["resourceType"] == "Slot" and body["id"] == str(slot_id)
    assert body["status"] == "free"
    assert body["schedule"] == {
        "reference": f"Schedule/{scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex}"
    }
    assert (body["start"], body["end"]) == ("2026-12-07T16:00:00Z", "2026-12-07T16:30:00Z")
    assert extension(body, "clinic-local-start")["valueDateTime"] == "2026-12-07T09:00:00-07:00"
    assert extension(body, "clinic-local-end")["valueDateTime"] == "2026-12-07T09:30:00-07:00"
    assert extension(body, "clinic-timezone")["valueCode"] == "America/Denver"
    assert body["meta"]["versionId"] == str(
        scalar(auth_env.engine, "SELECT version FROM slots WHERE slot_id = :i", i=slot_id)
    )
    assert body["specialty"] == [{"text": "Primary Care"}]


def test_slot_search_returns_exactly_what_availability_search_returns(auth_env) -> None:
    scenario = world(auth_env.engine, count=7)
    window = {
        "clinic_id": str(scenario.clinic.clinic_id),
        "start_date": MONDAY.isoformat(),
        "end_date": MONDAY.isoformat(),
    }
    expected = [
        slot["slot_id"]
        for slot in found(call(auth_env, "GET", "/availability", params=window))["slots"]
    ]

    bundle = found(
        fhir(
            auth_env,
            "/Slot",
            clinic=window["clinic_id"],
            start_date=window["start_date"],
            end_date=window["end_date"],
        )
    )

    assert ids(bundle) == expected == [str(slot) for slot in scenario.slots]
    assert {resource["status"] for resource in entries(bundle)} == {"free"}


def test_slot_search_pages_follow_the_same_order(auth_env) -> None:
    scenario = world(auth_env.engine, count=5)
    first = found(
        fhir(
            auth_env,
            "/Slot",
            clinic=str(scenario.clinic.clinic_id),
            start_date=MONDAY.isoformat(),
            end_date=MONDAY.isoformat(),
            limit=2,
        )
    )
    collected, bundle = ids(first), first
    while bundle["link"]:
        bundle = found(call(auth_env, "GET", bundle["link"][0]["url"]))
        collected += ids(bundle)

    assert collected == [str(slot) for slot in scenario.slots]


def test_slot_search_rejects_unknown_parameters_and_bad_windows(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    base = {
        "clinic": str(scenario.clinic.clinic_id),
        "start_date": MONDAY.isoformat(),
        "end_date": MONDAY.isoformat(),
    }

    for extra in ({"patient_id": str(uuid.uuid4())}, {"status": "busy"}, {"_count": "5"}):
        assert fhir(auth_env, "/Slot", **base, **extra).status_code == 422
    backwards = fhir(
        auth_env, "/Slot", **{**base, "end_date": (MONDAY - timedelta(days=1)).isoformat()}
    )
    assert backwards.status_code == 422
    too_long = fhir(
        auth_env, "/Slot", **{**base, "end_date": (MONDAY + timedelta(days=40)).isoformat()}
    )
    assert too_long.status_code == 422
    assert (
        fhir(
            auth_env, "/Slot", start_date=MONDAY.isoformat(), end_date=MONDAY.isoformat()
        ).status_code
        == 422
    )
    unknown = fhir(auth_env, "/Slot", **{**base, "clinic": str(uuid.uuid4())})
    assert unknown.status_code == 404


STATE_CHANGES = {
    "booked": ("UPDATE slots SET status = 'BOOKED' WHERE slot_id = :slot", "busy"),
    "held": ("UPDATE slots SET status = 'HELD' WHERE slot_id = :slot", "busy-tentative"),
    "removed": ("UPDATE slots SET removed_at = now() WHERE slot_id = :slot", "busy-unavailable"),
    "clinic_inactive": (
        "UPDATE clinics SET active = false WHERE clinic_id = :clinic",
        "busy-unavailable",
    ),
    "doctor_inactive": (
        "UPDATE doctors SET active = false WHERE doctor_id = :doctor",
        "busy-unavailable",
    ),
}


@pytest.mark.parametrize("change", list(STATE_CHANGES))
def test_slot_status_translation_for_staff_and_invisibility_for_the_public(
    auth_env, change: str
) -> None:
    scenario = world(auth_env.engine, count=1)
    slot_id = scenario.slots[0]
    statement, expected = STATE_CHANGES[change]
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(statement),
            {
                "slot": slot_id,
                "clinic": scenario.clinic.clinic_id,
                "doctor": scenario.doctor.doctor_id,
            },
        )
    staff = staff_session(auth_env)
    missing = fhir(auth_env, f"/Slot/{uuid.uuid4()}")

    assert found(fhir(auth_env, f"/Slot/{slot_id}", staff))["status"] == expected
    public = fhir(auth_env, f"/Slot/{slot_id}")
    assert public.status_code == 404 and public.json() == missing.json()  # Not revealed.
    searched = fhir(
        auth_env,
        "/Slot",
        clinic=str(scenario.clinic.clinic_id),
        start_date=MONDAY.isoformat(),
        end_date=MONDAY.isoformat(),
    )
    if change == "clinic_inactive":
        assert searched.status_code == 404  # Same as the availability search: no such clinic.
    else:
        assert ids(found(searched)) == []  # A slot that is not bookable is never searchable.


@pytest.mark.parametrize("cause", ["holiday", "leave", "past"])
def test_available_slots_that_are_not_bookable_are_busy_unavailable(auth_env, cause: str) -> None:
    if cause == "past":
        scenario = world(auth_env.engine, [(NEAR_CLOCK, time(7))])
    else:
        scenario = world(auth_env.engine, count=1)
        if cause == "holiday":
            add_holiday(auth_env.engine, scenario.clinic, MONDAY)
        else:
            add_leave(
                auth_env.engine,
                scenario.clinic,
                scenario.doctor,
                (MONDAY, time(8)),
                (MONDAY, time(10)),
            )
    slot_id = scenario.slots[0]
    staff = staff_session(auth_env)

    assert found(fhir(auth_env, f"/Slot/{slot_id}", staff))["status"] == "busy-unavailable"
    assert fhir(auth_env, f"/Slot/{slot_id}").status_code == 404


def test_slot_times_are_correct_in_other_timezones_and_across_daylight_saving(auth_env) -> None:
    kolkata = world(auth_env.engine, [(date(2026, 12, 8), time(15, 30))], timezone="Asia/Kolkata")
    auckland = world(
        auth_env.engine, [(date(2026, 12, 8), time(9, 0))], timezone="Pacific/Auckland"
    )
    gap = world(auth_env.engine, [(date(2026, 12, 9), time(12, 0))], timezone="America/Denver")
    lord_howe = world(
        auth_env.engine, [(date(2026, 12, 8), time(10, 30))], timezone="Australia/Lord_Howe"
    )
    cases = [
        (kolkata, "2026-12-08T10:00:00Z", "2026-12-08T15:30:00+05:30"),
        (auckland, "2026-12-07T20:00:00Z", "2026-12-08T09:00:00+13:00"),
        (gap, "2026-12-09T19:00:00Z", "2026-12-09T12:00:00-07:00"),
        (lord_howe, "2026-12-07T23:30:00Z", "2026-12-08T10:30:00+11:00"),
    ]

    for scenario, utc, local in cases:
        body = found(fhir(auth_env, f"/Slot/{scenario.slots[0]}"))
        assert body["start"] == utc
        assert extension(body, "clinic-local-start")["valueDateTime"] == local


def test_the_daylight_saving_change_day_has_correct_offsets(auth_env) -> None:
    # 2027-03-14 is the US spring-forward day: 02:00 does not exist in Denver.
    scenario = world(
        auth_env.engine,
        [(date(2027, 3, 14), time(1, 30)), (date(2027, 3, 14), time(3, 30))],
        timezone="America/Denver",
    )
    before, after = (found(fhir(auth_env, f"/Slot/{slot}")) for slot in scenario.slots)

    assert extension(before, "clinic-local-start")["valueDateTime"] == "2027-03-14T01:30:00-07:00"
    assert extension(after, "clinic-local-start")["valueDateTime"] == "2027-03-14T03:30:00-06:00"
    assert (before["start"], after["start"]) == ("2027-03-14T08:30:00Z", "2027-03-14T09:30:00Z")


# --- Appointment and Patient ------------------------------------------------------------------


def book_through_the_api(env, session, slot_id) -> dict:
    return confirm(env, session, prepared(env, session, book_body(slot_id))).json()


def test_a_confirmed_appointment_links_slot_patient_doctor_and_clinic(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    code = new_patient(auth_env, "Quinn Harbour")
    session = patient_session(auth_env, code)
    booked = book_through_the_api(auth_env, session, scenario.slots[0])

    body = found(fhir(auth_env, f"/Appointment/{booked['appointment_id']}", session))

    patient_id = str(patient_id_of(auth_env.engine, code))
    assert body["resourceType"] == "Appointment" and body["status"] == "booked"
    assert body["id"] == booked["appointment_id"]
    assert body["slot"] == [{"reference": f"Slot/{scenario.slots[0]}"}]
    assert [item["actor"].get("reference") for item in body["participant"]] == [
        f"Patient/{patient_id}",
        f"Practitioner/{scenario.doctor.doctor_id}",
        None,
    ]
    assert body["participant"][0]["actor"]["display"] == "Quinn Harbour"
    assert body["participant"][2]["actor"]["identifier"]["value"] == str(scenario.clinic.clinic_id)
    assert (body["start"], body["end"]) == ("2026-12-07T16:00:00Z", "2026-12-07T16:30:00Z")
    assert extension(body, "clinic-local-start")["valueDateTime"] == "2026-12-07T09:00:00-07:00"
    assert body["meta"]["versionId"] == "1"
    assert found(fhir(auth_env, f"/Slot/{scenario.slots[0]}", session))["status"] == "busy"


def test_reschedule_and_cancellation_translate_and_advance_the_version(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    code = new_patient(auth_env)
    session = patient_session(auth_env, code)
    appointment_id = book_through_the_api(auth_env, session, scenario.slots[0])["appointment_id"]

    confirm(
        auth_env,
        session,
        prepared(auth_env, session, reschedule_body(appointment_id, scenario.slots[2])),
    )
    moved = found(fhir(auth_env, f"/Appointment/{appointment_id}", session))
    assert moved["status"] == "booked" and moved["meta"]["versionId"] == "2"
    assert moved["slot"] == [{"reference": f"Slot/{scenario.slots[2]}"}]  # Same id, new slot.
    assert extension(moved, "clinic-local-start")["valueDateTime"] == "2026-12-07T10:00:00-07:00"

    confirm(auth_env, session, prepared(auth_env, session, cancel_body(appointment_id)))
    cancelled = found(fhir(auth_env, f"/Appointment/{appointment_id}", session))
    assert cancelled["status"] == "cancelled" and cancelled["meta"]["versionId"] == "3"
    assert extension(cancelled, "cancelled-at")["valueDateTime"].endswith("Z")
    assert "cancelled-at" not in json.dumps(moved)
    assert found(fhir(auth_env, f"/Slot/{scenario.slots[2]}", session))["status"] == "free"


def test_patients_search_only_their_own_appointments(auth_env) -> None:
    scenario = world(auth_env.engine, count=4)
    mine, theirs = new_patient(auth_env), new_patient(auth_env)
    me, other = patient_session(auth_env, mine), patient_session(auth_env, theirs)
    first = book_through_the_api(auth_env, me, scenario.slots[0])["appointment_id"]
    second = book_through_the_api(auth_env, me, scenario.slots[1])["appointment_id"]
    others = book_through_the_api(auth_env, other, scenario.slots[2])["appointment_id"]
    confirm(auth_env, me, prepared(auth_env, me, cancel_body(second)))

    everything = found(fhir(auth_env, "/Appointment", me))
    only_booked = found(fhir(auth_env, "/Appointment", me, status="booked"))
    only_cancelled = found(fhir(auth_env, "/Appointment", me, status="cancelled"))

    assert ids(everything) == [first, second] and others not in ids(everything)
    assert ids(only_booked) == [first] and ids(only_cancelled) == [second]
    assert ids(found(fhir(auth_env, "/Appointment", other))) == [others]
    paged = found(fhir(auth_env, "/Appointment", me, limit=1))
    assert ids(paged) == [first] and paged["link"]
    assert ids(found(call(auth_env, "GET", paged["link"][0]["url"], me))) == [second]


def test_patients_cannot_widen_an_appointment_search(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))

    for params in (
        {"patient": str(uuid.uuid4())},
        {"patient_id": str(uuid.uuid4())},
        {"clinic": str(scenario.clinic.clinic_id)},
        {"practitioner": str(scenario.doctor.doctor_id)},
        {"status": "fulfilled"},
    ):
        assert fhir(auth_env, "/Appointment", session, **params).status_code == 422, params


def test_another_patients_appointment_looks_exactly_like_a_missing_one(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    owner, snoop = new_patient(auth_env, "Rowan Fairweather"), new_patient(auth_env)
    appointment_id = book_through_the_api(
        auth_env, patient_session(auth_env, owner), scenario.slots[0]
    )["appointment_id"]
    session = patient_session(auth_env, snoop)

    denied = fhir(auth_env, f"/Appointment/{appointment_id}", session)
    missing = fhir(auth_env, f"/Appointment/{uuid.uuid4()}", session)

    assert denied.status_code == 404 and denied.json() == missing.json()
    for leaked in ("Rowan", "Fairweather", owner, appointment_id):
        assert leaked not in denied.text


def test_patient_records_are_isolated_and_minimal(auth_env) -> None:
    mine, theirs = new_patient(auth_env, "Ada Whitcombe"), new_patient(auth_env, "Bram Okonkwo")
    session = patient_session(auth_env, mine)
    my_id, their_id = patient_id_of(auth_env.engine, mine), patient_id_of(auth_env.engine, theirs)

    own = found(fhir(auth_env, f"/Patient/{my_id}", session))
    denied = fhir(auth_env, f"/Patient/{their_id}", session)
    missing = fhir(auth_env, f"/Patient/{uuid.uuid4()}", session)

    assert set(own) == {"resourceType", "id", "meta", "identifier", "active", "name"}
    assert own["identifier"][0]["value"] == mine and own["name"] == [{"text": "Ada Whitcombe"}]
    assert denied.status_code == 404 and denied.json() == missing.json()
    for leaked in ("Bram", "Okonkwo", theirs, str(their_id)):
        assert leaked not in denied.text


def test_staff_see_the_clinic_schedule_and_minimal_patient_identity(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    code = new_patient(auth_env, "Sasha Lindgren")
    session = patient_session(auth_env, code)
    appointment_id = book_through_the_api(auth_env, session, scenario.slots[0])["appointment_id"]
    staff = staff_session(auth_env)
    patient_id = patient_id_of(auth_env.engine, code)

    appointment = found(fhir(auth_env, f"/Appointment/{appointment_id}", staff))
    patient = found(fhir(auth_env, f"/Patient/{patient_id}", staff))
    listing = found(
        fhir(
            auth_env, "/Appointment", staff, clinic=str(scenario.clinic.clinic_id), status="booked"
        )
    )

    assert appointment["participant"][0]["actor"] == {
        "reference": f"Patient/{patient_id}",
        "display": "Sasha Lindgren",
    }
    assert ids(listing) == [appointment_id]
    assert set(patient) == {"resourceType", "id", "meta", "identifier", "active", "name"}
    assert found(fhir(auth_env, f"/Slot/{scenario.slots[0]}", staff))["status"] == "busy"


def test_staff_search_filters_and_pages_by_clinic_doctor_status_and_window(auth_env) -> None:
    scenario = world(auth_env.engine, count=5)
    session = patient_session(auth_env, new_patient(auth_env))
    booked = [
        book_through_the_api(auth_env, session, slot)["appointment_id"]
        for slot in scenario.slots[:3]
    ]
    confirm(auth_env, session, prepared(auth_env, session, cancel_body(booked[2])))
    staff = staff_session(auth_env)
    clinic = str(scenario.clinic.clinic_id)

    assert ids(found(fhir(auth_env, "/Appointment", staff, clinic=clinic))) == booked
    assert ids(found(fhir(auth_env, "/Appointment", staff, clinic=clinic, status="cancelled"))) == [
        booked[2]
    ]
    assert (
        ids(
            found(
                fhir(auth_env, "/Appointment", staff, practitioner=str(scenario.doctor.doctor_id))
            )
        )
        == booked
    )
    window = {"start_date": MONDAY.isoformat(), "end_date": MONDAY.isoformat()}
    assert ids(found(fhir(auth_env, "/Appointment", staff, clinic=clinic, **window))) == booked
    elsewhere = {"start_date": (MONDAY + timedelta(days=1)).isoformat()}
    assert ids(found(fhir(auth_env, "/Appointment", staff, clinic=clinic, **elsewhere))) == []
    first = found(fhir(auth_env, "/Appointment", staff, clinic=clinic, limit=2))
    second = found(call(auth_env, "GET", first["link"][0]["url"], staff))
    assert ids(first) + ids(second) == booked and not second["link"]


# --- authentication and authorization ---------------------------------------------------------


@pytest.mark.parametrize("path", ["/Patient/{p}", "/Appointment/{a}", "/Appointment"])
def test_public_callers_get_nothing_protected(auth_env, path: str) -> None:
    scenario = world(auth_env.engine, count=1)
    appointment_id = seed_appointment(auth_env.engine, "PT-1001", scenario.slots[0])
    patient_id = patient_id_of(auth_env.engine, "PT-1001")

    response = fhir(auth_env, path.format(p=patient_id, a=appointment_id))
    forged = call(
        auth_env, "GET", BASE + path.format(p=patient_id, a=appointment_id), token="forged"
    )

    for denied in (response, forged):
        assert denied.status_code == 401 and error_of(denied)["code"] == "AUTH_REQUIRED"
        assert "PT-1001" not in denied.text and str(appointment_id) not in denied.text


def test_expired_sessions_lose_protected_access(auth_env) -> None:
    session = patient_session(auth_env, new_patient(auth_env))
    assert fhir(auth_env, "/Appointment", session).status_code == 200
    auth_env.clock.advance(minutes=16)

    response = fhir(auth_env, "/Appointment", session)

    assert response.status_code == 401 and error_of(response)["code"] == "SESSION_EXPIRED"


def test_public_slot_reads_ignore_a_session_that_is_not_theirs(auth_env) -> None:
    scenario = world(auth_env.engine, count=2)
    owner = new_patient(auth_env)
    book_through_the_api(auth_env, patient_session(auth_env, owner), scenario.slots[0])
    stranger = patient_session(auth_env, new_patient(auth_env))

    assert fhir(auth_env, f"/Slot/{scenario.slots[0]}", stranger).status_code == 404
    assert (
        found(fhir(auth_env, f"/Slot/{scenario.slots[0]}", patient_session(auth_env, owner)))[
            "status"
        ]
        == "busy"
    )


# --- references -------------------------------------------------------------------------------


def walk(env, access: AccessContext, start: str) -> dict[str, dict]:
    """Follow every literal reference reachable from `start`, resolving each through the adapter."""
    resolved: dict[str, dict] = {}
    pending = [start]
    with adapter(env, access) as source:
        while pending:
            reference = pending.pop()
            if reference in resolved:
                continue
            resource = json.loads(source.resolve(reference).model_dump_json(exclude_none=True))
            assert reference == f"{resource['resourceType']}/{resource['id']}"
            resolved[reference] = resource
            pending.extend(references(resource))
    return resolved


def test_every_reference_in_an_appointment_resolves_for_staff_and_for_the_owner(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    code = new_patient(auth_env)
    session = patient_session(auth_env, code)
    appointment_id = book_through_the_api(auth_env, session, scenario.slots[0])["appointment_id"]
    patient_id = patient_id_of(auth_env.engine, code)
    schedule = f"Schedule/{scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex}"

    for access in (AccessContext.staff(), AccessContext.patient(patient_id)):
        graph = walk(auth_env, access, f"Appointment/{appointment_id}")
        assert set(graph) == {
            f"Appointment/{appointment_id}",
            f"Slot/{scenario.slots[0]}",
            f"Patient/{patient_id}",
            f"Practitioner/{scenario.doctor.doctor_id}",
            schedule,
        }
        # The graph is mutually consistent: the schedule's practitioner is the appointment's.
        assert (
            graph[schedule]["actor"][0]["reference"]
            == graph[f"Appointment/{appointment_id}"]["participant"][1]["actor"]["reference"]
        )
        assert graph[f"Slot/{scenario.slots[0]}"]["schedule"]["reference"] == schedule


def test_references_do_not_widen_access(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    owner = new_patient(auth_env)
    appointment_id = book_through_the_api(
        auth_env, patient_session(auth_env, owner), scenario.slots[0]
    )["appointment_id"]
    owner_id = patient_id_of(auth_env.engine, owner)
    stranger = AccessContext.patient(patient_id_of(auth_env.engine, new_patient(auth_env)))

    with adapter(auth_env, AccessContext.public()) as source:
        with pytest.raises(ResourceNotAvailableError):
            source.resolve(f"Slot/{scenario.slots[0]}")  # Booked slots are not public.
        for protected in (f"Patient/{owner_id}", f"Appointment/{appointment_id}"):
            with pytest.raises(AuthenticationRequiredError):
                source.resolve(protected)
        assert source.resolve(f"Practitioner/{scenario.doctor.doctor_id}")
    with adapter(auth_env, stranger) as source:
        for protected in (
            f"Patient/{owner_id}",
            f"Appointment/{appointment_id}",
            f"Slot/{scenario.slots[0]}",
        ):
            with pytest.raises(ResourceNotAvailableError):
                source.resolve(protected)


def test_missing_and_invalid_references_are_reported_safely(auth_env) -> None:
    with adapter(auth_env, AccessContext.staff()) as source:
        for reference in (
            f"Slot/{uuid.uuid4()}",
            f"Practitioner/{uuid.uuid4()}",
            f"Patient/{uuid.uuid4()}",
            f"Appointment/{uuid.uuid4()}",
            f"Schedule/{uuid.uuid4().hex}{uuid.uuid4().hex}",
        ):
            with pytest.raises(ResourceNotAvailableError):
                source.resolve(reference)
        for reference in ("Slot/not-a-uuid", "Location/abc", "Slot", "", f"Slot/{uuid.uuid4()}/x"):
            with pytest.raises(InvalidArgumentError):
                source.resolve(reference)


# --- consistency with PostgreSQL (EV-020) -----------------------------------------------------


def test_every_seeded_appointment_matches_its_authoritative_rows(auth_env) -> None:
    authoritative = rows(
        auth_env.engine,
        """SELECT a.appointment_id, a.status, a.patient_id, p.display_name, p.patient_code,
                  s.slot_id, s.starts_at AT TIME ZONE 'UTC', s.ends_at AT TIME ZONE 'UTC',
                  d.doctor_id, d.display_name, c.clinic_id, c.name, c.timezone, a.version,
                  to_char(s.starts_at AT TIME ZONE c.timezone, 'YYYY-MM-DD"T"HH24:MI:SS')
           FROM appointments a
           JOIN patients p USING (patient_id) JOIN slots s USING (slot_id)
           JOIN doctors d ON d.doctor_id = s.doctor_id JOIN clinics c ON c.clinic_id = s.clinic_id
           WHERE p.patient_code LIKE 'PT-10%' AND s.clinic_id IN (
               SELECT clinic_id FROM clinics WHERE name IN (
                   'Riverbend Family Health', 'Lakeside Community Clinic',
                   'Harborview Medical Associates'))
           ORDER BY a.appointment_id""",
    )
    assert len(authoritative) >= 10
    staff = staff_session(auth_env)

    for (
        aid,
        status,
        pid,
        pname,
        pcode,
        sid,
        start,
        end,
        did,
        dname,
        cid,
        cname,
        tz,
        version,
        local,
    ) in authoritative:
        body = found(fhir(auth_env, f"/Appointment/{aid}", staff))
        assert body["id"] == str(aid) and body["meta"]["versionId"] == str(version)
        assert body["status"] == {"CONFIRMED": "booked", "CANCELLED": "cancelled"}[status]
        assert body["slot"] == [{"reference": f"Slot/{sid}"}]
        assert body["start"] == start.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert body["end"] == end.strftime("%Y-%m-%dT%H:%M:%SZ")
        patient, practitioner, clinic = body["participant"]
        assert patient["actor"] == {"reference": f"Patient/{pid}", "display": pname}
        assert practitioner["actor"] == {"reference": f"Practitioner/{did}", "display": dname}
        assert (
            clinic["actor"]["identifier"]["value"] == str(cid)
            and clinic["actor"]["display"] == cname
        )
        assert extension(body, "clinic-timezone")["valueCode"] == tz
        assert extension(body, "clinic-local-start")["valueDateTime"].startswith(local)
        assert found(fhir(auth_env, f"/Patient/{pid}", staff))["identifier"][0]["value"] == pcode
        slot = found(fhir(auth_env, f"/Slot/{sid}", staff))
        assert slot["status"] == ("busy" if status == "CONFIRMED" else slot["status"])
        assert slot["schedule"]["reference"] == f"Schedule/{did.hex + cid.hex}"


# --- the adapter only reads -------------------------------------------------------------------

TABLES = (
    "slots",
    "appointments",
    "appointment_history",
    "outbox_events",
    "pending_actions",
    "idempotency_records",
    "patients",
    "patient_credentials",
    "staff_users",
    "staff_credentials",
    "doctors",
    "clinics",
    "doctor_clinics",
)


def state_of(env) -> dict[str, str]:
    return {
        table: scalar(
            env.engine,
            f"SELECT coalesce(md5(string_agg(t::text, ',' ORDER BY t::text)), '') FROM {table} t",
        )
        for table in TABLES
    }


def test_exercising_every_endpoint_with_every_role_changes_no_data(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    code = new_patient(auth_env)
    me = patient_session(auth_env, code)
    appointment_id = book_through_the_api(auth_env, me, scenario.slots[0])["appointment_id"]
    patient_id = patient_id_of(auth_env.engine, code)
    staff = staff_session(auth_env)
    before = state_of(auth_env)
    clinic = str(scenario.clinic.clinic_id)
    window = {"start_date": MONDAY.isoformat(), "end_date": MONDAY.isoformat()}
    schedule = scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex

    for session in (None, me, staff):
        for path, params in (
            (f"/Practitioner/{scenario.doctor.doctor_id}", {}),
            ("/Practitioner", {"clinic": clinic}),
            (f"/Schedule/{schedule}", {}),
            ("/Schedule", {"clinic": clinic}),
            (f"/Slot/{scenario.slots[0]}", {}),
            (f"/Slot/{scenario.slots[1]}", {}),
            ("/Slot", {"clinic": clinic, **window}),
            (f"/Patient/{patient_id}", {}),
            (f"/Appointment/{appointment_id}", {}),
            ("/Appointment", {}),
            (f"/Appointment/{uuid.uuid4()}", {}),
        ):
            fhir(auth_env, path, session, **params)

    assert state_of(auth_env) == before


def test_the_adapter_connection_cannot_write_even_if_misused(auth_env) -> None:
    with (
        adapter(auth_env, AccessContext.staff()) as source,
        pytest.raises(Exception, match="read-only"),
    ):
        source._connection.execute(text("UPDATE slots SET status = 'BOOKED'"))


def test_no_mutating_route_exists_under_the_adapter_prefix(auth_env) -> None:
    paths = {
        path: set(item)
        for path, item in auth_env.app.openapi()["paths"].items()
        if path.startswith(BASE)
    }

    assert paths and all(methods == {"get"} for methods in paths.values())
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    before = state_of(auth_env)
    attempts = [
        call(auth_env, method, f"{BASE}{path}", session, json={"action": "BOOK"})
        for method in ("POST", "PUT", "PATCH", "DELETE")
        for path in (f"/Appointment/{uuid.uuid4()}", "/Appointment", f"/Slot/{scenario.slots[0]}")
    ]

    assert {attempt.status_code for attempt in attempts} == {405}
    assert state_of(auth_env) == before
    confirm_paths = [path for path in auth_env.app.openapi()["paths"] if "confirm" in path]
    assert not any(path.startswith(BASE) for path in confirm_paths)


# --- no sensitive data ------------------------------------------------------------------------


def test_no_response_carries_credentials_tokens_or_other_patients_data(auth_env) -> None:
    scenario = world(auth_env.engine, count=3)
    mine, theirs = new_patient(auth_env, "Ines Valdivia"), new_patient(auth_env, "Jules Ashworth")
    me, other = patient_session(auth_env, mine), patient_session(auth_env, theirs)
    my_appointment = book_through_the_api(auth_env, me, scenario.slots[0])["appointment_id"]
    their_appointment = book_through_the_api(auth_env, other, scenario.slots[1])["appointment_id"]
    my_id, their_id = patient_id_of(auth_env.engine, mine), patient_id_of(auth_env.engine, theirs)
    staff = staff_session(auth_env)
    clinic = str(scenario.clinic.clinic_id)
    window = {"start_date": MONDAY.isoformat(), "end_date": MONDAY.isoformat()}

    def collect(session) -> str:
        responses = [
            fhir(auth_env, f"/Practitioner/{scenario.doctor.doctor_id}", session),
            fhir(auth_env, "/Schedule", session, clinic=clinic),
            fhir(auth_env, "/Slot", session, clinic=clinic, **window),
            fhir(auth_env, f"/Slot/{scenario.slots[0]}", session),
            fhir(auth_env, f"/Patient/{my_id}", session),
            fhir(auth_env, f"/Patient/{their_id}", session),
            fhir(auth_env, "/Appointment", session),
            fhir(auth_env, f"/Appointment/{my_appointment}", session),
            fhir(auth_env, f"/Appointment/{their_appointment}", session),
        ]
        return "\n".join(response.text for response in responses)

    everything = {
        name: collect(session) for name, session in (("public", None), ("me", me), ("staff", staff))
    }

    for name, text_seen in everything.items():
        lowered = text_seen.lower()
        for word in FORBIDDEN_WORDS:
            assert word not in lowered, (name, word)
        for secret in (auth_env.pin, auth_env.password, me.token, me.csrf, staff.token, staff.csrf):
            assert secret not in text_seen, name
        assert not re.search(r"\$argon2", text_seen)
    # The public sees neither patients nor appointments; a patient sees only their own.
    for private in ("Ines", "Jules", mine, theirs, my_appointment, their_appointment, str(my_id)):
        assert private not in everything["public"], private
    for private in ("Jules", "Ashworth", theirs, their_appointment, str(their_id)):
        assert private not in everything["me"], private
    assert "Ines Valdivia" in everything["me"] and "Jules Ashworth" in everything["staff"]
    # Staff see names and codes, not patient contact or clinical data (none exists in the model).
    for field in ("birthDate", "telecom", "address", "gender", "condition", "diagnosis"):
        assert field not in everything["staff"]


def test_resources_serve_no_field_beyond_the_simplified_models(auth_env) -> None:
    scenario = world(auth_env.engine, count=1)
    session = patient_session(auth_env, new_patient(auth_env))
    book_through_the_api(auth_env, session, scenario.slots[0])
    staff = staff_session(auth_env)
    seen: dict[str, set[str]] = {}

    appointment = entries(
        found(fhir(auth_env, "/Appointment", staff, clinic=str(scenario.clinic.clinic_id)))
    )[0]
    for resource in (
        appointment,
        found(fhir(auth_env, f"/Slot/{scenario.slots[0]}", staff)),
        found(fhir(auth_env, f"/Practitioner/{scenario.doctor.doctor_id}", staff)),
        found(
            fhir(
                auth_env,
                f"/Schedule/{scenario.doctor.doctor_id.hex + scenario.clinic.clinic_id.hex}",
                staff,
            )
        ),
    ):
        seen[resource["resourceType"]] = set(resource)

    assert seen == {
        "Appointment": {
            "resourceType",
            "id",
            "meta",
            "status",
            "slot",
            "specialty",
            "start",
            "end",
            "participant",
            "extension",
        },
        "Slot": {
            "resourceType",
            "id",
            "meta",
            "status",
            "schedule",
            "specialty",
            "start",
            "end",
            "extension",
        },
        "Practitioner": {"resourceType", "id", "meta", "active", "name", "qualification"},
        "Schedule": {"resourceType", "id", "meta", "active", "actor", "specialty", "extension"},
    }


def test_record_lookups_re_apply_ownership_even_for_ids_a_listing_returned(auth_env) -> None:
    """Defense in depth: an id from any listing is scoped to the caller again before mapping."""
    from app.fhir_like import queries

    scenario = world(auth_env.engine, count=1)
    owner = new_patient(auth_env)
    appointment_id = uuid.UUID(
        book_through_the_api(auth_env, patient_session(auth_env, owner), scenario.slots[0])[
            "appointment_id"
        ]
    )
    stranger = AccessContext.patient(patient_id_of(auth_env.engine, new_patient(auth_env)))
    owner_access = AccessContext.patient(patient_id_of(auth_env.engine, owner))

    with adapter(auth_env, stranger) as source:
        assert queries.appointments_by_id(source._connection, stranger, [appointment_id]) == []
    with (
        adapter(auth_env, AccessContext.public()) as source,
        pytest.raises(ResourceNotAvailableError),
    ):
        queries.appointments_by_id(source._connection, AccessContext.public(), [appointment_id])
    with adapter(auth_env, owner_access) as source:
        [record] = queries.appointments_by_id(source._connection, owner_access, [appointment_id])
        assert record.appointment_id == appointment_id


@pytest.mark.parametrize("role", ["public", "patient"])
def test_the_staff_appointment_query_refuses_callers_without_the_schedule_permission(
    auth_env, role: str
) -> None:
    from app.fhir_like import queries

    access = (
        AccessContext.public()
        if role == "public"
        else AccessContext.patient(patient_id_of(auth_env.engine, new_patient(auth_env)))
    )
    with adapter(auth_env, access) as source, pytest.raises(ResourceNotAvailableError):
        queries.search_staff_appointments(
            source._connection,
            access,
            clinic_id=None,
            doctor_id=None,
            status=None,
            start_date=None,
            end_date=None,
            cursor=None,
            limit=20,
        )


# --- a clinic-scoped practitioner search needs an active assignment at that clinic -----------


def test_clinic_scoped_practitioner_search_requires_an_active_assignment_there(auth_env) -> None:
    from scheduling_support import assign_doctor

    clinic_a, clinic_b = world(auth_env.engine, count=1), world(auth_env.engine, count=1)
    doctor = clinic_a.doctor
    assign_doctor(auth_env.engine, doctor, clinic_b.clinic)  # Works at both clinics...
    with auth_env.engine.begin() as connection:  # ...but the assignment at B is inactive.
        connection.execute(
            text(
                "UPDATE doctor_clinics SET active = false WHERE doctor_id = :d AND clinic_id = :c"
            ),
            {"d": doctor.doctor_id, "c": clinic_b.clinic.clinic_id},
        )
    staff = staff_session(auth_env)
    at_a = str(clinic_a.clinic.clinic_id)
    at_b = str(clinic_b.clinic.clinic_id)

    assert str(doctor.doctor_id) in ids(found(fhir(auth_env, "/Practitioner", clinic=at_a)))
    assert str(doctor.doctor_id) not in ids(found(fhir(auth_env, "/Practitioner", clinic=at_b)))
    assert ids(found(fhir(auth_env, "/Practitioner", clinic=at_b))) == [
        str(clinic_b.doctor.doctor_id)
    ]
    # The doctor is still a listed practitioner (active at clinic A), and staff asking about
    # clinic B can see the inactive assignment.
    assert fhir(auth_env, f"/Practitioner/{doctor.doctor_id}").status_code == 200
    assert str(doctor.doctor_id) in ids(found(fhir(auth_env, "/Practitioner", staff, clinic=at_b)))
    # The Schedule search agrees with the practitioner search.
    assert doctor.doctor_id.hex + clinic_b.clinic.clinic_id.hex not in ids(
        found(fhir(auth_env, "/Schedule", clinic=at_b))
    )


def test_clinic_scoped_practitioner_search_requires_an_active_clinic_and_doctor(auth_env) -> None:
    from scheduling_support import assign_doctor

    clinic_a, clinic_b = world(auth_env.engine, count=1), world(auth_env.engine, count=1)
    assign_doctor(auth_env.engine, clinic_a.doctor, clinic_b.clinic)
    inactive_doctor = world(auth_env.engine, count=1)
    with auth_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE clinics SET active = false WHERE clinic_id = :c"),
            {"c": clinic_b.clinic.clinic_id},
        )
        connection.execute(
            text("UPDATE doctors SET active = false WHERE doctor_id = :d"),
            {"d": inactive_doctor.doctor.doctor_id},
        )
    at_a = str(clinic_a.clinic.clinic_id)

    # The clinic is inactive, so its practitioners are not listed for it, even though the
    # assignment rows are active and the doctor is active at another clinic.
    assert ids(found(fhir(auth_env, "/Practitioner", clinic=str(clinic_b.clinic.clinic_id)))) == []
    assert ids(found(fhir(auth_env, "/Practitioner", clinic=at_a))) == [
        str(clinic_a.doctor.doctor_id)
    ]
    assert (
        ids(found(fhir(auth_env, "/Practitioner", clinic=str(inactive_doctor.clinic.clinic_id))))
        == []
    )
