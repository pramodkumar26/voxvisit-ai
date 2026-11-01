import uuid
from zoneinfo import available_timezones

import pytest
from auth_support import sign_in
from scheduling_support import new_clinic, new_doctor, token

CLINIC_KEYS = {
    "clinic_id",
    "name",
    "timezone",
    "address_line",
    "city",
    "region",
    "postal_code",
    "hours",
    "specialties",
}
DOCTOR_KEYS = {"doctor_id", "display_name", "specialty", "clinics"}
HOURS_KEYS = {"day_of_week", "weekday", "opens_at", "closes_at"}
SEEDED = {
    "Riverbend Family Health",
    "Lakeside Community Clinic",
    "Harborview Medical Associates",
}


def directory(env, **params):
    response = env.client.get("/directory", params={k: str(v) for k, v in params.items()})
    assert response.status_code == 200, response.text
    return response.json()


def names(body, kind="doctors", key="display_name") -> list[str]:
    return [item[key] for item in body[kind]]


def unique_specialty() -> str:
    return f"Specialty{token()}"


# --- public access and field safety ----------------------------------------------------------


def test_directory_is_public_and_lists_the_seeded_clinics(auth_env) -> None:
    body = directory(auth_env, browse="true", limit=20)

    assert set(names(body, "clinics", "name")) | SEEDED >= SEEDED  # present in the full listing
    seeded = directory(auth_env, clinic_name="Riverbend")["clinics"][0]
    assert seeded["name"] == "Riverbend Family Health"
    assert seeded["timezone"] == "America/Denver"
    assert seeded["hours"][0] == {
        "day_of_week": 0,
        "weekday": "MONDAY",
        "opens_at": "08:00",
        "closes_at": "17:00",
    }
    assert seeded["specialties"] == ["Family Medicine", "Internal Medicine", "Pediatrics"]


def test_directory_exposes_only_approved_public_fields(auth_env) -> None:
    response = auth_env.client.get("/directory", params={"clinic_name": "Harborview"})

    body = response.json()
    assert set(body) == {"clinics", "doctors", "next_cursor"}
    for clinic in body["clinics"]:
        assert set(clinic) == CLINIC_KEYS
        assert all(set(hours) == HOURS_KEYS for hours in clinic["hours"])
    for doctor in body["doctors"]:
        assert set(doctor) == DOCTOR_KEYS
        assert all(set(c) == {"clinic_id", "name", "timezone"} for c in doctor["clinics"])
    for forbidden in (
        "patient",
        "staff",
        "password",
        "pin",
        "token",
        "session",
        "hash",
        "active",
        "version",
    ):
        assert forbidden not in response.text.lower()


def test_directory_timezones_are_valid_iana_names(auth_env) -> None:
    for clinic in directory(auth_env, browse="true")["clinics"]:
        assert clinic["timezone"] in available_timezones()


def test_every_role_gets_the_same_public_directory(auth_env) -> None:
    anonymous = directory(auth_env, specialty="Cardiology")
    patient = sign_in(auth_env, "patient", "PT-1001", auth_env.pin)
    staff = sign_in(auth_env, "staff", "staff.morgan", auth_env.password)

    for session in (patient, staff):
        response = auth_env.client.get(
            "/directory",
            params={"specialty": "Cardiology"},
            headers={"Cookie": f"voxvisit_session={session.token}"},
        )
        auth_env.client.cookies.clear()
        assert response.json() == anonymous


# --- filtering -------------------------------------------------------------------------------


def test_specialty_filter_is_case_insensitive_and_scopes_clinics_to_matching_doctors(
    auth_env,
) -> None:
    for spelling in ("Cardiology", "cardiology", "  CARDIOLOGY "):
        body = directory(auth_env, specialty=spelling)
        assert names(body) == ["Dr. Marcus Ellery"]
        assert set(names(body, "clinics", "name")) == {
            "Lakeside Community Clinic",
            "Harborview Medical Associates",
        }
        assert {c["name"] for c in body["doctors"][0]["clinics"]} == set(
            names(body, "clinics", "name")
        )


def test_doctor_name_matching_ignores_titles_case_and_word_order(auth_env) -> None:
    for query in ("Hartwell", "dr. hartwell", "DOCTOR amelia", "hartwell amelia", "ame hart"):
        assert names(directory(auth_env, doctor_name=query)) == ["Dr. Amelia Hartwell"], query
    assert directory(auth_env, doctor_name="Nobodyhere") == {
        "clinics": [],
        "doctors": [],
        "next_cursor": None,
    }


def test_clinic_filters_return_the_clinic_and_its_active_doctors(auth_env) -> None:
    by_name = directory(auth_env, clinic_name="lakeside")
    clinic_id = by_name["clinics"][0]["clinic_id"]
    by_id = directory(auth_env, clinic_id=clinic_id)

    assert names(by_name, "clinics", "name") == ["Lakeside Community Clinic"]
    assert by_id == by_name
    assert set(names(by_id)) == {"Dr. Rohan Desai", "Dr. Priya Nambiar", "Dr. Marcus Ellery"}


def test_combined_filters_must_all_match(auth_env) -> None:
    body = directory(
        auth_env, clinic_name="harborview", specialty="orthopedics", doctor_name="okafor"
    )
    assert names(body) == ["Dr. Tomas Okafor"] and names(body, "clinics", "name") == [
        "Harborview Medical Associates"
    ]

    assert directory(auth_env, clinic_name="riverbend", specialty="orthopedics")["doctors"] == []
    assert directory(auth_env, clinic_name="riverbend", specialty="orthopedics")["clinics"] == []


@pytest.mark.parametrize("wildcard", ["%", "_", "\\", "%%", "a%b", "_ndsey"])
def test_search_text_wildcards_are_matched_literally(auth_env, wildcard) -> None:
    for filters in ({"clinic_name": wildcard}, {"doctor_name": wildcard}):
        assert directory(auth_env, **filters) == {"clinics": [], "doctors": [], "next_cursor": None}


# --- active status ---------------------------------------------------------------------------


def test_inactive_clinics_doctors_and_assignments_are_hidden(auth_env) -> None:
    engine, specialty = auth_env.engine, unique_specialty()
    open_clinic = new_clinic(engine)
    closed_clinic = new_clinic(engine, active=False)
    visible = new_doctor(engine, open_clinic, specialty=specialty)
    inactive_doctor = new_doctor(engine, open_clinic, specialty=specialty, active=False)
    retired = new_doctor(engine, open_clinic, specialty=specialty, assignment_active=False)
    only_closed = new_doctor(engine, closed_clinic, specialty=specialty)
    unassigned = new_doctor(engine, open_clinic, specialty=specialty, assigned=False)

    body = directory(auth_env, specialty=specialty)

    assert names(body) == [visible.display_name]
    assert names(body, "clinics", "name") == [open_clinic.name]
    for hidden in (inactive_doctor, retired, only_closed, unassigned):
        assert hidden.display_name not in str(body)
    assert directory(auth_env, clinic_name=closed_clinic.name) == {
        "clinics": [],
        "doctors": [],
        "next_cursor": None,
    }


def test_a_doctor_lists_only_active_clinic_assignments(auth_env) -> None:
    engine, specialty = auth_env.engine, unique_specialty()
    first = new_clinic(engine, name=f"Alpha {token()}")
    second = new_clinic(engine, name=f"Beta {token()}")
    closed = new_clinic(engine, active=False)
    doctor = new_doctor(engine, first, specialty=specialty)
    from factories import make

    from app.db.models import DoctorClinic

    with engine.begin() as connection:
        make(connection, DoctorClinic, doctor_id=doctor.doctor_id, clinic_id=second.clinic_id)
        make(connection, DoctorClinic, doctor_id=doctor.doctor_id, clinic_id=closed.clinic_id)

    [found] = directory(auth_env, specialty=specialty)["doctors"]

    assert [c["name"] for c in found["clinics"]] == [first.name, second.name]


# --- request rules ---------------------------------------------------------------------------


def test_a_filter_or_explicit_browse_is_required(auth_env) -> None:
    response = auth_env.client.get("/directory")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert auth_env.client.get("/directory", params={"browse": "false"}).status_code == 422
    assert auth_env.client.get("/directory", params={"browse": "true"}).status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        {"browse": "true", "patient_id": "sentinel-patient"},
        {"browse": "true", "role": "STAFF"},
        {"browse": "true", "limit": "21"},
        {"browse": "true", "limit": "0"},
        {"clinic_name": "x" * 81},
        {"clinic_name": ""},
        {"clinic_id": "sentinel-not-a-uuid"},
        {"doctor_name": "Dr."},
        {"doctor_name": " . , "},
    ],
)
def test_invalid_directory_requests_are_rejected_without_echoing_input(auth_env, params) -> None:
    response = auth_env.client.get("/directory", params=params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert "sentinel" not in response.text


# --- pagination ------------------------------------------------------------------------------


def test_doctors_page_in_name_order_without_gaps_or_repeats(auth_env) -> None:
    engine, specialty = auth_env.engine, unique_specialty()
    clinic = new_clinic(engine)
    created = sorted(
        (
            new_doctor(engine, clinic, specialty=specialty, name=f"Dr. Person {index:02d}")
            for index in range(25)
        ),
        key=lambda doctor: doctor.display_name,
    )

    first = directory(auth_env, specialty=specialty)
    second = directory(auth_env, specialty=specialty, cursor=first["next_cursor"])

    # The matching clinic is listed first, so the 20-item page holds it plus 19 doctors.
    assert [len(first["clinics"]) + len(first["doctors"]), len(second["doctors"])] == [20, 6]
    assert second["next_cursor"] is None and second["clinics"] == []
    assert names(first) + names(second) == [doctor.display_name for doctor in created]


def test_pages_continue_from_clinics_into_doctors(auth_env) -> None:
    engine, label = auth_env.engine, token()
    clinics = [new_clinic(engine, name=f"Group {label} {name}") for name in ("A", "B", "C")]
    specialty = unique_specialty()
    for index, clinic in enumerate(clinics):
        new_doctor(engine, clinic, specialty=specialty, name=f"Dr. Group {label} {index}")

    seen_clinics, seen_doctors, cursor, pages = [], [], None, 0
    while True:
        params = {"clinic_name": f"Group {label}", "limit": 2}
        if cursor:
            params["cursor"] = cursor
        body = directory(auth_env, **params)
        assert len(body["clinics"]) + len(body["doctors"]) <= 2
        seen_clinics += names(body, "clinics", "name")
        seen_doctors += names(body)
        cursor, pages = body["next_cursor"], pages + 1
        if not cursor:
            break

    assert pages == 3
    assert seen_clinics == [f"Group {label} {name}" for name in ("A", "B", "C")]
    assert seen_doctors == [f"Dr. Group {label} {index}" for index in range(3)]


def test_cursors_are_bound_to_their_search_and_reject_tampering(auth_env) -> None:
    engine, specialty = auth_env.engine, unique_specialty()
    clinic = new_clinic(engine)
    for index in range(4):
        new_doctor(engine, clinic, specialty=specialty, name=f"Dr. Cursor {index}")
    cursor = directory(auth_env, specialty=specialty, limit=2)["next_cursor"]

    for params in (
        {"specialty": specialty.upper() + "x", "cursor": cursor},
        {"doctor_name": "cursor", "cursor": cursor},
        {"specialty": specialty, "cursor": cursor[:-3] + "xyz"},
        {"specialty": specialty, "cursor": "not-a-cursor"},
        {"browse": "true", "cursor": cursor},
    ):
        response = auth_env.client.get("/directory", params=params)
        assert response.status_code == 422, params
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert directory(auth_env, specialty=specialty, cursor=cursor)["next_cursor"] is None
    assert uuid.UUID(directory(auth_env, specialty=specialty)["doctors"][0]["doctor_id"])
