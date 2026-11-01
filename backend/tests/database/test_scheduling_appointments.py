import uuid
from datetime import date, time, timedelta
from types import SimpleNamespace

import pytest
from auth_support import call, sign_in
from database_support import FAST_HASHER
from factories import make
from scheduling_support import MONDAY, add_slots, new_clinic, new_doctor, token
from sqlalchemy import text

from app.db.models import Appointment, Patient, PatientCredential

PATH = "/me/appointments"
ITEM_KEYS = {
    "appointment_id",
    "status",
    "clinic",
    "doctor",
    "starts_at_utc",
    "ends_at_utc",
    "starts_at_local",
    "ends_at_local",
    "clinic_timezone",
}
SEEDED_OWNERS = {
    "PT-1001": 2,
    "PT-1002": 1,
    "PT-1003": 1,
    "PT-1004": 1,
    "PT-1005": 1,
    "PT-1006": 1,
    "PT-1007": 1,
    "PT-1008": 1,
    "PT-1009": 1,
    "PT-1010": 1,
}


def ids_in_db(env, code: str) -> set[str]:
    with env.engine.connect() as connection:
        return {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT a.appointment_id FROM appointments a JOIN patients p USING (patient_id)"
                    " WHERE p.patient_code = :code"
                ),
                {"code": code},
            )
        }


def listing(env, code: str, **params):
    session = sign_in(env, "patient", code, env.pin)
    return call(env, "GET", PATH, session, params={k: str(v) for k, v in params.items()})


def returned_ids(response) -> set[str]:
    assert response.status_code == 200, response.text
    return {item["appointment_id"] for item in response.json()["appointments"]}


def patient_with_login(env) -> SimpleNamespace:
    """A new patient with their own PIN, so pagination tests own every record they read."""
    code = f"PT-{token()}"
    with env.engine.begin() as connection:
        patient = make(connection, Patient, patient_code=code)
        make(
            connection,
            PatientCredential,
            patient_id=patient.patient_id,
            pin_hash=FAST_HASHER.hash(env.pin),
        )
    return SimpleNamespace(patient_id=patient.patient_id, code=code)


def book_history(env, patient, count: int, *, hour: time = time(9)) -> list[uuid.UUID]:
    """Appointments on consecutive days: even index CONFIRMED, odd CANCELLED."""
    clinic = new_clinic(env.engine)
    doctor = new_doctor(env.engine, clinic)
    created = []
    with env.engine.begin() as connection:
        for index in range(count):
            confirmed = index % 2 == 0
            [slot_id] = add_slots(
                env.engine,
                clinic,
                doctor,
                [(MONDAY + timedelta(days=index), hour)],
                status="BOOKED" if confirmed else "AVAILABLE",
            )
            appointment = make(
                connection,
                Appointment,
                patient_id=patient.patient_id,
                slot_id=slot_id,
                status="CONFIRMED" if confirmed else "CANCELLED",
                cancelled_at=None if confirmed else env.clock.now,
            )
            created.append(appointment.appointment_id)
    return created


# --- session and role requirements -----------------------------------------------------------


def test_anonymous_and_forged_sessions_are_rejected(auth_env) -> None:
    anonymous = call(auth_env, "GET", PATH)
    forged = call(auth_env, "GET", PATH, token="forged-token")

    for response in (anonymous, forged):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTH_REQUIRED"
        assert "appointment" not in response.text.lower().replace("appointments", "")


def test_expired_sessions_are_rejected_with_a_distinct_code(auth_env) -> None:
    session = sign_in(auth_env, "patient", "PT-1001", auth_env.pin)
    auth_env.clock.advance(minutes=16)

    response = call(auth_env, "GET", PATH, session)

    assert response.status_code == 401 and response.json()["error"]["code"] == "SESSION_EXPIRED"


def test_staff_sessions_cannot_use_the_patient_endpoint(auth_env) -> None:
    staff = sign_in(auth_env, "staff", "staff.morgan", auth_env.password)

    response = call(auth_env, "GET", PATH, staff)

    assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN"
    for leaked in ("Avery", "PT-1001", "appointment_id"):
        assert leaked not in response.text


def test_the_patient_filter_cannot_be_supplied_by_the_caller(auth_env) -> None:
    victim_id = next(iter(ids_in_db(auth_env, "PT-1002")))
    with auth_env.engine.connect() as connection:
        victim = connection.execute(
            text("SELECT patient_id FROM patients WHERE patient_code = 'PT-1002'")
        ).scalar_one()
    session = sign_in(auth_env, "patient", "PT-1001", auth_env.pin)

    for param in ("patient_id", "patient_code", "role", "actor_id"):
        response = call(auth_env, "GET", PATH, session, params={param: str(victim)})
        assert response.status_code == 422, param
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
        assert victim_id not in response.text and str(victim) not in response.text


# --- seeded appointments and isolation -------------------------------------------------------


def test_a_patient_sees_exactly_their_own_seeded_appointments(auth_env) -> None:
    response = listing(auth_env, "PT-1001")

    body = response.json()
    assert (
        returned_ids(response) == ids_in_db(auth_env, "PT-1001") and len(body["appointments"]) == 2
    )
    assert response.headers["cache-control"] == "no-store"
    assert set(body) == {"appointments", "next_cursor"} and body["next_cursor"] is None
    starts = [item["starts_at_utc"] for item in body["appointments"]]
    assert starts == sorted(starts)
    for item in body["appointments"]:
        assert set(item) == ITEM_KEYS
        assert item["status"] == "CONFIRMED"
        assert set(item["clinic"]) == {"clinic_id", "name", "timezone"}
        assert set(item["doctor"]) == {"doctor_id", "display_name", "specialty"}
        assert item["clinic_timezone"] == item["clinic"]["timezone"]
        assert item["starts_at_local"].endswith(("-06:00", "-07:00", "-08:00", "-05:00"))


def test_responses_contain_no_patient_credential_or_internal_data(auth_env) -> None:
    response = listing(auth_env, "PT-1001")

    text_body = response.text
    for forbidden in (
        "PT-1001",
        "Avery",
        "patient",
        "pin",
        "hash",
        "token",
        "csrf",
        "session",
        "slot_id",
        "version",
        "cancelled_at",
        "argon2",
    ):
        assert forbidden not in text_body, forbidden


def test_every_seeded_patient_sees_only_their_own_records(auth_env) -> None:
    everything = set()
    for code, expected_count in SEEDED_OWNERS.items():
        mine = returned_ids(listing(auth_env, code))
        assert mine == ids_in_db(auth_env, code) and len(mine) == expected_count
        assert not mine & everything, f"{code} overlaps another patient"
        everything |= mine

    assert len(everything) == 11


def test_cancelled_appointments_are_kept_and_filterable(auth_env) -> None:
    cancelled = listing(auth_env, "PT-1008").json()["appointments"]

    assert [item["status"] for item in cancelled] == ["CANCELLED"]
    assert returned_ids(listing(auth_env, "PT-1008", status="CANCELLED")) == ids_in_db(
        auth_env, "PT-1008"
    )
    assert returned_ids(listing(auth_env, "PT-1008", status="CONFIRMED")) == set()
    assert returned_ids(listing(auth_env, "PT-1001", status="CANCELLED")) == set()


def test_patients_without_appointments_get_an_empty_page(auth_env) -> None:
    response = listing(auth_env, "PT-1014")

    assert response.status_code == 200
    assert response.json() == {"appointments": [], "next_cursor": None}


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"status": "PENDING"},
        {"status": "RESCHEDULED"},
        {"start_date": "2026-12-10", "end_date": "2026-12-09"},
        {"start_date": "10/12/2026"},
        {"limit": "21"},
        {"limit": "0"},
        {"cursor": "not-a-cursor"},
    ],
)
def test_invalid_listing_requests_are_rejected(auth_env, params) -> None:
    response = listing(auth_env, "PT-1001", **params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"


# --- pagination and date filtering -----------------------------------------------------------


def test_history_pages_in_start_order_and_cursors_cannot_cross_patients(auth_env) -> None:
    owner, other = patient_with_login(auth_env), patient_with_login(auth_env)
    created = book_history(auth_env, owner, 25)
    book_history(auth_env, other, 3)

    first = listing(auth_env, owner.code).json()
    second = listing(auth_env, owner.code, cursor=first["next_cursor"]).json()

    assert [len(first["appointments"]), len(second["appointments"])] == [20, 5]
    assert second["next_cursor"] is None and first["next_cursor"]
    flat = first["appointments"] + second["appointments"]
    assert [item["appointment_id"] for item in flat] == [str(i) for i in created]
    assert {item["status"] for item in flat} == {"CONFIRMED", "CANCELLED"}

    stolen = listing(auth_env, other.code, cursor=first["next_cursor"])
    assert stolen.status_code == 422 and stolen.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert returned_ids(listing(auth_env, other.code)).isdisjoint({str(i) for i in created})


def test_status_and_page_size_filters_combine(auth_env) -> None:
    owner = patient_with_login(auth_env)
    created = book_history(auth_env, owner, 10)

    confirmed = listing(auth_env, owner.code, status="CONFIRMED", limit=3).json()
    cancelled = listing(auth_env, owner.code, status="CANCELLED").json()

    assert [i["appointment_id"] for i in confirmed["appointments"]] == [
        str(i) for i in created[0::2][:3]
    ]
    assert confirmed["next_cursor"] is not None
    assert len(cancelled["appointments"]) == 5 and cancelled["next_cursor"] is None


def test_date_filters_use_each_appointments_clinic_local_date(auth_env) -> None:
    owner = patient_with_login(auth_env)
    clinic = new_clinic(auth_env.engine, timezone="America/Los_Angeles")
    doctor = new_doctor(auth_env.engine, clinic)
    day = date(2026, 12, 10)
    late, early = (day, time(23, 30)), (day + timedelta(days=1), time(0, 30))
    with auth_env.engine.begin() as connection:
        for moment in (late, early):
            [slot_id] = add_slots(auth_env.engine, clinic, doctor, [moment], status="BOOKED")
            make(connection, Appointment, patient_id=owner.patient_id, slot_id=slot_id)

    def local_dates(**params):
        body = listing(auth_env, owner.code, **params).json()
        return [item["starts_at_local"][:16] for item in body["appointments"]]

    # 23:30 PST on Dec 10 is already Dec 11 in UTC; the filter must follow the clinic's date.
    assert local_dates(start_date=day, end_date=day) == ["2026-12-10T23:30"]
    assert local_dates(start_date=day + timedelta(days=1), end_date=day + timedelta(days=1)) == [
        "2026-12-11T00:30"
    ]
    assert local_dates(start_date=day) == ["2026-12-10T23:30", "2026-12-11T00:30"]
    assert local_dates(end_date=day) == ["2026-12-10T23:30"]
    assert local_dates(start_date=day + timedelta(days=2)) == []
