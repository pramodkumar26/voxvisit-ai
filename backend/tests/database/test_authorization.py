import uuid

import pytest
from auth_support import ORIGIN, call, sign_in
from sqlalchemy import text


def patient_id_of(env, code: str) -> str:
    with env.engine.connect() as connection:
        return str(
            connection.execute(
                text("SELECT patient_id FROM patients WHERE patient_code = :code"), {"code": code}
            ).scalar_one()
        )


def patient_session(env, code: str):
    return sign_in(env, "patient", code, env.pin)


def staff_session(env, username: str = "staff.morgan"):
    return sign_in(env, "staff", username, env.password)


def error_code(response) -> str:
    return response.json()["error"]["code"]


# --- public access ---------------------------------------------------------------------------


def test_public_routes_work_without_a_session_and_ignore_bad_cookies(auth_env) -> None:
    assert call(auth_env, "GET", "/_probe/public").json() == {"actor": None}
    assert call(auth_env, "GET", "/_probe/public", token="forged").json() == {"actor": None}

    session = patient_session(auth_env, "PT-1001")
    assert call(auth_env, "GET", "/_probe/public", session).json() == {"actor": "PATIENT"}

    auth_env.clock.advance(minutes=16)
    assert call(auth_env, "GET", "/_probe/public", session).json() == {"actor": None}


def test_protected_routes_require_a_valid_session(auth_env) -> None:
    patient = patient_session(auth_env, "PT-1001")

    assert error_code(call(auth_env, "GET", "/_probe/me")) == "AUTH_REQUIRED"
    assert error_code(call(auth_env, "GET", "/_probe/me", token="forged")) == "AUTH_REQUIRED"
    auth_env.clock.advance(minutes=16)
    assert error_code(call(auth_env, "GET", "/_probe/me", patient)) == "SESSION_EXPIRED"


# --- patient isolation -----------------------------------------------------------------------


def test_each_patient_session_resolves_only_its_own_identity(auth_env) -> None:
    first, second = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")

    assert call(auth_env, "GET", "/_probe/me", first).json() == {
        "patient_id": patient_id_of(auth_env, "PT-1001")
    }
    assert call(auth_env, "GET", "/_probe/me", second).json() == {
        "patient_id": patient_id_of(auth_env, "PT-1002")
    }


def test_a_patient_can_read_their_own_profile(auth_env) -> None:
    session = patient_session(auth_env, "PT-1001")
    own_id = patient_id_of(auth_env, "PT-1001")

    response = call(auth_env, "GET", f"/_probe/patients/{own_id}/profile", session)

    assert response.status_code == 200
    assert response.json() == {"display_name": "Avery Calloway"}


def test_a_patient_cannot_read_another_patients_record(auth_env) -> None:
    session = patient_session(auth_env, "PT-1001")
    other_id = patient_id_of(auth_env, "PT-1002")

    denied = call(auth_env, "GET", f"/_probe/patients/{other_id}/profile", session)
    missing = call(auth_env, "GET", f"/_probe/patients/{uuid.uuid4()}/profile", session)

    assert denied.status_code == 404
    assert denied.json() == missing.json()
    assert error_code(denied) == "RESOURCE_NOT_AVAILABLE"
    for leaked in ("Jordan", "Fairweather", "PT-1002", other_id):
        assert leaked not in denied.text


def test_stolen_cookie_with_another_sessions_csrf_token_is_rejected(auth_env) -> None:
    first, second = patient_session(auth_env, "PT-1001"), patient_session(auth_env, "PT-1002")

    response = call(auth_env, "POST", "/_probe/patient/action", first, csrf=second.csrf)

    assert response.status_code == 403 and error_code(response) == "CSRF_FAILED"


# --- role separation -------------------------------------------------------------------------


def test_staff_sessions_cannot_use_patient_routes_and_vice_versa(auth_env) -> None:
    patient, staff = patient_session(auth_env, "PT-1001"), staff_session(auth_env)

    staff_on_patient_route = call(auth_env, "GET", "/_probe/me", staff)
    patient_on_staff_route = call(auth_env, "GET", "/_probe/staff/schedule", patient)

    assert (
        staff_on_patient_route.status_code == 403
        and error_code(staff_on_patient_route) == "FORBIDDEN"
    )
    assert (
        patient_on_staff_route.status_code == 403
        and error_code(patient_on_staff_route) == "FORBIDDEN"
    )
    denials = (
        auth_env.engine.connect()
        .execute(
            text(
                "SELECT actor_type, result FROM security_audit_events"
                " WHERE event_type = 'ACCESS_DENIED'"
            )
        )
        .all()
    )
    assert sorted(tuple(row) for row in denials) == [("PATIENT", "DENIED"), ("STAFF", "DENIED")]


def test_staff_may_use_the_staff_permissions_they_hold(auth_env) -> None:
    staff = staff_session(auth_env)

    assert call(auth_env, "GET", "/_probe/staff/schedule", staff).status_code == 200
    assert call(auth_env, "POST", "/_probe/staff/availability", staff).status_code == 200


@pytest.mark.parametrize("path", ["/_probe/appointments/cancel", "/_probe/waitlist/join"])
def test_staff_cannot_mutate_patient_appointments_or_waitlists(auth_env, path: str) -> None:
    staff = staff_session(auth_env)

    response = call(auth_env, "POST", path, staff)

    assert response.status_code == 403 and error_code(response) == "FORBIDDEN"


@pytest.mark.parametrize("path", ["/_probe/appointments/cancel", "/_probe/waitlist/join"])
def test_patients_may_use_their_own_mutation_permissions(auth_env, path: str) -> None:
    patient = patient_session(auth_env, "PT-1001")

    assert call(auth_env, "POST", path, patient).status_code == 200


def test_patients_cannot_manage_doctor_availability_or_view_schedules(auth_env) -> None:
    patient = patient_session(auth_env, "PT-1001")

    assert call(auth_env, "POST", "/_probe/staff/availability", patient).status_code == 403
    assert call(auth_env, "GET", "/_probe/staff/schedule", patient).status_code == 403


def test_a_staff_session_cannot_be_used_after_the_staff_account_is_deactivated(auth_env) -> None:
    staff = staff_session(auth_env)
    with auth_env.engine.begin() as connection:
        connection.execute(text("UPDATE staff_users SET active = false"))

    assert call(auth_env, "GET", "/_probe/staff/schedule", staff).status_code == 401


# --- mutation protection on authorized routes ------------------------------------------------


def test_authenticated_mutations_need_origin_and_csrf_token(auth_env) -> None:
    session = patient_session(auth_env, "PT-1001")
    path = "/_probe/patient/action"

    assert call(auth_env, "POST", path, session).status_code == 200
    for bad in (
        call(auth_env, "POST", path, session, csrf=None),
        call(auth_env, "POST", path, session, csrf="wrong"),
        call(auth_env, "POST", path, session, origin=None),
        call(auth_env, "POST", path, session, origin="https://evil.example"),
        call(auth_env, "POST", path, session, origin=ORIGIN.upper() + ".evil"),
    ):
        assert bad.status_code == 403 and error_code(bad) == "CSRF_FAILED"


def test_authorization_responses_never_contain_credentials_or_hashes(auth_env) -> None:
    patient = patient_session(auth_env, "PT-1001")
    responses = [
        call(auth_env, "GET", "/auth/session", patient),
        call(auth_env, "GET", "/_probe/me", patient),
        call(auth_env, "GET", "/_probe/staff/schedule", patient),
    ]

    for response in responses:
        for secret in (auth_env.pin, "argon2", patient.token):
            assert secret not in response.text
