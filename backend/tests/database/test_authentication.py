import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from auth_support import ORIGIN, build_app, call, login, sign_in
from database_support import FAST_HASHER
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.auth.tokens import hash_token

GENERIC_ERROR = {
    "error": {
        "code": "INVALID_CREDENTIALS",
        "message": "The credentials could not be verified.",
    }
}
PATIENT_A, PATIENT_B = "PT-1001", "PT-1002"
STAFF_A, STAFF_B = "staff.morgan", "staff.ellis"


def query(env: SimpleNamespace, sql: str, **params) -> list:
    with env.engine.connect() as connection:
        return [dict(row) for row in connection.execute(text(sql), params).mappings()]


def scalar(env: SimpleNamespace, sql: str, **params):
    rows = query(env, sql, **params)
    return next(iter(rows[0].values()))


def execute(env: SimpleNamespace, sql: str, **params) -> None:
    with env.engine.begin() as connection:
        connection.execute(text(sql), params)


def session_status(env: SimpleNamespace, token: str) -> dict:
    return query(
        env,
        "SELECT status, termination_reason, idle_expires_at, absolute_expires_at"
        " FROM auth_sessions WHERE token_hash = :hash",
        hash=hash_token(token),
    )[0]


def add_pending_action(env: SimpleNamespace, token: str) -> uuid.UUID:
    action_id = uuid.uuid4()
    execute(
        env,
        """INSERT INTO pending_actions (pending_action_id, session_id, action_type, target_type,
               summary, expected_versions, expires_at, request_id, created_at)
           SELECT :id, session_id, 'BOOK', 'SLOT', '{}'::jsonb, '{}'::jsonb,
                  :expires, :request, :created
           FROM auth_sessions WHERE token_hash = :hash""",
        id=action_id,
        hash=hash_token(token),
        request=uuid.uuid4(),
        created=env.clock.now,
        expires=env.clock.now.replace(minute=env.clock.now.minute + 10),
    )
    return action_id


def pending_status(env: SimpleNamespace, action_id: uuid.UUID) -> str:
    return scalar(
        env, "SELECT status FROM pending_actions WHERE pending_action_id = :id", id=action_id
    )


def audit_events(env: SimpleNamespace, event_type: str) -> list[dict]:
    return query(
        env,
        "SELECT * FROM security_audit_events WHERE event_type = :type ORDER BY occurred_at",
        type=event_type,
    )


# --- successful login ------------------------------------------------------------------------


def test_patient_login_creates_a_hashed_httponly_session(auth_env) -> None:
    response = login(auth_env, "patient", PATIENT_A, auth_env.pin)

    assert response.status_code == 200
    body = response.json()
    assert body["actor_type"] == "PATIENT" and body["login"] == PATIENT_A
    assert body["csrf_token"] and "token" not in {key for key in body if key != "csrf_token"}
    cookie = response.headers["set-cookie"]
    token = cookie.split(";")[0].split("=", 1)[1]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/" in cookie
    assert "Secure" not in cookie
    assert token not in response.text
    assert response.headers["cache-control"] == "no-store"

    [session] = query(auth_env, "SELECT * FROM auth_sessions")
    assert bytes(session["token_hash"]) == hashlib.sha256(token.encode()).digest()
    assert (session["idle_expires_at"] - auth_env.clock.now).total_seconds() == 15 * 60
    assert (session["absolute_expires_at"] - auth_env.clock.now).total_seconds() == 30 * 60
    assert session["status"] == "ACTIVE" and session["role"] == "PATIENT"
    for table in ("auth_sessions", "security_audit_events"):
        assert (
            scalar(
                auth_env,
                f"SELECT count(*) FROM {table} t WHERE t::text LIKE :token",
                token=f"%{token}%",
            )
            == 0
        )


def test_staff_login_uses_the_stricter_staff_lifetimes(auth_env) -> None:
    session = sign_in(auth_env, "staff", "STAFF.MORGAN", auth_env.password)

    body = session.response.json()
    assert body["actor_type"] == "STAFF" and body["login"] == STAFF_A
    status = session_status(auth_env, session.token)
    assert (status["idle_expires_at"] - auth_env.clock.now).total_seconds() == 10 * 60
    assert (status["absolute_expires_at"] - auth_env.clock.now).total_seconds() == 20 * 60


def test_production_cookies_are_secure_and_host_prefixed(auth_env) -> None:
    app = build_app(
        auth_env.engine,
        auth_env.clock,
        FAST_HASHER,
        environment="production",
        allowed_origins_csv="https://app.example.test",
    )
    with TestClient(app) as client:
        env = SimpleNamespace(**{**vars(auth_env), "client": client})
        response = login(env, "patient", PATIENT_A, auth_env.pin, origin="https://app.example.test")

    cookie = response.headers["set-cookie"]
    assert cookie.startswith("__Host-voxvisit_session=")
    assert "Secure" in cookie and "HttpOnly" in cookie and "Domain" not in cookie


# --- generic failures ------------------------------------------------------------------------

PATIENT_FAILURES = {
    "wrong-pin": ("PT-1003", "000000", None),
    "unknown-patient": ("PT-9999", None, None),
    "inactive-patient": (
        "PT-1004",
        None,
        "UPDATE patients SET active = false WHERE patient_code = 'PT-1004'",
    ),
    "disabled-credential": (
        "PT-1005",
        None,
        "UPDATE patient_credentials SET disabled = true WHERE patient_id ="
        " (SELECT patient_id FROM patients WHERE patient_code = 'PT-1005')",
    ),
    "malformed-pin": ("PT-1006", "12ab", None),
}
STAFF_FAILURES = {
    "wrong-password": ("staff.morgan", "definitely-not-the-password", None),
    "unknown-user": ("staff.nobody", None, None),
    "inactive-staff": (
        "staff.ellis",
        None,
        "UPDATE staff_users SET active = false WHERE username = 'staff.ellis'",
    ),
    "reset-required": (
        "staff.morgan",
        None,
        "UPDATE staff_credentials SET reset_required = true WHERE staff_id ="
        " (SELECT staff_id FROM staff_users WHERE username = 'staff.morgan')",
    ),
}


@pytest.mark.parametrize("case", PATIENT_FAILURES)
def test_patient_login_failures_are_indistinguishable(auth_env, case: str) -> None:
    identifier, secret, setup = PATIENT_FAILURES[case]
    if setup:
        execute(auth_env, setup)

    response = login(auth_env, "patient", identifier, secret or auth_env.pin)

    assert response.status_code == 401
    assert response.json() == GENERIC_ERROR
    assert "set-cookie" not in response.headers
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions") == 0


@pytest.mark.parametrize("case", STAFF_FAILURES)
def test_staff_login_failures_are_indistinguishable(auth_env, case: str) -> None:
    identifier, secret, setup = STAFF_FAILURES[case]
    if setup:
        execute(auth_env, setup)

    response = login(auth_env, "staff", identifier, secret or auth_env.password)

    assert response.status_code == 401
    assert response.json() == GENERIC_ERROR
    assert "set-cookie" not in response.headers
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions") == 0


def test_patient_credentials_cannot_sign_in_to_the_staff_portal(auth_env) -> None:
    assert login(auth_env, "staff", PATIENT_A, auth_env.pin).json() == GENERIC_ERROR
    assert login(auth_env, "patient", STAFF_A, auth_env.password).json() == GENERIC_ERROR


@pytest.mark.parametrize(
    "payload",
    [
        {"patient_code": "sentinel-code-AAA"},
        {"patient_code": "sentinel-code-BBB", "pin": 987654321},
        {"pin": "sentinel-pin-CCC", "extra": "sentinel-extra-DDD"},
    ],
)
def test_malformed_login_requests_never_echo_submitted_values(auth_env, payload: dict) -> None:
    response = auth_env.client.post("/auth/patient/login", json=payload, headers={"Origin": ORIGIN})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    for value in payload.values():
        assert str(value) not in response.text


# --- session lifecycle -----------------------------------------------------------------------


def test_missing_and_forged_cookies_are_rejected(auth_env) -> None:
    assert call(auth_env, "GET", "/auth/session").json()["error"]["code"] == "AUTH_REQUIRED"
    forged = call(auth_env, "GET", "/auth/session", token="not-a-real-session-token")
    assert forged.status_code == 401 and forged.json()["error"]["code"] == "AUTH_REQUIRED"


def test_current_session_reports_identity_without_secrets(auth_env) -> None:
    session = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)

    response = call(auth_env, "GET", "/auth/session", session)

    assert response.status_code == 200
    assert response.json()["login"] == PATIENT_A
    assert response.json()["csrf_token"] == session.csrf
    assert "argon2" not in response.text and session.token not in response.text


@pytest.mark.parametrize(
    ("kind", "identifier", "idle_minutes"), [("patient", PATIENT_A, 15), ("staff", STAFF_A, 10)]
)
def test_idle_sessions_expire_at_the_approved_timeout(
    auth_env, kind, identifier, idle_minutes
) -> None:
    secret = auth_env.pin if kind == "patient" else auth_env.password
    session = sign_in(auth_env, kind, identifier, secret)
    pending = add_pending_action(auth_env, session.token)

    auth_env.clock.advance(minutes=idle_minutes - 1, seconds=59)
    assert call(auth_env, "GET", "/auth/session", session).status_code == 200
    auth_env.clock.advance(minutes=idle_minutes)
    expired = call(auth_env, "GET", "/auth/session", session)

    assert expired.status_code == 401 and expired.json()["error"]["code"] == "SESSION_EXPIRED"
    assert session_status(auth_env, session.token)["termination_reason"] == "IDLE_TIMEOUT"
    assert pending_status(auth_env, pending) == "INVALIDATED"
    assert (
        call(auth_env, "GET", "/auth/session", session).json()["error"]["code"] == "SESSION_EXPIRED"
    )
    assert len(audit_events(auth_env, "SESSION_EXPIRED")) == 1


def test_a_lagging_application_clock_never_moves_session_activity_backwards(auth_env) -> None:
    """Instances do not share a perfectly synchronized clock; a lagging one must not fail or
    shorten a session (the stored activity time may not precede the creation time)."""
    session = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)
    before = session_status(auth_env, session.token)
    activity_before = query(
        auth_env,
        "SELECT last_activity_at FROM auth_sessions WHERE token_hash = :h",
        h=hash_token(session.token),
    )[0]["last_activity_at"]

    auth_env.clock.advance(minutes=-5)  # This instance's clock reads earlier than the login.
    response = call(auth_env, "GET", "/auth/session", session)

    assert response.status_code == 200
    after = session_status(auth_env, session.token)
    assert after["status"] == "ACTIVE"
    assert after["idle_expires_at"] == before["idle_expires_at"]  # Never shortened.
    assert (
        query(
            auth_env,
            "SELECT last_activity_at FROM auth_sessions WHERE token_hash = :h",
            h=hash_token(session.token),
        )[0]["last_activity_at"]
        == activity_before
    )


def test_activity_refreshes_idle_time_but_never_the_absolute_lifetime(auth_env) -> None:
    session = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)
    login_time = auth_env.clock.now
    absolute = session_status(auth_env, session.token)["absolute_expires_at"]

    auth_env.clock.advance(minutes=14)
    assert call(auth_env, "GET", "/auth/session", session).status_code == 200
    refreshed = session_status(auth_env, session.token)
    assert (refreshed["idle_expires_at"] - login_time).total_seconds() == 29 * 60
    assert refreshed["absolute_expires_at"] == absolute

    auth_env.clock.advance(minutes=14)
    assert call(auth_env, "GET", "/auth/session", session).status_code == 200
    assert session_status(auth_env, session.token)["idle_expires_at"] == absolute

    auth_env.clock.advance(minutes=3)
    expired = call(auth_env, "GET", "/auth/session", session)
    assert expired.json()["error"]["code"] == "SESSION_EXPIRED"
    assert session_status(auth_env, session.token)["termination_reason"] == "ABSOLUTE_TIMEOUT"


def test_staff_absolute_lifetime_is_twenty_minutes(auth_env) -> None:
    session = sign_in(auth_env, "staff", STAFF_A, auth_env.password)

    for _ in range(2):
        auth_env.clock.advance(minutes=9)
        assert call(auth_env, "GET", "/auth/session", session).status_code == 200
    auth_env.clock.advance(minutes=3)

    expired = call(auth_env, "GET", "/auth/session", session)
    assert expired.status_code == 401 and expired.json()["error"]["code"] == "SESSION_EXPIRED"
    assert session_status(auth_env, session.token)["termination_reason"] == "ABSOLUTE_TIMEOUT"


def test_logout_requires_csrf_then_revokes_the_token(auth_env) -> None:
    session = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)
    pending = add_pending_action(auth_env, session.token)

    assert call(auth_env, "POST", "/auth/logout", session, csrf=None).status_code == 403
    assert session_status(auth_env, session.token)["status"] == "ACTIVE"

    response = call(auth_env, "POST", "/auth/logout", session)

    assert response.status_code == 204
    assert (
        'voxvisit_session=""' in response.headers["set-cookie"]
        or "Max-Age=0" in response.headers["set-cookie"]
    )
    assert session_status(auth_env, session.token)["status"] == "LOGGED_OUT"
    assert pending_status(auth_env, pending) == "INVALIDATED"
    assert (
        call(auth_env, "GET", "/auth/session", session).json()["error"]["code"] == "AUTH_REQUIRED"
    )
    assert len(audit_events(auth_env, "LOGOUT")) == 1


def test_logout_without_a_session_is_harmless(auth_env) -> None:
    assert call(auth_env, "POST", "/auth/logout", csrf=None, origin=None).status_code == 204
    assert call(auth_env, "POST", "/auth/logout", token="forged", csrf=None).status_code == 204


def test_revoked_sessions_and_disabled_accounts_lose_access(auth_env) -> None:
    revoked = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)
    execute(
        auth_env,
        "UPDATE auth_sessions SET status = 'REVOKED', terminated_at = :now"
        " WHERE token_hash = :hash",
        now=auth_env.clock.now,
        hash=hash_token(revoked.token),
    )
    assert call(auth_env, "GET", "/auth/session", revoked).status_code == 401

    other = sign_in(auth_env, "patient", PATIENT_B, auth_env.pin)
    execute(
        auth_env, "UPDATE patients SET active = false WHERE patient_code = :code", code=PATIENT_B
    )
    assert call(auth_env, "GET", "/auth/session", other).status_code == 401
    assert session_status(auth_env, other.token)["termination_reason"] == "ACCOUNT_UNAVAILABLE"


# --- session limits --------------------------------------------------------------------------


def test_fourth_patient_login_revokes_the_oldest_session_and_its_pending_actions(auth_env) -> None:
    sessions, pending = [], []
    for _ in range(3):
        sessions.append(sign_in(auth_env, "patient", PATIENT_A, auth_env.pin))
        pending.append(add_pending_action(auth_env, sessions[-1].token))
        auth_env.clock.advance(seconds=1)

    fourth = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)

    assert call(auth_env, "GET", "/auth/session", sessions[0]).status_code == 401
    assert pending_status(auth_env, pending[0]) == "INVALIDATED"
    for survivor, action in zip([*sessions[1:], fourth], [*pending[1:], None], strict=True):
        assert call(auth_env, "GET", "/auth/session", survivor).status_code == 200
        if action:
            assert pending_status(auth_env, action) == "ACTIVE"
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'ACTIVE'") == 3
    assert session_status(auth_env, sessions[0].token)["termination_reason"] == "SESSION_LIMIT"
    assert len(audit_events(auth_env, "SESSION_REVOKED")) == 1


def test_expired_sessions_do_not_count_toward_the_patient_limit(auth_env) -> None:
    old = [sign_in(auth_env, "patient", PATIENT_A, auth_env.pin) for _ in range(3)]
    auth_env.clock.advance(minutes=20)

    fresh = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)

    assert call(auth_env, "GET", "/auth/session", fresh).status_code == 200
    statuses = {session_status(auth_env, s.token)["status"] for s in old}
    assert statuses == {"EXPIRED"}
    assert audit_events(auth_env, "SESSION_REVOKED") == []


def test_session_limits_are_per_account(auth_env) -> None:
    for _ in range(3):
        sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)

    other = sign_in(auth_env, "patient", PATIENT_B, auth_env.pin)

    assert call(auth_env, "GET", "/auth/session", other).status_code == 200
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'ACTIVE'") == 4


def test_new_staff_login_revokes_the_previous_session_first(auth_env) -> None:
    first = sign_in(auth_env, "staff", STAFF_A, auth_env.password)
    pending = add_pending_action(auth_env, first.token)
    other_staff = sign_in(auth_env, "staff", STAFF_B, auth_env.password)

    second = sign_in(auth_env, "staff", STAFF_A, auth_env.password)

    assert call(auth_env, "GET", "/auth/session", first).status_code == 401
    assert call(auth_env, "GET", "/auth/session", second).status_code == 200
    assert call(auth_env, "GET", "/auth/session", other_staff).status_code == 200
    assert pending_status(auth_env, pending) == "INVALIDATED"
    assert session_status(auth_env, first.token)["termination_reason"] == "REPLACED_BY_LOGIN"
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'ACTIVE'") == 2


# --- concurrency -----------------------------------------------------------------------------


def concurrently(count: int, work):
    """Run work(index) in parallel threads, each as a different network source.

    Distinct sources mean the per-source advisory lock cannot serialize the threads, so these
    tests exercise the account-level row locks on their own.
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(work, range(count)))


def service_login(env: SimpleNamespace, kind: str, identifier: str, secret: str, address=None):
    return env.service.login(
        kind, identifier, secret, network_address=address, request_id=uuid.uuid4()
    )


class SlowVerifier:
    """Delays verification so concurrent logins really overlap while they hold their locks."""

    def __init__(self, inner, delay: float = 0.03) -> None:
        self._inner, self._delay = inner, delay

    def verify(self, *args):
        time.sleep(self._delay)
        return self._inner.verify(*args)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def slow_hashing(env: SimpleNamespace) -> None:
    env.service._hasher = SlowVerifier(env.service._hasher)


def source(index: int) -> str:
    return f"198.51.100.{index}"


def test_concurrent_patient_logins_never_exceed_three_active_sessions(auth_env) -> None:
    slow_hashing(auth_env)
    results = concurrently(
        24, lambda i: service_login(auth_env, "PATIENT", PATIENT_A, auth_env.pin, source(i))
    )

    assert all(result.issued for result in results)
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'ACTIVE'") == 3
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'REVOKED'") == 21


def test_concurrent_staff_logins_leave_exactly_one_active_session(auth_env) -> None:
    slow_hashing(auth_env)
    results = concurrently(
        24, lambda i: service_login(auth_env, "STAFF", STAFF_A, auth_env.password, source(i))
    )

    assert all(result.issued for result in results)
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions WHERE status = 'ACTIVE'") == 1


def test_concurrent_wrong_pins_cannot_bypass_the_account_lockout(auth_env) -> None:
    slow_hashing(auth_env)
    concurrently(12, lambda i: service_login(auth_env, "PATIENT", PATIENT_A, "000000", source(i)))

    credential = query(
        auth_env,
        "SELECT failed_attempt_count, locked_until FROM patient_credentials"
        " WHERE patient_id = (SELECT patient_id FROM patients WHERE patient_code = :code)",
        code=PATIENT_A,
    )[0]
    reasons = query(
        auth_env,
        "SELECT details->>'reason' AS reason, count(*) AS n FROM security_audit_events"
        " WHERE event_type = 'LOGIN_FAILURE' GROUP BY 1",
    )
    assert credential["failed_attempt_count"] == 5 and credential["locked_until"] is not None
    assert {row["reason"]: row["n"] for row in reasons} == {
        "bad_credential": 5,
        "account_locked": 7,
    }
    assert len(audit_events(auth_env, "ACCOUNT_LOCKED")) == 1


def test_concurrent_attempts_cannot_bypass_the_network_limit(auth_env) -> None:
    results = concurrently(
        40, lambda _: service_login(auth_env, "PATIENT", "PT-9999", "000000", "203.0.113.7")
    )

    assert sum(1 for result in results if result.rate_limited) == 20
    assert len(audit_events(auth_env, "LOGIN_FAILURE")) == 20
    assert len(audit_events(auth_env, "LOGIN_RATE_LIMITED")) == 20


# --- account lockout -------------------------------------------------------------------------


def fail_login(env: SimpleNamespace, count: int, kind="patient", identifier=PATIENT_A) -> None:
    for _ in range(count):
        assert login(env, kind, identifier, "000000-wrong").status_code == 401


@pytest.mark.parametrize(
    ("kind", "identifier", "lock_minutes"), [("patient", PATIENT_A, 15), ("staff", STAFF_A, 30)]
)
def test_five_failures_lock_the_account_until_the_approved_time(
    auth_env, kind, identifier, lock_minutes
) -> None:
    secret = auth_env.pin if kind == "patient" else auth_env.password
    fail_login(auth_env, 4, kind, identifier)
    assert login(auth_env, kind, identifier, secret).status_code == 200
    fail_login(auth_env, 5, kind, identifier)

    locked = login(auth_env, kind, identifier, secret)
    wrong = login(auth_env, kind, identifier, "still-wrong-000")
    assert locked.status_code == 401 and locked.json() == wrong.json() == GENERIC_ERROR

    auth_env.clock.advance(minutes=lock_minutes - 1, seconds=59)
    assert login(auth_env, kind, identifier, secret).status_code == 401
    auth_env.clock.advance(seconds=1)
    assert login(auth_env, kind, identifier, secret).status_code == 200

    counters = query(
        auth_env,
        f"SELECT failed_attempt_count, locked_until FROM {kind}_credentials"
        " ORDER BY updated_at DESC LIMIT 1",
    )[0]
    assert counters["failed_attempt_count"] == 0 and counters["locked_until"] is None
    assert len(audit_events(auth_env, "ACCOUNT_LOCKED")) == 1


def test_a_successful_login_clears_earlier_failures(auth_env) -> None:
    fail_login(auth_env, 3)
    assert login(auth_env, "patient", PATIENT_A, auth_env.pin).status_code == 200

    fail_login(auth_env, 4)

    assert login(auth_env, "patient", PATIENT_A, auth_env.pin).status_code == 200


def test_failures_outside_the_fifteen_minute_window_do_not_accumulate(auth_env) -> None:
    fail_login(auth_env, 4)
    auth_env.clock.advance(minutes=16)

    fail_login(auth_env, 1)

    assert login(auth_env, "patient", PATIENT_A, auth_env.pin).status_code == 200
    assert audit_events(auth_env, "ACCOUNT_LOCKED") == []


def test_lockout_is_per_account(auth_env) -> None:
    fail_login(auth_env, 5)

    assert login(auth_env, "patient", PATIENT_B, auth_env.pin).status_code == 200


# --- network rate limiting -------------------------------------------------------------------


def test_network_source_is_limited_to_twenty_attempts_per_fifteen_minutes(auth_env) -> None:
    for _ in range(20):
        assert (
            login(auth_env, "patient", "PT-9999", "000000", address="198.51.100.1").status_code
            == 401
        )

    limited = login(auth_env, "patient", PATIENT_A, auth_env.pin, address="198.51.100.1")

    assert limited.status_code == 429 and limited.json()["error"]["code"] == "RATE_LIMITED"
    assert 1 <= int(limited.headers["retry-after"]) <= 15 * 60
    assert (
        login(auth_env, "patient", PATIENT_A, auth_env.pin, address="198.51.100.2").status_code
        == 200
    )
    assert len(audit_events(auth_env, "LOGIN_RATE_LIMITED")) == 1

    auth_env.clock.advance(minutes=15)
    assert (
        login(auth_env, "patient", PATIENT_A, auth_env.pin, address="198.51.100.1").status_code
        == 200
    )


def test_forwarded_addresses_are_ignored_unless_trusted(auth_env) -> None:
    app = build_app(auth_env.engine, auth_env.clock, FAST_HASHER, auth_trust_forwarded_for=False)
    with TestClient(app) as client:
        env = SimpleNamespace(**{**vars(auth_env), "client": client})
        for number in range(20):
            login(env, "patient", "PT-9999", "000000", address=f"198.51.100.{number}")

        spoofed = login(env, "patient", PATIENT_A, auth_env.pin, address="192.0.2.99")

    assert spoofed.status_code == 429


def test_audit_records_never_store_raw_network_addresses(auth_env) -> None:
    login(auth_env, "patient", PATIENT_A, auth_env.pin, address="203.0.113.50")

    sources = {
        row["network_source"]
        for row in query(auth_env, "SELECT network_source FROM security_audit_events")
    }

    assert sources and all(len(source) == 32 and "203.0.113.50" not in source for source in sources)


# --- CSRF, origin, and CORS ------------------------------------------------------------------


@pytest.mark.parametrize("origin", [None, "https://evil.example", "http://localhost:3001", "null"])
def test_login_rejects_missing_or_foreign_origins(auth_env, origin) -> None:
    response = login(auth_env, "patient", PATIENT_A, auth_env.pin, origin=origin)

    assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_FAILED"
    assert scalar(auth_env, "SELECT count(*) FROM auth_sessions") == 0
    assert len(audit_events(auth_env, "CSRF_REJECTED")) == 1


def test_logout_rejects_missing_wrong_and_foreign_session_csrf_tokens(auth_env) -> None:
    mine = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)
    theirs = sign_in(auth_env, "patient", PATIENT_B, auth_env.pin)

    rejected = [
        call(auth_env, "POST", "/auth/logout", mine, csrf=None),
        call(auth_env, "POST", "/auth/logout", mine, csrf="forged-token"),
        call(auth_env, "POST", "/auth/logout", mine, csrf=theirs.csrf),
        call(auth_env, "POST", "/auth/logout", mine, origin="https://evil.example"),
        call(auth_env, "POST", "/auth/logout", mine, origin=None),
    ]

    assert [r.status_code for r in rejected] == [403] * 5
    assert session_status(auth_env, mine.token)["status"] == "ACTIVE"
    assert call(auth_env, "POST", "/auth/logout", mine).status_code == 204
    assert len(audit_events(auth_env, "CSRF_REJECTED")) == 5


def test_safe_requests_do_not_need_csrf_tokens(auth_env) -> None:
    session = sign_in(auth_env, "patient", PATIENT_A, auth_env.pin)

    response = call(auth_env, "GET", "/auth/session", session, origin="https://evil.example")

    assert response.status_code == 200


def test_credentialed_cors_is_limited_to_the_allowed_origins(auth_env) -> None:
    def preflight(origin: str):
        return auth_env.client.options(
            "/auth/logout",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-CSRF-Token,Content-Type",
            },
        )

    allowed = preflight(ORIGIN)
    foreign = preflight("https://evil.example")

    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "x-csrf-token" in allowed.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-origin" not in foreign.headers
    assert allowed.headers["access-control-allow-origin"] != "*"


# --- audit and secret hygiene ----------------------------------------------------------------


def test_authentication_events_are_audited_without_secrets(auth_env, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    fail_login(auth_env, 5)
    session = sign_in(auth_env, "patient", PATIENT_B, auth_env.pin)
    for _ in range(3):
        sign_in(auth_env, "patient", PATIENT_B, auth_env.pin)
    call(auth_env, "POST", "/auth/logout", session)

    recorded = {
        row["event_type"] for row in query(auth_env, "SELECT event_type FROM security_audit_events")
    }
    assert {"LOGIN_FAILURE", "ACCOUNT_LOCKED", "LOGIN_SUCCESS", "SESSION_REVOKED"} <= recorded
    dump = " ".join(
        row["dump"]
        for row in query(auth_env, "SELECT t::text AS dump FROM security_audit_events t")
    )
    for secret in (
        auth_env.pin,
        auth_env.password,
        session.token,
        session.csrf,
        "argon2",
        "000000-wrong",
    ):
        assert secret not in dump, "audit table"
        assert secret not in caplog.text, "application log"
