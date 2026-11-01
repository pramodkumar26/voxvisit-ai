import pytest
from pydantic import SecretStr, ValidationError

from app.auth.passwords import (
    COMMON_PASSWORDS,
    CredentialPolicyError,
    create_hasher,
    validate_patient_pin,
    validate_staff_password,
    verify_secret,
)
from app.auth.permissions import ROLE_PERMISSIONS, Permission, has_permission
from app.auth.policy import PATIENT_POLICY, STAFF_POLICY
from app.auth.tokens import (
    csrf_token_for,
    hash_token,
    network_fingerprint,
    new_session_token,
    tokens_match,
)
from app.core.config import Settings

KEY = b"k" * 32
PRODUCTION = {
    "environment": "production",
    "auth_secret_key": SecretStr("production-secret-key-0123456789ab"),
    "allowed_origins_csv": "https://app.example.test",
}


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- approved policy values ------------------------------------------------------------------


def test_session_policies_match_the_approved_values() -> None:
    assert (
        PATIENT_POLICY.absolute_lifetime.total_seconds(),
        PATIENT_POLICY.idle_timeout.total_seconds(),
    ) == (1800, 900)
    assert (
        STAFF_POLICY.absolute_lifetime.total_seconds(),
        STAFF_POLICY.idle_timeout.total_seconds(),
    ) == (1200, 600)
    assert (PATIENT_POLICY.max_active_sessions, STAFF_POLICY.max_active_sessions) == (3, 1)
    assert (
        PATIENT_POLICY.lockout_duration.total_seconds(),
        STAFF_POLICY.lockout_duration.total_seconds(),
    ) == (900, 1800)


# --- tokens ----------------------------------------------------------------------------------


def test_session_tokens_are_unique_high_entropy_and_stored_only_as_hashes() -> None:
    tokens = {new_session_token() for _ in range(200)}

    assert len(tokens) == 200 and all(len(token) >= 43 for token in tokens)
    token = next(iter(tokens))
    assert hash_token(token) != token.encode() and len(hash_token(token)) == 32


def test_csrf_tokens_are_bound_to_one_session_and_one_server_key() -> None:
    first, second = hash_token("session-one"), hash_token("session-two")

    assert csrf_token_for(KEY, first) == csrf_token_for(KEY, first)
    assert csrf_token_for(KEY, first) != csrf_token_for(KEY, second)
    assert csrf_token_for(KEY, first) != csrf_token_for(b"x" * 32, first)
    assert tokens_match(csrf_token_for(KEY, first), csrf_token_for(KEY, first))
    assert not tokens_match(csrf_token_for(KEY, first), csrf_token_for(KEY, second))
    assert not tokens_match("expected", None) and not tokens_match("expected", "")


def test_network_fingerprints_hide_the_raw_address() -> None:
    fingerprint = network_fingerprint(KEY, "203.0.113.9")

    assert len(fingerprint) == 32 and "203" not in fingerprint
    assert fingerprint == network_fingerprint(KEY, "203.0.113.9")
    assert fingerprint != network_fingerprint(KEY, "203.0.113.10")
    assert fingerprint != network_fingerprint(b"y" * 32, "203.0.113.9")


# --- credential policy -----------------------------------------------------------------------


@pytest.mark.parametrize("pin", ["000000", "123456", "905173"])
def test_six_digit_pins_are_accepted(pin: str) -> None:
    validate_patient_pin(pin)


@pytest.mark.parametrize(
    "pin", ["", "12345", "1234567", "12345a", " 123456", "123456\n", "\uff11" * 6]
)
def test_other_pins_are_rejected(pin: str) -> None:
    with pytest.raises(CredentialPolicyError):
        validate_patient_pin(pin)


@pytest.mark.parametrize(
    "password",
    [
        "short",
        "elevenchars",
        "password1234",
        "Password12345",
        "PASSWORD123456",
        "qwertyuiop12",
        "Qwerty-123456",
        "abcdefghijkl",
        "1234567890123",
        "zyxwvutsrqpo",
        "qwertyuiopasdf",
        "aaaaaaaaaaaa",
        "ababababababab",
        "Welcome 2026 !!",
        "VoxVisit-Staff-1",
        "changeme123456",
    ],
)
def test_weak_or_common_staff_passwords_are_rejected(password: str) -> None:
    with pytest.raises(CredentialPolicyError) as exc_info:
        validate_staff_password(password)

    assert password not in str(exc_info.value)


@pytest.mark.parametrize(
    "password", ["Tw3lve-chars-ok", "purple tractor sings loudly", "k9#Lm2$vQx7!pZ"]
)
def test_strong_staff_passwords_are_accepted(password: str) -> None:
    validate_staff_password(password)


def test_staff_passwords_may_not_contain_the_username() -> None:
    with pytest.raises(CredentialPolicyError):
        validate_staff_password("morgan-Xk3-mm9-Q", username="Morgan")
    validate_staff_password("river-Xk3-mm9-Q", username="Morgan")


def test_common_password_list_is_normalized() -> None:
    assert all(
        entry == "".join(c for c in entry if c.isalnum()).lower() for entry in COMMON_PASSWORDS
    )


def test_argon2id_hashes_verify_and_reject_garbage() -> None:
    hasher = create_hasher()
    digest = hasher.hash("905173")

    assert digest.startswith("$argon2id$")
    assert verify_secret(hasher, digest, "905173")
    assert not verify_secret(hasher, digest, "905174")
    assert not verify_secret(hasher, "not-a-hash", "905173")


# --- permission matrix -----------------------------------------------------------------------


def test_staff_have_no_appointment_waitlist_or_patient_assistance_mutation_rights() -> None:
    forbidden = {
        Permission.VIEW_OWN_PROFILE,
        Permission.VIEW_OWN_APPOINTMENTS,
        Permission.MANAGE_OWN_APPOINTMENTS,
        Permission.MANAGE_OWN_WAITLIST,
        Permission.REQUEST_PATIENT_SPECIFIC_ASSISTANCE,
    }

    assert not forbidden & ROLE_PERMISSIONS["STAFF"]
    for permission in forbidden:
        assert not has_permission("STAFF", permission)


def test_patients_have_no_staff_rights() -> None:
    staff_only = ROLE_PERMISSIONS["STAFF"] - {Permission.REQUEST_GENERAL_ASSISTANCE}

    assert not staff_only & ROLE_PERMISSIONS["PATIENT"]


def test_unknown_roles_hold_no_permissions() -> None:
    assert not any(has_permission("ADMIN", permission) for permission in Permission)


# --- deployment settings ---------------------------------------------------------------------


def test_development_defaults_are_safe_for_local_use() -> None:
    development = settings()

    assert development.session_cookie_secure is False
    assert development.session_cookie_name == "voxvisit_session"
    assert development.auth_cookie_samesite == "lax"
    assert "*" not in development.allowed_origins


def test_production_cookies_are_secure_and_host_prefixed_by_default() -> None:
    production = settings(**PRODUCTION)

    assert production.session_cookie_secure is True
    assert production.session_cookie_name == "__Host-voxvisit_session"
    assert (
        settings(**PRODUCTION, auth_cookie_domain="example.test").session_cookie_name
        == "voxvisit_session"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"auth_secret_key": None},
        {"auth_cookie_secure": False},
        {"allowed_origins_csv": "http://app.example.test"},
        {"allowed_origins_csv": ""},
        {"allowed_origins_csv": "*"},
    ],
)
def test_production_rejects_unsafe_settings(overrides) -> None:
    with pytest.raises(ValidationError):
        settings(**{**PRODUCTION, **overrides})


@pytest.mark.parametrize(
    "origins", ["*", "https://*.example.test", "https://a.test/path", "ftp://a.test", "app.test"]
)
def test_allowed_origins_must_be_exact_origins(origins: str) -> None:
    with pytest.raises(ValidationError):
        settings(allowed_origins_csv=origins)


def test_cross_site_cookies_require_secure_and_short_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        settings(auth_cookie_samesite="none")
    assert (
        settings(auth_cookie_samesite="none", auth_cookie_secure=True).auth_cookie_samesite
        == "none"
    )
    with pytest.raises(ValidationError):
        settings(auth_secret_key=SecretStr("too-short"))


def test_allowed_origins_are_normalized() -> None:
    parsed = settings(allowed_origins_csv=" HTTPS://App.Example.test , http://localhost:3000 ")

    assert parsed.allowed_origins == ["https://app.example.test", "http://localhost:3000"]
