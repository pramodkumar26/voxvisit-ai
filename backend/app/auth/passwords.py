"""Credential policy and Argon2id hashing shared by the API and development tooling."""

import re

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

MIN_STAFF_PASSWORD_LENGTH = 12
PIN_PATTERN = re.compile(r"[0-9]{6}")

# A bundled stand-in for the approved common/compromised-password list. Entries are stored
# normalized (lowercase letters and digits only). Production should replace or extend it.
COMMON_PASSWORDS = frozenset(
    {
        "password1234",
        "password12345",
        "password123456",
        "passwordpassword",
        "passw0rd1234",
        "p4ssw0rd1234",
        "letmein123456",
        "letmeinletmein",
        "welcome12345",
        "welcome123456",
        "iloveyou1234",
        "iloveyou123456",
        "qwerty123456",
        "qwertyuiop12",
        "qwertyuiopas",
        "1q2w3e4r5t6y",
        "1qaz2wsx3edc",
        "zaq12wsxcde3",
        "abc123abc123",
        "administrator",
        "admin123456",
        "adminadmin12",
        "changeme1234",
        "changeme123456",
        "trustno1trustno1",
        "monkey123456",
        "dragon123456",
        "football1234",
        "baseball1234",
        "superman1234",
        "sunshine1234",
        "princess1234",
        "mustang12345",
        "shadow123456",
        "master123456",
        "hello1234567",
        "freedom12345",
        "whatever1234",
        "starwars1234",
        "charlie12345",
        "voxvisit1234",
        "voxvisit12345",
        "voxvisitvoxvisit",
        "staffpassword",
        "staffpassword1",
        "demopassword",
        "demopassword1",
        "demo12345678",
        "testpassword1",
        "testpassword12",
        "correcthorsebatterystaple",
    }
)
WEAK_STEMS = (
    "password",
    "passw0rd",
    "qwerty",
    "letmein",
    "welcome",
    "iloveyou",
    "admin",
    "changeme",
    "voxvisit",
)
KEYBOARD_RUNS = ("qwertyuiopasdfghjklzxcvbnm", "1234567890" * 2, "abcdefghijklmnopqrstuvwxyz")
PRODUCT_NAME = "voxvisit"
MIN_RUN_LENGTH = 8


class CredentialPolicyError(ValueError):
    """Raised for credentials that break policy. Messages never contain the value."""


def create_hasher() -> PasswordHasher:
    """Argon2id with the library's current recommended defaults."""
    return PasswordHasher()


def verify_secret(hasher: PasswordHasher, stored_hash: str, secret: str) -> bool:
    try:
        return hasher.verify(stored_hash, secret)
    except (VerificationError, InvalidHashError):
        return False


def validate_patient_pin(pin: str) -> None:
    if not PIN_PATTERN.fullmatch(pin):
        raise CredentialPolicyError("The patient PIN must be exactly six digits.")


def validate_staff_password(password: str, *, username: str | None = None) -> None:
    if len(password) < MIN_STAFF_PASSWORD_LENGTH:
        raise CredentialPolicyError(
            f"The staff password must be at least {MIN_STAFF_PASSWORD_LENGTH} characters."
        )
    normalized = _normalize(password)
    if (
        normalized in COMMON_PASSWORDS
        or _is_weak_stem_with_digits(normalized)
        or len(set(normalized)) <= 2
        or PRODUCT_NAME in normalized
        or _is_keyboard_or_sequence_run(normalized)
        or (username and username.casefold() in password.casefold())
    ):
        raise CredentialPolicyError("The staff password is too common or too easy to guess.")


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_weak_stem_with_digits(normalized: str) -> bool:
    return any(
        normalized.startswith(stem) and (normalized[len(stem) :].isdigit() or normalized == stem)
        for stem in WEAK_STEMS
    )


def _is_keyboard_or_sequence_run(normalized: str) -> bool:
    runs = [*KEYBOARD_RUNS, *(run[::-1] for run in KEYBOARD_RUNS)]
    return len(normalized) >= MIN_RUN_LENGTH and any(normalized in run for run in runs)
