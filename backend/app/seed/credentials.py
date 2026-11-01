from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.auth.passwords import (
    CredentialPolicyError,
    create_hasher,
    validate_patient_pin,
    validate_staff_password,
)

# Seeding fails with the same error type the shared policy raises.
SeedCredentialError = CredentialPolicyError

__all__ = [
    "SeedCredentialError",
    "SeedSettings",
    "create_hasher",
    "validate_patient_pin",
    "validate_staff_password",
]


class SeedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    patient_pin: SecretStr | None = Field(default=None, validation_alias="SEED_PATIENT_PIN")
    staff_password: SecretStr | None = Field(default=None, validation_alias="SEED_STAFF_PASSWORD")
