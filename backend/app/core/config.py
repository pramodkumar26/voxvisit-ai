import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEVELOPMENT_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
)
MIN_AUTH_SECRET_LENGTH = 32
# The repository's .env wherever a command is run from (backend/ or the repository root). A .env in
# the current folder is read after it and wins, so a local override still works.
REPOSITORY_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
ORIGIN_PATTERN = re.compile(r"https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPOSITORY_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="VoxVisit AI Backend", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )
    database_url: SecretStr = Field(validation_alias="DATABASE_URL")
    database_pool_size: int = Field(default=5, ge=1, le=20)
    database_max_overflow: int = Field(default=5, ge=0, le=20)
    database_pool_timeout_seconds: int = Field(default=5, ge=1, le=30)
    database_connect_timeout_seconds: int = Field(default=5, ge=1, le=30)

    auth_secret_key: SecretStr | None = Field(default=None, validation_alias="AUTH_SECRET_KEY")
    allowed_origins_csv: str = Field(
        default=DEVELOPMENT_ORIGINS,
        validation_alias="ALLOWED_ORIGINS",
    )
    auth_cookie_secure: bool | None = Field(default=None, validation_alias="AUTH_COOKIE_SECURE")
    auth_cookie_samesite: Literal["lax", "strict", "none"] = Field(
        default="lax",
        validation_alias="AUTH_COOKIE_SAMESITE",
    )
    auth_cookie_domain: str | None = Field(default=None, validation_alias="AUTH_COOKIE_DOMAIN")
    auth_trust_forwarded_for: bool = Field(
        default=False,
        validation_alias="AUTH_TRUST_FORWARDED_FOR",
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use the postgresql+psycopg driver")
        return value

    @field_validator("auth_secret_key")
    @classmethod
    def validate_auth_secret_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < MIN_AUTH_SECRET_LENGTH:
            raise ValueError(
                f"AUTH_SECRET_KEY must be at least {MIN_AUTH_SECRET_LENGTH} characters"
            )
        return value

    @field_validator("allowed_origins_csv")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        for origin in _split_origins(value):
            if not ORIGIN_PATTERN.fullmatch(origin):
                raise ValueError(
                    "ALLOWED_ORIGINS entries must be exact http(s) origins, no wildcards"
                )
        return value

    @model_validator(mode="after")
    def validate_cookie_and_production_settings(self) -> "Settings":
        if self.auth_cookie_samesite == "none" and not self.session_cookie_secure:
            raise ValueError("SameSite=None requires secure cookies")
        if self.environment == "production":
            if self.auth_cookie_secure is False:
                raise ValueError("Secure cookies cannot be disabled in production")
            if self.auth_secret_key is None:
                raise ValueError("AUTH_SECRET_KEY is required in production")
            if not self.allowed_origins or any(
                not origin.startswith("https://") for origin in self.allowed_origins
            ):
                raise ValueError("Production ALLOWED_ORIGINS must be explicit https origins")
        return self

    @property
    def allowed_origins(self) -> list[str]:
        return _split_origins(self.allowed_origins_csv)

    @property
    def session_cookie_secure(self) -> bool:
        if self.auth_cookie_secure is not None:
            return self.auth_cookie_secure
        return self.environment == "production"

    @property
    def session_cookie_name(self) -> str:
        if self.session_cookie_secure and not self.auth_cookie_domain:
            return "__Host-voxvisit_session"
        return "voxvisit_session"


def _split_origins(value: str) -> list[str]:
    return [origin.strip().lower() for origin in value.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
