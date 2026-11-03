import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_database_url_requires_psycopg_driver() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=SecretStr("sqlite:///local.db"), _env_file=None)


def test_database_url_is_masked_in_settings() -> None:
    password = "settings-secret"
    settings = Settings(
        database_url=SecretStr(f"postgresql+psycopg://voxvisit:{password}@127.0.0.1:5432/voxvisit"),
        _env_file=None,
    )

    assert password not in repr(settings)
    assert str(settings.database_url) == "**********"


def test_the_repository_env_file_is_found_from_any_folder() -> None:
    from app.core.config import REPOSITORY_ENV_FILE

    # It points at the repository root (the folder holding backend/), not at the current folder.
    assert (REPOSITORY_ENV_FILE.parent / "backend" / "app" / "core" / "config.py").is_file()
    assert REPOSITORY_ENV_FILE.name == ".env"
    assert Settings.model_config["env_file"][0] == REPOSITORY_ENV_FILE
