import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

if os.getenv("RUN_DATABASE_TESTS") == "1":
    for key, value in dotenv_values(Path(__file__).parents[2] / ".env").items():
        if value is not None:
            os.environ.setdefault(key, value)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://voxvisit:test-only-password@127.0.0.1:5432/voxvisit",
)
os.environ.setdefault("APP_ENV", "test")

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
