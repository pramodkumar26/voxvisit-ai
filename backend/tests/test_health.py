import os

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app


def test_application_startup_initializes_database_engine(
    client: TestClient,
) -> None:
    assert client.app.state.db_engine is not None


def test_health_returns_application_availability(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "VoxVisit AI Backend",
        "version": "0.1.0",
    }


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DATABASE_TESTS") != "1",
    reason="local PostgreSQL integration tests are disabled",
)
def test_ready_succeeds_when_postgresql_is_available(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "available"}


def test_ready_returns_safe_error_when_database_is_unavailable() -> None:
    password = "response-secret"
    settings = Settings(
        database_url=SecretStr(f"postgresql+psycopg://voxvisit:{password}@127.0.0.1:1/voxvisit"),
        database_connect_timeout_seconds=1,
        _env_file=None,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "DATABASE_UNAVAILABLE",
            "message": "Database connectivity check failed.",
        }
    }
    assert password not in response.text
