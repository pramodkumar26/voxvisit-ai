from functools import partial

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.api.routes import scheduling as scheduling_routes
from app.core.errors import DependencyUnavailableError
from app.services import directory
from app.services.read_access import READ_TIMEOUT_MS, read_only_connection


def test_the_default_read_timeout_is_the_approved_eight_seconds(auth_env) -> None:
    assert READ_TIMEOUT_MS == 8000

    with read_only_connection(auth_env.engine) as connection:
        assert connection.execute(text("SHOW statement_timeout")).scalar_one() == "8s"
        assert connection.execute(text("SHOW transaction_timeout")).scalar_one() == "8s"
        assert connection.execute(text("SHOW transaction_read_only")).scalar_one() == "on"


def test_read_services_cannot_write(auth_env) -> None:
    with pytest.raises(DBAPIError), read_only_connection(auth_env.engine) as connection:
        connection.execute(text("UPDATE clinics SET active = active"))


def test_slow_queries_are_cancelled_and_reported_as_a_dependency_failure(auth_env) -> None:
    with (
        pytest.raises(DependencyUnavailableError) as exc_info,
        read_only_connection(auth_env.engine, timeout_ms=50) as connection,
    ):
        connection.execute(text("SELECT pg_sleep(2)"))

    assert exc_info.value.code == "DEPENDENCY_UNAVAILABLE" and exc_info.value.status_code == 503


def test_a_timed_out_endpoint_returns_a_safe_503(auth_env, monkeypatch) -> None:
    monkeypatch.setattr(
        scheduling_routes, "read_only_connection", partial(read_only_connection, timeout_ms=50)
    )
    monkeypatch.setattr(
        directory,
        "search_directory",
        lambda connection, **_: connection.execute(text("SELECT pg_sleep(2)")),
    )

    response = auth_env.client.get("/directory", params={"browse": "true"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "DEPENDENCY_UNAVAILABLE",
            "message": "The service is temporarily unavailable. Please try again.",
        }
    }
    for leak in ("pg_sleep", "statement", "psycopg", "SELECT"):
        assert leak not in response.text


def test_the_database_is_reachable_again_after_a_timeout(auth_env) -> None:
    with (
        pytest.raises(DependencyUnavailableError),
        read_only_connection(auth_env.engine, timeout_ms=50) as connection,
    ):
        connection.execute(text("SELECT pg_sleep(2)"))

    with read_only_connection(auth_env.engine) as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1


def test_the_read_timeout_bounds_the_whole_transaction(auth_env) -> None:
    with (
        pytest.raises(DependencyUnavailableError),
        read_only_connection(auth_env.engine, timeout_ms=600) as connection,
    ):
        for _ in range(10):
            connection.execute(text("SELECT pg_sleep(0.2)"))

    with read_only_connection(auth_env.engine) as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
