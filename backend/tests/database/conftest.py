import os
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from auth_support import build_app
from database_support import (
    FAST_HASHER,
    FakeClock,
    disposable_database,
    random_credentials,
    run_migration,
)
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, text

from app.evaluation.graph_store import CheckpointStore
from app.seed.loader import seed_database


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_reason = "local PostgreSQL integration tests are disabled"
    for item in items:
        if Path(item.fspath).parent != Path(__file__).parent:
            continue
        item.add_marker(pytest.mark.integration)
        if os.getenv("RUN_DATABASE_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason=skip_reason))


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    with disposable_database() as engine:
        run_migration(engine, "upgrade", "head")
        yield engine


@pytest.fixture
def fresh_database() -> Iterator[Engine]:
    with disposable_database() as engine:
        yield engine


@pytest.fixture
def migrate() -> Callable[[Engine, str, str], None]:
    return run_migration


@pytest.fixture
def conn(migrated_engine: Engine) -> Iterator[Connection]:
    with migrated_engine.connect() as connection:
        transaction = connection.begin()
        yield connection
        transaction.rollback()


@pytest.fixture(scope="module")
def auth_database() -> Iterator[SimpleNamespace]:
    credentials = random_credentials()
    with disposable_database() as engine:
        run_migration(engine, "upgrade", "head")
        seed_database(
            engine,
            patient_pin=credentials.pin,
            staff_password=credentials.password,
            anchor_date=date(2026, 10, 28),
            hasher=FAST_HASHER,
        )
        yield SimpleNamespace(engine=engine, pin=credentials.pin, password=credentials.password)


@pytest.fixture
def auth_env(auth_database: SimpleNamespace) -> Iterator[SimpleNamespace]:
    engine = auth_database.engine
    clock = FakeClock(datetime(2026, 11, 2, 15, 0, tzinfo=UTC))
    with engine.begin() as connection:
        for statement in (
            "DELETE FROM security_audit_events",
            "DELETE FROM auth_sessions",
            "UPDATE patients SET active = true",
            "UPDATE staff_users SET active = true",
            "UPDATE patient_credentials SET failed_attempt_count = 0,"
            " failed_attempt_window_started_at = NULL, locked_until = NULL, disabled = false",
            "UPDATE staff_credentials SET failed_attempt_count = 0,"
            " failed_attempt_window_started_at = NULL, locked_until = NULL, reset_required = false",
        ):
            connection.execute(text(statement))
    app = build_app(engine, clock, FAST_HASHER)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            engine=engine,
            clock=clock,
            service=app.state.auth_service,
            pin=auth_database.pin,
            password=auth_database.password,
        )


@pytest.fixture
def graph_store(auth_env: SimpleNamespace) -> Iterator[CheckpointStore]:
    store = CheckpointStore(auth_env.engine.url, retention=timedelta(hours=12)).open()
    store.delete_all()
    try:
        yield store
    finally:
        store.close()
