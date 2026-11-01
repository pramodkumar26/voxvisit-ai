"""The world a scenario runs in: a disposable PostgreSQL database with the synthetic seed data, the
real FastAPI application (for the trusted UI), and a controllable clock."""

import secrets
import string
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from app.auth.service import Principal
from app.core.config import Settings, get_settings
from app.evaluation.fixtures import EVAL_HASHER
from app.main import create_app
from app.seed.loader import seed_database

BACKEND_DIR = Path(__file__).parents[2]
DISPOSABLE_PREFIX = "voxvisit_eval_"
SEED_ANCHOR = date(2026, 10, 28)
EVALUATION_START = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
FAST_HASHER = EVAL_HASHER


class EvalClock:
    """A settable clock, so time-dependent rules are exercised exactly and repeatably."""

    def __init__(self, start: datetime = EVALUATION_START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@dataclass(frozen=True)
class EvalEnvironment:
    engine: Engine
    clock: Any  # Callable returning the time, with .advance(**delta).
    client: Any  # A FastAPI TestClient on the real application, used only by the trusted UI.
    authenticate: Callable[[str], Principal]  # session token -> verified principal
    pin: str  # The synthetic patients' shared demo PIN. Never written to any output.
    origin: str
    cookie_name: str

    @classmethod
    def from_parts(
        cls, *, engine: Engine, clock: Any, client: Any, auth_service: Any, pin: str
    ) -> "EvalEnvironment":
        settings = client.app.state.settings
        return cls(
            engine=engine,
            clock=clock,
            client=client,
            authenticate=auth_service.authenticate,
            pin=pin,
            origin=settings.allowed_origins[0],
            cookie_name=settings.session_cookie_name,
        )


def _run_migrations(engine: Engine) -> None:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


@contextmanager
def create_disposable_environment() -> Iterator[EvalEnvironment]:
    """Create a fresh database, migrate it, seed the synthetic data, start the application, and
    drop everything afterwards. The development database is never touched."""
    from fastapi.testclient import TestClient

    base_url = make_url(get_settings().database_url.get_secret_value())
    name = f"{DISPOSABLE_PREFIX}{uuid.uuid4().hex[:12]}"
    admin = create_engine(base_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = base_url.set(database=name)
    engine = create_engine(url)
    try:
        _run_migrations(engine)
        pin = "".join(secrets.choice(string.digits) for _ in range(6))
        seed_database(
            engine,
            patient_pin=pin,
            staff_password=secrets.token_urlsafe(18),
            anchor_date=SEED_ANCHOR,
            hasher=FAST_HASHER,
        )
        clock = EvalClock()
        settings = Settings(
            _env_file=None,
            database_url=SecretStr(url.render_as_string(hide_password=False)),
            environment="test",
            auth_secret_key=SecretStr(secrets.token_urlsafe(48)),
        )
        app = create_app(settings, clock=clock, hasher=FAST_HASHER)
        with TestClient(app) as client:
            yield EvalEnvironment.from_parts(
                engine=engine,
                clock=clock,
                client=client,
                auth_service=app.state.auth_service,
                pin=pin,
            )
    finally:
        engine.dispose()
        assert name.startswith(DISPOSABLE_PREFIX)
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
