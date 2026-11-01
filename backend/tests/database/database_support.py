import secrets
import string
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from argon2 import PasswordHasher
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import Settings

BACKEND_DIR = Path(__file__).parents[2]
TEST_DATABASE_PREFIX = "voxvisit_test_"


@contextmanager
def disposable_database() -> Iterator[Engine]:
    base_url = make_url(Settings(_env_file=None).database_url.get_secret_value())
    name = f"{TEST_DATABASE_PREFIX}{uuid.uuid4().hex[:12]}"
    assert name != base_url.database

    admin = create_engine(base_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(base_url.set(database=name))
    try:
        yield engine
    finally:
        engine.dispose()
        assert name.startswith(TEST_DATABASE_PREFIX)
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def run_migration(engine: Engine, action: str, revision: str) -> None:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        getattr(command, action)(config, revision)


FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)


class FakeClock:
    """A settable clock so expiry, lockout, and rate-limit behavior is deterministic."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def random_credentials() -> SimpleNamespace:
    pin = "".join(secrets.choice(string.digits) for _ in range(6))
    return SimpleNamespace(pin=pin, password=secrets.token_urlsafe(18))
