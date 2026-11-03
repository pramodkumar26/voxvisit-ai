"""LG-5B: durable, session-scoped conversation state for the LangGraph orchestrator.

The conversation lives in LangGraph's own PostgreSQL checkpointer (`langgraph-checkpoint-postgres`),
in a schema of its own (`agent_checkpoints`) of the application's database. Nothing here touches
the application's tables or Alembic history: the library versions its own tables
(`checkpoint_migrations`), `CheckpointStore.open()` creates the schema and runs that setup, and
dropping the schema removes every trace.

What this module guarantees, and what it leaves to others:

- A checkpoint is *state*, not authority. Whoever holds a thread id gets nothing from it: every
  turn is authenticated by the backend first, the thread id is derived from the verified session,
  and the state records which patient it belongs to (`owner_of`), so a state that does not match
  the caller is discarded rather than used.
- Only plain values are written (see `AgentState`): no credentials, tokens, tool results, patient
  words or reply text. They are read back with a strict serializer: no pickle, and no msgpack
  type outside LangGraph's built-in safe list, so a row written by anyone else cannot make the
  reader import or run code.
- One turn at a time per conversation, across threads and processes: a PostgreSQL advisory lock
  on a connection of its own (not the pool's, so lock holders can never starve the saver).
- Retention is explicit: history is pruned to the latest checkpoint after each turn, a
  conversation is deleted at sign-out, and `expire` removes conversations that are idle past the
  retention period or that belong to a session which is over.
"""

import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout
from sqlalchemy import Engine, select
from sqlalchemy.engine import URL, make_url

logger = logging.getLogger("voxvisit.graph_store")

SCHEMA = "agent_checkpoints"
RETENTION = timedelta(hours=12)  # Idle conversations are removed after this.
LOCK_WAIT_SECONDS = 8.0
POOL_SIZE = (1, 6)


# What a database that cannot be reached or used raises.
STORE_FAULTS = (psycopg.Error, PoolTimeout, PoolClosed, OSError)


class CheckpointStoreError(RuntimeError):
    """The store could not be used. The turn fails closed: nothing is changed on a guess."""


class ConversationBusy(CheckpointStoreError):
    """Another turn of the same conversation is still running."""


def strict_serializer() -> JsonPlusSerializer:
    """No pickle, and only LangGraph's built-in safe types may be rebuilt from a stored row."""
    return JsonPlusSerializer(
        pickle_fallback=False, allowed_json_modules=None, allowed_msgpack_modules=None
    )


def thread_for(session_id: object) -> str:
    """The checkpoint thread of a verified session. Opaque, and never the session id itself."""
    return hashlib.sha256(f"session:{session_id}".encode()).hexdigest()[:32]


def owner_of(patient_id: object) -> str:
    """What a checkpoint records of whom it belongs to: an opaque digest, not an identifier."""
    return hashlib.sha256(f"patient:{patient_id}".encode()).hexdigest()[:32]


def _conninfo(url: URL | str) -> str:
    parsed = make_url(url) if isinstance(url, str) else url
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


@dataclass(frozen=True)
class ThreadInfo:
    thread_id: str
    last_checkpoint: datetime
    checkpoints: int


class CheckpointStore:
    """The PostgreSQL checkpointer, its pool, and the operations a conversation store needs."""

    def __init__(
        self,
        database_url: URL | str,
        *,
        pool_size: tuple[int, int] = POOL_SIZE,
        retention: timedelta = RETENTION,
        lock_wait_seconds: float = LOCK_WAIT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._conninfo = _conninfo(database_url)
        self._retention = retention
        self._lock_wait = lock_wait_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pool = ConnectionPool(
            self._conninfo,
            min_size=pool_size[0],
            max_size=pool_size[1],
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
                "options": f"-c search_path={SCHEMA}",
            },
            open=False,
        )
        self.saver = PostgresSaver(self._pool, serde=strict_serializer())  # type: ignore[arg-type]
        self._opened = False

    # -- lifecycle --------------------------------------------------------------------------

    def open(self) -> "CheckpointStore":
        """Create the schema and the library's tables if they are missing, then open the pool.
        Safe to call again and from several processes."""
        try:
            with psycopg.connect(self._conninfo, autocommit=True) as connection:
                connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            self._pool.open(wait=True, timeout=10)
            self.saver.setup()
        except Exception as error:
            raise CheckpointStoreError(f"checkpoint store unavailable: {_kind(error)}") from error
        self._opened = True
        return self

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._pool.close(timeout=5)
        self._opened = False

    def __enter__(self) -> "CheckpointStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def healthy(self) -> bool:
        try:
            with self._pool.connection(timeout=3) as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    def drop_everything(self) -> None:
        """Remove the schema and all that is in it. For tests and decommissioning."""
        self.close()
        with psycopg.connect(self._conninfo, autocommit=True) as connection:
            connection.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")

    # -- one turn at a time -----------------------------------------------------------------

    @contextmanager
    def turn(self, thread_id: str) -> Iterator[None]:
        """Hold the conversation for one turn. A second turn of the same conversation, in this
        process or another, waits briefly and then fails with `ConversationBusy`, never runs
        beside it. The lock sits on its own connection: it is released if that connection or this
        process dies."""
        key = _lock_key(thread_id)
        try:
            connection = psycopg.connect(self._conninfo, autocommit=True)
        except Exception as error:
            raise CheckpointStoreError(f"checkpoint store unavailable: {_kind(error)}") from error
        try:
            deadline = time.monotonic() + self._lock_wait
            while True:
                got = connection.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()
                if got and got[0]:
                    break
                if time.monotonic() >= deadline:
                    raise ConversationBusy("the conversation is busy with another message")
                time.sleep(0.02)
            try:
                yield
            finally:
                with contextlib.suppress(Exception):
                    connection.execute("SELECT pg_advisory_unlock(%s)", (key,))
        finally:
            with contextlib.suppress(Exception):
                connection.close()

    # -- retention --------------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        self.saver.delete_thread(thread_id)

    def prune_history(self, thread_id: str) -> None:
        """Keep only the latest checkpoint of a conversation: earlier states are not needed to
        continue it, and are state that need not be kept."""
        self.saver.prune([thread_id], strategy="keep_latest")

    def delete_all(self) -> None:
        """Remove every stored conversation (the schema stays). For tests and for resets."""
        with self._pool.connection() as connection:
            connection.execute("TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes")

    def threads(self) -> list[ThreadInfo]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT thread_id, max(checkpoint->>'ts') AS last, count(*) AS n"
                " FROM checkpoints GROUP BY thread_id"
            ).fetchall()
        return [
            ThreadInfo(r["thread_id"], datetime.fromisoformat(r["last"]), int(r["n"]))
            for r in rows
            if r["last"]
        ]

    def count_rows(self) -> dict[str, int]:
        with self._pool.connection() as connection:
            return {
                table: connection.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
            }

    def expire(self, engine: Engine | None = None, *, now: datetime | None = None) -> list[str]:
        """Delete idle conversations, and those of sessions that are over. Returns the threads
        deleted. `engine` is the application database, for the session check; without it only
        idleness is applied."""
        now = now or self._clock()
        known = self.threads()
        doomed = {t.thread_id for t in known if now - t.last_checkpoint > self._retention}
        if engine is not None and known:
            doomed |= self._of_finished_sessions(engine, {t.thread_id for t in known}, now)
        for thread_id in sorted(doomed):
            self.delete_thread(thread_id)
        return sorted(doomed)

    def _of_finished_sessions(self, engine: Engine, known: set[str], now: datetime) -> set[str]:
        from app.db.models import AuthSession  # Imported late: the store itself needs no models.

        window = now - 2 * self._retention
        with engine.connect() as connection:
            rows = connection.execute(
                select(
                    AuthSession.session_id,
                    AuthSession.status,
                    AuthSession.idle_expires_at,
                    AuthSession.absolute_expires_at,
                ).where(AuthSession.last_activity_at > window)
            ).all()
        over = {
            thread_for(r.session_id)
            for r in rows
            if r.status != "ACTIVE" or now >= r.idle_expires_at or now >= r.absolute_expires_at
        }
        return over & known


def _lock_key(thread_id: str) -> int:
    return int.from_bytes(hashlib.sha256(thread_id.encode()).digest()[:8], "big", signed=True)


def _kind(error: Exception) -> str:
    return type(error).__name__  # Only the kind of fault is ever logged or raised on.


def new_anonymous_thread() -> str:
    return uuid.uuid4().hex


def main(argv: list[str] | None = None) -> int:
    """Set up, inspect, sweep or remove the checkpoint schema of the configured database.

    python -m app.evaluation.graph_store setup | stats | expire | drop
    """
    import argparse

    from sqlalchemy import create_engine

    from app.core.config import get_settings

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["setup", "stats", "expire", "drop"])
    command = parser.parse_args(argv).command
    url = make_url(get_settings().database_url.get_secret_value())
    store = CheckpointStore(url)
    if command == "drop":
        store.drop_everything()
        print(f"dropped schema {SCHEMA}")
        return 0
    store.open()
    try:
        if command == "setup":
            print(f"schema {SCHEMA} ready")
        elif command == "stats":
            print({"conversations": len(store.threads()), **store.count_rows()})
        else:
            engine = create_engine(url)
            try:
                print({"deleted": len(store.expire(engine))})
            finally:
                engine.dispose()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
