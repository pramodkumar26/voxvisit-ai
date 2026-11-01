"""Time-limited READ COMMITTED transactions for state-changing services."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import InternalError, OperationalError

from app.core.errors import DependencyUnavailableError

logger = logging.getLogger(__name__)
WRITE_TIMEOUT_MS = 12000  # The approved 12-second pending-action and mutation tool timeout.
TRANSACTION_TIMEOUT_SQLSTATE = "25P04"  # PostgreSQL 17 transaction_timeout expired.


@contextmanager
def write_transaction(
    engine: Engine, *, timeout_ms: int = WRITE_TIMEOUT_MS
) -> Iterator[Connection]:
    """One atomic transaction. Any exception rolls back every statement it executed.

    The timeout bounds the whole transaction (PostgreSQL 17 `transaction_timeout`), including every
    wait for a row lock, so a mutation cannot outlive the tool's 12 seconds by chaining several
    statements that each stay under a per-statement limit. `statement_timeout` is kept as well so
    a single slow statement is cancelled without terminating the connection. Timeouts, deadlock
    victims, and connection failures become DEPENDENCY_UNAVAILABLE, which is safe to retry with the
    same idempotency key because nothing was committed.
    """
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
            connection.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            connection.execute(text(f"SET LOCAL transaction_timeout = {int(timeout_ms)}"))
            yield connection
    except OperationalError as exc:
        logger.warning("write transaction failed", extra={"error_type": type(exc.orig).__name__})
        raise DependencyUnavailableError() from None
    except InternalError as exc:
        if getattr(exc.orig, "sqlstate", None) != TRANSACTION_TIMEOUT_SQLSTATE:
            raise
        logger.warning("write transaction timed out")
        raise DependencyUnavailableError() from None
