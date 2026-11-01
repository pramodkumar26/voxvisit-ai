"""Read-only, time-limited database access for the scheduling read services."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import InternalError, OperationalError

from app.core.errors import DependencyUnavailableError

logger = logging.getLogger(__name__)
READ_TIMEOUT_MS = 8000  # The approved 8-second read-only tool timeout.
TRANSACTION_TIMEOUT_SQLSTATE = "25P04"  # PostgreSQL 17 transaction_timeout expired.


@contextmanager
def read_only_connection(
    engine: Engine, *, timeout_ms: int = READ_TIMEOUT_MS
) -> Iterator[Connection]:
    """A read-only transaction that PostgreSQL cancels if it runs past the timeout, as a whole.

    Timeouts and connection failures become DEPENDENCY_UNAVAILABLE. Other errors are bugs and
    propagate to the generic 500 handler without exposing details.
    """
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            connection.execute(text(f"SET LOCAL transaction_timeout = {int(timeout_ms)}"))
            yield connection
    except OperationalError as exc:
        logger.warning("read query failed", extra={"error_type": type(exc.orig).__name__})
        raise DependencyUnavailableError() from None
    except InternalError as exc:
        if getattr(exc.orig, "sqlstate", None) != TRANSACTION_TIMEOUT_SQLSTATE:
            raise
        logger.warning("read transaction timed out")
        raise DependencyUnavailableError() from None
