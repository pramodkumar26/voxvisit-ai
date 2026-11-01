"""Checks against the job's disposable PostgreSQL service. Usage: check_postgres.py MODE

preflight                  The server is PostgreSQL 17 or newer, ships btree_gist, and the role can
                           create a database and the extension in it, so the test suite can create
                           and drop its own disposable databases.
no-leaked-test-databases   No voxvisit_test_* database survived the test run.

Output uses GitHub Actions annotations so a failure names its cause in the job summary.
"""

import os
import sys
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

MINIMUM_SERVER_VERSION_NUM = 170000
TEST_DATABASE_PREFIX = "voxvisit_test_"


def fail(title: str, message: str) -> None:
    print(f"::error title={title}::{message}")
    raise SystemExit(1)


def admin_engine() -> Engine:
    url = make_url(os.environ["DATABASE_URL"]).set(database="postgres")
    return create_engine(url, isolation_level="AUTOCOMMIT")


def preflight() -> None:
    engine = admin_engine()
    scratch = f"voxvisit_ci_preflight_{uuid.uuid4().hex[:10]}"
    with engine.connect() as connection:
        version_num = int(connection.execute(text("SHOW server_version_num")).scalar_one())
        version = connection.execute(text("SHOW server_version")).scalar_one()
        if version_num < MINIMUM_SERVER_VERSION_NUM:
            fail("PostgreSQL version", f"PostgreSQL 17 or newer is required, found {version}.")
        print(f"PostgreSQL {version}")

        available = connection.execute(
            text("SELECT count(*) FROM pg_available_extensions WHERE name = 'btree_gist'")
        ).scalar_one()
        if not available:
            fail("btree_gist", "The btree_gist extension is not available on this server.")
        print("btree_gist is available")

        can_create = connection.execute(
            text("SELECT rolcreatedb OR rolsuper FROM pg_roles WHERE rolname = current_user")
        ).scalar_one()
        if not can_create:
            fail("Database role", "The role cannot create databases, so tests cannot run.")

        connection.execute(text(f'CREATE DATABASE "{scratch}"'))
    try:
        url = make_url(os.environ["DATABASE_URL"]).set(database=scratch)
        scratch_engine = create_engine(url, isolation_level="AUTOCOMMIT")
        with scratch_engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION btree_gist"))
        scratch_engine.dispose()
        print("a disposable database can be created with btree_gist")
    finally:
        with engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))
        engine.dispose()
    print("PostgreSQL preflight passed")


def no_leaked_test_databases() -> None:
    engine = admin_engine()
    with engine.connect() as connection:
        leaked = (
            connection.execute(
                text("SELECT datname FROM pg_database WHERE datname LIKE :pattern ORDER BY 1"),
                {"pattern": f"{TEST_DATABASE_PREFIX}%"},
            )
            .scalars()
            .all()
        )
    engine.dispose()
    if leaked:
        fail("Leaked databases", f"Test databases were not removed: {', '.join(leaked)}")
    print("no disposable test databases remain")


MODES = {"preflight": preflight, "no-leaked-test-databases": no_leaked_test_databases}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        raise SystemExit(f"usage: check_postgres.py {{{'|'.join(MODES)}}}")
    MODES[sys.argv[1]]()


if __name__ == "__main__":
    main()
