import argparse
from collections.abc import Sequence
from datetime import date

from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from app.core.config import get_settings
from app.db.session import create_database_engine
from app.seed.credentials import SeedCredentialError, SeedSettings
from app.seed.loader import SeedResult, seed_database

EXIT_FAILURE = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed",
        description="Load fictional demonstration data into the configured local database.",
    )
    parser.add_argument(
        "--anchor-date",
        type=date.fromisoformat,
        help="YYYY-MM-DD clinic-local date to plan from (default: today in each clinic's zone).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full seed inside a transaction, report the counts, and roll it back.",
    )
    return parser


def print_summary(result: SeedResult, *, dry_run: bool) -> None:
    print(f"{'table':<28}{'inserted':>10}{'existing':>10}")
    for table, counts in result.tables.items():
        print(f"{table:<28}{counts.inserted:>10}{counts.existing:>10}")
    for key in result.skipped_appointments:
        print(f"skipped appointment {key}: its slot is no longer available")
    print("Dry run: nothing was saved." if dry_run else "Seed committed.")


def describe_failure(exc: Exception) -> str:
    if isinstance(exc, DBAPIError):
        return str(exc.orig).strip().splitlines()[0]
    if isinstance(exc, ValueError):
        return str(exc)
    return type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
        seed_settings = SeedSettings()
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
        print(f"Invalid or missing configuration: {fields}")
        return EXIT_USAGE

    if settings.environment == "production":
        print("Refusing to seed demonstration data when APP_ENV is production.")
        return EXIT_USAGE
    if seed_settings.patient_pin is None or seed_settings.staff_password is None:
        print("Set SEED_PATIENT_PIN and SEED_STAFF_PASSWORD in the local .env file.")
        return EXIT_USAGE

    engine = create_database_engine(settings)
    try:
        result = seed_database(
            engine,
            patient_pin=seed_settings.patient_pin.get_secret_value(),
            staff_password=seed_settings.staff_password.get_secret_value(),
            anchor_date=args.anchor_date,
            dry_run=args.dry_run,
        )
    except SeedCredentialError as exc:
        print(str(exc))
        return EXIT_USAGE
    except Exception as exc:
        print(f"Seeding failed and was rolled back: {describe_failure(exc)}")
        return EXIT_FAILURE
    finally:
        engine.dispose()

    print_summary(result, dry_run=args.dry_run)
    return 0
