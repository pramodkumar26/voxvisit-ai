import uuid
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from sqlalchemy import column
from sqlalchemy.dialects import postgresql

from app.core.errors import InvalidArgumentError
from app.models.scheduling import AvailabilityQuery, DirectoryQuery, MyAppointmentsQuery
from app.services.availability import MAX_WINDOW_DAYS, validate_window
from app.services.pagination import (
    MAX_PAGE_SIZE,
    decode_cursor,
    encode_cursor,
    fingerprint,
    parse_timestamp_position,
    timestamp_position,
)
from app.services.search_text import contains_all, normalize_specialty, search_tokens
from app.services.timeutil import local_window_utc, time_fields

DENVER = ZoneInfo("America/Denver")


# --- cursors ---------------------------------------------------------------------------------


def test_cursors_round_trip_their_position() -> None:
    search = fingerprint(clinic="a", start=date(2026, 12, 7))
    cursor = encode_cursor("availability", search, {"t": "x", "i": "y"})

    assert decode_cursor(cursor, "availability", search) == {"t": "x", "i": "y"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c[:-2],
        lambda c: c + "AAAA",
        lambda c: "!!!" + c,
        lambda c: "",
        lambda c: "A" * 600,
        lambda c: "e30",  # {}
        lambda c: "W10",  # []
        lambda c: "bnVsbA",  # null
    ],
)
def test_malformed_cursors_are_rejected(mutate) -> None:
    search = fingerprint(x=1)
    cursor = encode_cursor("directory", search, {"i": "1"})

    with pytest.raises(InvalidArgumentError):
        decode_cursor(mutate(cursor), "directory", search)


def test_cursors_cannot_cross_searches_or_endpoints() -> None:
    cursor = encode_cursor("directory", fingerprint(x=1), {"i": "1"})

    with pytest.raises(InvalidArgumentError):
        decode_cursor(cursor, "directory", fingerprint(x=2))
    with pytest.raises(InvalidArgumentError):
        decode_cursor(cursor, "availability", fingerprint(x=1))


def test_fingerprints_are_stable_and_filter_sensitive() -> None:
    patient = uuid.uuid4()

    assert fingerprint(a=1, b=patient) == fingerprint(b=patient, a=1)
    assert fingerprint(a=1, b=patient) != fingerprint(a=1, b=uuid.uuid4())
    assert fingerprint(a=None) != fingerprint(a="other")


def test_timestamp_positions_round_trip_and_reject_naive_values() -> None:
    row_id = uuid.uuid4()
    moment = datetime(2026, 12, 7, 16, 0, tzinfo=DENVER)

    parsed_time, parsed_id = parse_timestamp_position(timestamp_position(moment, row_id))

    assert parsed_time == moment and parsed_id == row_id
    for bad in (
        {"t": "2026-12-07T16:00:00", "i": str(row_id)},
        {"t": "x", "i": str(row_id)},
        {"t": "2026-12-07T16:00:00Z", "i": "nope"},
        {},
    ):
        with pytest.raises(InvalidArgumentError):
            parse_timestamp_position(bad)


# --- search text -----------------------------------------------------------------------------


def test_search_text_is_normalized_into_tokens() -> None:
    assert search_tokens("  Dr.  Amelia   HARTWELL ", drop_titles=True) == ["amelia", "hartwell"]
    assert search_tokens("Dr. Smith") == ["dr", "smith"]
    assert search_tokens("Ｓｍｉｔｈ") == ["smith"]  # noqa: RUF001
    assert normalize_specialty("  Family   MEDICINE ") == "family medicine"


@pytest.mark.parametrize("value", ["", "   ", ".", " , . "])
def test_search_text_needs_at_least_one_token(value: str) -> None:
    with pytest.raises(InvalidArgumentError):
        search_tokens(value)


def test_dropping_titles_can_leave_nothing_to_search() -> None:
    with pytest.raises(InvalidArgumentError):
        search_tokens("Dr.", drop_titles=True)


def test_like_wildcards_in_user_text_are_escaped() -> None:
    [condition] = contains_all(column("name"), ["50%_off\\"])
    compiled = str(
        condition.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    # The driver doubles literal percent signs in compiled text; undo that to read the pattern.
    assert compiled.replace("%%", "%") == "lower(name) LIKE '%50\\%\\_off\\\\%' ESCAPE '\\'"


# --- windows and time zones ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "hours"),
    [
        (date(2026, 11, 1), 25),  # US fall-back
        (date(2027, 3, 14), 23),  # US spring-forward
        (date(2026, 12, 7), 24),
    ],
)
def test_clinic_local_days_have_their_real_length(day: date, hours: int) -> None:
    lower, upper = local_window_utc(DENVER, day, day)

    assert upper - lower == timedelta(hours=hours)


def test_multi_day_windows_start_and_end_at_local_midnights() -> None:
    lower, upper = local_window_utc(DENVER, date(2026, 10, 31), date(2026, 11, 2))

    assert lower == datetime(2026, 10, 31, 6, 0, tzinfo=UTC)  # 00:00 MDT
    assert upper == datetime(2026, 11, 3, 7, 0, tzinfo=UTC)  # 00:00 MST the day after
    assert upper - lower == timedelta(hours=73)


def test_time_fields_carry_utc_and_local_offsets() -> None:
    fields = time_fields(
        datetime(2026, 11, 2, 16, 0, tzinfo=UTC),
        datetime(2026, 11, 2, 16, 30, tzinfo=UTC),
        "America/Denver",
    )

    assert fields["starts_at_utc"].utcoffset() == timedelta(0)
    assert fields["starts_at_local"].isoformat() == "2026-11-02T09:00:00-07:00"
    assert fields["clinic_timezone"] == "America/Denver"


def test_search_windows_allow_at_most_31_inclusive_days() -> None:
    start = date(2026, 12, 1)

    validate_window(start, start)
    validate_window(start, start + timedelta(days=MAX_WINDOW_DAYS - 1))
    for end in (start + timedelta(days=MAX_WINDOW_DAYS), start - timedelta(days=1)):
        with pytest.raises(InvalidArgumentError):
            validate_window(start, end)


# --- query models ----------------------------------------------------------------------------


def test_page_size_is_capped_at_twenty() -> None:
    assert MAX_PAGE_SIZE == 20
    assert DirectoryQuery(browse=True).limit == 20
    for model, extra in (
        (DirectoryQuery, {}),
        (MyAppointmentsQuery, {}),
        (
            AvailabilityQuery,
            {"clinic_id": uuid.uuid4(), "start_date": "2026-12-07", "end_date": "2026-12-07"},
        ),
    ):
        with pytest.raises(ValidationError):
            model(limit=21, **extra)
        with pytest.raises(ValidationError):
            model(limit=0, **extra)
        assert model(limit=20, **extra).limit == 20


@pytest.mark.parametrize("model", [DirectoryQuery, MyAppointmentsQuery])
@pytest.mark.parametrize("field", ["patient_id", "patient_code", "role", "session", "actor_id"])
def test_identity_fields_are_never_accepted_from_callers(model, field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model(**{field: "sentinel-value"})

    [error] = exc_info.value.errors()
    assert error["type"] == "extra_forbidden" and error["loc"] == (field,)
