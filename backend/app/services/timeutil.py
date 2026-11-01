from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MORNING = (time(8), time(12))
AFTERNOON = (time(12), time(17))
WEEKDAYS = ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY")


def local_day_start_utc(zone: ZoneInfo, day: date) -> datetime:
    """The UTC instant at which a clinic-local calendar day begins."""
    return datetime.combine(day, time.min).replace(tzinfo=zone).astimezone(UTC)


def local_window_utc(zone: ZoneInfo, start: date, end: date) -> tuple[datetime, datetime]:
    """Half-open UTC bounds covering clinic-local days start..end inclusive.

    Days are 23 or 25 hours long across daylight-saving changes, so bounds come from local
    midnights, never from adding fixed 24-hour spans.
    """
    return local_day_start_utc(zone, start), local_day_start_utc(zone, end + timedelta(days=1))


def time_fields(starts_at: datetime, ends_at: datetime, timezone: str) -> dict:
    """Unambiguous UTC instants plus clinic-local equivalents (ISO with explicit offsets)."""
    zone = ZoneInfo(timezone)
    return {
        "starts_at_utc": starts_at.astimezone(UTC),
        "ends_at_utc": ends_at.astimezone(UTC),
        "starts_at_local": starts_at.astimezone(zone),
        "ends_at_local": ends_at.astimezone(zone),
        "clinic_timezone": timezone,
    }
