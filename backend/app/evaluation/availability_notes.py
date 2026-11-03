"""What to tell a patient when a search for openings comes back empty.

An empty result has three different meanings, and the patient is told which one it is:

- the clinic has no doctor in the specialty that was asked for (say so, and list the specialties it
  does have);
- the clinic has openings that day, but not ones that match every filter the patient set: specialty,
  part of the day, or a named doctor (say which filter, and what else is open);
- the clinic has no openings at all that day (say so, and give the next days that really have some).

A filter is never dropped to make a search succeed. The patient hears what excluded the openings and
chooses what to change. This module is plain text and arithmetic over what the tools returned; it
calls nothing and decides nothing. Dates come only from slots the backend listed.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date

LOOK_AHEAD_DAYS = 31  # The most a single availability search may span.
MAX_DAYS_NAMED = 3

# Search arguments that narrow a search to what the patient asked for. Without these, the same
# search shows everything the clinic has open.
FILTER_ARGUMENTS = ("specialty", "doctor_id", "time_preference")


def same_specialty(asked: str | None, listed: str | None) -> bool:
    """ "Cardiologist" and "Cardiology" name one specialty: compare the stems."""
    a, b = (asked or "").casefold().strip(), (listed or "").casefold().strip()
    return bool(a and b and a[:5] == b[:5])


def or_list(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def and_list(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def short_day(iso: str) -> str:
    """ "Thursday, November 5": short enough to listen to; the year is the clinic's current one."""
    day = date.fromisoformat(iso[:10])
    return f"{day:%A}, {day:%B} {day.day}"


def specialties_listed(clinic: Mapping[str, object] | None) -> list[str]:
    """The specialties a directory entry says the clinic has, as the backend wrote them."""
    return _distinct(str(s) for s in (clinic or {}).get("specialties") or [] if s)


def specialties_open(slots: Iterable[Mapping[str, object]]) -> list[str]:
    """The specialties of the doctors who have these openings, in the order first seen."""
    return _distinct(str((slot.get("doctor") or {}).get("specialty") or "") for slot in slots)


def next_days(slots: Iterable[Mapping[str, object]], limit: int = MAX_DAYS_NAMED) -> list[str]:
    """The first days (ISO dates, earliest first) on which these listed slots fall."""
    ordered = sorted(slots, key=lambda slot: str(slot["starts_at_utc"]))
    return _distinct(str(slot["starts_at_local"])[:10] for slot in ordered)[:limit]


def _distinct(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item and item.casefold() not in {k.casefold() for k in seen}:
            seen[item] = None
    return list(seen)


def kind_of_openings(
    specialty: str | None = None, part_of_day: str | None = None, doctor: str | None = None
) -> str:
    """ "General Practice morning openings with Dr. Rao": what the patient was looking for."""
    kind = " ".join(word for word in (specialty, part_of_day) if word)
    kind = f"{kind} openings" if kind else "openings"
    return f"{kind} with {doctor}" if doctor else kind


def changeable(specialty: str | None, part_of_day: str | None, doctor: str | None) -> list[str]:
    return [
        f"the {name}"
        for name, active in (
            ("specialty", specialty),
            ("time of day", part_of_day),
            ("doctor", doctor),
        )
        if active
    ]


# --- the three meanings, as sentences ---------------------------------------------------------


def no_such_specialty(clinic: str, specialty: str, listed: Sequence[str]) -> str:
    return (
        f"{clinic} doesn't have a {specialty} doctor. Its doctors are in {and_list(listed)}. "
        "Which specialty would you like, or would you like a different clinic?"
    )


def filtered_out(
    clinic: str,
    when: str,
    *,
    specialty: str | None = None,
    part_of_day: str | None = None,
    doctor: str | None = None,
    open_specialties: Sequence[str] = (),
) -> str:
    """The clinic has openings then, but none that match what was asked."""
    kind = kind_of_openings(specialty, part_of_day, doctor)
    other = f" with {or_list(open_specialties)}" if specialty and open_specialties else ""
    change = " or ".join(changeable(specialty, part_of_day, doctor))
    return (
        f"I don't see any {kind} at {clinic}{when}, though there are other openings then{other}. "
        f"Would you like to change {change}, pick another day, or try another clinic?"
    )


def later_days(
    clinic: str,
    when: str,
    days: Sequence[str],
    *,
    specialty: str | None = None,
    part_of_day: str | None = None,
    doctor: str | None = None,
) -> str:
    """Nothing at all then; these are the next days the backend lists openings for."""
    kind = kind_of_openings(specialty, part_of_day, doctor)
    named = and_list([short_day(day) for day in days])
    return (
        f"I don't see any openings at {clinic}{when}. The next {kind} are on {named}. "
        "Which day would you like?"
    )


def nothing_ahead(
    clinic: str,
    when: str,
    *,
    specialty: str | None = None,
    part_of_day: str | None = None,
    doctor: str | None = None,
) -> str:
    """Nothing then, and nothing matching what was asked for in the whole look-ahead."""
    kind = kind_of_openings(specialty, part_of_day, doctor)
    change = changeable(specialty, part_of_day, doctor)
    options = ["try another clinic"]
    if change:
        options.insert(0, f"change {' or '.join(change)}")
    ahead = f"in the next {LOOK_AHEAD_DAYS} days"
    seen = f"no {kind} {ahead}" if change else f"none {ahead}"
    return (
        f"I don't see any openings at {clinic}{when}, and {seen}. "
        f"Would you like to {or_list(options)}?"
    )
