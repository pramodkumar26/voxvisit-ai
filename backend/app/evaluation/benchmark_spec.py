"""The frozen architecture benchmark: its manifest schema and the facts it implies.

The manifest (`backend/tests/agent-architecture-benchmark-v1.json`) is a specification, not agent
code. Each scenario states, in plain data, the patient's request, the exact initial state of the
world, the patient's true intent, the correct end state, and what a correct reply must contain.
Nothing in it names a tool or a call order, so no approach is favored by how the expected answer is
written, and both approaches can be given an identical request and an identical starting database.

This module also *derives* what the correct answers must be from the fixtures (the earliest
eligible slot, whether a doctor's surname is ambiguous, which appointment a cancellation refers to)
and `validate_manifest` checks the declared expectations against that derivation. A mistake in the
frozen manifest is therefore found before any model runs, not argued about afterwards.

Nothing here imports agent, model, or tool code.
"""

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_PATH = Path(__file__).parents[2] / "tests" / "agent-architecture-benchmark-v1.json"
VERSION = "agent-architecture-benchmark-v1"
# The first day of every run, in the clinics' time zone: the evaluation clock's local date.
REFERENCE_DATE = date(2026, 11, 2)
SLOT_MINUTES = 30
MORNING = ("08:00", "12:00")  # Start inclusive, end exclusive: the availability tool's windows.
AFTERNOON = ("12:00", "17:00")
WINDOW_EDGES = frozenset({"08:00", "12:00", "17:00"})

Category = Literal[
    "BOOKING",
    "CONFLICT",
    "CANCELLATION",
    "RESCHEDULING",
    "NO_AVAILABILITY",
    "AMBIGUITY",
    "UNAUTHENTICATED",
]
Outcome = Literal[
    "BOOKED",
    "CANCELLED",
    "RESCHEDULED",
    "CONFLICT",
    "CLARIFICATION",
    "NO_AVAILABILITY",
    "NO_MATCH",
    "AUTH_REQUIRED",
]
ResponseKind = Literal[
    "REVIEW_PROMPT",
    "COMMITTED",
    "CONFLICT_REPORT",
    "CLARIFICATION",
    "NO_AVAILABILITY",
    "NO_MATCH",
    "AUTH_REQUIRED",
]
# Replies that describe an action: every detail in them must belong to that action.
ACTION_KINDS = frozenset({"REVIEW_PROMPT", "COMMITTED"})
PLACEHOLDERS = frozenset(
    {
        "clinic",
        "doctor",
        "surname",
        "specialty",
        "time",
        "day",
        "day_long",
        "day_dmy",
        "day2",
        "day2_long",
        "day2_dmy",
    }
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DoctorSpec(Strict):
    key: str
    name: str
    specialty: str


class ClinicSpec(Strict):
    key: str
    name: str
    doctors: list[DoctorSpec]


class SlotSpec(Strict):
    key: str
    clinic: str
    doctor: str
    day: str
    time: str  # Clinic-local HH:MM; a 30-minute slot.
    # none: open. patient/other: booked by the verified patient / by someone else.
    # patient_cancelled: open again, with a cancelled appointment of the patient's on record.
    held_by: Literal["none", "patient", "other", "patient_cancelled"] = "none"


class FixtureSpec(Strict):
    timezone: str
    days: dict[str, int]  # Day key -> days after REFERENCE_DATE.
    clinics: list[ClinicSpec]
    slots: list[SlotSpec]


class IntentSpec(Strict):
    """What the patient actually wants, independent of how any approach reads it."""

    operation: Literal["BOOK", "CANCEL", "RESCHEDULE"]
    clinic: str | None = None
    doctor: str | None = None
    specialty: str | None = None
    day: str | None = None
    new_day: str | None = None
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    selection: Literal["EARLIEST", "SPECIFIC_TIME", "LATER", "EARLIER", "ANY", "UNSPECIFIED"] = (
        "UNSPECIFIED"
    )
    time: str | None = None
    ambiguity: Literal["DOCTOR_NAME", "APPOINTMENT_TARGET", "MISSING_DATE"] | None = None
    options: list[str] = Field(default_factory=list)  # Doctor keys or appointment (slot) keys.


class PreparedSpec(Strict):
    action: Literal["BOOK", "CANCEL", "RESCHEDULE"]
    slot: str | None = None  # BOOK: the slot to book.
    appointment: str | None = None  # CANCEL, RESCHEDULE: the patient's appointment (its slot key).
    new_slot: str | None = None  # RESCHEDULE: the preferred new slot.
    acceptable_new_slots: list[str] = Field(default_factory=list)  # Any of these is correct.


class FinalSpec(Strict):
    """Slots and appointments whose state differs from the start. Nothing else may differ."""

    booked_by_patient: list[str] = Field(default_factory=list)
    released: list[str] = Field(default_factory=list)  # Patient's slots that became open again.
    booked_by_other: list[str] = Field(default_factory=list)  # Taken by the world, not the patient.
    cancelled_appointments: list[str] = Field(default_factory=list)


class ExpectedSpec(Strict):
    outcome: Outcome
    prepared: PreparedSpec | None = None
    ui: Literal["COMMITTED", "SLOT_UNAVAILABLE", "NOT_REACHED"] = "NOT_REACHED"
    final: FinalSpec = Field(default_factory=FinalSpec)


class WorldEvent(Strict):
    when: Literal["after_prepare"]
    event: Literal["other_patient_books"]
    slot: str


class ResponseSpec(Strict):
    kind: ResponseKind
    subject: list[str] = Field(default_factory=list)  # Slot keys the reply describes.
    options: list[str] = Field(default_factory=list)  # What the patient must be able to choose.
    must_cover: list[list[str]] = Field(default_factory=list)  # Each group: any one word suffices.
    required_details: list[Literal["doctor", "time", "date", "clinic"]] = Field(
        default_factory=list
    )


class ResponsesSpec(Strict):
    review: ResponseSpec | None = None  # The reply before the patient presses Confirm.
    final: ResponseSpec


class StrataSpec(Strict):
    phrasing: str
    date_format: Literal["iso", "long", "dmy", "none"]


class ScenarioSpec(Strict):
    id: str
    category: Category
    title: str
    actor: Literal["patient", "public"]
    request: str
    stated: dict[str, str]
    strata: StrataSpec
    fixture: FixtureSpec
    intent: IntentSpec
    expected: ExpectedSpec
    world: list[WorldEvent] = Field(default_factory=list)
    responses: ResponsesSpec


class ManifestSpec(Strict):
    version: str
    status: Literal["FROZEN_BEFORE_EXECUTION"]
    frozen_on: str
    protocol: str
    purpose: str
    rules: dict[str, bool]
    development_evidence: list[str]
    scenarios: list[ScenarioSpec]


def manifest_bytes(path: Path = MANIFEST_PATH) -> bytes:
    return path.read_bytes()


def manifest_sha256(path: Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(manifest_bytes(path)).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> ManifestSpec:
    return ManifestSpec.model_validate(json.loads(path.read_text()))


# --- facts the fixtures imply ------------------------------------------------------------------


def _minutes(clock: str) -> int:
    hour, minute = clock.split(":")
    return int(hour) * 60 + int(minute)


def in_window(clock: str, window: tuple[str, str]) -> bool:
    return _minutes(window[0]) <= _minutes(clock) < _minutes(window[1])


def day_of(fixture: FixtureSpec, key: str) -> date:
    return REFERENCE_DATE + timedelta(days=fixture.days[key])


def slot_index(fixture: FixtureSpec) -> dict[str, SlotSpec]:
    return {slot.key: slot for slot in fixture.slots}


def doctor_index(fixture: FixtureSpec) -> dict[str, tuple[DoctorSpec, ClinicSpec]]:
    return {doctor.key: (doctor, clinic) for clinic in fixture.clinics for doctor in clinic.doctors}


def clinic_index(fixture: FixtureSpec) -> dict[str, ClinicSpec]:
    return {clinic.key: clinic for clinic in fixture.clinics}


def surname(name: str) -> str:
    return name.split()[-1]


def eligible_slots(
    fixture: FixtureSpec,
    *,
    clinic: str,
    day: str,
    doctor: str | None = None,
    specialty: str | None = None,
    time_preference: str = "ANYTIME",
) -> list[SlotSpec]:
    """Open slots a search with these constraints returns, earliest first."""
    doctors = doctor_index(fixture)
    found = []
    for slot in fixture.slots:
        if slot.held_by not in ("none", "patient_cancelled"):
            continue
        if slot.clinic != clinic or slot.day != day:
            continue
        if doctor and slot.doctor != doctor:
            continue
        if specialty and doctors[slot.doctor][0].specialty.casefold() != specialty.casefold():
            continue
        if time_preference == "MORNING" and not in_window(slot.time, MORNING):
            continue
        if time_preference == "AFTERNOON" and not in_window(slot.time, AFTERNOON):
            continue
        found.append(slot)
    return sorted(found, key=lambda slot: _minutes(slot.time))


def patient_appointments(fixture: FixtureSpec) -> list[SlotSpec]:
    return [slot for slot in fixture.slots if slot.held_by == "patient"]


def matching_appointments(fixture: FixtureSpec, intent: IntentSpec) -> list[SlotSpec]:
    """The patient's confirmed appointments that the intent's stated constraints select."""
    found = []
    for slot in patient_appointments(fixture):
        if intent.clinic and slot.clinic != intent.clinic:
            continue
        if intent.doctor and slot.doctor != intent.doctor:
            continue
        if intent.day and slot.day != intent.day:
            continue
        found.append(slot)
    return found


def doctors_named(fixture: FixtureSpec, clinic: str, surname_text: str) -> list[str]:
    """Doctor keys at the clinic whose name contains the text (the directory's matching rule)."""
    needle = surname_text.casefold()
    return [
        doctor.key
        for doctor in clinic_index(fixture)[clinic].doctors
        if needle in doctor.name.casefold()
    ]


def format_time(clock: str) -> str:
    hour, minute = int(clock[:2]), clock[3:]
    return f"{hour % 12 or 12}:{minute} {'AM' if hour < 12 else 'PM'}"


def render_request(scenario: ScenarioSpec) -> str:
    """The words the patient says: the template with each stated value written out."""
    fixture = scenario.fixture
    doctors = doctor_index(fixture)
    values: dict[str, str] = {}
    stated = scenario.stated
    if "clinic" in stated:
        values["clinic"] = clinic_index(fixture)[stated["clinic"]].name
    if "doctor" in stated:
        values["doctor"] = doctors[stated["doctor"]][0].name
        values["surname"] = surname(doctors[stated["doctor"]][0].name)
    if "specialty" in stated:
        values["specialty"] = stated["specialty"]
    if "time" in stated:
        values["time"] = format_time(stated["time"])
    for base in ("day", "day2"):
        if base in stated:
            when = day_of(fixture, stated[base])
            values[base] = when.isoformat()
            values[f"{base}_long"] = f"{when.strftime('%B')} {when.day}, {when.year}"
            values[f"{base}_dmy"] = f"{when.day} {when.strftime('%B %Y')}"
    return scenario.request.format(**values)


def request_placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))


# --- the correct end state, derived -------------------------------------------------------------


def derived_final(scenario: ScenarioSpec, new_slot: str | None = None) -> FinalSpec:
    """What the world must look like at the end, from the expectation and the world events."""
    expected = scenario.expected
    final = FinalSpec()
    for event in scenario.world:
        final.booked_by_other.append(event.slot)
    prepared = expected.prepared
    if prepared is None or expected.ui != "COMMITTED":
        return final
    if prepared.action == "BOOK":
        final.booked_by_patient.append(prepared.slot or "")
    elif prepared.action == "CANCEL":
        final.released.append(prepared.appointment or "")
        final.cancelled_appointments.append(prepared.appointment or "")
    else:
        final.released.append(prepared.appointment or "")
        final.booked_by_patient.append(new_slot or prepared.new_slot or "")
    return final


def initial_state(fixture: FixtureSpec) -> dict[str, Any]:
    """The world as data: no database identifiers, so it is the same for every approach and run."""
    doctors = doctor_index(fixture)
    return {
        "timezone": fixture.timezone,
        "clinics": [
            {
                "name": clinic.name,
                "doctors": [[d.name, d.specialty] for d in clinic.doctors],
            }
            for clinic in fixture.clinics
        ],
        "slots": sorted(
            [
                clinic_index(fixture)[slot.clinic].name,
                doctors[slot.doctor][0].name,
                day_of(fixture, slot.day).isoformat(),
                slot.time,
                slot.held_by,
            ]
            for slot in fixture.slots
        ),
    }


def initial_state_fingerprint(fixture: FixtureSpec) -> str:
    canonical = json.dumps(initial_state(fixture), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def local_datetime(fixture: FixtureSpec, slot: SlotSpec) -> datetime:
    when = day_of(fixture, slot.day)
    return datetime.fromisoformat(f"{when.isoformat()}T{slot.time}:00")


# --- validation ---------------------------------------------------------------------------------


def search_matches(query: str, name: str) -> bool:
    """Whether a name search for `query` returns `name`: every word of the query is found inside
    the name (the directory's rule: app.services.search_text), titles ignored."""
    titles = {"dr", "doctor"}
    words = [w for w in re.sub(r"[^\w ]+", " ", query.casefold()).split() if w not in titles]
    haystack = name.casefold()
    return bool(words) and all(word in haystack for word in words)


def validate_manifest(
    manifest: ManifestSpec,
    seeded_clinics: frozenset[str] = frozenset(),
    seeded_doctors: frozenset[str] = frozenset(),
) -> list[str]:
    """Every way the frozen manifest contradicts itself or its own fixtures. Empty if consistent.

    `seeded_*` are the names the disposable database already holds (the demonstration seed): a
    search the patient's words would make must never find one of them.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    for scenario in manifest.scenarios:
        sid = scenario.id
        if sid in seen_ids:
            problems.append(f"{sid}: duplicate id")
        seen_ids.add(sid)
        found = _scenario_problems(scenario) + _collision_problems(
            scenario, seeded_clinics, seeded_doctors
        )
        problems += [f"{sid}: {problem}" for problem in found]
    return problems


def _collision_problems(
    scenario: ScenarioSpec, seeded_clinics: frozenset[str], seeded_doctors: frozenset[str]
) -> list[str]:
    """Searches the patient's words imply, checked against every name the database will hold."""
    problems: list[str] = []
    fixture = scenario.fixture
    queries: list[tuple[str, str, frozenset[str]]] = []
    for clinic in fixture.clinics:
        queries.append(("clinic", clinic.name, seeded_clinics))
    stated = scenario.stated
    if "doctor" in stated:
        doctor_name = doctor_index(fixture)[stated["doctor"]][0].name
        queries.append(("doctor", surname(doctor_name), seeded_doctors))
        queries.append(("doctor", doctor_name, seeded_doctors))
    for kind, query, seeded in queries:
        hits = [name for name in seeded if search_matches(query, name)]
        if hits:
            problems.append(f"a {kind} search for {query!r} would also find seeded {hits}")
    # Within the scenario, a doctor's full name must find only that doctor (a surname may not).
    for clinic in fixture.clinics:
        for doctor in clinic.doctors:
            rivals = [
                other.name
                for other_clinic in fixture.clinics
                for other in other_clinic.doctors
                if other.key != doctor.key and search_matches(doctor.name, other.name)
            ]
            if rivals:
                problems.append(f"the full name {doctor.name!r} also finds {rivals}")
    return problems


def _scenario_problems(scenario: ScenarioSpec) -> list[str]:
    problems: list[str] = []
    fixture, intent, expected = scenario.fixture, scenario.intent, scenario.expected
    clinics, doctors, slots = clinic_index(fixture), doctor_index(fixture), slot_index(fixture)

    def known(kind: str, key: str | None, table: dict) -> None:
        if key is not None and key not in table:
            problems.append(f"unknown {kind} {key!r}")

    # -- structure and references
    if len(slots) != len(fixture.slots):
        problems.append("duplicate slot keys")
    if len(clinics) != len(fixture.clinics) or len(doctors) != sum(
        len(c.doctors) for c in fixture.clinics
    ):
        problems.append("duplicate clinic or doctor keys")
    seen_times: set[tuple[str, str, str]] = set()
    for slot in fixture.slots:
        known("clinic", slot.clinic, clinics)
        known("doctor", slot.doctor, doctors)
        known("day", slot.day, fixture.days)
        if (
            slot.doctor in doctors
            and slot.clinic in clinics
            and doctors[slot.doctor][1].key != slot.clinic
        ):
            problems.append(f"slot {slot.key}: doctor does not work at its clinic")
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", slot.time):
            problems.append(f"slot {slot.key}: bad time {slot.time!r}")
        elif (
            _minutes(slot.time) % SLOT_MINUTES
            or not 8 * 60 + 30 <= _minutes(slot.time) <= 16 * 60 + 30
        ):
            problems.append(f"slot {slot.key}: time off the grid or near a window edge")
        elif slot.time in WINDOW_EDGES:
            problems.append(f"slot {slot.key}: time on a morning or afternoon window edge")
        marker = (slot.doctor, slot.day, slot.time)
        if marker in seen_times:
            problems.append(f"slot {slot.key}: the doctor already has a slot then")
        seen_times.add(marker)
    for key, offset in fixture.days.items():
        if not 7 <= offset <= 40:
            problems.append(f"day {key}: {offset} days ahead is outside the booking horizon")
        elif (REFERENCE_DATE + timedelta(days=offset)).weekday() >= 5:
            problems.append(f"day {key}: {offset} days ahead is a weekend")

    # -- the request
    placeholders = request_placeholders(scenario.request)
    if not placeholders <= PLACEHOLDERS:
        problems.append(f"unknown placeholders {sorted(placeholders - PLACEHOLDERS)}")
    try:
        text = render_request(scenario)
    except (KeyError, IndexError):
        problems.append("the request cannot be rendered from its stated values")
        text = ""
    base_of = {
        "clinic": "clinic",
        "doctor": "doctor",
        "surname": "doctor",
        "specialty": "specialty",
        "time": "time",
        "day": "day",
        "day_long": "day",
        "day_dmy": "day",
        "day2": "day2",
        "day2_long": "day2",
        "day2_dmy": "day2",
    }
    used = {base_of[name] for name in placeholders if name in base_of}
    if used != set(scenario.stated):
        problems.append("stated values and request placeholders do not match")
    known("clinic", scenario.stated.get("clinic"), clinics)
    known("doctor", scenario.stated.get("doctor"), doctors)
    known("day", scenario.stated.get("day"), fixture.days)
    known("day", scenario.stated.get("day2"), fixture.days)
    formats = {
        "iso": any(p == "day" for p in placeholders),
        "long": any(p.endswith("_long") for p in placeholders),
        "dmy": any(p.endswith("_dmy") for p in placeholders),
    }
    declared = scenario.strata.date_format
    if declared == "none" and any(formats.values()):
        problems.append("date_format none, but the request states a date")
    if declared != "none" and not formats.get(declared):
        problems.append(f"date_format {declared!r} is not what the request uses")

    # -- the intent
    known("clinic", intent.clinic, clinics)
    known("doctor", intent.doctor, doctors)
    known("day", intent.day, fixture.days)
    known("day", intent.new_day, fixture.days)
    if intent.clinic != scenario.stated.get("clinic"):
        problems.append("the intent's clinic is not the clinic the patient named")
    if intent.doctor != scenario.stated.get("doctor") and not (
        intent.ambiguity == "DOCTOR_NAME" and intent.doctor is None
    ):
        problems.append("the intent's doctor is not the doctor the patient named")
    if intent.specialty != scenario.stated.get("specialty"):
        problems.append("the intent's specialty is not the specialty the patient named")
    if intent.day != scenario.stated.get("day"):
        problems.append("the intent's day is not the day the patient named")

    # -- the expectation, re-derived from the fixture
    prepared = expected.prepared
    outcome = expected.outcome
    appointments = {slot.key for slot in patient_appointments(fixture)}
    if outcome in {"BOOKED", "CANCELLED", "RESCHEDULED", "CONFLICT"} and prepared is None:
        problems.append(f"outcome {outcome} needs a prepared action")
    if outcome not in {"BOOKED", "CANCELLED", "RESCHEDULED", "CONFLICT"} and prepared is not None:
        problems.append(f"outcome {outcome} must not prepare anything")
    if (outcome in {"BOOKED", "CANCELLED", "RESCHEDULED"}) != (expected.ui == "COMMITTED"):
        problems.append("only a successful outcome ends in a committed confirmation")
    if outcome == "CONFLICT" and expected.ui != "SLOT_UNAVAILABLE":
        problems.append("a conflict ends in SLOT_UNAVAILABLE")
    if (
        outcome not in {"BOOKED", "CANCELLED", "RESCHEDULED", "CONFLICT"}
        and expected.ui != "NOT_REACHED"
    ):
        problems.append("without a prepared action the confirmation is never reached")

    if prepared and prepared.action == "BOOK":
        problems += _book_problems(scenario, prepared, slots)
    if prepared and prepared.action == "CANCEL":
        problems += _cancel_problems(scenario, prepared)
    if prepared and prepared.action == "RESCHEDULE":
        problems += _reschedule_problems(scenario, prepared, slots)
    if outcome == "NO_AVAILABILITY":
        found = eligible_slots(
            fixture,
            clinic=intent.clinic or "",
            day=intent.day or "",
            doctor=intent.doctor,
            specialty=intent.specialty,
            time_preference=intent.time_preference,
        )
        if found:
            problems.append("NO_AVAILABILITY, but the fixture has matching open slots")
        if intent.operation != "BOOK":
            problems.append("NO_AVAILABILITY scenarios are booking requests")
        if not any(s.held_by in ("none", "patient_cancelled") for s in fixture.slots):
            problems.append("a no-availability fixture should still have open slots elsewhere")
    if outcome == "NO_MATCH":
        if matching_appointments(fixture, intent):
            problems.append("NO_MATCH, but an appointment matches the request")
        if not appointments:
            problems.append("NO_MATCH needs another appointment to be mistaken for the target")
    if outcome == "CLARIFICATION":
        problems += _ambiguity_problems(scenario, text)
    elif intent.ambiguity or intent.options:
        problems.append("only a clarification scenario declares an ambiguity")
    if outcome == "AUTH_REQUIRED":
        if scenario.actor != "public":
            problems.append("AUTH_REQUIRED needs an unauthenticated actor")
        if intent.operation != "BOOK" and not appointments:
            problems.append("an unauthenticated change needs a protected appointment to leak")
    elif scenario.actor == "public":
        problems.append("an unauthenticated actor can only end in AUTH_REQUIRED")

    # -- the end state, derived and compared with the declared one
    for event in scenario.world:
        known("slot", event.slot, slots)
        if prepared is None or prepared.slot != event.slot:
            problems.append("a world event must take the slot that was prepared")
    if prepared is None or prepared.action != "RESCHEDULE":
        derived = derived_final(scenario)
        for field in FinalSpec.model_fields:
            if sorted(getattr(derived, field)) != sorted(getattr(expected.final, field)):
                problems.append(f"declared final {field} differs from the derived end state")
    else:
        derived = derived_final(scenario, new_slot=prepared.new_slot)
        declared = expected.final
        for field in ("released", "cancelled_appointments", "booked_by_other"):
            if sorted(getattr(derived, field)) != sorted(getattr(declared, field)):
                problems.append(f"declared final {field} differs from the derived end state")
        if len(declared.booked_by_patient) != 1 or declared.booked_by_patient[0] not in (
            prepared.acceptable_new_slots or [prepared.new_slot]
        ):
            problems.append("the declared reschedule target is not an acceptable new slot")

    # -- the replies
    problems += _response_problems(scenario)
    if scenario.fixture.timezone != "America/Denver":
        problems.append("all benchmark clinics share one time zone")
    return problems


def _book_problems(scenario: ScenarioSpec, prepared: PreparedSpec, slots: dict) -> list[str]:
    problems: list[str] = []
    fixture, intent = scenario.fixture, scenario.intent
    if intent.operation != "BOOK":
        problems.append("a BOOK action needs a BOOK intent")
    slot = slots.get(prepared.slot or "")
    if slot is None:
        return [*problems, "the prepared slot does not exist"]
    found = eligible_slots(
        fixture,
        clinic=intent.clinic or "",
        day=intent.day or "",
        doctor=intent.doctor,
        specialty=intent.specialty,
        time_preference=intent.time_preference,
    )
    if slot not in found:
        problems.append("the prepared slot is not eligible under the patient's constraints")
    elif intent.selection == "EARLIEST" and found[0] != slot:
        problems.append("the prepared slot is not the earliest eligible one")
    elif intent.selection == "SPECIFIC_TIME":
        if slot.time != intent.time or sum(s.time == intent.time for s in found) != 1:
            problems.append("the requested time does not select exactly the prepared slot")
        if scenario.stated.get("time") != intent.time:
            problems.append("the time the patient states is not the intent's time")
    if intent.selection not in {"EARLIEST", "SPECIFIC_TIME"}:
        problems.append("a booking scenario needs EARLIEST or SPECIFIC_TIME")
    return problems


def _cancel_problems(scenario: ScenarioSpec, prepared: PreparedSpec) -> list[str]:
    problems: list[str] = []
    if scenario.intent.operation != "CANCEL":
        problems.append("a CANCEL action needs a CANCEL intent")
    matches = matching_appointments(scenario.fixture, scenario.intent)
    if [slot.key for slot in matches] != [prepared.appointment]:
        problems.append("the request must select exactly the cancelled appointment")
    return problems


def _reschedule_problems(scenario: ScenarioSpec, prepared: PreparedSpec, slots: dict) -> list[str]:
    problems: list[str] = []
    fixture, intent = scenario.fixture, scenario.intent
    if intent.operation != "RESCHEDULE":
        problems.append("a RESCHEDULE action needs a RESCHEDULE intent")
    matches = matching_appointments(fixture, intent)
    if [slot.key for slot in matches] != [prepared.appointment]:
        return [*problems, "the request must select exactly the rescheduled appointment"]
    current = matches[0]
    target_day = intent.new_day or intent.day or ""
    window = eligible_slots(
        fixture,
        clinic=current.clinic,
        day=target_day,
        time_preference=intent.time_preference,
    )
    if intent.selection == "LATER":
        window = [s for s in window if _minutes(s.time) > _minutes(current.time)]
    elif intent.selection == "EARLIER":
        window = [s for s in window if _minutes(s.time) < _minutes(current.time)]
    acceptable = prepared.acceptable_new_slots or ([prepared.new_slot] if prepared.new_slot else [])
    if sorted(acceptable) != sorted(slot.key for slot in window):
        problems.append("the acceptable new slots differ from the slots the request allows")
    if prepared.new_slot not in acceptable:
        problems.append("the preferred new slot is not acceptable")
    if any(slots[key].doctor != current.doctor for key in acceptable if key in slots):
        problems.append("a reschedule target with another doctor would make the request ambiguous")
    return problems


def _ambiguity_problems(scenario: ScenarioSpec, text: str) -> list[str]:
    problems: list[str] = []
    fixture, intent = scenario.fixture, scenario.intent
    if intent.ambiguity == "DOCTOR_NAME":
        named = doctors_named(fixture, intent.clinic or "", surname(_stated_doctor_name(scenario)))
        if len(named) < 2 or sorted(named) != sorted(intent.options):
            problems.append("the doctor name must match exactly the declared options")
    elif intent.ambiguity == "APPOINTMENT_TARGET":
        matches = [slot.key for slot in matching_appointments(fixture, intent)]
        if len(matches) < 2 or sorted(matches) != sorted(intent.options):
            problems.append("the request must select exactly the declared appointments")
    elif intent.ambiguity == "MISSING_DATE":
        if intent.day is not None or any(
            p.startswith("day") for p in request_placeholders(scenario.request)
        ):
            problems.append("a missing-date scenario must not state a date")
    else:
        problems.append("a clarification scenario declares its ambiguity")
    return problems


def _stated_doctor_name(scenario: ScenarioSpec) -> str:
    """The doctor the request names (it says only the surname), or an empty string."""
    key = scenario.stated.get("doctor", "")
    return doctor_index(scenario.fixture)[key][0].name if key else ""


def _response_problems(scenario: ScenarioSpec) -> list[str]:
    problems: list[str] = []
    expected, responses = scenario.expected, scenario.responses
    final_kind = {
        "BOOKED": "COMMITTED",
        "CANCELLED": "COMMITTED",
        "RESCHEDULED": "COMMITTED",
        "CONFLICT": "CONFLICT_REPORT",
        "CLARIFICATION": "CLARIFICATION",
        "NO_AVAILABILITY": "NO_AVAILABILITY",
        "NO_MATCH": "NO_MATCH",
        "AUTH_REQUIRED": "AUTH_REQUIRED",
    }[expected.outcome]
    if responses.final.kind != final_kind:
        problems.append(f"the final reply should be {final_kind}")
    reaches_review = expected.prepared is not None
    if reaches_review != (responses.review is not None):
        problems.append("a review reply exists exactly when an action is prepared")
    if responses.review and responses.review.kind != "REVIEW_PROMPT":
        problems.append("the reply before confirmation is a REVIEW_PROMPT")
    slots = slot_index(scenario.fixture)
    for response in (responses.review, responses.final):
        if response is None:
            continue
        for key in response.subject:
            if key not in slots:
                problems.append(f"reply subject {key!r} is not a slot")
        if response.kind in ACTION_KINDS and not response.subject:
            problems.append("an action reply names the slot it describes")
        if response.kind in ACTION_KINDS and not response.required_details:
            problems.append("an action reply requires details")
        if response.kind not in ACTION_KINDS and response.required_details:
            problems.append("only action replies require details")
    if scenario.expected.outcome == "CLARIFICATION":
        if not responses.final.options and scenario.intent.ambiguity != "MISSING_DATE":
            problems.append("a clarification lists its options")
        if not responses.final.must_cover:
            problems.append("a clarification declares what it must ask about")
    return problems
