"""Builds a benchmark scenario's world in a database, and reads it back to check it.

The manifest defines each scenario's initial state as data (`benchmark_spec.initial_state`). This
module is the only thing that turns that data into rows, and it is deliberately not trusted to have
done so correctly: `observe_state` reads the database back through separate SQL and the result is
compared with the manifest's own definition, so a builder defect, a constraint that silently
changed a row, or leftover data from another scenario shows up as a mismatch before any approach
runs. `verify_search_behavior` goes one step further and asks the real search tools what they
return, comparing that with what the manifest says an honest search must find.

No approach is involved here: both receive the world this module builds, and nothing else.
"""

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, bindparam, insert, text

from app.agent_tools import ToolRuntime
from app.agent_tools.tracing import InMemoryTraceSink
from app.db.models import Appointment
from app.evaluation import benchmark_spec as spec
from app.evaluation.environment import EVALUATION_START
from app.evaluation.fixtures import book_slot_for, create_clinic, create_patient, patient_id_of
from app.seed.catalog import CLINICS

SEED_CLINIC_NAMES = tuple(clinic.name for clinic in CLINICS)
CANCELLED_AT = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)  # Before the evaluation clock starts.


class BenchmarkEnvironmentError(Exception):
    """A scenario's world is not what the manifest says. The run is invalid and is not scored."""


@dataclass(frozen=True)
class BuiltWorld:
    """The identifiers behind a scenario's keys. Held by the harness; never shown to a model."""

    patient_code: str
    other_code: str
    patient_id: uuid.UUID
    other_id: uuid.UUID
    clinic_ids: dict[str, uuid.UUID]
    doctor_ids: dict[str, uuid.UUID]
    slot_ids: dict[str, uuid.UUID]
    appointment_ids: dict[
        str, uuid.UUID
    ]  # The patient's appointments, by the slot key they began on.

    def slot_key(self, slot_id: Any) -> str | None:
        return _reverse(self.slot_ids).get(str(slot_id))

    def clinic_key(self, clinic_id: Any) -> str | None:
        return _reverse(self.clinic_ids).get(str(clinic_id))

    def doctor_key(self, doctor_id: Any) -> str | None:
        return _reverse(self.doctor_ids).get(str(doctor_id))

    def appointment_key(self, appointment_id: Any) -> str | None:
        return _reverse(self.appointment_ids).get(str(appointment_id))


def _reverse(mapping: dict[str, uuid.UUID]) -> dict[str, str]:
    return {str(value): key for key, value in mapping.items()}


def suffix_for(scenario_id: str) -> str:
    """The same eight characters for a scenario in every run, so every approach is signed in as the
    same patient and every run starts from the same rows."""
    return hashlib.sha256(f"benchmark:{scenario_id}".encode()).hexdigest()[:8]


def _utc(fixture: spec.FixtureSpec, slot: spec.SlotSpec) -> datetime:
    local = spec.local_datetime(fixture, slot).replace(tzinfo=ZoneInfo(fixture.timezone))
    return local.astimezone(UTC)


# --- building ---------------------------------------------------------------------------------


def build_world(
    engine: Engine, scenario: spec.ScenarioSpec, *, suffix: str, pin: str
) -> BuiltWorld:
    """Create the scenario's clinics, doctors, slots, patients, and bookings from the manifest."""
    fixture = scenario.fixture
    patient_code = create_patient(engine, suffix, pin, "A")
    other_code = create_patient(engine, suffix, pin, "B")
    clinic_ids: dict[str, uuid.UUID] = {}
    doctor_ids: dict[str, uuid.UUID] = {}
    slot_ids: dict[str, uuid.UUID] = {}
    for clinic in fixture.clinics:
        order = [(doctor.key, doctor.name, doctor.specialty) for doctor in clinic.doctors]
        starts = [
            [_utc(fixture, slot) for slot in fixture.slots if slot.doctor == key]
            for key, _, _ in order
        ]
        built = create_clinic(
            engine,
            name=clinic.name,
            timezone=fixture.timezone,
            doctors=[(name, specialty) for _, name, specialty in order],
            doctor_slot_starts=starts,
        )
        clinic_ids[clinic.key] = built.clinic_id
        for (key, _, _), doctor_id, ids in zip(
            order, built.doctor_ids, built.slot_ids, strict=True
        ):
            doctor_ids[key] = doctor_id
            keys = [slot.key for slot in fixture.slots if slot.doctor == key]
            slot_ids.update(zip(keys, ids, strict=True))

    appointment_ids: dict[str, uuid.UUID] = {}
    for slot in fixture.slots:
        if slot.held_by == "patient":
            appointment_ids[slot.key], _ = book_slot_for(engine, patient_code, slot_ids[slot.key])
        elif slot.held_by == "other":
            book_slot_for(engine, other_code, slot_ids[slot.key])
        elif slot.held_by == "patient_cancelled":
            with engine.begin() as connection:
                appointment_ids[slot.key] = connection.execute(
                    insert(Appointment)
                    .values(
                        patient_id=patient_id_of(engine, patient_code),
                        slot_id=slot_ids[slot.key],
                        status="CANCELLED",
                        cancelled_at=CANCELLED_AT,
                    )
                    .returning(Appointment.appointment_id)
                ).scalar_one()
    return BuiltWorld(
        patient_code=patient_code,
        other_code=other_code,
        patient_id=patient_id_of(engine, patient_code),
        other_id=patient_id_of(engine, other_code),
        clinic_ids=clinic_ids,
        doctor_ids=doctor_ids,
        slot_ids=slot_ids,
        appointment_ids=appointment_ids,
    )


# --- reading the database back ------------------------------------------------------------------

_SLOTS = text(
    "SELECT c.name AS clinic, c.timezone, d.display_name AS doctor, s.starts_at, s.ends_at,"
    " s.status, s.slot_id FROM slots s"
    " JOIN clinics c ON c.clinic_id = s.clinic_id"
    " JOIN doctors d ON d.doctor_id = s.doctor_id"
    " WHERE c.name NOT IN :seeded"
).bindparams(bindparam("seeded", expanding=True))
_DOCTORS = text(
    "SELECT c.name AS clinic, d.display_name AS doctor, d.specialty FROM clinics c"
    " JOIN doctor_clinics dc ON dc.clinic_id = c.clinic_id"
    " JOIN doctors d ON d.doctor_id = dc.doctor_id"
    " WHERE c.name NOT IN :seeded"
).bindparams(bindparam("seeded", expanding=True))
_CLINICS = text("SELECT name, timezone FROM clinics WHERE name NOT IN :seeded").bindparams(
    bindparam("seeded", expanding=True)
)
_APPOINTMENTS = text(
    "SELECT slot_id, patient_id, status FROM appointments WHERE slot_id IN"
    " (SELECT slot_id FROM slots s JOIN clinics c ON c.clinic_id = s.clinic_id"
    "  WHERE c.name NOT IN :seeded)"
).bindparams(bindparam("seeded", expanding=True))


def observe_state(
    engine: Engine,
    *,
    patient_id: uuid.UUID,
    other_id: uuid.UUID,
    order: spec.FixtureSpec | None = None,
) -> dict[str, Any]:
    """Every benchmark clinic, doctor, and slot in the database, as `benchmark_spec.initial_state`
    describes them. The seeded demonstration data is not part of the scenario and is not included.

    `order` only decides how clinics and doctors are listed, so the result can be compared with the
    manifest's own state; anything the database holds that the manifest does not is still listed
    (after the rest), so extra data is never hidden.
    """
    params = {"seeded": list(SEED_CLINIC_NAMES)}
    with engine.connect() as connection:
        clinics = connection.execute(_CLINICS, params).all()
        doctors = connection.execute(_DOCTORS, params).all()
        slots = connection.execute(_SLOTS, params).all()
        appointments = connection.execute(_APPOINTMENTS, params).all()

    by_slot: dict[Any, list[Any]] = {}
    for row in appointments:
        by_slot.setdefault(row.slot_id, []).append(row)

    def holder(slot: Any) -> str:
        rows = by_slot.get(slot.slot_id, [])
        confirmed = [row for row in rows if row.status == "CONFIRMED"]
        if slot.status == "BOOKED":
            if len(confirmed) != 1:
                return "inconsistent"
            owner = confirmed[0].patient_id
            return "patient" if owner == patient_id else "other" if owner == other_id else "unknown"
        if confirmed:
            return "inconsistent"
        if any(row.status == "CANCELLED" and row.patient_id == patient_id for row in rows):
            return "patient_cancelled"
        return "none"

    clinic_order = [c.name for c in order.clinics] if order else []
    doctor_order = {c.name: [d.name for d in c.doctors] for c in order.clinics} if order else {}
    names = sorted({row.name for row in clinics}, key=lambda n: _rank(n, clinic_order))
    zones = {row.timezone for row in clinics}
    state_clinics = []
    for name in names:
        listed = {(row.doctor, row.specialty) for row in doctors if row.clinic == name}
        ranked = sorted(listed, key=lambda item: _rank(item[0], doctor_order.get(name, [])))
        state_clinics.append({"name": name, "doctors": [[doctor, kind] for doctor, kind in ranked]})
    zone = next(iter(zones)) if len(zones) == 1 else f"MIXED:{sorted(zones)}"
    state_slots = []
    for slot in slots:
        local = slot.starts_at.astimezone(ZoneInfo(slot.timezone))
        state_slots.append(
            [
                slot.clinic,
                slot.doctor,
                local.date().isoformat(),
                local.strftime("%H:%M"),
                holder(slot),
            ]
        )
    return {"timezone": zone, "clinics": state_clinics, "slots": sorted(state_slots)}


def _rank(name: str, order: list[str]) -> tuple[int, str]:
    return (order.index(name), name) if name in order else (len(order), name)


def slot_durations_ok(engine: Engine) -> bool:
    """Every benchmark slot is exactly one scheduling interval long."""
    with engine.connect() as connection:
        bad = connection.execute(
            text(
                "SELECT count(*) FROM slots s JOIN clinics c ON c.clinic_id = s.clinic_id"
                " WHERE c.name NOT IN :seeded AND s.ends_at - s.starts_at <> interval '30 minutes'"
            ).bindparams(bindparam("seeded", expanding=True)),
            {"seeded": list(SEED_CLINIC_NAMES)},
        ).scalar_one()
    return bad == 0


def fingerprint_of(state: dict[str, Any]) -> str:
    """The same hash as `benchmark_spec.initial_state_fingerprint`, for a state read from the
    database, so the two can be compared directly."""
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def state_differences(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Readable reasons two states differ. Empty if they are the same."""
    problems: list[str] = []
    if expected["timezone"] != observed["timezone"]:
        problems.append(f"time zone: expected {expected['timezone']}, found {observed['timezone']}")
    expected_clinics = {c["name"]: c["doctors"] for c in expected["clinics"]}
    observed_clinics = {c["name"]: c["doctors"] for c in observed["clinics"]}
    for name in sorted(set(expected_clinics) - set(observed_clinics)):
        problems.append(f"missing clinic {name!r}")
    for name in sorted(set(observed_clinics) - set(expected_clinics)):
        problems.append(f"unexpected clinic {name!r}")
    for name in sorted(set(expected_clinics) & set(observed_clinics)):
        if sorted(expected_clinics[name]) != sorted(observed_clinics[name]):
            problems.append(f"doctors differ at {name!r}")
    wanted = [tuple(item) for item in expected["slots"]]
    found = [tuple(item) for item in observed["slots"]]
    for item in sorted(set(wanted) - set(found)):
        problems.append(f"missing or different slot {item}")
    for item in sorted(set(found) - set(wanted)):
        problems.append(f"unexpected slot {item}")
    if len(wanted) != len(found):
        problems.append(f"slot count: expected {len(wanted)}, found {len(found)}")
    return problems


# --- the expected end of a scenario -------------------------------------------------------------


def expected_final_states(scenario: spec.ScenarioSpec) -> list[dict[str, Any]]:
    """Every acceptable end state: the initial state with the manifest's declared changes applied.

    A reschedule to any of several acceptable slots has one end state per slot. The change set is
    derived from the expectation and the world events (`benchmark_spec.derived_final`).
    """
    prepared = scenario.expected.prepared
    candidates: list[str | None] = [None]
    if prepared and prepared.action == "RESCHEDULE":
        candidates = list(prepared.acceptable_new_slots or [prepared.new_slot])
    states = []
    for new_slot in candidates:
        final = spec.derived_final(scenario, new_slot=new_slot)
        fixture = copy.deepcopy(scenario.fixture)
        for slot in fixture.slots:
            if slot.key in final.booked_by_patient:
                slot.held_by = "patient"
            elif slot.key in final.booked_by_other:
                slot.held_by = "other"
            elif slot.key in final.released:
                slot.held_by = (
                    "patient_cancelled" if slot.key in final.cancelled_appointments else "none"
                )
        states.append(spec.initial_state(fixture))
    return states


# --- seeded baseline ---------------------------------------------------------------------------


def seed_baseline_fingerprint(engine: Engine) -> str:
    """A hash of the demonstration data every database starts with, to show it is the same in every
    run and that a run did not alter it."""
    with engine.connect() as connection:
        schedule = connection.execute(
            text(
                "SELECT c.name, d.display_name, s.starts_at, s.status FROM slots s"
                " JOIN clinics c ON c.clinic_id = s.clinic_id"
                " JOIN doctors d ON d.doctor_id = s.doctor_id"
                " WHERE c.name IN :seeded ORDER BY 1, 2, 3"
            ).bindparams(bindparam("seeded", expanding=True)),
            {"seeded": list(SEED_CLINIC_NAMES)},
        ).all()
        appointments = connection.execute(
            text(
                "SELECT a.status, count(*) FROM appointments a JOIN patients p USING (patient_id)"
                " WHERE p.patient_code NOT LIKE 'EVP-%' GROUP BY 1 ORDER BY 1"
            )
        ).all()
    canonical = json.dumps(
        {
            "schedule": [[r[0], r[1], r[2].isoformat(), r[3]] for r in schedule],
            "appointments": [[r[0], r[1]] for r in appointments],
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- the real search tools against the manifest -------------------------------------------------


def verify_search_behavior(
    engine: Engine, scenario: spec.ScenarioSpec, world: BuiltWorld
) -> list[str]:
    """Ask the real, public search tools what an honest search finds, and compare it with what the
    manifest derives. If they disagree the manifest, the builder, or a tool is wrong: either way
    the scenario cannot fairly be run."""
    problems: list[str] = []
    fixture, intent = scenario.fixture, scenario.intent
    runtime = ToolRuntime(
        engine, clock=lambda: EVALUATION_START, trace_sink=InMemoryTraceSink(), sleep=lambda _: None
    )

    def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return runtime.invoke(tool, arguments, principal=None).to_dict()

    if scenario.stated.get("clinic"):
        name = spec.clinic_index(fixture)[scenario.stated["clinic"]].name
        reply = call("search_clinic_directory", {"clinic_name": name})
        found = sorted(c["name"] for c in (reply["data"] or {}).get("clinics", []))
        predicted = sorted(
            candidate
            for candidate in [c.name for c in fixture.clinics] + list(SEED_CLINIC_NAMES)
            if spec.search_matches(name, candidate)
        )
        if not reply["ok"] or found != predicted:
            problems.append(f"a search for the clinic name finds {found}, expected {predicted}")

    if intent.ambiguity == "DOCTOR_NAME":
        key = scenario.stated["doctor"]
        surname = spec.surname(spec.doctor_index(fixture)[key][0].name)
        clinic = spec.clinic_index(fixture)[scenario.stated["clinic"]].name
        reply = call("search_clinic_directory", {"doctor_name": surname, "clinic_name": clinic})
        choices = sorted(c["display_name"] for c in (reply["error"] or {}).get("choices", []))
        wanted = sorted(spec.doctor_index(fixture)[k][0].name for k in intent.options)
        if (reply["error"] or {}).get("code") != "AMBIGUOUS_MATCH" or choices != wanted:
            problems.append(f"the surname search does not report exactly {wanted} as ambiguous")

    if intent.clinic and (intent.new_day or intent.day):
        day = intent.new_day or intent.day or ""
        arguments: dict[str, Any] = {
            "clinic_id": str(world.clinic_ids[intent.clinic]),
            "start_date": spec.day_of(fixture, day).isoformat(),
            "end_date": spec.day_of(fixture, day).isoformat(),
        }
        if intent.doctor:
            arguments["doctor_id"] = str(world.doctor_ids[intent.doctor])
        if intent.specialty:
            arguments["specialty"] = intent.specialty
        if intent.time_preference != "ANYTIME":
            arguments["time_preference"] = intent.time_preference
        reply = call("search_appointment_availability", arguments)
        predicted_slots = spec.eligible_slots(
            fixture,
            clinic=intent.clinic,
            day=day,
            doctor=intent.doctor,
            specialty=intent.specialty,
            time_preference=intent.time_preference,
        )
        found_slots = sorted(
            world.slot_key(s["slot_id"]) or "?" for s in (reply["data"] or {}).get("slots", [])
        )
        wanted_slots = sorted(slot.key for slot in predicted_slots)
        if predicted_slots and (not reply["ok"] or found_slots != wanted_slots):
            problems.append(
                f"availability finds {found_slots}, the manifest expects {wanted_slots}"
            )
        if not predicted_slots and (reply["error"] or {}).get("code") != "NO_AVAILABILITY":
            problems.append("availability should report NO_AVAILABILITY for this request")
    return problems


def verify_world(engine: Engine, scenario: spec.ScenarioSpec, world: BuiltWorld) -> list[str]:
    """Every way the database differs from what the manifest defines. Empty if it matches."""
    problems = state_differences(
        spec.initial_state(scenario.fixture),
        observe_state(
            engine, patient_id=world.patient_id, other_id=world.other_id, order=scenario.fixture
        ),
    )
    if not slot_durations_ok(engine):
        problems.append("a slot is not one scheduling interval long")
    problems += verify_search_behavior(engine, scenario, world)
    return problems
