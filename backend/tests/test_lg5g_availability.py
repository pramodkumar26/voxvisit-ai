"""An empty search is explained, never papered over (found in the first microphone trial).

A patient said "General Physician"; the model wrote `specialty: "General Practice"`; the clinic's
doctors are Cardiology, Pediatrics and Internal Medicine; the filter stayed on silently and the
assistant said "no openings" three times for days that had many. These tests pin the repair:

- a specialty the clinic does not have is explained, with the specialties it does have, and the
  filter stays until the patient chooses;
- "nothing matches" is told apart from "nothing at all";
- alternative days are days the backend listed, never made up;
- an explicit change of clinic, date, doctor or specialty updates the saved request;
- nothing says an appointment is booked before a trusted confirmation.

Model readings are scripted; the tools are a fake with specialties. The real database and the full
Lakeside conversation are in tests/database/test_lakeside_regression.py."""

import ast
import re
from datetime import date
from pathlib import Path

import pytest
from test_conversation_unit import NOW, RetiringBackend, ok
from test_orchestrator import ask, book, sched, session

from app.evaluation import availability_notes as notes

LAKE = "Lakeside Community Clinic"
TUE, WED, THU, SAT = "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-07"
BOOKED_WORDS = re.compile(r"\b(booked|scheduled|rescheduled|cancel+ed|confirmed|reserved)\b", re.I)


class SpecialtyBackend(RetiringBackend):
    """The fake tools plus doctors with specialties, a clinic that lists them, and the specialty
    filter of the real availability search. Lakeside has no General Practice doctor."""

    def __init__(self):
        super().__init__()
        self.clinics["c-lake"] = {
            "clinic_id": "c-lake",
            "name": LAKE,
            "timezone": "America/Denver",
        }
        for key, name in (
            ("d-ellery", "Dr. Marcus Ellery"),
            ("d-nambiar", "Dr. Priya Nambiar"),
            ("d-desai", "Dr. Rohan Desai"),
        ):
            self.doctors[key] = {"doctor_id": key, "display_name": name, "clinic": "c-lake"}
        self.specialty = {
            "d-ellery": "Cardiology",
            "d-nambiar": "Pediatrics",
            "d-desai": "Internal Medicine",
        }
        day = date.fromisoformat
        self.slot("l-ellery-tue", "c-lake", "d-ellery", day(TUE), 9)
        self.slot("l-nambiar-tue", "c-lake", "d-nambiar", day(TUE), 10)
        self.slot("l-desai-tue-am", "c-lake", "d-desai", day(TUE), 8)
        self.slot("l-desai-tue-pm", "c-lake", "d-desai", day(TUE), 13)
        self.slot("l-desai-wed", "c-lake", "d-desai", day(WED), 9)
        self.slot("l-ellery-mon", "c-lake", "d-ellery", date(2026, 11, 9), 10)

    def specialty_of(self, doctor: str) -> str:
        return self.specialty.get(doctor, "Family Medicine")

    def view(self, clinic, doctor, when):
        found = super().view(clinic, doctor, when)
        found["doctor"]["specialty"] = self.specialty_of(doctor)
        return found

    def search_clinic_directory(self, args, principal):
        response = super().search_clinic_directory(args, principal)
        data = response.to_dict().get("data") or {}
        for clinic in data.get("clinics", []):
            clinic["specialties"] = sorted(
                {
                    self.specialty_of(d["doctor_id"])
                    for d in self.doctors.values()
                    if d["clinic"] == clinic["clinic_id"]
                }
            )
        for doctor in data.get("doctors", []):
            doctor["specialty"] = self.specialty_of(doctor["doctor_id"])
        return response

    def search_appointment_availability(self, args, principal):
        response = super().search_appointment_availability(args, principal)
        wanted = args.get("specialty")
        body = response.to_dict()
        if not wanted or not body["ok"]:
            return response
        slots = [
            s
            for s in body["data"]["slots"]
            if notes.same_specialty(wanted, s["doctor"]["specialty"])
        ]
        return ok({"slots": slots}) if slots else self._none()

    @staticmethod
    def _none():
        from test_conversation_unit import fail

        return fail("NO_AVAILABILITY")

    def availability_calls(self):
        return [a for tool, a, _ in self.calls if tool == "search_appointment_availability"]

    def tool_names(self):
        return [tool for tool, *_ in self.calls]


def _is_regex_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
    )


def lake(**fields):
    return {"clinic_name": LAKE, "start_date": TUE, "end_date": TUE, **fields}


def asked_for_general_practice(*then):
    """The failure: a clinic with no General Practice doctor, asked for a morning."""
    backend = SpecialtyBackend()
    made, wire = session(
        backend,
        book(**lake(specialty="General Practice", time_preference="MORNING")),
        *then,
    )
    reply = made.respond("I want a general physician at Lakeside Community Clinic tomorrow morning")
    return backend, made, wire, reply


# --- a specialty the clinic does not have -----------------------------------------------------


def test_a_specialty_the_clinic_does_not_have_is_explained_and_the_filter_stays() -> None:
    backend, made, _, reply = asked_for_general_practice()

    assert f"{LAKE} doesn't have a General Practice doctor" in reply.text
    for listed in ("Cardiology", "Internal Medicine", "Pediatrics"):
        assert listed in reply.text
    assert "Which specialty would you like, or would you like a different clinic?" in reply.text
    assert backend.prepared == [] and not reply.awaiting_confirmation
    assert "I don't see any openings" not in reply.text  # Not the false claim from the trial.
    saved = made.snapshot()
    assert saved["request"]["specialty"] == "General Practice"  # Kept, not silently removed.
    assert saved["dialogue"]["awaiting"] == "SPECIALTY"
    assert len(backend.availability_calls()) == 1  # No hidden second search without the filter.


def test_asking_again_gets_the_same_honest_explanation_not_a_loop_of_no_openings() -> None:
    _, made, _, _ = asked_for_general_practice(book())  # "Can you find me an appointment?"

    again = made.respond("Can you find me an appointment?")

    assert "doesn't have a General Practice doctor" in again.text
    assert made.snapshot()["request"]["specialty"] == "General Practice"


def test_choosing_a_specialty_the_clinic_has_continues_the_same_request_to_a_review() -> None:
    backend, made, _, _ = asked_for_general_practice(sched(specialty="Internal Medicine"))

    reply = made.respond("Internal Medicine")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "l-desai-tue-am"}]
    request = made.snapshot()["request"]
    assert request["specialty"] == "Internal Medicine" and request["clinic_name"] == LAKE
    assert request["start_date"] == TUE and request["time_preference"] == "MORNING"
    assert "8:00 AM" in reply.text and "Dr. Rohan Desai" in reply.text


def test_only_the_patient_saying_any_drops_the_specialty_and_then_it_asks_which_time() -> None:
    backend, made, _, _ = asked_for_general_practice(sched())

    reply = made.respond("Any specialty is fine")

    assert made.snapshot()["request"]["specialty"] is None
    assert backend.prepared == [] and "8:00 AM" in reply.text and "9:00 AM" in reply.text


def test_an_answer_that_is_not_about_the_specialty_leaves_it_in_force() -> None:
    backend, made, _, _ = asked_for_general_practice(sched(start_date=THU, end_date=THU))

    reply = made.respond("How about Thursday instead?")

    saved = made.snapshot()["request"]
    assert saved["start_date"] == THU and saved["specialty"] == "General Practice"
    assert "doesn't have a General Practice doctor" in reply.text and backend.prepared == []


# --- explicit changes reach the saved request -------------------------------------------------


def test_choosing_another_clinic_updates_the_request_and_checks_the_specialty_there() -> None:
    _, made, _, _ = asked_for_general_practice(sched(clinic_name="Pine Clinic"))

    reply = made.respond("Actually, Pine Clinic")

    saved = made.snapshot()["request"]
    assert saved["clinic_name"] == "Pine Clinic" and saved["doctor_name"] is None
    assert saved["specialty"] == "General Practice"
    assert "Pine Clinic doesn't have a General Practice doctor" in reply.text


def test_naming_a_doctor_chooses_that_doctor_over_the_unmatched_specialty() -> None:
    backend, made, _, _ = asked_for_general_practice(sched(doctor_name="Dr. Priya Nambiar"))

    reply = made.respond("Dr. Priya Nambiar then")

    saved = made.snapshot()["request"]
    assert saved["doctor_name"] == "Dr. Priya Nambiar" and saved["specialty"] is None
    assert reply.awaiting_confirmation and "Dr. Priya Nambiar" in reply.text
    assert backend.prepared == [{"action": "BOOK", "slot_id": "l-nambiar-tue"}]


def test_a_doctor_name_that_matches_nobody_does_not_remove_the_specialty() -> None:
    backend, made, _, _ = asked_for_general_practice(
        sched(doctor_name="Dr. Zed Nobody"), sched(clinic_name=LAKE)
    )

    reply = made.respond("Dr. Zed Nobody then")

    assert "couldn't find a doctor called Dr. Zed Nobody" in reply.text
    assert made.snapshot()["request"]["specialty"] == "General Practice"  # Kept: no doctor chosen.
    assert backend.prepared == []


def test_a_role_heard_as_a_doctors_name_is_the_specialty_and_is_explained() -> None:
    """The second microphone trial: Whisper heard "gender physician", then the model read "General
    Physician at Lakeside" as the name of a doctor and the assistant asked which doctor."""
    backend = SpecialtyBackend()
    made, _ = session(
        backend,
        book(specialty="Gender Medicine", start_date=TUE, end_date=TUE),
        sched(clinic_name=LAKE, doctor_name="General Physician"),
    )
    made.respond("I want to book an appointment with a gender physician tomorrow.")

    reply = made.respond("General Physician at Lakeside Community Clinic")

    assert f"{LAKE} doesn't have a General Physician doctor" in reply.text
    assert "Internal Medicine" in reply.text and "couldn't find a doctor" not in reply.text
    saved = made.snapshot()["request"]
    assert saved["doctor_name"] is None and saved["specialty"] == "General Physician"
    assert saved["clinic_name"] == LAKE and backend.prepared == []


def test_changing_the_day_while_a_specialty_is_set_searches_that_day_with_it() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend,
        book(**lake(start_date=WED, end_date=WED, specialty="Cardiology")),
        sched(start_date=TUE, end_date=TUE),
    )
    first = made.respond("Book a cardiologist at Lakeside on Wednesday")
    assert "I don't see any Cardiology openings" in first.text and backend.prepared == []

    reply = made.respond("Tuesday instead")

    saved = made.snapshot()["request"]
    assert saved["start_date"] == TUE and saved["specialty"] == "Cardiology"
    assert backend.availability_calls()[-1] == {
        "clinic_id": "c-lake",
        "start_date": TUE,
        "end_date": TUE,
        "specialty": "Cardiology",
    }
    assert reply.awaiting_confirmation and backend.prepared == [
        {"action": "BOOK", "slot_id": "l-ellery-tue"}
    ]


# --- nothing that matches, versus nothing at all ----------------------------------------------


def test_openings_that_exist_but_do_not_match_the_specialty_are_said_to_exist() -> None:
    backend = SpecialtyBackend()
    made, _ = session(backend, book(**lake(start_date=WED, end_date=WED, specialty="Cardiology")))

    reply = made.respond("Book a cardiologist at Lakeside on Wednesday")

    assert "I don't see any Cardiology openings" in reply.text
    assert "though there are other openings then with Internal Medicine" in reply.text
    assert "change the specialty" in reply.text and backend.prepared == []
    unfiltered = backend.availability_calls()[1]
    assert "specialty" not in unfiltered  # The comparison search; it was never used to book.
    assert made.snapshot()["request"]["specialty"] == "Cardiology"
    assert made.snapshot()["dialogue"]["awaiting"] == "SPECIALTY"


def test_a_part_of_the_day_with_nothing_is_named_as_the_filter_not_as_no_openings() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend, book(**lake(start_date=WED, end_date=WED, time_preference="AFTERNOON"))
    )

    reply = made.respond("Book Lakeside Wednesday afternoon")

    assert "I don't see any afternoon openings" in reply.text
    assert "though there are other openings then" in reply.text
    assert "change the time of day" in reply.text and backend.prepared == []
    assert made.snapshot()["request"]["time_preference"] == "AFTERNOON"


def test_a_day_with_nothing_at_all_says_so_and_names_only_days_the_backend_listed() -> None:
    backend = SpecialtyBackend()
    made, _ = session(backend, book(**lake(start_date=SAT, end_date=SAT)))

    reply = made.respond("Book Lakeside on Saturday")

    assert "I don't see any openings at" in reply.text and "November 7, 2026" in reply.text
    assert "though there are other openings" not in reply.text  # Not the "no match" wording.
    named = set(re.findall(r"(?:Tuesday|Wednesday|Monday), November \d+", reply.text))
    listed = {"Tuesday, November 3", "Wednesday, November 4", "Monday, November 9"}
    assert named == listed and "Which day would you like?" in reply.text
    assert "Saturday, November 7" not in named


def test_alternative_days_keep_the_patients_specialty_and_leave_out_other_doctors() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend, book(**lake(start_date=SAT, end_date=SAT, specialty="Internal Medicine"))
    )

    reply = made.respond("Book an internist at Lakeside on Saturday")

    assert "The next Internal Medicine openings are on" in reply.text
    assert "Tuesday, November 3 and Wednesday, November 4" in reply.text
    assert "November 9" not in reply.text  # Only a Cardiology opening is open then.


def test_nothing_in_the_whole_look_ahead_is_said_plainly() -> None:
    backend = SpecialtyBackend()
    backend.slots = []
    made, _ = session(backend, book(**lake(start_date=SAT, end_date=SAT)))

    reply = made.respond("Book Lakeside on Saturday")

    assert "none in the next 31 days" in reply.text and "try another clinic" in reply.text
    assert not re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday), November", reply.text)


# --- the question path says the same things ---------------------------------------------------


def test_a_lookup_for_a_specialty_the_clinic_lacks_lists_what_it_has_and_asks() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name=LAKE, specialty="General Practice", start_date=TUE),
    )

    reply = made.respond("Does Lakeside have openings with a general practitioner tomorrow?")

    assert f"{LAKE} doesn't have a General Practice doctor" in reply.text
    assert "Internal Medicine" in reply.text and backend.prepared == []
    assert "Which specialty would you like" in reply.text


def test_a_lookup_with_openings_for_other_specialties_says_they_exist() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name=LAKE, specialty="Cardiology", start_date=WED, end_date=WED),
    )

    reply = made.respond("Any cardiology openings at Lakeside on Wednesday?")

    assert "I don't see any Cardiology openings" in reply.text
    assert "other openings then with Internal Medicine" in reply.text


def test_a_lookup_for_an_empty_day_names_real_days() -> None:
    backend = SpecialtyBackend()
    made, _ = session(backend, ask("OPENINGS", clinic_name=LAKE, start_date=SAT, end_date=SAT))

    reply = made.respond("Any openings at Lakeside on Saturday?")

    assert "I don't see any openings at" in reply.text
    assert "Tuesday, November 3" in reply.text and "Wednesday, November 4" in reply.text
    assert "Saturday, November 7" in reply.text  # The empty day itself, named once, as asked.


def test_a_lookup_continues_with_the_specialty_the_patient_then_chooses() -> None:
    backend = SpecialtyBackend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name=LAKE, specialty="General Practice", start_date=TUE),
        ask("OPENINGS", specialty="Internal Medicine"),
    )
    first = made.respond("Do you have openings with a general practitioner at Lakeside tomorrow?")
    assert "doesn't have a General Practice doctor" in first.text

    reply = made.respond("Internal Medicine")

    assert "8:00 AM" in reply.text and "1:00 PM" in reply.text  # Dr. Desai's Tuesday openings.
    assert "Dr. Marcus Ellery" not in reply.text and "doesn't have" not in reply.text
    assert backend.prepared == [] and not reply.awaiting_confirmation


# --- wording that must never suggest a booking exists -----------------------------------------


def test_the_summary_of_an_unfinished_request_names_the_specialty_and_is_no_appointment() -> None:
    backend = SpecialtyBackend()
    made, _ = session(backend, book(specialty="General Practice", start_date=TUE, end_date=TUE))

    reply = made.respond("I want to book an appointment with a general physician tomorrow")

    assert "So far I have a request to book an appointment for General Practice" in reply.text
    assert "So far I have an appointment" not in reply.text
    assert not BOOKED_WORDS.search(reply.text)


def test_no_reply_in_the_whole_empty_search_family_claims_a_booking() -> None:
    backend, made, _, first = asked_for_general_practice(
        sched(specialty="Cardiology", start_date=WED, end_date=WED),
        sched(start_date=SAT, end_date=SAT, specialty="Internal Medicine"),
    )
    replies = [
        first,
        made.respond("Cardiology Wednesday"),
        made.respond("Internal Medicine on Saturday"),
    ]

    assert all(not BOOKED_WORDS.search(r.text) for r in replies)
    assert backend.prepared == []


def test_only_the_trusted_event_handler_ever_words_a_booking_as_done() -> None:
    """Every string in the agent's reply code that could claim a booking, a change or a
    confirmation is inside `_after_event`, which runs only for the trusted UI's report."""
    root = Path(__file__).resolve().parents[1] / "app" / "evaluation"
    offenders = []
    for module in ("hybrid", "planner", "orchestrator", "availability_notes"):
        tree = ast.parse((root / f"{module}.py").read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }

        def visit(node, inside, module=module, docstrings=docstrings):
            for child in ast.iter_child_nodes(node):
                if _is_regex_call(child):
                    continue  # A pattern the code matches against, not words it says.
                name = child.name if isinstance(child, ast.FunctionDef) else inside
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and id(child) not in docstrings
                    and not child.value.isupper()  # A status code such as "CONFIRMED".
                    and BOOKED_WORDS.search(child.value)
                    and inside != "_after_event"
                ):
                    offenders.append((module, child.lineno, child.value[:60]))
                visit(child, name)

        visit(tree, "<module>")

    assert offenders == []


@pytest.mark.parametrize(
    ("said", "is_role"),
    [
        ("General Physician", True),
        ("a family doctor", True),
        ("primary care", True),
        ("general practitioner", True),
        ("Dr. Rao", False),
        ("Anita Rao", False),
        ("Dr. General", False),  # "Dr." makes it a name, whatever the word after it.
        ("any doctor", False),  # No preference: handled as such, not as a specialty.
        ("General Hospital", False),
        (None, False),
    ],
)
def test_only_a_kind_of_doctor_is_taken_for_a_specialty_never_a_name(said, is_role) -> None:
    from app.evaluation.hybrid import _role_phrase

    assert _role_phrase(said) is is_role


# --- the helpers ------------------------------------------------------------------------------


def test_same_specialty_compares_stems_and_never_matches_nothing() -> None:
    assert notes.same_specialty("Cardiologist", "Cardiology")
    assert not notes.same_specialty("General Practice", "Internal Medicine")
    assert not notes.same_specialty(None, "Cardiology") and not notes.same_specialty("", "")


def test_the_days_named_come_only_from_the_slots_given_and_are_earliest_first() -> None:
    slots = [
        {"starts_at_utc": "2026-11-09T17:00:00Z", "starts_at_local": "2026-11-09T10:00:00-07:00"},
        {"starts_at_utc": "2026-11-03T15:00:00Z", "starts_at_local": "2026-11-03T08:00:00-07:00"},
        {"starts_at_utc": "2026-11-03T20:00:00Z", "starts_at_local": "2026-11-03T13:00:00-07:00"},
        {"starts_at_utc": "2026-11-04T16:00:00Z", "starts_at_local": "2026-11-04T09:00:00-07:00"},
    ]

    assert notes.next_days(slots) == ["2026-11-03", "2026-11-04", "2026-11-09"]
    assert notes.next_days(slots, limit=2) == ["2026-11-03", "2026-11-04"]
    assert notes.next_days([]) == []


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ((None, None, None), "openings"),
        (("Cardiology", None, None), "Cardiology openings"),
        (("Cardiology", "morning", "Dr. Kim"), "Cardiology morning openings with Dr. Kim"),
    ],
)
def test_the_kind_of_opening_wanted_is_said_back_as_asked(asked, expected) -> None:
    assert notes.kind_of_openings(*asked) == expected


def test_the_clock_is_the_one_the_fixtures_assume() -> None:
    assert NOW.date() == date(2026, 11, 2)  # Monday: Tuesday is "tomorrow" in these tests.
