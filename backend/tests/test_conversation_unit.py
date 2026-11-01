"""The conversation layer without a database or a model: dates and times as patients write them,
the extractor's recovery steps, how the controller keeps and drops conversation state, and how
conversations are kept apart. A fake backend stands in for the tools; the real tools are covered by
tests/database/test_conversation.py."""

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.auth.service import Principal
from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.evaluation.conversation import (
    FALLBACK_REPLY,
    MAX_TURNS_PER_SESSION,
    REVIEW_STILL_OPEN_NOTICE,
    Awaiting,
    ConversationRegistry,
    TextSession,
)
from app.evaluation.hybrid import (
    _OPERATION_WORDS,
    CREDENTIAL_NOTICE,
    SIGN_IN_NOTICE,
    HybridControllerModel,
    PatientIntent,
    StructuredIntentExtractor,
    date_mentions,
    minutes_of,
    stated_dates,
)
from app.evaluation.llm import LlmConfig
from app.evaluation.model import MalformedModelResponse

TODAY = date(2026, 11, 2)  # A Monday.
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)


# --- dates and times as patients write them -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Book it for November 17.", {"2026-11-17"}),
        ("on Nov 17th please", {"2026-11-17"}),
        ("the 17 of November", {"2026-11-17"}),
        ("17 November 2027", {"2027-11-17"}),
        ("November 17, 2026", {"2026-11-17"}),
        ("2026-11-17", {"2026-11-17"}),
        ("tomorrow", {"2026-11-03"}),
        ("the day after tomorrow", {"2026-11-04"}),
        ("next Friday", set()),  # Two reasonable meanings on a Monday: asked, not guessed.
        ("on Monday", {"2026-11-09"}),  # Today is Monday: "Monday" means the next one.
        ("January 5", {"2027-01-05"}),  # Already past this year: the next occurrence.
        ("November 2", {"2026-11-02"}),
        ("February 30", set()),  # Not a date.
        ("at 10:30 on 2026-11-17", {"2026-11-17"}),
        ("I'd like 11 AM", set()),
        ("May I book?", set()),
    ],
)
def test_dates_are_read_the_way_patients_write_them(text, expected) -> None:
    assert stated_dates(text, TODAY) == expected


def test_a_bare_day_number_means_that_day_of_the_month_being_discussed() -> None:
    assert stated_dates("no, I meant the 18th", TODAY, anchor=date(2026, 11, 16)) == {"2026-11-18"}
    assert stated_dates("make it the 4th", TODAY) == {"2026-11-04"}


def test_each_date_is_returned_with_where_it_was_said() -> None:
    found = date_mentions("move my November 16 visit to November 20", TODAY)

    assert [m.iso for m in found] == ["2026-11-16", "2026-11-20"]
    assert found[0].position < found[1].position


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("10:30 AM", 630),
        ("2 pm", 840),
        ("09 AM", 540),
        ("12 PM", 720),
        ("12 AM", 0),
        ("14:30", 870),
        ("noon", 720),
        ("13 pm", None),
        ("later", None),
        (None, None),
    ],
)
def test_clock_times(text, minutes) -> None:
    assert minutes_of(text) == minutes


# --- the extractor: strict reading, salvage, repair ----------------------------------------------


class Provider:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.payloads = []

    def __call__(self, path, payload, timeout_s):
        self.payloads.append(payload)
        content = self.contents.pop(0)
        if isinstance(content, Exception):
            raise content
        return {"message": {"content": content}, "eval_count": 5, "total_duration": 1_000_000}


def reading(**fields) -> str:
    return json.dumps({"operation": "BOOK", **fields})


def extractor(*contents):
    provider = Provider(*contents)
    return StructuredIntentExtractor(LlmConfig(), transport=provider), provider


def test_a_reply_that_states_something_the_message_does_not_is_rejected_whole() -> None:
    reader, _ = extractor(reading(clinic_name="Invented Clinic"))

    with pytest.raises(MalformedModelResponse):
        reader.extract("Book me something on November 17.", today=TODAY)


def test_salvage_keeps_what_the_message_supports_and_nulls_the_rest() -> None:
    reader, _ = extractor(
        reading(
            clinic_name="Invented Clinic",
            doctor_name="Rao",
            start_date="2026-11-18",
            end_date="2026-11-18",
            time_preference="MORNING",
            selection="EARLIEST",
        )
    )
    message = "Book me with Dr. Rao on November 17."
    with pytest.raises(MalformedModelResponse):
        reader.extract(message, today=TODAY)

    kept = reader.salvage(message, TODAY)

    assert kept is not None
    assert kept.operation == "BOOK" and kept.doctor_name == "Rao"
    assert kept.clinic_name is None  # Not in the message.
    assert kept.start_date is None and kept.end_date is None  # Not the date the message states.
    assert kept.time_preference == "ANYTIME" and kept.selection == "UNSPECIFIED"


def test_salvage_drops_a_field_of_the_wrong_shape_but_not_the_whole_reply() -> None:
    reader, _ = extractor(reading(start_date="soon", clinic_name="Pine Clinic"))
    with pytest.raises(MalformedModelResponse):
        reader.extract("Book at Pine Clinic.", today=TODAY)

    kept = reader.salvage("Book at Pine Clinic.", TODAY)

    assert kept is not None and kept.clinic_name == "Pine Clinic" and kept.start_date is None


@pytest.mark.parametrize("content", ["not json at all", "[1, 2]", '{"clinic_name": "X"}', "   "])
def test_a_reply_that_is_not_the_schema_cannot_be_salvaged(content) -> None:
    reader, _ = extractor(content)
    with pytest.raises(MalformedModelResponse):
        reader.extract("Book at X.", today=TODAY)

    assert reader.salvage("Book at X.", TODAY) is None


def test_nothing_is_salvaged_after_a_provider_fault() -> None:
    reader, _ = extractor(TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        reader.extract("Book at X.", today=TODAY)

    assert reader.salvage("Book at X.", TODAY) is None


def test_the_prompt_carries_today_the_conversation_so_far_and_a_repair_note_when_asked() -> None:
    reader, provider = extractor(reading(), reading())

    reader.extract("hi", context='the assistant just asked: "Which clinic?"', today=TODAY)
    reader.extract("hi", today=TODAY, repair=True)

    first, second = (p["messages"][0]["content"] for p in provider.payloads)
    assert "Today is 2026-11-02 (Monday)" in first
    assert 'the assistant just asked: "Which clinic?"' in first
    assert "could not be used" not in first and "could not be used" in second
    assert "ClinicName" not in first  # The prompt never shows an internal name.


@pytest.mark.parametrize(
    "bad",
    [
        {"selection": "EARLIEST"},  # "Book an appointment" does not say earliest.
        {"option_number": 2},  # No ordinal was said.
        {"time": "3 PM"},  # No 3 PM in the message.
    ],
)
def test_values_the_message_does_not_support_are_not_kept(bad) -> None:
    reader, _ = extractor(reading(**bad))
    message = "Book an appointment at Pine Clinic."
    with pytest.raises(MalformedModelResponse):
        reader.extract(message, today=TODAY)

    kept = reader.salvage(message, TODAY)

    assert kept is not None and all(getattr(kept, k) in (None, "UNSPECIFIED") for k in bad)


# --- a fake backend for the controller -----------------------------------------------------------


def local(day: date, hour: int, minute: int = 0) -> str:
    return f"{day.isoformat()}T{hour:02d}:{minute:02d}:00-07:00"


class Response:
    def __init__(self, body):
        self.body = body

    def to_dict(self):
        return self.body


def ok(data):
    return Response({"ok": True, "data": data, "error": None})


def fail(code, **extra):
    return Response({"ok": False, "data": None, "error": {"code": code, **extra}})


class Backend:
    """A small stand-in for the tool runtime: two clinics, three doctors, a few slots and
    appointments, and the same error codes the real tools use."""

    def __init__(self):
        self.calls = []
        self.clinics = {
            "c-cedar": {"clinic_id": "c-cedar", "name": "Linden Court Heart Center"},
            "c-cedar2": {"clinic_id": "c-cedar2", "name": "Linden Court Family Clinic"},
            "c-pine": {"clinic_id": "c-pine", "name": "Pine Clinic"},
            "c-rivers": {"clinic_id": "c-rivers", "name": "Two Rivers Clinic"},
            "c-kiwi": {
                "clinic_id": "c-kiwi",
                "name": "Kauri Coast Clinic",
                "timezone": "Pacific/Auckland",
            },
        }
        for clinic in self.clinics.values():
            clinic.setdefault("timezone", "America/Denver")
        self.doctors = {
            "d-rao": {"doctor_id": "d-rao", "display_name": "Dr. Anita Rao", "clinic": "c-pine"},
            "d-kim": {"doctor_id": "d-kim", "display_name": "Dr. Aiden Kim", "clinic": "c-pine"},
            "d-lee1": {"doctor_id": "d-lee1", "display_name": "Dr. Lee North", "clinic": "c-pine"},
            "d-lee2": {"doctor_id": "d-lee2", "display_name": "Dr. Lee South", "clinic": "c-pine"},
            "d-cedar": {
                "doctor_id": "d-cedar",
                "display_name": "Dr. Ivo Reed",
                "clinic": "c-cedar",
            },
            "d-kiwi": {"doctor_id": "d-kiwi", "display_name": "Dr. Tui Hart", "clinic": "c-kiwi"},
            "d-rivers": {
                "doctor_id": "d-rivers",
                "display_name": "Dr. Ana Wren",
                "clinic": "c-rivers",
            },
        }
        self.slots = []  # (slot_id, clinic, doctor, local string)
        self.appointments = []  # (appointment_id, clinic, doctor, local string)
        self.prepared = []
        self.fail = {}  # tool -> Response

    def slot(self, slot_id, clinic, doctor, day, hour, minute=0, offset="-07:00"):
        when = local(day, hour, minute)[:-6] + offset
        self.slots.append((slot_id, clinic, doctor, when))

    def appointment(self, appointment_id, clinic, doctor, day, hour, offset="-07:00"):
        self.appointments.append((appointment_id, clinic, doctor, local(day, hour)[:-6] + offset))

    def view(self, clinic, doctor, when):
        return {
            "clinic": {
                "clinic_id": clinic,
                "name": self.clinics[clinic]["name"],
                "timezone": self.clinics[clinic]["timezone"],
            },
            "doctor": {
                "doctor_id": doctor,
                "display_name": self.doctors[doctor]["display_name"],
            },
            "starts_at_local": when,
            "starts_at_utc": datetime.fromisoformat(when).astimezone(UTC).isoformat(),
        }

    def invoke(self, tool, arguments, *, principal):
        self.calls.append((tool, dict(arguments), principal))
        if tool in self.fail:
            return self.fail[tool]
        return getattr(self, tool)(arguments, principal)

    def search_clinic_directory(self, args, principal):
        def words(value):
            return value.casefold().replace(".", " ").split()

        clinics = [
            c
            for c in self.clinics.values()
            if all(w in c["name"].casefold() for w in words(args.get("clinic_name", "")))
        ]
        doctors = []
        if args.get("doctor_name"):
            wanted = [w for w in words(args["doctor_name"]) if w not in ("dr", "doctor")]
            doctors = [
                d
                for d in self.doctors.values()
                if all(w in d["display_name"].casefold() for w in wanted)
                and (not clinics or d["clinic"] in {c["clinic_id"] for c in clinics})
            ]
            if not doctors:
                return fail("NO_RESULTS")
            if len(doctors) > 1 and not any(
                words(d["display_name"])[1:] == wanted for d in doctors
            ):
                return fail(
                    "AMBIGUOUS_MATCH",
                    choices=[
                        {"doctor_id": d["doctor_id"], "display_name": d["display_name"]}
                        for d in doctors
                    ],
                )
        if not clinics:
            return fail("NO_RESULTS")
        if len(clinics) > 1 and not any(
            c["name"].casefold() == args["clinic_name"].casefold() for c in clinics
        ):
            return fail(
                "AMBIGUOUS_MATCH",
                choices=[{"clinic_id": c["clinic_id"], "name": c["name"]} for c in clinics],
            )
        return ok({"clinics": clinics, "doctors": doctors})

    def search_appointment_availability(self, args, principal):
        found = [
            self.view(clinic, doctor, when) | {"slot_id": slot_id}
            for slot_id, clinic, doctor, when in self.slots
            if clinic == args["clinic_id"]
            and args["start_date"] <= when[:10] <= args["end_date"]
            and (not args.get("doctor_id") or doctor == args["doctor_id"])
            and (
                args.get("time_preference", "ANYTIME") == "ANYTIME"
                or (args["time_preference"] == "MORNING") == (int(when[11:13]) < 12)
            )
        ]
        return ok({"slots": found}) if found else fail("NO_AVAILABILITY")

    def get_my_appointments(self, args, principal):
        if principal is None:
            return fail("AUTH_REQUIRED")
        found = [
            self.view(clinic, doctor, when) | {"appointment_id": appointment_id}
            for appointment_id, clinic, doctor, when in self.appointments
            if "start_date" not in args or args["start_date"] <= when[:10] <= args["end_date"]
        ]
        return ok({"appointments": found})

    def prepare_appointment_action(self, args, principal):
        if principal is None:
            return fail("AUTH_REQUIRED")
        self.prepared.append(args)
        return ok({"pending_action_id": f"pending-{len(self.prepared)}"})


class Script:
    """Stands in for the model's reading: one prepared answer per message."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []
        self.usage = None

    def extract(self, request, *, context="", today=None, anchor=None, repair=False):
        self.seen.append((request, context, repair))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def salvage(self, *_args):
        return None


def patient(patient_id=None, session_id=None) -> Principal:
    return Principal(
        session_id=session_id or uuid.uuid4(),
        actor_type="PATIENT",
        login="synthetic",
        display_name="Synthetic Patient",
        patient_id=patient_id or uuid.uuid4(),
        staff_id=None,
        absolute_expires_at=NOW + timedelta(hours=8),
        idle_expires_at=NOW + timedelta(minutes=30),
        csrf_token="synthetic",
    )


def intent(**fields) -> PatientIntent:
    return PatientIntent(**{"operation": "UNKNOWN", **fields})


def talk(backend, *answers, principal=True, now=NOW):
    reader = Script(*answers)
    model = HybridControllerModel(reader, clock=lambda: now)
    session = TextSession(model, backend, patient() if principal else None)
    return session, model, reader


NOV16, NOV17, NOV18 = date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18)


def book(**fields):
    return intent(operation="BOOK", **fields)


def tools_called(backend):
    return [name for name, *_ in backend.calls]


# --- missing information -------------------------------------------------------------------------


def test_a_request_with_no_time_where_one_is_needed_asks_for_it_and_never_shows_none() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", NOV16, 9)
    session, _, _ = talk(
        backend,
        book(
            clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16, selection="SPECIFIC_TIME"
        ),
    )

    reply = session.respond("Book the slot at Pine Clinic on November 16.")

    assert "What time would you like?" in reply.text
    assert "None" not in reply.text and "null" not in reply.text
    assert backend.calls == []  # Nothing is searched until the intent is complete.


def test_a_time_with_no_opening_lists_what_is_open() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", NOV16, 9)
    backend.slot("s2", "c-pine", "d-rao", NOV16, 10, 30)
    session, _, _ = talk(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date=NOV16,
            end_date=NOV16,
            selection="SPECIFIC_TIME",
            time="11 AM",
        ),
    )

    reply = session.respond("Book 11 AM at Pine Clinic on November 16.")

    assert "I don't see an opening at 11:00 AM" in reply.text
    assert "9:00 AM" in reply.text and "10:30 AM" in reply.text
    assert backend.prepared == []


@pytest.mark.parametrize(
    ("answers", "question"),
    [
        ([intent()], "book, move, or cancel"),
        ([book()], "Which clinic"),
        ([book(clinic_name="Pine Clinic")], "Which date"),
    ],
)
def test_missing_information_is_asked_for_before_any_tool_is_called(answers, question) -> None:
    backend = Backend()
    session, _, _ = talk(backend, *answers)

    reply = session.respond("I'd like to book something.")

    assert question in reply.text and backend.calls == []


def test_a_clarification_is_answered_by_the_next_message() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", NOV17, 11)
    session, _model, _ = talk(
        backend,
        book(start_date=NOV17, end_date=NOV17),
        intent(clinic_name="Pine Clinic"),
    )

    first = session.respond("Book an appointment for November 17.")
    second = session.respond("Pine Clinic.")

    assert "Which clinic" in first.text
    assert "I prepared booking 11:00 AM" in second.text and second.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


# --- doctors and appointments are matched from trusted data --------------------------------------


def test_a_partial_doctor_name_finds_the_appointment_with_that_doctor() -> None:
    backend = Backend()
    backend.appointment("a1", "c-pine", "d-kim", NOV16, 9)
    backend.appointment("a2", "c-pine", "d-rao", NOV16, 14)
    session, _, _ = talk(
        backend,
        intent(operation="CANCEL", doctor_name="Dr. Kim", start_date=NOV16, end_date=NOV16),
    )

    reply = session.respond("I can't make it to see Dr. Kim on November 16, please cancel.")

    assert backend.prepared == [{"action": "CANCEL", "appointment_id": "a1"}]
    assert "Aiden Kim" in reply.text


def test_a_shortened_clinic_name_still_finds_the_appointment_but_never_a_different_one() -> None:
    backend = Backend()
    backend.appointment("a1", "c-pine", "d-kim", NOV16, 9)
    session, _, _ = talk(
        backend,
        intent(operation="CANCEL", clinic_name="Pine", start_date=NOV16, end_date=NOV16),
        intent(operation="CANCEL", clinic_name="Cedar", start_date=NOV16, end_date=NOV16),
    )

    session.respond("Cancel my appointment at Pine on November 16.")
    other = session.respond("Cancel my appointment at Cedar on November 16.")

    assert backend.prepared == [{"action": "CANCEL", "appointment_id": "a1"}]
    assert "couldn't find an active appointment" in other.text


def test_a_partial_doctor_name_that_fits_two_doctors_asks_and_does_not_guess() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-lee1", NOV16, 9)
    backend.slot("s2", "c-pine", "d-lee2", NOV16, 10)
    session, _model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", doctor_name="Lee", start_date=NOV16, end_date=NOV16),
        intent(doctor_name="Lee South"),
    )

    first = session.respond("Book with Dr. Lee at Pine Clinic on November 16.")
    second = session.respond("Dr. Lee South.")

    assert "Lee North" in first.text and "Lee South" in first.text
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s2"}]
    assert "Lee South" in second.text


def test_a_doctor_who_is_not_at_the_clinic_is_asked_about_not_swapped_for_another() -> None:
    backend = Backend()
    backend.slot("s1", "c-cedar", "d-cedar", NOV16, 9)
    session, _, _ = talk(
        backend,
        book(clinic_name="Linden Court Heart Center", doctor_name="Rao", start_date=NOV16),
    )

    reply = session.respond("Book Dr. Rao at Linden Court Heart Center on November 16.")

    assert "couldn't find a doctor called Rao" in reply.text and backend.prepared == []
    assert tools_called(backend) == ["search_clinic_directory", "search_clinic_directory"]


def test_a_clinic_name_that_fits_two_clinics_is_put_to_the_patient() -> None:
    backend = Backend()
    backend.slot("s1", "c-cedar2", "d-cedar", NOV16, 9)
    session, _, _ = talk(
        backend,
        book(clinic_name="Linden Court", start_date=NOV16, end_date=NOV16),
        intent(clinic_name="Linden Court Family Clinic"),
    )

    first = session.respond("Book at Linden Court on November 16.")
    session.respond("Linden Court Family Clinic")

    assert "Heart Center" in first.text and "Family Clinic" in first.text
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


def test_a_choice_by_position_means_what_was_offered_in_that_position() -> None:
    backend = Backend()
    backend.slot("s1", "c-cedar2", "d-cedar", NOV16, 9)
    session, _, _ = talk(
        backend,
        book(clinic_name="Linden Court", start_date=NOV16, end_date=NOV16),
        intent(option_number=2),
    )

    first = session.respond("Book at Linden Court on November 16.")
    session.respond("The second one.")

    assert first.text.index("Heart Center") < first.text.index("Family Clinic")
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]  # The Family Clinic's.


def test_a_position_that_was_never_offered_is_not_followed() -> None:
    backend = Backend()
    session, _, _ = talk(
        backend,
        book(clinic_name="Linden Court", start_date=NOV16, end_date=NOV16),
        intent(option_number=5),
    )

    session.respond("Book at Linden Court on November 16.")
    reply = session.respond("The fifth one.")

    assert backend.prepared == [] and "Which clinic do you mean" in reply.text


# --- a request that changes ----------------------------------------------------------------------


def two_day_backend():
    backend = Backend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    backend.slot("s17", "c-pine", "d-rao", NOV17, 10)
    backend.slot("s18", "c-pine", "d-rao", NOV18, 15)
    return backend


def test_a_changed_date_searches_again_and_prepares_a_new_action() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(start_date=NOV17, end_date=NOV17),
    )

    first = session.respond("Book at Pine Clinic on November 16.")
    assert model.state.prepared.pending_action_id == "pending-1"
    second = session.respond("Actually, make it November 17.")

    searches = [a for tool, a, _ in backend.calls if tool == "search_appointment_availability"]
    assert [a["start_date"] for a in searches] == ["2026-11-16", "2026-11-17"]  # Fresh each time.
    assert backend.prepared == [
        {"action": "BOOK", "slot_id": "s16"},
        {"action": "BOOK", "slot_id": "s17"},
    ]
    assert "11:00 AM" in first.text and "10:00 AM" in second.text
    assert model.state.prepared.pending_action_id == "pending-2"  # The earlier one is not current.


def test_the_second_day_is_a_move_only_when_the_message_says_where_to() -> None:
    backend = two_day_backend()
    backend.appointment("a1", "c-pine", "d-rao", NOV16, 9)
    session, model, _ = talk(
        backend,
        intent(operation="RESCHEDULE", clinic_name="Pine Clinic", start_date=date(2026, 11, 20)),
    )

    session.respond("Please move my appointment at Pine Clinic to November 18.")

    # Whatever the model put in the date fields, "to November 18" says where it moves to.
    assert model.state.draft.new_date == NOV18 and model.state.draft.start_date == NOV16
    assert backend.prepared == [
        {"action": "RESCHEDULE", "appointment_id": "a1", "new_slot_id": "s18"}
    ]


def test_after_a_reschedule_review_a_lone_new_date_is_where_to_not_which_appointment() -> None:
    backend = two_day_backend()
    backend.appointment("a1", "c-pine", "d-rao", NOV16, 9)
    session, model, _ = talk(
        backend,
        intent(operation="RESCHEDULE", clinic_name="Pine Clinic", start_date=NOV16, new_date=NOV17),
        intent(start_date=NOV18, end_date=NOV18),
    )

    session.respond("Move my November 16 appointment at Pine Clinic to November 17.")
    session.respond("Actually, November 18 works better.")

    assert [a["new_slot_id"] for a in backend.prepared] == ["s17", "s18"]
    assert model.state.draft.start_date == NOV16 and model.state.draft.new_date == NOV18


def test_a_changed_clinic_forgets_the_old_clinics_doctor() -> None:
    backend = Backend()
    backend.slot("sp", "c-pine", "d-rao", NOV16, 9)
    backend.slot("sc", "c-cedar", "d-cedar", NOV16, 9)
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", doctor_name="Rao", start_date=NOV16, end_date=NOV16),
        intent(clinic_name="Linden Court Heart Center"),
    )

    session.respond("Book Dr. Rao at Pine Clinic on November 16.")
    session.respond("Sorry, Linden Court Heart Center.")

    assert model.state.draft.doctor_name is None
    assert backend.prepared[-1] == {"action": "BOOK", "slot_id": "sc"}


def test_once_the_request_changes_the_earlier_review_is_no_longer_current() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(start_date=date(2026, 11, 25), end_date=date(2026, 11, 25)),
    )
    first = session.respond("Book at Pine Clinic on November 16.")
    assert first.awaiting_confirmation

    changed = session.respond("Actually, November 25.")  # Nothing is open that day.
    agreed = session.respond("yes")

    assert not changed.awaiting_confirmation and model.state.prepared is None
    assert "haven't prepared anything" in agreed.text and "can't confirm" not in agreed.text
    stale = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )
    assert "11:00 AM" not in stale.text  # No details of an action that is no longer current.


def test_a_message_that_changes_nothing_does_not_prepare_a_second_action() -> None:
    backend = two_day_backend()
    session, _, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
    )

    session.respond("Book at Pine Clinic on November 16.")
    reply = session.respond("Book at Pine Clinic on November 16.")

    assert len(backend.prepared) == 1 and "already prepared" in reply.text


def test_a_new_operation_starts_a_new_request_and_does_not_inherit_the_old_one() -> None:
    backend = two_day_backend()
    backend.appointment("a1", "c-pine", "d-rao", NOV16, 9)
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV17, end_date=NOV17),
        intent(operation="CANCEL"),
    )

    session.respond("Book at Pine Clinic on November 17.")
    session.respond("Actually, cancel my appointment instead.")

    assert model.state.draft.operation == "CANCEL"
    assert model.state.draft.clinic_name is None and model.state.draft.start_date is None
    assert backend.prepared[-1] == {"action": "CANCEL", "appointment_id": "a1"}


def test_an_operation_the_message_has_no_words_for_is_not_a_change_of_operation() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        # The model reads "Sorry, I meant the 17th" as a move; nothing in the words says so.
        intent(operation="RESCHEDULE", new_date=NOV17),
    )

    session.respond("Book at Pine Clinic on November 16.")
    session.respond("Sorry, I meant the 17th.")

    assert model.state.draft.operation == "BOOK" and model.state.draft.start_date == NOV17
    assert "operation_without_words" in model.recoveries


# --- agreement, dropping a request, unrelated questions ------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["Yes", "yes please", "OK", "Sounds good", "Go ahead", "Book it", "I confirm", "Confirm"],
)
def test_saying_yes_never_confirms_anything(phrase) -> None:
    backend = two_day_backend()
    session, model, reader = talk(
        backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )
    session.respond("Book at Pine Clinic on November 16.")
    calls_before = len(backend.calls)

    reply = session.respond(phrase)

    assert "I can't confirm for you" in reply.text and "Confirm in the app" in reply.text
    assert len(backend.calls) == calls_before  # No tool at all, and in particular no commit.
    assert len(reader.seen) == 1  # Not even read by the model.
    assert model.state.prepared is not None  # Still waiting for the app.


def test_nothing_in_the_tool_runtime_can_confirm() -> None:
    from app.agent_tools import build_default_registry

    assert not [n for n in build_default_registry().names() if "confirm" in n.lower()]


def test_dropping_a_request_forgets_it_and_says_not_to_confirm_the_review() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )
    session.respond("Book at Pine Clinic on November 16.")

    reply = session.respond("Never mind, forget it.")

    assert model.state.draft is None and model.state.prepared is None
    assert "nothing was changed" in reply.text and "don't confirm" in reply.text
    assert not reply.awaiting_confirmation


def test_dropping_works_when_the_model_says_so_and_the_words_are_unusual() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(message_type="ABANDON"),
    )
    session.respond("Book at Pine Clinic on November 16.")

    session.respond("On second thought, I'll sort it out some other way.")

    assert model.state.draft is None and model.state.prepared is None


def test_an_unrelated_question_ends_the_old_request() -> None:
    backend = two_day_backend()
    session, _model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic"),
        intent(message_type="OFF_TOPIC"),
        intent(time="10 AM", selection="SPECIFIC_TIME"),
    )
    session.respond("Book at Pine Clinic.")

    off = session.respond("What time does the pharmacy close?")
    later = session.respond("10 AM please.")

    assert "I can only help with booking" in off.text and "set aside" in off.text
    assert "book, move, or cancel" in later.text  # It did not carry on with the old booking.
    assert backend.prepared == []


def test_a_message_with_content_is_a_request_even_if_the_model_calls_it_off_topic() -> None:
    backend = two_day_backend()
    session, _, _ = talk(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date=NOV16,
            end_date=NOV16,
            message_type="OFF_TOPIC",
        ),
    )

    session.respond("Do you validate parking? Also book at Pine Clinic on November 16.")

    assert len(backend.prepared) == 1


# --- credentials and signing in ------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "My PIN is 4821",
        "here is my password: hunter2",
        "my login code is abc",
        "access token xyz",
        "482913",
        "the one-time code is 123456",
    ],
)
def test_a_credential_in_a_message_is_never_read_stored_or_repeated(message) -> None:
    backend = Backend()
    session, model, reader = talk(backend)

    reply = session.respond(message)

    assert reply.text == CREDENTIAL_NOTICE
    assert reader.seen == [] and backend.calls == []
    assert not any(ch.isdigit() for ch in reply.text)
    assert message not in reply.text and model.state.draft is None


def test_not_signed_in_sends_the_patient_to_the_sign_in_screen_and_asks_for_no_secret() -> None:
    backend = Backend()
    session, _, _ = talk(
        backend,
        intent(operation="CANCEL", clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        principal=False,
    )

    reply = session.respond("Cancel my appointment at Pine Clinic on November 16.")

    assert reply.text == SIGN_IN_NOTICE
    assert "secure sign-in screen" in reply.text and "can't take a PIN" in reply.text
    assert backend.prepared == [] and backend.calls[0][2] is None  # Called as the public.


@pytest.mark.parametrize("code", ["AUTH_REQUIRED", "SESSION_EXPIRED", "FORBIDDEN"])
def test_an_authorization_failure_during_preparation_is_a_sign_in_prompt(code) -> None:
    backend = two_day_backend()
    backend.fail["prepare_appointment_action"] = fail(code)
    session, model, _ = talk(
        backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )

    reply = session.respond("Book at Pine Clinic on November 16.")

    assert reply.text == SIGN_IN_NOTICE and model.state.prepared is None


# --- when the model's reading fails --------------------------------------------------------------


def test_a_model_that_cannot_be_reached_still_gets_a_useful_reply() -> None:
    backend = Backend()
    session, model, _ = talk(backend, TimeoutError("down"))

    reply = session.respond("hello?")

    assert reply.text.strip() and "didn't quite catch" in reply.text
    assert model.recoveries == ["rule_based_reading"]


def test_an_answer_to_a_clinic_question_is_used_as_the_clinic_when_the_model_is_down() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", NOV16, 9)
    session, model, _ = talk(
        backend,
        book(start_date=NOV16, end_date=NOV16),
        TimeoutError("down"),
    )

    session.respond("Book an appointment on November 16.")
    reply = session.respond("Pine Clinic")

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]  # The directory checked it.
    assert "I prepared booking" in reply.text and "rule_based_reading" in model.recoveries


def test_a_name_that_is_not_a_clinic_found_by_rules_is_checked_and_refused_by_the_directory() -> (
    None
):
    backend = Backend()
    session, _, _ = talk(
        backend,
        book(start_date=NOV16, end_date=NOV16),
        TimeoutError("down"),
    )

    session.respond("Book an appointment on November 16.")
    reply = session.respond("the blue one near the lake")

    assert "couldn't find a clinic called the blue one near the lake" in reply.text
    assert backend.prepared == []


def test_without_the_model_a_cancellation_is_never_guessed_even_with_one_match() -> None:
    backend = Backend()
    backend.appointment("a1", "c-pine", "d-rao", NOV16, 9)
    session, _model, _ = talk(backend, TimeoutError("down"), TimeoutError("down"))

    first = session.respond("Please cancel my appointment on November 16.")
    assert backend.prepared == [] and "won't guess" in first.text
    second = session.respond("The first one.")

    assert backend.prepared == [{"action": "CANCEL", "appointment_id": "a1"}]
    assert "cancelling your appointment" in second.text


def test_without_the_model_dates_and_the_kind_of_request_are_still_understood() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", NOV17, 9)
    session, model, _ = talk(backend, TimeoutError("down"), TimeoutError("down"))

    session.respond("I want to book an appointment for November 17")
    assert model.state.draft.operation == "BOOK" and model.state.draft.start_date == NOV17
    assert model.state.awaiting == Awaiting.CLINIC


def test_an_invalid_first_reading_is_repaired_by_one_more_reading() -> None:
    backend = two_day_backend()
    session, model, reader = talk(
        backend,
        MalformedModelResponse("bad"),
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
    )

    reply = session.respond("Book at Pine Clinic on November 16.")

    assert [repair for _, _, repair in reader.seen] == [False, True]
    assert model.recoveries == ["repaired_reply"] and "I prepared booking" in reply.text


def test_two_invalid_readings_fall_back_to_rules_and_never_a_third_call() -> None:
    backend = Backend()
    session, model, reader = talk(
        backend, MalformedModelResponse("bad"), MalformedModelResponse("bad again")
    )

    reply = session.respond("blah")

    assert len(reader.seen) == 2 and model.recoveries == ["rule_based_reading"]
    assert reply.text.strip()


def test_a_salvaged_reading_is_used_without_another_call() -> None:
    backend = Backend()

    class Salvaging(Script):
        def salvage(self, *_args):
            return book(clinic_name="Pine Clinic")

    reader = Salvaging(MalformedModelResponse("unsupported"))
    session = TextSession(HybridControllerModel(reader, clock=lambda: NOW), backend, patient())

    reply = session.respond("Book at Pine Clinic")

    assert len(reader.seen) == 1 and "Which date" in reply.text


# --- tool failures are said plainly and nothing is invented --------------------------------------


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("DEPENDENCY_UNAVAILABLE", "trouble reaching the scheduling system"),
        ("RATE_LIMITED", "trouble reaching the scheduling system"),
        ("INTERNAL_ERROR", "could not complete"),
        ("SELF_SERVICE_CUTOFF", "contact the clinic"),
        ("RESOURCE_NOT_AVAILABLE", "any more"),
    ],
)
def test_a_failed_preparation_never_sounds_like_a_success(code, fragment) -> None:
    backend = two_day_backend()
    backend.fail["prepare_appointment_action"] = fail(code)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16))

    reply = session.respond("Book at Pine Clinic on November 16.")

    assert fragment in reply.text
    assert "prepared" not in reply.text and not reply.awaiting_confirmation


def test_a_slot_taken_between_search_and_prepare_is_searched_again_once() -> None:
    backend = two_day_backend()
    backend.slot("s16b", "c-pine", "d-rao", NOV16, 12)
    taken = {"once": True}
    original = backend.prepare_appointment_action

    def flaky(args, principal):
        if taken.pop("once", False):
            return fail("SLOT_UNAVAILABLE")
        return original(args, principal)

    backend.prepare_appointment_action = flaky
    session, _, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16, selection="EARLIEST"),
    )

    reply = session.respond("Book the first one at Pine Clinic on November 16.")

    assert "I prepared booking" in reply.text
    assert tools_called(backend).count("search_appointment_availability") == 2


def test_a_slot_that_stays_unavailable_is_reported_after_one_retry() -> None:
    backend = two_day_backend()
    backend.fail["prepare_appointment_action"] = fail("SLOT_UNAVAILABLE")
    session, _, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16, selection="EARLIEST"),
    )

    reply = session.respond("Book the first one at Pine Clinic on November 16.")

    assert (
        "just taken" in reply.text
        and tools_called(backend).count("prepare_appointment_action") == 2
    )


def test_no_availability_offers_another_day_and_the_next_date_is_used() -> None:
    backend = two_day_backend()
    session, _model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=date(2026, 11, 20), end_date=date(2026, 11, 20)),
        intent(start_date=NOV17, end_date=NOV17),
    )

    first = session.respond("Book at Pine Clinic on November 20.")
    second = session.respond("How about November 17?")

    assert "Would another day work?" in first.text and "November 20, 2026" in first.text
    assert "I prepared booking 10:00 AM" in second.text


# --- what is said about what happened ------------------------------------------------------------


def test_only_a_reported_commit_of_the_current_action_is_described_as_done() -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )
    session.respond("Book at Pine Clinic on November 16.")

    done = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert "is booked" in done.text and "11:00 AM" in done.text
    assert model.state.draft is None and model.state.prepared is None


def test_a_commit_of_an_out_of_date_action_is_never_described_with_the_current_details() -> None:
    backend = two_day_backend()
    session, _, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(start_date=NOV17, end_date=NOV17),
    )
    session.respond("Book at Pine Clinic on November 16.")
    session.respond("Actually, make it November 17.")

    reply = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert "10:00 AM" not in reply.text and "11:00 AM" not in reply.text
    assert "as confirmed in the app" in reply.text


@pytest.mark.parametrize("kind", ["CONFIRMATION_REJECTED", "CONFIRMATION_WITHHELD"])
def test_a_confirmation_that_did_not_happen_is_never_described_as_done(kind) -> None:
    backend = two_day_backend()
    session, model, _ = talk(
        backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )
    session.respond("Book at Pine Clinic on November 16.")

    reply = session.confirmation_event({"type": kind})

    assert "nothing was changed" in reply.text and "booked" not in reply.text
    assert model.state.prepared is None


def test_no_reply_is_empty_or_shows_an_internal_placeholder() -> None:
    from app.evaluation.hybrid import GENERIC_NOTICE

    for bad in ("", "   ", "Opening at None", "value is null", "{'a': 1}", "NaN"):
        assert HybridControllerModel._say(bad).text == GENERIC_NOTICE
    assert HybridControllerModel._say("Fine.").text == "Fine."


def test_a_long_conversation_starts_over_rather_than_growing_without_end() -> None:
    backend = Backend()
    session, model, _ = talk(backend, *[intent() for _ in range(MAX_TURNS_PER_SESSION + 2)])

    for _ in range(MAX_TURNS_PER_SESSION + 1):
        session.respond("hello")

    assert model.state.turns == 1


# --- sessions are kept apart ---------------------------------------------------------------------


def factory():
    return HybridControllerModel(Script(*[intent() for _ in range(20)]), clock=lambda: NOW)


def test_each_session_has_its_own_conversation() -> None:
    registry = ConversationRegistry(factory)
    first, second = patient(), patient()

    one = registry.session_for(Backend(), first)
    two = registry.session_for(Backend(), second)

    assert one is not two and one.model is not two.model
    assert registry.session_for(Backend(), first) is one  # The same session continues.


def test_one_sessions_state_is_not_visible_to_another() -> None:
    registry = ConversationRegistry(
        lambda: HybridControllerModel(Script(book(clinic_name="Pine Clinic")), clock=lambda: NOW)
    )
    one = registry.session_for(Backend(), patient())
    two = registry.session_for(Backend(), patient())

    one.respond("Book at Pine Clinic.")

    assert one.model.state.draft is not None and two.model.state.draft is None


def test_a_session_id_that_now_belongs_to_another_patient_gets_a_new_conversation() -> None:
    registry = ConversationRegistry(factory)
    shared = uuid.uuid4()
    first = registry.session_for(Backend(), patient(session_id=shared))
    first.model.state.draft = intent(operation="BOOK", clinic_name="Pine Clinic")

    second = registry.session_for(Backend(), patient(session_id=shared))

    assert second is not first and second.model.state.draft is None


def test_signing_in_does_not_carry_an_anonymous_conversation_into_the_session() -> None:
    registry = ConversationRegistry(factory)
    anonymous = registry.session_for(Backend(), None, anonymous_id="visitor-1")
    anonymous.model.state.draft = intent(operation="CANCEL")

    signed_in = registry.session_for(Backend(), patient())

    assert signed_in is not anonymous and signed_in.model.state.draft is None


def test_an_anonymous_conversation_needs_an_id() -> None:
    with pytest.raises(ValueError):
        ConversationRegistry(factory).session_for(Backend(), None)


def test_idle_conversations_expire_and_the_total_is_bounded() -> None:
    clock = {"now": 0.0}
    registry = ConversationRegistry(
        factory, max_sessions=3, idle_seconds=60, timer=lambda: clock["now"]
    )
    old = patient()
    kept = registry.session_for(Backend(), old)
    clock["now"] = 61.0
    assert registry.session_for(Backend(), old) is not kept  # Expired: a new conversation.

    for _ in range(5):
        registry.session_for(Backend(), patient())
    assert len(registry) == 3


def test_ending_a_session_forgets_its_conversation() -> None:
    registry = ConversationRegistry(factory)
    principal = patient()
    first = registry.session_for(Backend(), principal)

    registry.end(principal)

    assert registry.session_for(Backend(), principal) is not first


def test_the_session_uses_its_own_principal_for_every_call_and_the_model_never_supplies_one() -> (
    None
):
    backend = two_day_backend()
    principal = patient()
    reader = Script(book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16))
    session = TextSession(HybridControllerModel(reader, clock=lambda: NOW), backend, principal)

    session.respond("Book at Pine Clinic on November 16.")

    assert {call[2] for call in backend.calls} == {principal}
    assert all(
        "patient_id" not in args and "session_id" not in args for _, args, _ in backend.calls
    )


def test_a_fault_in_the_model_still_leaves_the_patient_a_reply() -> None:
    class Broken:
        name = "broken"

        def decide(self, transcript):
            raise RuntimeError("boom")

    session = TextSession(Broken(), Backend(), patient())

    reply = session.respond("hello")

    assert reply.text.strip() and not reply.awaiting_confirmation
    assert "boom" not in reply.text


# --- the conversation work is kept apart from the frozen benchmark ----------


def test_no_conversation_fixture_or_test_shares_a_name_with_the_frozen_benchmark() -> None:
    """The conversation scenarios and their tests were written without the benchmark in view: no
    clinic or doctor name in them is one of the benchmark's, whole or by a distinctive word."""
    import re
    from pathlib import Path

    from app.evaluation import benchmark_spec as spec

    names = set()
    for scenario in spec.load_manifest().scenarios:
        for clinic in scenario.fixture.clinics:
            names.add(clinic.name)
            names |= {doctor.name for doctor in clinic.doctors}
    generic = {
        *("clinic", "health", "medical", "center", "family", "care", "medicine", "associates"),
        *("partners", "group", "practice", "walk-in", "community", "primary", "wellness"),
        *("internal", "specialty", "women's", "and", "creek", "falls", "valley", "ridge"),
        *("heart", "hollow", "west", "pine", "cedar", "maple", "willow", "prairie", "juniper"),
        *("granite", "summit", "aspen", "birch", "meadow", "harbor", "copper"),  # Place words.
    }
    distinctive = {
        word.strip(".,'").casefold()
        for name in names
        for word in re.split(r"[\s-]", name)
        if len(word) > 3 and word.strip(".,'").casefold() not in generic
    }
    root = Path(__file__).parent
    files = [
        root.parent / "app/evaluation" / name
        for name in ("conversation.py", "conversation_scenarios.py", "conversation_eval.py")
    ] + [
        root / "test_conversation_unit.py",
        root / "database/test_conversation.py",
        root / "database/conversation_support.py",
    ]
    for path in files:
        text = path.read_text()
        assert not [name for name in names if name in text], path.name
        words = {w.strip(".,\"'():;").casefold() for w in re.split(r"\s+", text)}
        assert not (words & distinctive), (path.name, sorted(words & distinctive))


# --- a dropped review is retired in the backend --------------------------------


class RetiringBackend(Backend):
    """The fake backend, plus the trusted discard the session calls when a review is dropped."""

    def __init__(self):
        super().__init__()
        self.discarded = []
        self.discard_fails = False

    def discard_pending_action(self, principal, pending_action_id):
        if self.discard_fails:
            raise RuntimeError("database down")
        self.discarded.append((principal, pending_action_id))
        return True


def prepared_session(*more_answers):
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    backend.slot("s17", "c-pine", "d-rao", NOV17, 10)
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        *more_answers,
    )
    assert session.respond("Book at Pine Clinic on November 16.").awaiting_confirmation
    return session, model, backend


def test_dropping_a_request_retires_the_review_it_prepared() -> None:
    session, _, backend = prepared_session()

    session.respond("Never mind, forget it.")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]
    assert backend.discarded[0][0] is session.principal


def test_an_unrelated_question_retires_the_review_too() -> None:
    session, _, backend = prepared_session(intent(message_type="OFF_TOPIC"))

    session.respond("What time does the pharmacy close?")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]


def test_a_change_that_ends_without_a_new_review_retires_the_old_one() -> None:
    session, _, backend = prepared_session(
        intent(start_date=date(2026, 11, 25), end_date=date(2026, 11, 25))
    )

    reply = session.respond("Actually, November 25.")  # Nothing is open that day.

    assert not reply.awaiting_confirmation
    assert [pending for _, pending in backend.discarded] == ["pending-1"]


def test_a_change_that_makes_a_new_review_leaves_the_retiring_to_the_backend() -> None:
    session, _, backend = prepared_session(intent(start_date=NOV17, end_date=NOV17))

    reply = session.respond("Actually, make it November 17.")

    assert reply.awaiting_confirmation and backend.discarded == []
    assert session.prepared_action_id == "pending-2"


def test_a_review_that_is_still_current_is_not_retired() -> None:
    session, _, backend = prepared_session()

    session.respond("yes")  # Agreement is not a change.
    session.respond("sounds good")

    assert backend.discarded == [] and session.prepared_action_id == "pending-1"


def test_a_finished_review_is_not_retired_after_the_app_reports_it() -> None:
    session, _, backend = prepared_session()

    session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert backend.discarded == [] and session.prepared_action_id is None


def test_a_failed_retirement_is_told_to_the_patient_and_tried_again_on_the_next_turn() -> None:
    session, _, backend = prepared_session()
    backend.discard_fails = True

    reply = session.respond("Never mind, forget it.")

    # Not reported as retired: the patient is told the earlier review may still be open.
    assert "nothing was changed" in reply.text and not reply.awaiting_confirmation
    assert reply.text.endswith(REVIEW_STILL_OPEN_NOTICE)
    assert session.prepared_action_id == "pending-1"  # Still tracked, so it is tried again.

    backend.discard_fails = False
    session.respond("hello")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]
    assert session.prepared_action_id is None


@pytest.mark.parametrize("error", [AuthenticationRequiredError(), SessionExpiredError()])
def test_a_session_that_has_ended_has_no_review_left_to_warn_about(error) -> None:
    session, _, backend = prepared_session()
    backend.discard = lambda *_: (_ for _ in ()).throw(error)
    backend.discard_pending_action = lambda *_: (_ for _ in ()).throw(error)

    reply = session.respond("Never mind, forget it.")

    assert REVIEW_STILL_OPEN_NOTICE not in reply.text and session.prepared_action_id is None


def test_a_failed_retirement_logs_only_the_kind_of_error(caplog) -> None:
    session, _, backend = prepared_session()
    backend.discard_fails = True

    with caplog.at_level("WARNING"):
        session.respond("Never mind, forget it.")

    records = [r for r in caplog.records if "retire" in r.getMessage()]
    assert records and records[0].error_type == "RuntimeError"
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "database down" not in logged and "pending-1" not in logged


def test_a_fault_during_the_turn_still_retires_the_review_it_left_behind() -> None:
    class Broken:
        name = "broken"
        state = SimpleNamespace(prepared=None)  # The conversation no longer holds a review.

        def decide(self, transcript):
            raise RuntimeError("boom")

    backend = RetiringBackend()
    session = TextSession(Broken(), backend, patient())
    session.prepared_action_id = "pending-9"

    reply = session.respond("hello")

    assert reply.text == FALLBACK_REPLY
    assert [pending for _, pending in backend.discarded] == ["pending-9"]


def test_a_session_without_the_trusted_discard_still_works() -> None:
    backend = two_day_backend()  # The plain fake has no discard method.
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16))
    session.respond("Book at Pine Clinic on November 16.")

    reply = session.respond("Never mind, forget it.")

    assert "nothing was changed" in reply.text


# --- spoken dates and times, on the clinic's own calendar -----------------------------------------


def availability_calls(backend):
    return [args for tool, args, _ in backend.calls if tool == "search_appointment_availability"]


def pine_day(*hours, day=date(2026, 11, 17), **kwargs):
    backend = Backend()
    for number, hour in enumerate(hours):
        whole, _, half = str(hour).partition(".")
        backend.slot(f"s{number}", "c-pine", "d-rao", day, int(whole), 30 if half else 0, **kwargs)
    return backend


def test_a_spoken_date_is_understood_even_when_the_model_missed_it() -> None:
    backend = pine_day(9)
    session, _model, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond("Book me at Pine Clinic on November seventeenth.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-17"
    assert "I prepared booking 9:00 AM on Tuesday, November 17, 2026" in reply.text


def test_the_twenty_fourth_of_november_is_the_twenty_fourth() -> None:
    backend = pine_day(9, day=date(2026, 11, 24))
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond("Book Pine Clinic, the twenty-fourth of November, please.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-24"
    assert reply.awaiting_confirmation


def test_eleven_thirty_in_the_morning_picks_that_time() -> None:
    backend = pine_day(9, 11.5, 14)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond(
        "Book Pine Clinic on November seventeenth at eleven thirty in the morning."
    )

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]
    assert "11:30 AM" in reply.text


@pytest.mark.parametrize(
    ("phrase", "slot", "time"),
    [
        ("half past three in the afternoon", "s1", "3:30 PM"),
        ("quarter to nine in the morning", "s0", "8:45 AM"),
        ("ten o'clock in the morning", "s2", "10:00 AM"),
        ("two forty-five in the afternoon", "s3", "2:45 PM"),
    ],
)
def test_clock_phrases_that_say_which_half_of_the_day_are_taken_as_said(phrase, slot, time) -> None:
    backend = pine_day(8.75 - 0.25 if False else 8, 15.5, 10, 14.75)
    backend.slots[0] = ("s0", "c-pine", "d-rao", "2026-11-17T08:45:00-07:00")
    backend.slots[3] = ("s3", "c-pine", "d-rao", "2026-11-17T14:45:00-07:00")
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond(f"Book Pine Clinic on November seventeenth at {phrase}.")

    assert backend.prepared == [{"action": "BOOK", "slot_id": slot}]
    assert time in reply.text


@pytest.mark.parametrize("answer", ["PM", "in the afternoon", "2:15 PM", "p.m. please"])
def test_quarter_past_two_is_asked_about_and_the_answer_settles_it(answer) -> None:
    backend = pine_day(2.25, 14.25)
    backend.slots[0] = ("s0", "c-pine", "d-rao", "2026-11-17T02:15:00-07:00")
    backend.slots[1] = ("s1", "c-pine", "d-rao", "2026-11-17T14:15:00-07:00")
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), intent())

    asked = session.respond("Book Pine Clinic on November seventeenth at quarter past two.")

    assert "Do you mean 2:15 AM or 2:15 PM?" in asked.text
    assert backend.prepared == [] and availability_calls(backend) == []  # Nothing was guessed.
    done = session.respond(answer)
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]
    assert "2:15 PM" in done.text


def test_the_ambiguous_time_can_be_answered_by_position() -> None:
    backend = pine_day(9)
    backend.slots[0] = ("s0", "c-pine", "d-rao", "2026-11-17T02:15:00-07:00")
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), intent(option_number=1))

    session.respond("Book Pine Clinic on November seventeenth at quarter past two.")
    session.respond("The first one.")

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s0"}]  # 2:15 AM, as chosen.


def test_a_bare_hour_with_no_am_or_pm_is_never_guessed() -> None:
    backend = pine_day(10, 22)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), intent())

    asked = session.respond("Book Pine Clinic on November seventeenth at ten.")

    assert "Do you mean 10:00 AM or 10:00 PM?" in asked.text
    assert backend.prepared == [] and availability_calls(backend) == []
    session.respond("PM")
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


def test_tomorrow_afternoon_means_the_clinics_tomorrow_not_utc_tomorrow() -> None:
    kiwi = Backend()
    kiwi.slot("k1", "c-kiwi", "d-kiwi", date(2026, 11, 4), 14, offset="+13:00")
    kiwi.slot("k0", "c-kiwi", "d-kiwi", date(2026, 11, 3), 14, offset="+13:00")
    # 15:00 UTC on Nov 2 is already 04:00 on Nov 3 in Auckland, so the clinic's tomorrow is Nov 4.
    session, _, _ = talk(kiwi, book(clinic_name="Kauri Coast Clinic"))

    reply = session.respond("Book Kauri Coast Clinic tomorrow afternoon.")

    assert tools_called(kiwi)[0] == "search_clinic_directory"
    assert availability_calls(kiwi)[0]["start_date"] == "2026-11-04"
    assert availability_calls(kiwi)[0]["time_preference"] == "AFTERNOON"
    assert kiwi.prepared == [{"action": "BOOK", "slot_id": "k1"}]
    assert "Wednesday, November 4, 2026" in reply.text


def test_tomorrow_at_a_clinic_behind_utc_is_still_that_clinics_tomorrow() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", date(2026, 11, 2), 21)
    late = datetime(2026, 11, 2, 3, 0, tzinfo=UTC)  # 20:00 on Nov 1 in Denver; Nov 2 in UTC.
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), now=late)

    session.respond("Book Pine Clinic tomorrow.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-02"  # Not Nov 3.


def test_a_date_with_no_year_is_its_next_occurrence_on_the_clinics_calendar() -> None:
    backend = Backend()
    backend.slot("s1", "c-pine", "d-rao", date(2026, 11, 1), 21)
    late = datetime(2026, 11, 2, 3, 0, tzinfo=UTC)  # Still November 1 in Denver.
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), now=late)

    session.respond("Book Pine Clinic on November first.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-01"  # Not next year's.


def test_next_tuesday_said_on_a_monday_is_asked_about_and_the_answer_is_used() -> None:
    backend = pine_day(9, day=date(2026, 11, 10))
    session, _model, _ = talk(backend, book(clinic_name="Pine Clinic"), intent())

    asked = session.respond("Book Pine Clinic next Tuesday morning.")

    assert "Tuesday, November 3, 2026" in asked.text and "Tuesday, November 10, 2026" in asked.text
    assert backend.calls == []  # Nothing is searched on a guess.
    session.respond("the tenth")
    assert availability_calls(backend)[0]["start_date"] == "2026-11-10"
    assert (
        availability_calls(backend)[0]["time_preference"] == "MORNING"
    )  # Kept across the question.


def test_the_ambiguous_day_can_be_answered_by_position() -> None:
    backend = pine_day(9, day=date(2026, 11, 3))
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), intent(option_number=1))

    session.respond("Book Pine Clinic next Tuesday.")
    session.respond("The first one.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-03"


def test_next_tuesday_said_on_a_wednesday_is_not_ambiguous() -> None:
    backend = pine_day(9, day=date(2026, 11, 10))
    wednesday = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), now=wednesday)

    reply = session.respond("Book Pine Clinic next Tuesday.")

    assert (
        availability_calls(backend)[0]["start_date"] == "2026-11-10" and reply.awaiting_confirmation
    )


def test_the_first_available_appointment_after_ten_asks_which_ten_then_picks_the_earliest() -> None:
    backend = pine_day(9, 10.5, 14)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic", selection="EARLIEST"), intent())

    asked = session.respond(
        "Book Pine Clinic on November seventeenth, the first available appointment after ten."
    )

    assert "Do you mean after 10:00 AM or after 10:00 PM?" in asked.text
    reply = session.respond("AM")
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]  # 10:30, not 9:00.
    assert "10:30 AM" in reply.text


def test_after_ten_in_the_morning_needs_no_question() -> None:
    backend = pine_day(9, 10.5, 14)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    session.respond(
        "Book Pine Clinic on November seventeenth, the first one after ten in the morning."
    )

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


def test_nothing_after_the_time_lists_what_is_open_and_prepares_nothing() -> None:
    backend = pine_day(9, 9.5)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond("Book Pine Clinic on November seventeenth after ten in the morning.")

    assert "I don't see an opening after 10:00 AM" in reply.text
    assert "9:00 AM" in reply.text and backend.prepared == []


def test_before_a_time_limits_the_search_from_above() -> None:
    backend = pine_day(9, 13)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    session.respond("Book Pine Clinic on November seventeenth, anything before noon.")

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s0"}]


def test_a_spoken_correction_changes_the_date_and_searches_again() -> None:
    backend = Backend()
    backend.slot("a", "c-pine", "d-rao", date(2026, 11, 17), 11)
    backend.slot("b", "c-pine", "d-rao", date(2026, 11, 24), 10)
    backend.slot("c", "c-pine", "d-rao", date(2026, 11, 3), 15)
    session, model, _ = talk(backend, book(clinic_name="Pine Clinic"), intent(), intent())

    session.respond("Book Pine Clinic, November seventeenth.")
    session.respond("Actually, the twenty-fourth.")
    third = session.respond("No, make it tomorrow.")

    assert [c["start_date"] for c in availability_calls(backend)] == [
        "2026-11-17",
        "2026-11-24",
        "2026-11-03",
    ]
    assert [p["slot_id"] for p in backend.prepared] == ["a", "b", "c"]
    assert model.state.draft.start_spec == "rel:tomorrow" and "3:00 PM" in third.text


@pytest.mark.parametrize(
    "phrase", ["February thirtieth", "November thirty-first", "the thirty-second of November"]
)
def test_a_date_that_does_not_exist_is_asked_about_again(phrase) -> None:
    backend = pine_day(9)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond(f"Book Pine Clinic on {phrase}.")

    assert "Which date would you like?" in reply.text and backend.calls == []


def test_a_date_already_past_on_the_clinics_calendar_is_not_searched() -> None:
    backend = pine_day(9)
    session, model, _ = talk(
        backend,
        book(clinic_name="Pine Clinic", start_date=date(2026, 10, 20), end_date=date(2026, 10, 20)),
    )

    reply = session.respond("Book Pine Clinic on 2026-10-20.")

    assert "has already passed" in reply.text and availability_calls(backend) == []
    assert model.state.draft.start_date is None  # So the answer replaces it.


def test_a_search_longer_than_the_backend_allows_is_narrowed_not_attempted() -> None:
    backend = pine_day(9)
    session, _, _ = talk(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date=date(2026, 11, 17),
            end_date=date(2026, 12, 31),
        ),
    )

    reply = session.respond("Book Pine Clinic from November seventeenth to December thirty-first.")

    assert "up to 31 days" in reply.text and availability_calls(backend) == []


def test_a_clinic_name_with_a_number_word_is_left_alone() -> None:
    backend = Backend()
    backend.slot("s1", "c-rivers", "d-rivers", date(2026, 11, 17), 9)
    session, _, reader = talk(backend, book(clinic_name="Two Rivers Clinic"))

    reply = session.respond("Book at Two Rivers Clinic on November seventeenth.")

    assert reader.seen[0][0] == "Book at Two Rivers Clinic on November seventeenth."
    assert backend.calls[0][1]["clinic_name"] == "Two Rivers Clinic"
    assert reply.awaiting_confirmation


@pytest.mark.parametrize("phrase", ["four eight two one", "one two three four five six", "0 0 0 0"])
def test_digits_said_aloud_are_treated_as_a_credential_and_never_sent_to_the_model(phrase) -> None:
    backend = Backend()
    session, _, reader = talk(backend)

    reply = session.respond(phrase)

    assert reply.text == CREDENTIAL_NOTICE and reader.seen == [] and backend.calls == []


def test_a_time_said_in_words_is_not_mistaken_for_a_credential() -> None:
    backend = pine_day(9)
    session, _, reader = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond("Book Pine Clinic on November seventeenth, ten thirty in the morning.")

    assert reply.text != CREDENTIAL_NOTICE and len(reader.seen) == 1


def test_cancelling_tomorrow_finds_the_appointment_on_that_clinics_tomorrow() -> None:
    backend = Backend()
    backend.appointment("a-kiwi", "c-kiwi", "d-kiwi", date(2026, 11, 4), 10, offset="+13:00")
    backend.appointment("a-far", "c-pine", "d-rao", date(2026, 11, 20), 10)
    session, _, _ = talk(backend, intent(operation="CANCEL"))

    reply = session.respond("Please cancel my appointment tomorrow.")

    assert backend.prepared == [{"action": "CANCEL", "appointment_id": "a-kiwi"}]
    assert "Wednesday, November 4, 2026" in reply.text


def test_rescheduling_to_tomorrow_uses_the_appointments_clinic_calendar() -> None:
    backend = Backend()
    backend.appointment("a-kiwi", "c-kiwi", "d-kiwi", date(2026, 11, 10), 10, offset="+13:00")
    backend.slot("k-new", "c-kiwi", "d-kiwi", date(2026, 11, 4), 9, offset="+13:00")
    session, _, _ = talk(
        backend,
        intent(operation="RESCHEDULE", clinic_name="Kauri Coast", start_date=date(2026, 11, 10)),
    )

    session.respond("Move my November tenth appointment at Kauri Coast to tomorrow.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-04"
    assert backend.prepared == [
        {"action": "RESCHEDULE", "appointment_id": "a-kiwi", "new_slot_id": "k-new"}
    ]


def test_when_the_clocks_go_back_the_same_time_twice_is_told_apart() -> None:
    backend = Backend()
    first, second = date(2026, 11, 1), date(2026, 11, 1)
    backend.slot("early", "c-pine", "d-rao", first, 1, 30, offset="-06:00")  # 01:30 MDT
    backend.slot("late", "c-pine", "d-rao", second, 1, 30, offset="-07:00")  # 01:30 MST
    before = datetime(2026, 10, 31, 18, 0, tzinfo=UTC)
    session, _, _ = talk(
        backend, book(clinic_name="Pine Clinic"), intent(option_number=2), now=before
    )

    asked = session.respond("Book Pine Clinic on November first at one thirty in the morning.")

    assert "1:30 AM (UTC-06:00)" in asked.text and "1:30 AM (UTC-07:00)" in asked.text
    assert backend.prepared == []
    session.respond("The second one.")
    assert backend.prepared == [{"action": "BOOK", "slot_id": "late"}]


def test_a_time_that_the_clocks_skip_has_no_opening_and_the_real_times_are_offered() -> None:
    backend = Backend()
    day = date(2027, 3, 14)  # Clocks go from 02:00 to 03:00.
    backend.slot("before", "c-pine", "d-rao", day, 1, 30, offset="-07:00")
    backend.slot("after", "c-pine", "d-rao", day, 3, 30, offset="-06:00")
    early = datetime(2027, 3, 1, 15, 0, tzinfo=UTC)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"), now=early)

    reply = session.respond("Book Pine Clinic on March fourteenth at two thirty in the morning.")

    assert "I don't see an opening at 2:30 AM" in reply.text
    assert "1:30 AM" in reply.text and "3:30 AM" in reply.text and backend.prepared == []


def test_an_evening_request_is_asked_about_because_only_two_parts_of_day_are_searchable() -> None:
    backend = pine_day(9)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond("Book Pine Clinic on November seventeenth in the evening.")

    assert "mornings, afternoons, or a particular time" in reply.text and backend.calls == []


def test_a_clinic_whose_name_holds_a_date_word_is_not_read_as_a_date() -> None:
    backend = Backend()
    backend.clinics["c-tomorrow"] = {
        "clinic_id": "c-tomorrow",
        "name": "Tomorrow Family Clinic",
        "timezone": "America/Denver",
    }
    backend.doctors["d-tomorrow"] = {
        "doctor_id": "d-tomorrow",
        "display_name": "Dr. Sam Fern",
        "clinic": "c-tomorrow",
    }
    backend.slot("t1", "c-tomorrow", "d-tomorrow", date(2026, 11, 3), 15)
    session, _, _ = talk(backend, book(clinic_name="Tomorrow Family Clinic"))

    session.respond("Book Tomorrow Family Clinic tomorrow afternoon.")

    assert availability_calls(backend)[0]["start_date"] == "2026-11-03"  # One date, not two.
    assert backend.prepared == [{"action": "BOOK", "slot_id": "t1"}]


def test_first_available_after_ten_is_remembered_across_the_question_without_the_model() -> None:
    backend = pine_day(9, 10.5, 14)
    session, model, _ = talk(backend, book(clinic_name="Pine Clinic"), intent())

    session.respond("Book Pine Clinic on November seventeenth, the first available one after ten.")
    session.respond("AM")

    assert model.state.draft.selection == "EARLIEST"
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


# --- natural ways of asking to be seen ---------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Could I get in at Pine Clinic on November seventeenth?",
        "Can you fit me in at Pine Clinic on November seventeenth?",
        "Please squeeze me in at Pine Clinic on November seventeenth.",
        "Any chance you could work me in at Pine Clinic on November seventeenth?",
        "Could you sign me up at Pine Clinic for November seventeenth?",
    ],
)
def test_natural_ways_of_asking_to_be_seen_are_booking_requests(phrase) -> None:
    backend = pine_day(9)
    session, _, _ = talk(backend, book(clinic_name="Pine Clinic"))

    reply = session.respond(phrase)

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s0"}]


@pytest.mark.parametrize(
    ("phrase", "intent_read", "asks"),
    [
        ("Could I get in?", book(), "Which clinic"),
        ("Can you fit me in tomorrow?", book(), "Which clinic"),
        ("Could I get in at Pine Clinic?", book(clinic_name="Pine Clinic"), "Which date"),
        ("Could you fit me in at Pine Clinic on Tuesday?", book(clinic_name="Pine Clinic"), None),
    ],
)
def test_a_request_to_be_fitted_in_never_assumes_what_it_did_not_say(
    phrase, intent_read, asks
) -> None:
    backend = pine_day(9, day=date(2026, 11, 3))
    session, _, _ = talk(backend, intent_read)

    reply = session.respond(phrase)

    if asks:
        assert asks in reply.text and backend.calls == []
    else:
        assert availability_calls(backend)[0]["start_date"] == "2026-11-03"  # What was said.


@pytest.mark.parametrize(
    "phrase",
    [
        "Are there any openings at Pine Clinic on Tuesday?",
        "Do you have any appointments at Pine Clinic on Tuesday?",
        "I'd like to make a booking at Pine Clinic.",
        "Any visits free at Pine Clinic on Tuesday?",
        "Are there slots at Pine Clinic on Tuesday?",
    ],
)
def test_plural_and_progressive_forms_of_booking_words_are_booking_words(phrase) -> None:
    assert _OPERATION_WORDS["BOOK"].search(phrase) is not None


@pytest.mark.parametrize(
    "phrase",
    [
        "I'd like to get in touch with the clinic",
        "How do I get into the building",
        "Can I come in through the side door",
        "What time do you open",
    ],
)
def test_phrases_that_are_not_requests_for_an_appointment_are_not_taken_for_one(phrase) -> None:
    assert _OPERATION_WORDS["BOOK"].search(phrase) is None


def test_a_request_to_be_fitted_in_is_understood_without_the_model_too() -> None:
    backend = pine_day(9)
    session, model, _ = talk(backend, TimeoutError("down"))

    reply = session.respond("Could you fit me in on November seventeenth?")

    assert model.state.draft.operation == "BOOK" and "Which clinic" in reply.text
    assert backend.calls == []
