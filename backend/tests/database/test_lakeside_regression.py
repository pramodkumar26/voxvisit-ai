"""The Lakeside / General Practice failure from the first microphone trial, replayed end to end.

A patient asked for a "General Physician". The model wrote `specialty: "General Practice"`. The
clinic's doctors are Cardiology, Pediatrics and Internal Medicine, and the assistant answered "I
don't see any openings" for days that had many, over and over, without ever saying why.

Here the same six turns run through `GraphConversations` (the path voice mode uses), the real
authentication, the real tools on PostgreSQL, the real PostgreSQL checkpoints and the UI-only
confirmation. The model's readings are scripted as that conversation produced them; everything the
patient hears is built by the code. The repair under test: the mismatch is explained, nothing is
dropped silently, the patient switches to Internal Medicine, a valid review is prepared, and only
the authenticated UI can commit it, after which (and only then) a booking is announced."""

import re

import pytest
from test_graph_persistence import active_reviews, service_over, sign_in, sql
from test_orchestrator import Wire, book, sched

from app.evaluation.fixtures import create_clinic, local_start, unique_place_name
from app.evaluation.graph_store import thread_for
from app.evaluation.lg4_scenarios import NOW, ZONE

TUESDAY, WEDNESDAY, SATURDAY = "2026-11-03", "2026-11-04", "2026-11-07"
BOOKED = re.compile(r"\b(booked|scheduled|rescheduled|confirmed|reserved)\b", re.I)


class ReadingsWire(Wire):
    """One scripted reading per message. A turn may also ask the model for a second look at a
    request in progress (a `Revision`); that is answered "no change", so the script stays one
    reading per patient message however many looks a turn takes."""

    def __call__(self, path, payload, timeout_s):
        if (payload.get("format") or {}).get("title") == "Revision":
            self.payloads.append(payload)
            return {
                "message": {"content": '{"kind": "NONE"}'},
                "eval_count": 1,
                "total_duration": 1,
            }
        return super().__call__(path, payload, timeout_s)


def at(days: int, hour: int):
    return local_start(NOW, ZONE, days_ahead=days, hour=hour, minute=0)


@pytest.fixture
def evaluation(auth_env):
    from app.evaluation.environment import EvalEnvironment

    return EvalEnvironment.from_parts(
        engine=auth_env.app.state.db_engine,
        clock=auth_env.clock,
        client=auth_env.client,
        auth_service=auth_env.service,
        pin=auth_env.pin,
    )


def lakeside(evaluation, suffix="0a1"):
    """Lakeside as the trial found it: Cardiology, Pediatrics and Internal Medicine, no General
    Practice. Tomorrow (Tuesday) has one Internal Medicine morning opening and several others."""
    return create_clinic(
        evaluation.engine,
        name=unique_place_name(evaluation.engine, "Lakeside Community Clinic", suffix),
        timezone=ZONE,
        doctors=[
            ("Dr. Marcus Ellery", "Cardiology"),
            ("Dr. Priya Nambiar", "Pediatrics"),
            ("Dr. Rohan Desai", "Internal Medicine"),
        ],
        doctor_slot_starts=[
            [at(1, 9), at(7, 10)],  # Ellery: Tuesday 9:00 and the Monday after
            [at(1, 10)],  # Nambiar: Tuesday 10:00
            [at(1, 8), at(1, 13), at(2, 9)],  # Desai: Tuesday 8:00 and 1:00 PM, Wednesday 9:00
        ],
    )


def saved_request(evaluation, store, cookie) -> dict:
    session_id = evaluation.authenticate(cookie.token).session_id
    saved = store.saver.get_tuple({"configurable": {"thread_id": thread_for(session_id)}})
    return dict(saved.checkpoint["channel_values"].get("request") or {})


def appointments_on(evaluation, slot_id) -> list[str]:
    return [
        r[0]
        for r in sql(
            evaluation, "SELECT status FROM appointments WHERE slot_id = :s", s=str(slot_id)
        )
    ]


def test_the_lakeside_conversation_explains_the_mismatch_then_books_internal_medicine(
    evaluation, graph_store
) -> None:
    clinic = lakeside(evaluation)
    _world, ui, cookie = sign_in(evaluation)
    name = clinic.name
    ellery_tue, ellery_later = clinic.slot_ids[0]
    nambiar_tue = clinic.slot_ids[1][0]
    desai_tue_am, desai_tue_pm, desai_wed = clinic.slot_ids[2]
    wire = ReadingsWire(
        book(specialty="General Practice", start_date=TUESDAY, end_date=TUESDAY),
        book(),
        book(clinic_name=name, start_date=TUESDAY, end_date=TUESDAY, time_preference="MORNING"),
        book(),
        sched(specialty="Internal Medicine"),
    )
    service, _ = service_over(evaluation, graph_store, transport=wire)
    said = []

    def turn(words):
        reply = service.turn(cookie.token, words)
        said.append(reply.text)
        return reply

    first = turn("Hi, I want to book an appointment with a general physician tomorrow.")
    assert "a request to book an appointment for General Practice" in first.text
    assert "Which clinic" in first.text and not first.awaiting_confirmation

    second = turn("Okay, book that appointment for me.")
    assert not second.awaiting_confirmation and active_reviews(evaluation, _world.patient_id) == []

    third = turn(f"I want to book at {name} tomorrow morning")
    assert f"{name} doesn't have a General Practice doctor" in third.text
    assert all(s in third.text for s in ("Cardiology", "Internal Medicine", "Pediatrics"))
    assert "I don't see any openings" not in third.text
    assert saved_request(evaluation, graph_store, cookie)["specialty"] == "General Practice"

    fourth = turn("Can you find me an appointment?")
    assert "doesn't have a General Practice doctor" in fourth.text  # The same honest answer.
    assert active_reviews(evaluation, _world.patient_id) == []

    fifth = turn("Internal Medicine")
    assert fifth.awaiting_confirmation and "8:00 AM" in fifth.text
    assert "Dr. Rohan Desai" in fifth.text
    request = saved_request(evaluation, graph_store, cookie)
    assert request["specialty"] == "Internal Medicine" and request["clinic_name"] == name
    assert request["start_date"] == TUESDAY and request["time_preference"] == "MORNING"

    (pending,) = active_reviews(evaluation, _world.patient_id)
    assert all(not BOOKED.search(text) for text in said)  # Nothing claimed a booking so far.
    for slot in (desai_tue_am, desai_tue_pm, desai_wed, ellery_tue, nambiar_tue, ellery_later):
        assert appointments_on(evaluation, slot) == []  # Nothing was written by the agent.

    outcome = ui.confirm(cookie, pending)  # Only the authenticated UI commits.
    assert outcome.committed
    assert appointments_on(evaluation, desai_tue_am) == ["CONFIRMED"]
    done = service.event(
        cookie.token,
        {"type": "APPOINTMENT_COMMITTED", "result": outcome.result, "pending_action_id": pending},
    )
    # Announced only now, after the trusted confirmation, and said to be the app's doing.
    assert "booked" in done.text and "as confirmed in the app" in done.text
    assert wire.replies == []  # Every scripted reading was used: five turns, five readings.


def test_openings_that_exist_but_do_not_match_are_told_apart_from_none_at_all(
    evaluation, graph_store
) -> None:
    clinic = lakeside(evaluation, "0a2")
    _world, _ui, cookie = sign_in(evaluation)
    name = clinic.name
    service, _ = service_over(
        evaluation,
        graph_store,
        book(clinic_name=name, specialty="Cardiology", start_date=WEDNESDAY, end_date=WEDNESDAY),
        book(clinic_name=name, start_date=SATURDAY, end_date=SATURDAY),
        book(clinic_name=name, start_date=SATURDAY, end_date=SATURDAY, specialty="Pediatrics"),
    )

    no_match = service.turn(cookie.token, f"Book a cardiologist at {name} on Wednesday")
    nothing = service.turn(cookie.token, f"Book {name} on Saturday")
    ahead = service.turn(cookie.token, "A pediatrician on Saturday")

    # Wednesday has an Internal Medicine opening but no Cardiology one: a filter, not an empty day.
    assert "I don't see any Cardiology openings" in no_match.text
    assert "other openings then with Internal Medicine" in no_match.text
    # Saturday has nothing at all, and the days offered are the days the database lists.
    assert "I don't see any openings at" in nothing.text
    assert "though there are other openings" not in nothing.text
    listed = {
        f"{d:%A}, {d:%B} {d.day}"
        for (d,) in sql(
            evaluation,
            "SELECT DISTINCT (starts_at AT TIME ZONE :z)::date FROM slots"
            " WHERE clinic_id = :c AND status = 'AVAILABLE'",
            z=ZONE,
            c=str(clinic.clinic_id),
        )
    }
    named = set(
        re.findall(r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day, [A-Z][a-z]+ \d+", nothing.text)
    )
    assert named - {"Saturday, November 7"} and (named - {"Saturday, November 7"}) <= listed
    # Pediatrics has nothing on Saturday either; the days named are Pediatrics days.
    assert "The next Pediatrics openings are on Tuesday, November 3" in ahead.text
    assert "Wednesday" not in ahead.text.split("The next")[1]
    assert active_reviews(evaluation, _world.patient_id) == []
