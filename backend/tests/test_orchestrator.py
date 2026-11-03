"""LG-3: the conversation lives in the graph checkpoint. The model's readings are scripted here
(the real model is measured by the LG-4 evaluation); these tests are about state, tools and safety
whatever the model says."""

import dataclasses
import json
import uuid
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_conversation_unit import NOW, RetiringBackend, fail, patient, pine_day

from app.evaluation.conversation import ConversationRegistry
from app.evaluation.llm import LlmConfig
from app.evaluation.orchestrator import OrchestratedSession, bare_time_choice
from app.evaluation.planner import GoalPlanner
from app.evaluation.turn_reader import TurnReader

MONDAY, THURSDAY = date(2026, 11, 9), date(2026, 11, 12)
TODAY = NOW.date()


class Wire:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, path, payload, timeout_s):
        self.payloads.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"message": {"content": content}, "eval_count": 3, "total_duration": 1_000_000}


def sched(**fields):
    return {"route": "SCHEDULING", **fields}


def book(**fields):
    return sched(operation="BOOK", **fields)


def ask(*subjects, **fields):
    return {"route": "LOOKUP", "subjects": list(subjects), **fields}


def days_backend():
    backend = RetiringBackend()
    for number, (day, hour, minute) in enumerate(
        [(MONDAY, 9, 0), (MONDAY, 9, 30), (MONDAY, 11, 0), (THURSDAY, 14, 0), (THURSDAY, 15, 0)]
    ):
        backend.slot(f"s{number}", "c-pine", "d-rao", day, hour, minute)
    return backend


def session(backend, *readings, principal=None, saver=None, quick=False):
    wire = Wire(*readings)
    config = LlmConfig()
    made = OrchestratedSession(
        backend,
        principal or patient(),
        config=config,
        reader=TurnReader(config, transport=wire),
        planner=GoalPlanner(config, transport=Wire()),
        checkpointer=saver or InMemorySaver(),
        clock=lambda: NOW,
        quick_paths=quick,
    )
    return made, wire


def tools(backend):
    return [name for name, *_ in backend.calls]


# --- the checkpoint is the conversation -------------------------------------------------------


def test_the_scheduler_keeps_nothing_between_turns_the_checkpoint_does() -> None:
    backend = days_backend()
    one, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
    )
    one.respond("I need an appointment at Pine Clinic on November 9, morning.")
    state = one.snapshot()

    assert state["request"]["clinic_name"] == "Pine Clinic"
    assert state["dialogue"]["awaiting"] == "TIME" and len(state["dialogue"]["offered"]) == 3
    # A second session object over the same checkpoint continues the conversation: nothing else
    # held the request.
    saver = one._saver
    two, _ = session(
        backend,
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        principal=one.principal,
        saver=saver,
    )
    reply = two.respond("9:30 AM")

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


def test_read_only_availability_can_page_through_three_spoken_options() -> None:
    backend = days_backend()
    made, wire = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-12",
        ),
        quick=True,
    )

    first = made.respond("What is open at Pine Clinic from November 9 to November 12?")
    second = made.respond("Show me more options.")

    assert all(time in first.text for time in ("9:00 AM", "9:30 AM", "11:00 AM"))
    assert "2:00 PM" not in first.text and "3:00 PM" not in first.text
    assert "2:00 PM" in second.text and "3:00 PM" in second.text
    assert len(wire.payloads) == 1
    availability = [call for call in backend.calls if call[0] == "search_appointment_availability"]
    assert len(availability) == 2
    assert made.snapshot()["context"]["shown"]["offer_offset"] == 3
    assert backend.prepared == []


def test_scheduling_more_options_survive_the_checkpoint_and_only_spoken_choices_can_be_picked() -> (
    None
):
    backend = pine_day(9, 9.5, 10, 10.5, 11, 11.5)
    made, wire = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-17",
            end_date="2026-11-17",
        ),
        sched(option_number=2),
        quick=True,
    )
    made.respond("Book Pine Clinic on November 17.")

    more = made.respond("Show me more options.")

    assert all(time in more.text for time in ("10:30 AM", "11:00 AM", "11:30 AM"))
    assert made.snapshot()["dialogue"]["offer_offset"] == 3
    chosen = made.respond("The second option.")
    assert chosen.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]
    assert len(wire.payloads) == 2


def test_editing_the_checkpoint_changes_what_the_agent_does() -> None:
    backend = days_backend()
    one, _ = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    one.respond("Book Pine Clinic November 9.")
    request = dict(one.snapshot()["request"], clinic_name="Two Rivers Clinic")
    one.graph.update_state(one.config, {"request": request})

    one.respond("9:30 AM")

    # The request was Two Rivers by then: no 9:30 there, so nothing was prepared.
    assert backend.prepared == []


def test_a_review_survives_in_the_checkpoint_and_the_ui_report_describes_it() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-12",
            end_date="2026-11-12",
            selection="EARLIEST",
        ),
    )
    first = made.respond("Book the earliest at Pine Clinic on November 12.")
    assert (
        first.awaiting_confirmation
        and made.snapshot()["review"]["pending_action_id"] == "pending-1"
    )

    done = made.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert "is booked" in done.text and "2:00 PM" in done.text
    assert made.snapshot()["review"] is None and made.snapshot()["request"] == {}


def test_no_secret_identifier_or_spoken_word_is_checkpointed() -> None:
    who = dataclasses.replace(patient(), csrf_token="csrf-" + uuid.uuid4().hex)
    backend = days_backend()
    made, wire = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        principal=who,
    )
    made.respond("Book Pine Clinic November 9, thanks very much indeed.")

    stored = repr(made._saver.get_tuple(made.config))

    for secret in (who.csrf_token, str(who.session_id), str(who.patient_id), who.login):
        assert secret not in stored and secret not in json.dumps(wire.payloads)
    assert (
        "thanks very much" not in stored
    )  # The question asked is state; the patient's words are not.


# --- one call per message ---------------------------------------------------------------------


def test_a_scheduling_message_costs_one_model_call() -> None:
    backend = days_backend()
    made, wire = session(
        backend, book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09")
    )

    made.respond("Book Pine Clinic November 9.")

    assert len(wire.payloads) == 1 and made.model_calls == 1


def test_an_unusable_reading_falls_back_to_the_extractor_ladder() -> None:
    backend = days_backend()
    made, _wire = session(backend, "not json")
    made.extractor._post = lambda *_: {"message": {"content": "{}"}}  # Never usable either.

    reply = made.respond("Book Pine Clinic November 9.")

    assert made.reader.fallbacks == 1 and reply.text  # A reply, from the rule-based reading.
    assert backend.prepared == []


# --- questions are answered from tools and leave the request alone ----------------------------


def test_a_compound_question_runs_both_lookups_in_order_and_grounds_both_answers() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "DOCTORS",
            "OPENINGS",
            clinic_name="Pine Clinic",
            start_date="2026-11-12",
            end_date="2026-11-12",
        ),
    )

    reply = made.respond("Who are the doctors at Pine Clinic and what's open on November 12?")

    assert "Dr. Anita Rao" in reply.text and "2:00 PM" in reply.text and "3:00 PM" in reply.text
    assert tools(backend)[0] == "search_clinic_directory"
    assert "search_appointment_availability" in tools(backend) and backend.prepared == []


def test_a_question_in_the_middle_of_a_request_keeps_the_request_and_asks_again() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(start_date="2026-11-12", end_date="2026-11-12"),
        ask("CLINICS"),
    )
    made.respond("I need an appointment on November 12.")
    before = made.snapshot()

    reply = made.respond("Which places can I go?")
    after = made.snapshot()

    assert "Pine Clinic in Boulder" in reply.text and reply.text.endswith(
        "Which clinic would you like?"
    )
    assert after["request"] == before["request"] and after["dialogue"] == before["dialogue"]
    assert after["interruptions"][-1]["resumed"] is False


def test_what_a_lookup_showed_is_what_i_ll_take_2_pm_refers_to() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        sched(time="2 PM", selection="SPECIFIC_TIME"),
    )
    made.respond("What does Pine Clinic have open on November 12?")

    reply = made.respond("I'll take 2 PM")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s3"}]


def test_what_was_shown_is_used_once() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        sched(),
        sched(time="2 PM", selection="SPECIFIC_TIME"),
    )
    made.respond("What does Pine Clinic have open on November 12?")
    made.respond("hmm thanks")
    assert made.snapshot()["context"]["shown"] is None

    made.respond("2 PM then")

    assert backend.prepared == []


def test_an_unrelated_remark_keeps_the_request() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        {"route": "OUT_OF_SCOPE"},
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    made.respond("Pine Clinic November 9 morning, book it.")
    aside = made.respond("is the parking free?")
    done = made.respond("9:30 AM")

    assert "Which time would you like?" in aside.text and done.awaiting_confirmation


def test_a_waiting_review_is_not_retired_by_a_question() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            selection="EARLIEST",
        ),
        ask("CLINICS"),
    )
    made.respond("Book the earliest at Pine Clinic on November 9.")

    reply = made.respond("what other clinics are there?")

    assert reply.awaiting_confirmation and backend.discarded == []
    assert made.snapshot()["review"]["pending_action_id"] == "pending-1"


def test_dropping_a_request_retires_the_review_and_forgets_it() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            selection="EARLIEST",
        ),
        sched(message_type="ABANDON"),
    )
    made.respond("Book the earliest at Pine Clinic on November 9.")

    made.respond("never mind")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]
    state = made.snapshot()
    assert state["review"] is None and state["request"] == {} and state["retiring"] is None


def test_a_failed_retirement_is_remembered_in_the_checkpoint_and_retried() -> None:
    backend = days_backend()
    backend.discard_fails = True
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            selection="EARLIEST",
        ),
        sched(message_type="ABANDON"),
        sched(),
    )
    made.respond("Book the earliest at Pine Clinic on November 9.")

    told = made.respond("never mind")
    assert "please don't confirm it in the app" in told.text
    assert made.snapshot()["retiring"] == "pending-1"

    backend.discard_fails = False
    made.respond("hello again")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]
    assert made.snapshot()["retiring"] is None


def test_a_correction_that_sounds_like_a_move_stays_a_correction() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        {
            "route": "SCHEDULING",
            "same_request": True,
            "operation": "RESCHEDULE",
            "start_date": "2026-11-12",
            "end_date": "2026-11-12",
            "time_preference": "AFTERNOON",
        },
    )
    made.respond("Pine Clinic Monday morning, book it.")

    reply = made.respond("sorry, I meant Thursday November 12 afternoon instead")

    assert "get_my_appointments" not in tools(backend) and "2:00 PM" in reply.text


def test_a_clear_request_worded_as_a_wish_for_an_appointment_starts_a_booking() -> None:
    backend = days_backend()
    starts = sched(
        starts_booking=True,
        clinic_name="Pine Clinic",
        start_date="2026-11-09",
        end_date="2026-11-09",
    )
    made, wire = session(backend, starts)

    reply = made.respond("I'd like an appointment at Pine Clinic on November 9, mornings")

    assert made.snapshot()["request"]["operation"] == "BOOK" and "Which time" in reply.text
    assert made.snapshot()["request"]["vouched"] is True
    assert len(wire.payloads) == 1  # No second reading takes part.


def test_no_model_reading_starts_a_booking_by_itself() -> None:
    backend = days_backend()
    starts = sched(
        starts_booking=True,
        clinic_name="Pine Clinic",
        start_date="2026-11-09",
        end_date="2026-11-09",
    )
    made, wire = session(backend, starts, {"asks_for": "BOOK"})  # Even if one said so.

    reply = made.respond("Pine Clinic on November 9, I can only do mornings")

    assert "9:00 AM" in reply.text  # Answered as a question about what is open.
    assert not reply.awaiting_confirmation and backend.prepared == []
    assert made.snapshot()["request"] == {} and len(wire.payloads) == 1


@pytest.mark.parametrize(
    ("text", "offered", "expected"),
    [
        ("11", ["9:00 AM", "11:00 AM"], "11:00 AM"),
        ("the 9:30", ["9:00 AM", "9:30 AM"], "9:30 AM"),
        ("2 please", ["2:00 PM", "3:00 PM"], "2:00 PM"),
        ("10", ["9:00 AM", "11:00 AM"], None),
        ("9", ["9:00 AM", "9:00 PM"], None),
        ("Pine Clinic", ["9:00 AM"], None),
    ],
)
def test_a_bare_number_means_the_one_offered_time_it_reads_as(text, offered, expected) -> None:
    dialogue = {
        "awaiting": "TIME",
        "offered": [{"label": f"{t} with Dr. X", "key": "k"} for t in offered],
    }

    assert bare_time_choice(text, dialogue) == expected


def test_a_bare_number_means_nothing_when_no_time_was_asked_for() -> None:
    assert bare_time_choice("11", {"awaiting": "DATE", "offered": []}) is None


# --- sessions, isolation and cleanup ----------------------------------------------------------


def registry(saver, backend_readings=()):
    config = LlmConfig()
    return ConversationRegistry(
        lambda: None,
        session_factory=lambda _model, runtime, principal: OrchestratedSession(
            runtime,
            principal,
            config=config,
            reader=TurnReader(config, transport=Wire(*backend_readings)),
            planner=GoalPlanner(config, transport=Wire()),
            checkpointer=saver,
            clock=lambda: NOW,
        ),
    )


def test_two_sessions_never_share_a_conversation_and_ending_one_deletes_only_its_own() -> None:
    saver, backend = InMemorySaver(), days_backend()
    sessions = registry(
        saver, [book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09")]
    )
    first, second = patient(), patient()
    mine = sessions.session_for(backend, first)
    theirs = sessions.session_for(backend, second)
    mine.respond("Book Pine Clinic November 9.")

    assert theirs.snapshot() == {} and mine.snapshot()["request"]["clinic_name"] == "Pine Clinic"
    sessions.end(first)
    assert saver.get_tuple(mine.config) is None and sessions.session_for(backend, second) is theirs


def test_a_session_id_now_owned_by_another_patient_gets_a_fresh_conversation() -> None:
    saver, backend = InMemorySaver(), days_backend()
    sessions = registry(
        saver, [book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09")]
    )
    old = patient()
    first = sessions.session_for(backend, old)
    first.respond("Book Pine Clinic November 9.")

    second = sessions.session_for(backend, dataclasses.replace(old, patient_id=uuid.uuid4()))

    assert second is not first and saver.get_tuple(first.config) is None and second.snapshot() == {}


def test_a_lookup_failure_says_so_and_changes_nothing() -> None:
    backend = days_backend()
    backend.fail["search_clinic_directory"] = fail("DEPENDENCY_UNAVAILABLE")
    made, _ = session(backend, ask("CLINICS"))

    reply = made.respond("which clinics?")

    assert "trouble looking that up" in reply.text and backend.prepared == []


@pytest.mark.parametrize("route", ["LOOKUP", "CLARIFY", "OUT_OF_SCOPE"])
def test_no_route_but_scheduling_can_prepare_anything(route) -> None:
    backend = days_backend()
    reading = {
        "route": route,
        "subjects": ["OPENINGS"],
        "clinic_name": "Pine Clinic",
        "operation": "BOOK",
        "selection": "EARLIEST",
        "start_date": "2026-11-09",
        "end_date": "2026-11-09",
    }
    made, _ = session(backend, reading)

    made.respond("Please book the earliest at Pine Clinic on November 9.")

    assert backend.prepared == [] and "prepare_appointment_action" not in tools(backend)


def test_a_credential_never_reaches_the_model() -> None:
    backend = days_backend()
    made, wire = session(backend)

    reply = made.respond("my pin is 4821")

    assert wire.payloads == [] and "don't share a PIN" in reply.text


def test_a_signed_out_visitor_can_look_things_up_but_cannot_prepare() -> None:
    backend = pine_day(9, day=MONDAY)
    config = LlmConfig()
    anonymous = OrchestratedSession(
        backend,
        None,
        config=config,
        reader=TurnReader(
            config,
            transport=Wire(
                ask("CLINICS"),
                book(
                    clinic_name="Pine Clinic",
                    start_date="2026-11-09",
                    end_date="2026-11-09",
                    selection="EARLIEST",
                ),
            ),
        ),
        planner=GoalPlanner(config, transport=Wire()),
        clock=lambda: NOW,
    )

    listed = anonymous.respond("which clinics are there?")
    refused = anonymous.respond("book the earliest at Pine Clinic on November 9")

    assert "Pine Clinic in Boulder" in listed.text
    assert "sign in" in refused.text.lower() and backend.prepared == []


def test_a_repeat_of_the_same_scheduling_message_prepares_nothing_twice() -> None:
    backend = days_backend()
    reading = book(
        clinic_name="Pine Clinic",
        start_date="2026-11-09",
        end_date="2026-11-09",
        selection="EARLIEST",
    )
    made, _ = session(backend, reading, reading)
    made.respond("Book the earliest at Pine Clinic on November 9.")

    again = made.respond("Book the earliest at Pine Clinic on November 9.")

    assert len(backend.prepared) == 1 and "already prepared" in again.text
