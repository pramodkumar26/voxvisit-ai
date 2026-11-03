"""LG-2: the planner beside the controller. The model's replies are scripted here; the real model
is measured by tests/graph_eval.py. The point of these tests is what the planner may and may not
do whatever the model says."""

import json
from datetime import date

import pytest
from test_conversation_unit import (
    NOW,
    RetiringBackend,
    Script,
    book,
    fail,
    intent,
    patient,
)

from app.evaluation.graph_session import GraphSession
from app.evaluation.hybrid import HybridControllerModel
from app.evaluation.llm import LlmConfig
from app.evaluation.planner import GoalPlanner, PlannerDecision, ground, merge_open

MONDAY = date(2026, 11, 9)
THURSDAY = date(2026, 11, 12)
TODAY = date(2026, 11, 2)


class Wire:
    """The planner's transport: one scripted reply per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, path, payload, timeout_s):
        self.payloads.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"message": {"content": content}, "eval_count": 3, "total_duration": 2_000_000}


def route(name="SCHEDULING", **fields):
    return {"route": name, **fields}


def lookup(subject, **fields):
    return route("LOOKUP", subject=subject, **fields)


def backend_with_days():
    backend = RetiringBackend()
    for number, (day, hour, minute) in enumerate(
        [(MONDAY, 9, 0), (MONDAY, 9, 30), (THURSDAY, 14, 0), (THURSDAY, 15, 0)]
    ):
        backend.slot(f"s{number}", "c-pine", "d-rao", day, hour, minute)
    return backend


def session_with(backend, plans, *intents, principal=None):
    wire = Wire(*plans)
    planner = GoalPlanner(LlmConfig(), transport=wire)
    model = HybridControllerModel(Script(*intents), clock=lambda: NOW)
    session = GraphSession(
        model, backend, principal or patient(), planner=planner, clock=lambda: NOW
    )
    return session, model, planner, wire


def tools(backend):
    return [name for name, *_ in backend.calls]


# --- grounding what the model returns ---------------------------------------------------------


def test_a_value_the_message_does_not_contain_is_dropped() -> None:
    decision = PlannerDecision(
        route="LOOKUP",
        subject="OPENINGS",
        clinic_name="Invented Clinic",
        doctor_name="Rao",
        start_date="2026-11-12",
        end_date="2026-11-12",
        time_preference="MORNING",
        week="NEXT",
    )

    grounded = ground(decision, "What is open at the Pine Clinic with Dr. Rao?", TODAY)

    assert grounded.clinic_name is None and grounded.doctor_name == "Rao"
    assert grounded.start_date is None and grounded.time_preference == "ANYTIME"
    assert grounded.week == "NONE"


def test_a_lookup_with_nothing_to_look_up_goes_to_the_controller() -> None:
    decision = PlannerDecision(route="LOOKUP", subject="NONE")

    assert ground(decision, "hmm", TODAY).route == "SCHEDULING"


def test_the_model_cannot_name_a_tool_or_an_identifier() -> None:
    wire = Wire(
        {"route": "LOOKUP", "subject": "CLINICS", "tool": "prepare_appointment_action"},
        {"route": "LOOKUP", "subject": "CLINICS", "clinic_id": "c-pine"},
        "not json at all",
        RuntimeError("provider down"),
    )
    planner = GoalPlanner(LlmConfig(), transport=wire)

    decisions = [planner.plan("hello", context="", today=TODAY, anchor=None) for _ in range(4)]

    assert all(d.route == "SCHEDULING" for d in decisions) and planner.fallbacks == 4


def test_an_open_lookup_is_completed_by_the_answer_and_dropped_for_a_different_question() -> None:
    open_one = PlannerDecision(
        route="LOOKUP", subject="OPENINGS", start_date=THURSDAY, end_date=THURSDAY
    )
    answer = PlannerDecision(route="LOOKUP", subject="OPENINGS", clinic_name="Pine Clinic")
    other = PlannerDecision(route="LOOKUP", subject="CLINICS")

    merged = merge_open(answer, open_one)

    assert merged.clinic_name == "Pine Clinic" and merged.start_date == THURSDAY
    assert merge_open(other, open_one) == other


# --- lookups ----------------------------------------------------------------------------------


def test_a_clinic_question_is_answered_from_the_directory_and_the_request_is_untouched() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), lookup("CLINICS")],
        book(start_date=THURSDAY, end_date=THURSDAY),
    )
    session.respond("I need an appointment on Thursday.")
    before = (model.state.draft, model.state.awaiting, model.state.last_question, model.state.turns)

    reply = session.respond("Is there more than one place I could go?")

    assert "Pine Clinic in Boulder" in reply.text and reply.text.endswith(
        "Which clinic would you like?"
    )
    assert (
        model.state.draft,
        model.state.awaiting,
        model.state.last_question,
        model.state.turns,
    ) == before
    assert tools(backend) == ["search_clinic_directory"] and backend.prepared == []
    assert session.snapshot()["interruptions"][-1]["kind"] == "lookup"


def test_open_times_are_looked_up_through_the_directory_then_availability() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [
            lookup(
                "OPENINGS",
                clinic_name="Pine Clinic",
                start_date="2026-11-12",
                end_date="2026-11-12",
                time_preference="AFTERNOON",
            )
        ],
    )

    reply = session.respond("What's open at Pine Clinic on November 12 in the afternoon?")

    assert tools(backend) == ["search_clinic_directory", "search_appointment_availability"]
    searched = backend.calls[1][1]
    assert searched["clinic_id"] == "c-pine" and searched["start_date"] == "2026-11-12"
    assert searched["time_preference"] == "AFTERNOON"
    assert "2:00 PM" in reply.text and "3:00 PM" in reply.text and "9:00 AM" not in reply.text
    assert backend.prepared == []


def test_next_week_is_searched_as_the_clinics_whole_week_and_spoken_concisely() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend, [lookup("OPENINGS", clinic_name="Pine Clinic", week="NEXT")]
    )

    reply = session.respond("Does Pine Clinic have anything next week?")

    searched = backend.calls[-1][1]
    assert (searched["start_date"], searched["end_date"]) == ("2026-11-09", "2026-11-15")
    assert all(time in reply.text for time in ("9:00 AM", "9:30 AM", "2:00 PM"))
    assert "3:00 PM" not in reply.text
    assert "more options" in reply.text


def test_the_clinic_and_date_of_the_request_in_progress_fill_a_lookup_that_names_neither() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), lookup("OPENINGS")],
        book(
            clinic_name="Pine Clinic",
            start_date=THURSDAY,
            end_date=THURSDAY,
            time_preference="AFTERNOON",
        ),
    )
    session.respond("I need an appointment at Pine Clinic Thursday afternoon.")

    reply = session.respond("What else is open that day?")

    assert backend.calls[-1][1]["clinic_id"] == "c-pine"
    assert backend.calls[-1][1]["start_date"] == "2026-11-12"
    assert "2:00 PM" in reply.text and model.state.draft.clinic_name == "Pine Clinic"


def test_a_missing_detail_is_asked_for_and_the_answer_finishes_the_lookup() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [
            lookup("OPENINGS", start_date="2026-11-12", end_date="2026-11-12"),
            lookup("OPENINGS", clinic_name="Pine Clinic"),
        ],
    )

    asked = session.respond("What's open on November 12?")
    assert asked.text.startswith("Which clinic should I check?")
    assert tools(backend) == ["search_clinic_directory"]  # Only to see whether a clinic was named.
    assert session.snapshot()["planner"]["open_lookup"]["subject"] == "OPENINGS"

    answered = session.respond("Pine Clinic")

    assert "2:00 PM" in answered.text and session.snapshot()["planner"]["open_lookup"] is None


def test_an_ambiguous_clinic_name_is_put_to_the_patient_not_chosen() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [
            lookup(
                "OPENINGS",
                clinic_name="Linden Court",
                start_date="2026-11-12",
                end_date="2026-11-12",
            )
        ],
    )

    reply = session.respond("What is open at Linden Court on November 12?")

    assert "Linden Court Heart Center" in reply.text and "Linden Court Family Clinic" in reply.text
    assert "search_appointment_availability" not in tools(backend)


def test_where_a_doctor_works_is_answered_from_the_directory() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend, [lookup("DOCTORS", clinic_name="Pine Clinic", doctor_name="Ivo Reed")]
    )

    reply = session.respond("Does Dr. Ivo Reed work at Pine Clinic?")

    assert "isn't listed at Pine Clinic" in reply.text
    assert "Linden Court Heart Center" in reply.text


def test_who_are_the_doctors_is_scoped_to_the_clinic_being_booked() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [route(), lookup("DOCTORS")],
        book(clinic_name="Pine Clinic"),
    )
    session.respond("Book at Pine Clinic.")

    reply = session.respond("Who would I see there?")

    assert backend.calls[-1][1] == {"clinic_name": "Pine Clinic"}
    assert "Dr. Anita Rao" in reply.text and "Dr. Ivo Reed" not in reply.text


def test_a_failed_lookup_says_so_and_changes_nothing() -> None:
    backend = backend_with_days()
    backend.fail["search_clinic_directory"] = fail("DEPENDENCY_UNAVAILABLE")
    session, _model, _, _ = session_with(backend, [lookup("CLINICS")])

    reply = session.respond("Which clinics do you have?")

    assert "trouble looking that up" in reply.text and backend.prepared == []


def test_a_range_longer_than_the_backend_allows_is_narrowed_before_asking_it() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [
            lookup(
                "OPENINGS",
                clinic_name="Pine Clinic",
                start_date="2026-11-10",
                end_date="2027-01-30",
            )
        ],
    )

    reply = session.respond("Pine Clinic from 2026-11-10 to 2027-01-30?")

    assert "up to 31 days" in reply.text and "search_appointment_availability" not in tools(backend)


# --- everything else stays with the controller ------------------------------------------------


def test_an_unrelated_remark_leaves_the_request_and_the_next_answer_completes_it() -> None:
    backend = backend_with_days()
    session, _model, _, _ = session_with(
        backend,
        [route(), route("OUT_OF_SCOPE"), route()],
        book(
            clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY, time_preference="MORNING"
        ),
        intent(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    session.respond("An appointment at Pine Clinic Monday morning.")

    aside = session.respond("Do you validate parking?")
    done = session.respond("9:30 please")

    assert (
        "can only help with appointments" in aside.text
        and "Which time would you like?" in aside.text
    )
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s1"}] and done.awaiting_confirmation


def test_an_unclear_message_gets_a_fixed_question_and_nothing_else_happens() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(backend, [route("CLARIFY", unclear_about="GOAL")])

    reply = session.respond("my knee hurts and I don't know")

    assert "book, move, or cancel" in reply.text and backend.calls == []
    assert model.state.draft is None


def test_a_waiting_review_survives_a_lookup_and_is_not_retired() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), lookup("CLINICS")],
        book(clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY, selection="EARLIEST"),
    )
    first = session.respond("Book the earliest at Pine Clinic on November 9.")
    assert first.awaiting_confirmation

    reply = session.respond("Which clinics are there, out of interest?")

    assert reply.awaiting_confirmation and "press Confirm in the app" in reply.text
    assert backend.discarded == [] and model.state.prepared is not None


def test_a_scheduling_message_reaches_the_controller_unchanged() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend,
        [route()],
        book(
            clinic_name="Pine Clinic",
            start_date=THURSDAY,
            end_date=THURSDAY,
            time_preference="AFTERNOON",
        ),
    )

    reply = session.respond("Book Pine Clinic Thursday afternoon.")

    assert "Which time would you like?" in reply.text and backend.prepared == []


def test_a_message_with_a_credential_never_reaches_the_planner() -> None:
    backend = backend_with_days()
    session, _, _, wire = session_with(backend, [])

    reply = session.respond("my pin is 4821")

    assert wire.payloads == [] and "don't share a PIN" in reply.text


def test_nothing_the_planner_is_told_holds_a_secret_or_an_identifier() -> None:
    backend = backend_with_days()
    who = patient()
    session, _, _, wire = session_with(
        backend, [route(), route()], book(clinic_name="Pine Clinic"), principal=who
    )
    session.respond("Book at Pine Clinic.")
    session.respond("Which doctors are there?")

    sent = json.dumps(wire.payloads)

    for secret in (who.csrf_token, str(who.session_id), str(who.patient_id)):
        assert secret not in sent


def test_a_confirmation_event_never_goes_through_the_planner() -> None:
    backend = backend_with_days()
    session, _, _, wire = session_with(
        backend,
        [route()],
        book(clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY, selection="EARLIEST"),
    )
    session.respond("Book the earliest at Pine Clinic on November 9.")
    calls = len(wire.payloads)

    done = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert "is booked" in done.text and len(wire.payloads) == calls


@pytest.mark.parametrize("name", ["LOOKUP", "CLARIFY", "OUT_OF_SCOPE"])
def test_no_route_can_prepare_a_change(name) -> None:
    backend = backend_with_days()
    plan = (
        lookup("OPENINGS", clinic_name="Pine Clinic", week="NEXT")
        if name == "LOOKUP"
        else route(name)
    )
    session, _, _, _ = session_with(backend, [plan])

    session.respond("Please book the first one at Pine Clinic next week.")

    assert backend.prepared == [] and "prepare_appointment_action" not in tools(backend)


def test_the_checkpoint_keeps_the_planner_state_but_not_the_words_or_the_model_reply() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(
        backend, [lookup("OPENINGS", start_date="2026-11-12", end_date="2026-11-12")]
    )
    session.respond("Anything on November 12, please, whichever clinic")

    stored = repr(session._saver.get_tuple(session.config))

    assert "please, whichever clinic" not in stored and "Which clinic should I check" not in stored
    assert session.snapshot()["planner"]["route"] == "LOOKUP"


# --- what the planner understood is passed on as a hint, never as a decision ------------------


def test_dates_come_from_the_patients_words_not_from_the_model() -> None:
    decision = PlannerDecision(
        route="LOOKUP", subject="OPENINGS", start_date="2031-01-01", end_date="2031-01-01"
    )

    grounded = ground(decision, "what's open on Thursday?", TODAY)

    assert grounded.start_date == grounded.end_date == date(2026, 11, 5)


def test_a_correction_in_words_that_sound_like_a_move_stays_a_correction() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), route(same_request=True)],
        book(
            clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY, time_preference="MORNING"
        ),
        PatientIntentFor.reschedule(
            start_date=THURSDAY, end_date=THURSDAY, time_preference="AFTERNOON"
        ),
    )
    session.respond("An appointment at Pine Clinic Monday morning.")

    reply = session.respond("sorry, I meant Thursday afternoon instead")

    assert model.state.draft.operation == "BOOK"
    assert "get_my_appointments" not in tools(backend)
    assert "2:00 PM" in reply.text and backend.prepared == []


class PatientIntentFor:
    @staticmethod
    def reschedule(**fields):
        return intent(operation="RESCHEDULE", **fields)


def test_without_the_planners_word_the_controller_reads_it_as_before() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), route(same_request=False)],
        book(clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY),
        PatientIntentFor.reschedule(start_date=THURSDAY, end_date=THURSDAY),
    )
    session.respond("An appointment at Pine Clinic Monday.")

    session.respond("sorry, let me change that to Thursday")

    assert model.state.draft.operation == "RESCHEDULE"


def test_openings_just_shown_for_one_day_are_what_i_ll_take_2_pm_refers_to() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [
            lookup("OPENINGS", clinic_name="Pine Clinic"),
            route(),
        ],
        intent(time="2 PM", selection="SPECIFIC_TIME"),
    )
    session.respond("What does Pine Clinic have open on November 12?")

    reply = session.respond("I'll take 2 PM")

    assert backend.prepared == [{"action": "BOOK", "slot_id": "s2"}] and reply.awaiting_confirmation
    assert model.state.draft.clinic_name == "Pine Clinic"
    assert model.state.draft.start_date == THURSDAY


def test_what_was_shown_is_not_carried_into_a_different_clinic_or_a_week() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [
            lookup("OPENINGS", clinic_name="Pine Clinic", week="NEXT"),
            route(),
        ],
        book(clinic_name="Two Rivers Clinic"),
    )
    session.respond("Anything at Pine Clinic next week?")

    session.respond("Book Two Rivers Clinic.")

    assert model.state.draft.clinic_name == "Two Rivers Clinic"
    assert model.state.draft.start_date is None  # A week is not a day: the date is asked.


def test_what_was_shown_is_used_once() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [lookup("OPENINGS", clinic_name="Pine Clinic"), route(), route()],
        intent(),
        intent(),
    )
    session.respond("What does Pine Clinic have open on November 12?")
    session.respond("hmm, thanks")  # No schedule words: nothing is started from it.

    assert model.state.draft is None or model.state.draft.clinic_name is None


def test_an_off_topic_reading_by_the_controller_no_longer_ends_the_request_on_this_path() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(), route()],
        book(clinic_name="Pine Clinic"),
        intent(message_type="OFF_TOPIC"),
    )
    session.respond("Book at Pine Clinic.")

    reply = session.respond("I wonder what the weather is like")

    assert model.state.draft.clinic_name == "Pine Clinic" and "Which date" in reply.text


def test_a_weekday_said_after_a_week_was_listed_is_that_weeks_day() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [lookup("OPENINGS", clinic_name="Pine Clinic", week="NEXT"), route()],
        book(start_date=date(2026, 11, 5), end_date=date(2026, 11, 5)),
    )
    session.respond("Anything at Pine Clinic next week?")

    session.respond("Thursday, then")

    assert model.state.draft.clinic_name == "Pine Clinic"
    assert model.state.draft.start_date == THURSDAY  # Not the 5th: the week shown was the 9th-15th.


def test_a_clinic_the_message_names_but_the_model_missed_is_found_in_the_directory() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(backend, [lookup("OPENINGS", week="THIS")])

    reply = session.respond("Does Pine Clinic have anything this week?")

    assert backend.calls[0][1] == {"browse": True}
    assert backend.calls[-1][1]["clinic_id"] == "c-pine" and "Which clinic" not in reply.text


def test_two_clinic_names_in_the_message_are_not_guessed_between() -> None:
    backend = backend_with_days()
    session, _, _, _ = session_with(backend, [lookup("OPENINGS", week="THIS")])

    reply = session.respond("Pine Clinic or Two Rivers Clinic, which has anything this week?")

    assert "Which clinic should I check?" in reply.text
    assert "search_appointment_availability" not in tools(backend)


def test_an_appointment_asked_for_without_a_booking_word_starts_a_request() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [route(starts_booking=True)],
        intent(clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY),
    )

    reply = session.respond("Pine Clinic on Monday, I can only do mornings")

    assert model.state.draft.operation == "BOOK" and "Which time would you like?" in reply.text


def test_without_the_planners_word_a_message_with_no_booking_word_still_asks_what_to_do() -> None:
    backend = backend_with_days()
    session, _model, _, _ = session_with(
        backend,
        [route(starts_booking=False)],
        intent(clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY),
    )

    reply = session.respond("Pine Clinic on Monday, I can only do mornings")

    assert "book, move, or cancel" in reply.text and backend.prepared == []


def test_a_doctor_just_asked_about_is_the_doctor_of_the_booking_that_follows() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend,
        [lookup("DOCTORS", clinic_name="Two Rivers Clinic", doctor_name="Wren"), route()],
        book(start_date=THURSDAY, end_date=THURSDAY, doctor_name="that doctor"),
    )
    session.respond("Is Dr. Wren with Two Rivers Clinic?")

    session.respond("book November 12 with that doctor")

    draft = model.state.draft
    assert (draft.clinic_name, draft.doctor_name) == ("Two Rivers Clinic", "Dr. Ana Wren")


def test_that_doctor_is_never_taken_for_a_name() -> None:
    backend = backend_with_days()
    session, model, _, _ = session_with(
        backend, [route()], book(clinic_name="Pine Clinic", doctor_name="that doctor")
    )

    session.respond("Book Pine Clinic with that doctor")

    assert model.state.draft.doctor_name is None
