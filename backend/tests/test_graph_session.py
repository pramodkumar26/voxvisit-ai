"""LG-1: the LangGraph path runs the same conversation as the controller, with a session-scoped
checkpoint that holds a projection of the state and nothing secret. The fake backend and scripted
reader are those of test_conversation_unit."""

import dataclasses
import uuid
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_conversation_unit import (
    NOV16,
    NOV17,
    NOW,
    Backend,
    RetiringBackend,
    Script,
    book,
    intent,
    patient,
    pine_day,
)

from app.evaluation.conversation import FALLBACK_REPLY, Awaiting, ConversationRegistry, TextSession
from app.evaluation.graph_session import GraphSession, thread_id_for
from app.evaluation.hybrid import HybridControllerModel
from app.evaluation.model import ToolAction

MONDAY = date(2026, 11, 9)


def make(session_class, backend, *answers, principal=None, **options):
    model = HybridControllerModel(Script(*answers), clock=lambda: NOW)
    return session_class(model, backend, principal or patient(), **options), model


def morning_backend():
    return pine_day(9, 9.5, 11, day=MONDAY)


MORNING = book(
    clinic_name="Pine Clinic", start_date=MONDAY, end_date=MONDAY, time_preference="MORNING"
)
CONVERSATION = [
    ("I need an appointment at Pine Clinic Monday morning.", MORNING),
    ("Which clinics are available?", intent()),
    ("9:30 please", intent(time="9:30 AM", selection="SPECIFIC_TIME")),
]


def run(session_class, **options):
    backend = morning_backend()
    session, _ = make(session_class, backend, *[answer for _, answer in CONVERSATION], **options)
    replies = [session.respond(text) for text, _ in CONVERSATION]
    return backend, session, replies


# --- the graph runs the same conversation ---------------------------------------------------------


def test_the_graph_path_says_and_does_exactly_what_the_controller_path_does() -> None:
    old_backend, _, old = run(TextSession)
    new_backend, _, new = run(GraphSession)

    assert [r.text for r in new] == [r.text for r in old]
    assert [r.awaiting_confirmation for r in new] == [r.awaiting_confirmation for r in old]
    assert [(t, a) for t, a, _ in new_backend.calls] == [(t, a) for t, a, _ in old_backend.calls]
    assert new_backend.prepared == old_backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]


def test_a_model_fault_still_leaves_the_patient_a_reply_and_the_graph_usable() -> None:
    backend = Backend()
    session, model = make(GraphSession, backend, book(clinic_name="Pine Clinic"))
    original = model.decide
    model.decide = lambda transcript: (_ for _ in ()).throw(RuntimeError("provider down"))

    assert session.respond("Book at Pine Clinic.").text == FALLBACK_REPLY

    model.decide = original
    assert "Which date" in session.respond("Book at Pine Clinic.").text


def test_a_turn_that_runs_too_long_ends_with_the_fallback_reply() -> None:
    session, model = make(GraphSession, Backend())
    model.decide = lambda transcript: ToolAction("get_my_appointments", {})

    assert session.respond("hello").text == FALLBACK_REPLY


# --- state transitions ----------------------------------------------------------------------------


def test_the_checkpoint_holds_the_request_the_question_and_the_choices_offered() -> None:
    backend = morning_backend()
    other, _ = make(GraphSession, backend, MORNING)
    other.respond("I need an appointment at Pine Clinic Monday morning.")

    state = other.snapshot()

    assert state["goal"] == {
        "operation": "BOOK",
        "clinic_name": "Pine Clinic",
        "start_date": "2026-11-09",
        "end_date": "2026-11-09",
    }
    assert state["preferences"] == {"time_preference": "MORNING"}
    assert state["clarification"]["awaiting"] == str(Awaiting.TIME)
    assert [o["label"].split(" on")[0] for o in state["offered"]] == [
        "9:00 AM",
        "9:30 AM",
        "11:00 AM",
    ]
    assert all(
        "Monday, November 9, 2026 with Dr. Anita Rao" in o["label"] for o in state["offered"]
    )
    assert state["review"] is None and state["turn"] == 1


def test_the_projection_follows_the_controller_and_never_leads_it() -> None:
    _, session, _ = run(GraphSession)

    state = session.snapshot()
    controller = session.model.state

    assert state["goal"]["clinic_name"] == controller.draft.clinic_name
    assert state["preferences"]["time"] == controller.draft.time == "9:30 AM"
    assert state["review"] == {"pending_action_id": "pending-1"}
    # Changing the checkpoint does not change what the agent does: the controller owns the state.
    session.graph.update_state(session.config, {"goal": {"clinic_name": "Elsewhere"}})
    assert session.model.state.draft.clinic_name == "Pine Clinic"


def test_a_correction_updates_the_goal_and_drops_the_old_review_from_the_checkpoint() -> None:
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    backend.slot("s17", "c-pine", "d-rao", NOV17, 10)
    session, _ = make(
        GraphSession,
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(start_date=NOV17, end_date=NOV17),
    )
    session.respond("Book at Pine Clinic on November 16.")
    assert session.snapshot()["review"] == {"pending_action_id": "pending-1"}

    session.respond("Actually, November 17.")

    state = session.snapshot()
    assert state["goal"]["start_date"] == "2026-11-17"
    assert state["review"] == {"pending_action_id": "pending-2"}


def test_a_reported_commit_clears_the_review_and_the_request() -> None:
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    session, _ = make(
        GraphSession, backend, book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16)
    )
    session.respond("Book at Pine Clinic on November 16.")

    done = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED", "pending_action_id": "pending-1"}
    )

    assert "is booked" in done.text
    state = session.snapshot()
    assert state["review"] is None and state["goal"] == {}


# --- interruption and resumption ------------------------------------------------------------------


def test_a_directory_question_is_recorded_as_an_interruption_and_then_resumed() -> None:
    _, session, _ = run(GraphSession)
    # The second message was the aside; the third went back to the request.
    interruptions = session.snapshot()["interruptions"]
    assert interruptions == [
        {"kind": "directory_question", "turn": 2, "awaiting": "TIME", "resumed": True}
    ]


def test_while_the_aside_is_open_the_request_and_its_question_are_unchanged() -> None:
    backend = morning_backend()
    session, _ = make(GraphSession, backend, MORNING, intent())
    session.respond("I need an appointment at Pine Clinic Monday morning.")
    before = session.snapshot()

    session.respond("Which clinics are available?")
    during = session.snapshot()

    assert during["goal"] == before["goal"] and during["offered"] == before["offered"]
    assert during["clarification"] == before["clarification"]
    assert during["interruptions"][-1]["resumed"] is False


def test_an_aside_with_a_review_waiting_keeps_the_review_reference() -> None:
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    session, _ = make(
        GraphSession,
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(),
    )
    session.respond("Book at Pine Clinic on November 16.")

    session.respond("Which clinics are available?")

    assert session.snapshot()["review"] == {"pending_action_id": "pending-1"}
    assert backend.discarded == []


# --- the review is retired exactly as before ---------------------------------------------------


def test_dropping_the_request_retires_the_review_and_the_checkpoint_forgets_it() -> None:
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    session, _ = make(
        GraphSession,
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(message_type="ABANDON"),
    )
    session.respond("Book at Pine Clinic on November 16.")

    session.respond("Never mind, forget it.")

    assert [pending for _, pending in backend.discarded] == ["pending-1"]
    assert backend.discarded[0][0] is session.principal
    assert session.snapshot()["review"] is None


def test_a_failed_retirement_is_told_to_the_patient_on_the_graph_path_too() -> None:
    backend = RetiringBackend()
    backend.slot("s16", "c-pine", "d-rao", NOV16, 11)
    backend.discard_fails = True
    session, _ = make(
        GraphSession,
        backend,
        book(clinic_name="Pine Clinic", start_date=NOV16, end_date=NOV16),
        intent(message_type="ABANDON"),
    )
    session.respond("Book at Pine Clinic on November 16.")

    reply = session.respond("Never mind, forget it.")

    assert "please don't confirm it in the app" in reply.text


# --- isolation and secrets ------------------------------------------------------------------------


def test_two_sessions_sharing_one_saver_keep_separate_checkpoints() -> None:
    saver = InMemorySaver()
    first, _ = make(GraphSession, morning_backend(), MORNING, checkpointer=saver)
    second, _ = make(
        GraphSession,
        Backend(),
        book(clinic_name="Two Rivers Clinic"),
        checkpointer=saver,
    )

    first.respond("I need an appointment at Pine Clinic Monday morning.")
    second.respond("Book at Two Rivers Clinic.")

    assert first.config != second.config
    assert first.snapshot()["goal"]["clinic_name"] == "Pine Clinic"
    assert second.snapshot()["goal"]["clinic_name"] == "Two Rivers Clinic"
    assert first.snapshot()["clarification"]["awaiting"] == "TIME"
    assert second.snapshot()["clarification"]["awaiting"] == "DATE"


def test_the_thread_is_a_hash_of_the_session_and_never_the_session_id() -> None:
    who = patient()
    thread = thread_id_for(who, "x")

    assert str(who.session_id) not in thread and thread == thread_id_for(who, "other")
    assert thread != thread_id_for(patient(), "x")
    assert thread_id_for(None, "a") != thread_id_for(None, "b")


def test_no_credential_identifier_or_spoken_word_is_ever_checkpointed() -> None:
    who = dataclasses.replace(patient(), csrf_token="csrf-" + uuid.uuid4().hex)
    session, _ = make(
        GraphSession, morning_backend(), MORNING, principal=who, checkpointer=InMemorySaver()
    )
    session.respond("I need an appointment at Pine Clinic Monday morning, thanks very much.")

    stored = session._saver.get_tuple(session.config)
    everything = repr(stored)

    for secret in (who.csrf_token, str(who.session_id), str(who.patient_id), who.login):
        assert secret not in everything
    assert "thanks very much" not in everything
    assert "principal" not in everything and "runtime" not in everything


# --- cleanup --------------------------------------------------------------------------------------


def registry(saver, **options):
    return ConversationRegistry(
        lambda: HybridControllerModel(Script(*[MORNING] * 4), clock=lambda: NOW),
        session_factory=lambda model, runtime, principal: GraphSession(
            model, runtime, principal, checkpointer=saver
        ),
        **options,
    )


def saved(saver, session) -> bool:
    return saver.get_tuple(session.config) is not None


def test_a_registry_hands_back_the_same_graph_session_to_the_same_session() -> None:
    saver, backend, who = InMemorySaver(), morning_backend(), patient()
    sessions = registry(saver)

    one = sessions.session_for(backend, who)
    one.respond("I need an appointment at Pine Clinic Monday morning.")

    assert sessions.session_for(backend, who) is one
    assert isinstance(one, GraphSession) and saved(saver, one)


def test_ending_a_session_deletes_its_checkpoints_and_no_one_elses() -> None:
    saver, backend = InMemorySaver(), morning_backend()
    sessions = registry(saver)
    keep, drop = patient(), patient()
    kept = sessions.session_for(backend, keep)
    dropped = sessions.session_for(backend, drop)
    for item in (kept, dropped):
        item.respond("I need an appointment at Pine Clinic Monday morning.")

    sessions.end(drop)

    assert not saved(saver, dropped) and saved(saver, kept) and len(sessions) == 1


def test_idle_expiry_and_eviction_delete_checkpoints() -> None:
    saver, backend, clock = InMemorySaver(), morning_backend(), [0.0]
    sessions = registry(saver, idle_seconds=60, max_sessions=2, timer=lambda: clock[0])
    a, b, c = patient(), patient(), patient()
    first = sessions.session_for(backend, a)
    first.respond("I need an appointment at Pine Clinic Monday morning.")
    second = sessions.session_for(backend, b)
    second.respond("I need an appointment at Pine Clinic Monday morning.")

    third = sessions.session_for(backend, c)  # Over the limit: the oldest goes.
    third.respond("I need an appointment at Pine Clinic Monday morning.")
    assert not saved(saver, first) and saved(saver, second)

    clock[0] = 120.0  # Everything idle expires on the next lookup.
    sessions.session_for(backend, patient())
    assert not saved(saver, second) and not saved(saver, third)


def test_a_session_id_that_now_belongs_to_another_patient_gets_a_fresh_graph() -> None:
    saver, backend = InMemorySaver(), morning_backend()
    sessions = registry(saver)
    old = patient()
    first = sessions.session_for(backend, old)
    first.respond("I need an appointment at Pine Clinic Monday morning.")
    taken_over = dataclasses.replace(old, patient_id=uuid.uuid4())

    second = sessions.session_for(backend, taken_over)

    assert second is not first and not saved(saver, first)
    assert second.snapshot() == {}


def test_closing_twice_or_before_any_turn_is_harmless() -> None:
    session, _ = make(GraphSession, Backend())

    session.close()
    session.close()

    assert session.snapshot() == {}


def test_the_controller_path_is_unchanged_by_the_registry_hooks() -> None:
    sessions = ConversationRegistry(lambda: HybridControllerModel(Script(), clock=lambda: NOW))
    who = patient()

    session = sessions.session_for(Backend(), who)
    sessions.end(who)

    assert type(session) is TextSession and len(sessions) == 0


@pytest.mark.parametrize("anonymous", ["a", "b"])
def test_anonymous_conversations_have_their_own_threads(anonymous) -> None:
    saver, backend = InMemorySaver(), Backend()
    sessions = registry(saver)

    session = sessions.session_for(backend, None, anonymous_id=anonymous)

    assert session.principal is None and session.config["configurable"]["thread_id"]
