"""LG-5F voice mode, stage 1, end to end except the microphone and the model.

Fake Whisper transcripts go through the real voice loop and the real adapter into the real
`GraphConversations`, authentication, tools, PostgreSQL checkpoints and the UI-only confirm driver.
The model's readings are scripted (the real model is measured by the LG-5 evaluations); what is
under test is that a spoken conversation behaves as the typed one does, and that voice adds no way
to commit, to resume a dead session, or to leak a credential."""

import logging
import uuid
from types import SimpleNamespace

import pytest
from test_graph_persistence import (
    active_reviews,
    monday_reading,
    morning_of_monday,
    service_over,
    sign_in,
    sql,
    stored_text,
)
from test_orchestrator import Wire, ask, book, sched
from test_voice_agent import Screen, agent, transcripts

from app.evaluation.environment import EvalEnvironment
from app.evaluation.llm import ModelTimeoutError, ModelUnavailableError
from app.voice.lg5f_session import SESSION_ENDED_REPLY, TROUBLE_REPLY, GraphVoiceSession

THURSDAY, FRIDAY, MONDAY = "2026-11-05", "2026-11-06", "2026-11-09"
SLOT_TIME = {"time": "9:30 AM", "selection": "SPECIFIC_TIME"}


@pytest.fixture
def evaluation(auth_env) -> EvalEnvironment:
    return EvalEnvironment.from_parts(
        engine=auth_env.app.state.db_engine,
        clock=auth_env.clock,
        client=auth_env.client,
        auth_service=auth_env.service,
        pin=auth_env.pin,
    )


class Call:
    """One signed-in synthetic patient talking by (fake) voice.

    `script` is a function of the patient's world that returns the model's scripted readings, one
    per model call, so a reading can name this world's own clinics."""

    def __init__(self, evaluation, store, script=lambda world: (), transport=None):
        self.evaluation, self.store = evaluation, store
        self.world, self.ui, self.cookie = sign_in(evaluation)
        self.service, self.wire = service_over(
            evaluation, store, *script(self.world), transport=transport
        )
        self.session = GraphVoiceSession(self.service, self.cookie.token)
        self.spoken: list[str] = []
        self.screens: list[Screen] = []

    def say(self, *heard: str, answers=None):
        """Whisper 'hears' each text; the person accepts it as it is unless `answers` says."""
        screen = Screen(answers or [""] * len(heard))
        self.screens.append(screen)
        speaker = SimpleNamespace(
            speaking=False, speak=lambda words: self.spoken.append(words) or 1.0
        )
        voice = agent(self.session, screen, transcripts(*heard), speaker)
        return [voice.run_turn() for _ in heard]

    def listen_until_it_stops(self, *heard: str) -> Screen:
        screen = Screen([""] * len(heard))
        speaker = SimpleNamespace(
            speaking=False, speak=lambda words: self.spoken.append(words) or 1.0
        )
        agent(self.session, screen, transcripts(*heard), speaker).run()
        return screen

    def appointments(self, slot: str) -> int:
        return sql(
            self.evaluation,
            "SELECT count(*) FROM appointments WHERE slot_id = :s",
            s=self.world.slots[slot],
        )[0][0]

    def reviews(self) -> list[str]:
        return active_reviews(self.evaluation, self.world.patient_id)

    def sign_out_elsewhere(self) -> None:
        principal = self.evaluation.authenticate(self.cookie.token)
        self.evaluation.client.app.state.auth_service.logout(principal, request_id=uuid.uuid4())

    @property
    def model_readings(self) -> int:
        return len(self.wire.payloads)


def pine_monday_then_930(world):
    return [monday_reading(world), sched(**SLOT_TIME)]


# --- a spoken conversation is the same conversation -------------------------------------------


def test_a_spoken_booking_continues_across_turns_prepares_a_review_and_commits_nothing(
    evaluation, graph_store
) -> None:
    call = Call(evaluation, graph_store, pine_monday_then_930)

    first, second = call.say(morning_of_monday(call.world), "9:30 AM")

    assert not first.awaiting_confirmation
    assert second.awaiting_confirmation and len(call.reviews()) == 1
    assert "9:30 AM" in second.reply and "confirm in the app" in second.reply
    assert call.appointments("pine_mon_0930") == 0
    assert "booked" not in second.reply.lower()
    assert len(graph_store.threads()) == 1  # One conversation, in the checkpoint store.


def test_a_review_confirmed_in_the_app_is_not_announced_by_voice_without_the_trusted_event(
    evaluation, graph_store
) -> None:
    call = Call(evaluation, graph_store, lambda w: [*pine_monday_then_930(w), sched()])
    call.say(morning_of_monday(call.world), "9:30 AM")
    (pending,) = call.reviews()

    assert call.ui.confirm(call.cookie, pending).committed  # The authenticated UI, not voice.
    (after,) = call.say("thanks")

    assert call.appointments("pine_mon_0930") == 1
    assert "booked" not in after.reply.lower() and not after.awaiting_confirmation


def test_saying_yes_or_confirm_by_voice_commits_nothing(evaluation, graph_store) -> None:
    call = Call(evaluation, graph_store, pine_monday_then_930)
    call.say(morning_of_monday(call.world), "9:30 AM")

    (heard,) = call.say("yes, confirm it, go ahead")

    assert call.appointments("pine_mon_0930") == 0 and len(call.reviews()) == 1
    assert "app" in heard.reply.lower() and not call.session.ended


# --- lookup versus booking, spoken ------------------------------------------------------------


def test_eh10_a_clinic_and_day_after_the_clinic_list_is_a_lookup_not_a_review(
    evaluation, graph_store
) -> None:
    def script(world):
        return [
            ask("CLINICS"),
            book(clinic_name=world.names["heart"], start_date=MONDAY),
            {"asks_for": "BOOK"},  # A second reading that would agree: it must not be asked for.
        ]

    call = Call(evaluation, graph_store, script)

    listed, heard = call.say("Which clinics do you have?", f"{call.world.names['heart']}, Monday.")

    assert not listed.awaiting_confirmation
    assert not heard.awaiting_confirmation and "1:00 PM" in heard.reply
    assert call.reviews() == [] and call.appointments("heart_mon_1300") == 0
    assert call.model_readings == 2  # One reading per message: no second model look.


def test_an_indirect_spoken_request_to_be_fitted_in_starts_a_booking(
    evaluation, graph_store
) -> None:
    def script(world):
        return [book(clinic_name=world.names["rivers"], start_date=THURSDAY, end_date=THURSDAY)]

    call = Call(evaluation, graph_store, script)

    (heard,) = call.say(f"Can you fit me in at {call.world.names['rivers']} on Thursday?")

    assert heard.awaiting_confirmation and len(call.reviews()) == 1
    assert "10:00 AM" in heard.reply and call.appointments("rivers_thu_1000") == 0


def test_a_spoken_yes_to_the_assistants_own_offer_to_book_starts_the_booking(
    evaluation, graph_store
) -> None:
    def script(world):
        listing = ask(
            "OPENINGS", clinic_name=world.names["rivers"], start_date=FRIDAY, end_date=FRIDAY
        )
        return [listing]  # The yes is settled by rules: no second reading is scripted or needed.

    call = Call(evaluation, graph_store, script)

    listed, yes = call.say(f"Are there openings at {call.world.names['rivers']} on Friday?", "Yes.")

    assert not listed.awaiting_confirmation and call.model_readings == 1
    assert yes.awaiting_confirmation and len(call.reviews()) == 1
    assert "1:00 PM" in yes.reply and call.appointments("rivers_fri_1300") == 0


def test_a_spoken_yes_with_nothing_offered_prepares_nothing(evaluation, graph_store) -> None:
    call = Call(evaluation, graph_store, lambda world: [sched(message_type="AGREEMENT")])

    (heard,) = call.say("Yes")

    assert not heard.awaiting_confirmation and call.reviews() == []


def test_a_spoken_correction_replaces_the_review_and_the_old_one_cannot_be_confirmed(
    evaluation, graph_store
) -> None:
    def script(world):
        rivers = world.names["rivers"]
        return [
            book(clinic_name=rivers, start_date=THURSDAY, end_date=THURSDAY),
            sched(same_request=True, clinic_name=rivers, start_date=FRIDAY, end_date=FRIDAY),
        ]

    call = Call(evaluation, graph_store, script)

    first, fixed = call.say(f"Book {call.world.names['rivers']} on Thursday", "Actually, Friday.")

    assert first.awaiting_confirmation and fixed.awaiting_confirmation
    assert "1:00 PM" in fixed.reply and "10:00 AM" not in fixed.reply
    (current,) = call.reviews()  # Only the new review is live.
    states = sql(
        evaluation,
        "SELECT p.status FROM pending_actions p JOIN auth_sessions s USING (session_id)"
        " WHERE s.patient_id = :p ORDER BY p.created_at",
        p=call.world.patient_id,
    )
    assert [s[0] for s in states] == ["INVALIDATED", "ACTIVE"]
    assert call.appointments("rivers_thu_1000") == 0 and call.appointments("rivers_fri_1300") == 0
    assert call.ui.confirm(call.cookie, current).committed
    assert call.appointments("rivers_fri_1300") == 1  # The corrected one, and only through the UI.


def test_what_the_person_edits_after_hearing_is_what_the_orchestrator_reads(
    evaluation, graph_store
) -> None:
    def script(world):
        return [book(clinic_name=world.names["rivers"], start_date=THURSDAY, end_date=THURSDAY)]

    call = Call(evaluation, graph_store, script)
    corrected = f"Book {call.world.names['rivers']} on Thursday"

    (sent,) = call.say("Book river clinic on the fifth", answers=["e", corrected, ""])

    assert sent.heard == corrected and sent.awaiting_confirmation
    assert "river clinic" not in str(call.wire.payloads)


# --- sessions and models that fail ------------------------------------------------------------


def test_an_expired_session_is_told_once_stops_the_loop_and_leaves_no_usable_review(
    evaluation, graph_store
) -> None:
    call = Call(evaluation, graph_store, pine_monday_then_930)
    call.say(morning_of_monday(call.world), "9:30 AM")
    (pending,) = call.reviews()
    readings = call.model_readings

    evaluation.clock.advance(minutes=16)  # Past the idle timeout.
    screen = call.listen_until_it_stops("yes", "never reached")

    assert call.spoken[-1] == SESSION_ENDED_REPLY and call.session.ended
    assert len(screen.prompts) == 1 and call.model_readings == readings  # No model was reached.
    assert not call.ui.confirm(call.cookie, pending).committed
    assert call.appointments("pine_mon_0930") == 0 and call.reviews() == []


def test_a_session_signed_out_elsewhere_cannot_be_spoken_into(evaluation, graph_store) -> None:
    call = Call(evaluation, graph_store, lambda world: [monday_reading(world)])
    call.say(morning_of_monday(call.world))
    call.sign_out_elsewhere()
    readings = call.model_readings

    (heard,) = call.say("Book Pine Clinic on Monday")

    assert heard.reply == SESSION_ENDED_REPLY and call.session.ended
    assert call.model_readings == readings and call.reviews() == []


@pytest.mark.parametrize("failure", [ModelUnavailableError(), ModelTimeoutError()])
def test_a_model_that_fails_or_times_out_gets_a_spoken_reply_and_changes_nothing(
    evaluation, graph_store, failure
) -> None:
    call = Call(evaluation, graph_store, transport=Wire(failure, failure, failure))

    (heard,) = call.say(morning_of_monday(call.world))

    assert heard.reply and not heard.awaiting_confirmation
    assert call.spoken == [heard.reply]  # The patient is never left in silence.
    assert call.reviews() == [] and call.appointments("pine_mon_0900") == 0
    assert not call.session.ended


def test_a_conversation_store_that_cannot_be_used_fails_closed_with_a_spoken_reply(
    evaluation, graph_store
) -> None:
    call = Call(evaluation, graph_store, lambda world: [monday_reading(world)])
    graph_store.close()  # The checkpoint database goes away.

    (heard,) = call.say(morning_of_monday(call.world))

    assert heard.reply == TROUBLE_REPLY and not heard.awaiting_confirmation
    assert call.reviews() == [] and not call.session.ended


# --- credentials and cleanup ------------------------------------------------------------------


def test_a_spoken_pin_never_reaches_the_model_the_checkpoint_or_the_log(
    evaluation, graph_store, caplog
) -> None:
    call = Call(evaluation, graph_store, lambda world: [monday_reading(world)])
    call.say(morning_of_monday(call.world))
    readings = call.model_readings

    with caplog.at_level(logging.DEBUG):
        (heard,) = call.say("My pin is four eight two nine one three, sign me in, 482913")

    assert call.model_readings == readings  # The credential was screened before any model.
    assert "482913" not in heard.reply
    held = stored_text(graph_store)
    for forbidden in (
        call.cookie.token,
        call.cookie.csrf,
        evaluation.pin,
        "482913",
        "four eight two",
        call.world.patient_code,
    ):
        assert forbidden not in held, forbidden
    logged = " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert "482913" not in logged and call.cookie.token not in logged
    for payload in call.wire.payloads:
        assert evaluation.pin not in str(payload) and call.cookie.token not in str(payload)


def test_signing_out_removes_the_voice_conversation_and_nothing_resumes(
    evaluation, graph_store
) -> None:
    call = Call(evaluation, graph_store, lambda world: [monday_reading(world)])
    call.say(morning_of_monday(call.world))
    assert len(graph_store.threads()) == 1

    principal = evaluation.authenticate(call.cookie.token)
    call.sign_out_elsewhere()
    call.service.sign_out(principal)  # What the voice agent's release step does.

    assert graph_store.threads() == []
    (heard,) = call.say("hello again")
    assert heard.reply == SESSION_ENDED_REPLY and graph_store.threads() == []


def test_the_real_wiring_signs_in_runs_a_turn_and_cleans_up_at_sign_out(
    auth_env, monkeypatch
) -> None:
    """`open_graph_voice` as the launcher uses it: the real login service, a store opened from the
    engine, the verified token. Only the model's warm-up is stubbed; the credential turn below is
    screened before any model is asked."""
    from app.evaluation.graph_store import CheckpointStore
    from app.evaluation.turn_reader import TurnReader
    from app.voice.lg5f_session import open_graph_voice
    from app.voice.voice_agent import sign_in as voice_sign_in

    warmed = []
    monkeypatch.setattr(TurnReader, "warm_up", lambda self, today: warmed.append(today) or 0.0)
    issued = voice_sign_in(auth_env.service, "PT-1001", auth_env.pin)
    engine = auth_env.app.state.db_engine

    session, release = open_graph_voice(engine, auth_env.service, issued, clock=auth_env.clock)
    try:
        reply = session.respond("my pin is 482913, sign me in")
        assert warmed and reply.text and not reply.awaiting_confirmation
        assert repr(session) == "GraphVoiceSession()" and issued.token not in repr(session)
    finally:
        auth_env.service.logout(issued.principal, request_id=uuid.uuid4())
        release()

    with CheckpointStore(engine.url) as check:
        assert check.threads() == []  # The conversation went with the sign-out.
    assert session.respond("hello").text == SESSION_ENDED_REPLY and session.ended
