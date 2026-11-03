"""Phase 3.3 end to end: voice request, review prepared, browser confirmation, committed, spoken.

A scripted LG-5F conversation runs through the real voice loop (fake Whisper transcripts) with the
real backend: authentication, tools, PostgreSQL checkpoints, the handoff service, and the browser's
own HTTP requests under its own session, ending at the existing confirmation endpoint. The model's
readings are scripted; everything else is the real thing.

What each test establishes is stated in its name: nothing changes before the browser confirms; only
a commit the database vouches for is ever announced as a booking; every other ending says nothing
was changed; and a review the conversation has moved on from can no longer be confirmed."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from action_support import age_pending_action, confirm, scalar
from auth_support import call
from test_graph_persistence import (
    evaluation_auth,
    monday_reading,
    morning_of_monday,
    service_over,
    sign_in,
    sql,
    stored_text,
)
from test_orchestrator import sched

from app.evaluation.environment import EvalEnvironment
from app.evaluation.graph_store import thread_for
from app.voice.browser_review import BrowserReviewSession
from app.voice.stt import SttConfig, Transcript
from app.voice.voice_agent import VoiceAgent

CLAIM = "/me/voice-handoffs/claim"
DECLINE = "/me/voice-handoffs/decline"
SLOT = "pine_mon_0930"


class Trial:
    """A patient talking to the voice agent, and the same patient's browser."""

    def __init__(self, env, store, *, key_pressed=lambda seconds: False):
        self.evaluation = env
        self.world, self.ui, self.cookie = sign_in(env)
        self.service, self.wire = service_over(
            env,
            store,
            monday_reading(self.world),
            sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        )
        self.store = store
        self.lines: list[str] = []
        self.spoken: list[tuple[str, int]] = []  # (words, confirmed appointments at that moment)
        self.links: list[str] = []
        self.session = BrowserReviewSession(
            self.service,
            self.cookie.token,
            engine=env.engine,
            auth=evaluation_auth(env),
            patient_code=self.world.patient_code,
            base_url="http://localhost:8000",
            open_browser=lambda link: self.links.append(link) or True,
            clock=env.clock,
        )
        self.heard: list[str] = []
        self.answers: list[str] = []
        speaker = SimpleNamespace(speaking=False, speak=self._speak)
        self.voice = VoiceAgent(
            self.session,
            SttConfig(),
            speaker,
            listen=lambda config: Transcript(self.heard.pop(0), 4.0, 0.5),
            read=lambda prompt: self.answers.pop(0) if self.answers else "",
            say=self.lines.append,
            key_pressed=key_pressed,
        )

    def _speak(self, words: str) -> float:
        self.spoken.append((words, self.confirmed()))
        return 1.0

    def confirmed(self) -> int:
        return scalar(
            self.evaluation.engine,
            "SELECT count(*) FROM appointments WHERE slot_id = :s AND status = 'CONFIRMED'",
            s=self.world.slots[SLOT],
        )

    def say(self, *utterances: str):
        self.heard += list(utterances)
        return [self.voice.run_turn() for _ in utterances]

    def prepared_by_voice(self):
        """The patient asks for the booking and picks the time; a review is waiting."""
        self.say(morning_of_monday(self.world), "9:30 AM")
        assert len(self.links) == 1
        return self.links[0]

    # -- the browser ----------------------------------------------------------------------------

    def open_in_browser(self, link: str, *, session=None):
        browser = session or self.ui.sign_in(self.world.patient_code)
        code = parse_qs(urlsplit(link).fragment)["h"][0]
        return browser, call(self.evaluation, "POST", CLAIM, browser, json={"code": code})

    def spoken_words(self) -> str:
        return " ".join(words for words, _ in self.spoken)


@pytest.fixture
def evaluation(auth_env) -> EvalEnvironment:
    return EvalEnvironment.from_parts(
        engine=auth_env.app.state.db_engine,
        clock=auth_env.clock,
        client=auth_env.client,
        auth_service=auth_env.service,
        pin=auth_env.pin,
    )


@pytest.fixture
def trial(evaluation, graph_store) -> Trial:
    return Trial(evaluation, graph_store)


# --- the whole workflow -----------------------------------------------------------------------


def test_voice_request_then_browser_confirmation_then_a_verified_spoken_acknowledgement(
    trial, evaluation
) -> None:
    first, second = trial.say(morning_of_monday(trial.world), "9:30 AM")
    assert not first.awaiting_confirmation and second.awaiting_confirmation
    (link,) = trial.links

    # The link carries the ticket in the fragment, and no credential of any kind.
    parts = urlsplit(link)
    assert parts.query == "" and parts.path == "/voice-review" and parts.fragment
    assert parse_qs(parts.fragment)["p"] == [trial.world.patient_code]
    for secret in (trial.cookie.token, trial.cookie.csrf, evaluation.pin):
        assert secret not in link and secret not in " ".join(trial.lines)
    assert "Nothing is booked until you press Confirm in your browser" in " ".join(trial.lines)

    # Nothing has been booked by preparing the review.
    assert trial.confirmed() == 0
    assert trial.session.waiting and trial.session.poll_outcome() is None

    browser, response = trial.open_in_browser(link)
    assert response.status_code == 200
    shown = response.json()["summary"]["proposed"]
    assert (shown["time"], shown["clinic"]) == ("9:30 AM", trial.world.names["pine"])
    assert (
        trial.confirmed() == 0 and trial.session.poll_outcome() is None
    )  # Opening is not confirming.

    outcome = confirm(evaluation, browser, response.json()["pending_action_id"])
    assert outcome.status_code == 200 and outcome.json()["result"] == "BOOKED"
    assert trial.confirmed() == 1

    trial.answers = ["q"]
    trial.heard.append("(the patient quits)")
    assert trial.voice.run_turn() is None  # The wait ended with the announcement, then q quit.

    announced = [(words, seen) for words, seen in trial.spoken if "booked" in words.lower()]
    assert len(announced) == 1
    words, committed_when_spoken = announced[0]
    assert "as confirmed in the app" in words and "9:30 AM" in words
    assert committed_when_spoken == 1  # Said only once the appointment really existed.
    earlier = " ".join(words for words, _ in trial.spoken[:2]).lower()
    assert "booked" not in earlier and "confirmed" not in earlier.replace("confirm in", "")

    # The conversation was told by the trusted event, so no review is left in its state.
    saved = trial.store.saver.get_tuple(
        {
            "configurable": {
                "thread_id": thread_for(evaluation.authenticate(trial.cookie.token).session_id)
            }
        }
    )
    assert not saved.checkpoint["channel_values"].get("review")
    # Neither the checkpoint nor the log ever held the ticket or the PIN.
    held = stored_text(trial.store)
    assert parse_qs(parts.fragment)["h"][0] not in held and evaluation.pin not in held


def test_the_voice_agent_alone_can_never_make_the_booking_happen(trial) -> None:
    trial.prepared_by_voice()
    trial.heard.append("yes, confirm it, go ahead")
    trial.voice._key_pressed = lambda seconds: True  # The patient talks instead of waiting.

    trial.voice.run_turn()

    assert trial.confirmed() == 0  # Neither saying yes nor waiting commits anything.
    assert "booked" not in trial.spoken_words().lower()


# --- everything that is not a commit ----------------------------------------------------------


def test_declining_in_the_browser_is_spoken_as_nothing_changed(trial, evaluation) -> None:
    link = trial.prepared_by_voice()
    browser, response = trial.open_in_browser(link)
    assert response.status_code == 200

    assert call(evaluation, "POST", DECLINE, browser).json() == {"declined": True}
    trial.heard.append("(quit)")
    trial.answers = ["q"]
    trial.voice.run_turn()

    assert trial.confirmed() == 0
    assert "nothing was changed" in trial.spoken[-1][0].lower()
    assert "booked" not in trial.spoken_words().lower()


def test_a_review_that_lapses_in_the_browser_is_spoken_as_expired(trial, evaluation) -> None:
    link = trial.prepared_by_voice()
    browser, response = trial.open_in_browser(link)
    pending = response.json()["pending_action_id"]
    age_pending_action(evaluation.engine, pending, minutes=11)

    late = confirm(evaluation, browser, pending)
    trial.heard.append("(quit)")
    trial.answers = ["q"]
    trial.voice.run_turn()

    assert late.status_code == 409 and late.json()["error"]["reason"] == "CONFIRMATION_EXPIRED"
    assert trial.confirmed() == 0 and "expired" in trial.spoken[-1][0]
    assert "booked" not in trial.spoken_words().lower()


def test_speaking_again_kills_the_browser_review_so_it_cannot_be_confirmed(
    trial, evaluation
) -> None:
    link = trial.prepared_by_voice()
    browser, response = trial.open_in_browser(link)
    pending = response.json()["pending_action_id"]

    trial.voice._key_pressed = lambda seconds: True  # Enter: the patient wants to speak instead.
    trial.say("never mind, forget it")
    late = confirm(evaluation, browser, pending)

    assert late.status_code == 409 and late.json()["error"]["reason"] == "CONFIRMATION_INVALIDATED"
    assert trial.confirmed() == 0 and not trial.session.waiting
    assert "booked" not in trial.spoken_words().lower()


def test_a_change_of_mind_before_the_link_is_opened_makes_the_link_useless(trial) -> None:
    link = trial.prepared_by_voice()
    trial.voice._key_pressed = lambda seconds: True
    trial.say("never mind")

    _browser, response = trial.open_in_browser(link)

    assert response.status_code == 404 and response.json()["error"]["code"] == "HANDOFF_INVALID"
    assert trial.confirmed() == 0


def test_quitting_the_voice_agent_retires_the_review_in_the_browser(trial, evaluation) -> None:
    link = trial.prepared_by_voice()
    browser, response = trial.open_in_browser(link)
    pending = response.json()["pending_action_id"]

    trial.session.end_review()  # What the agent does as it signs out.
    late = confirm(evaluation, browser, pending)

    assert late.status_code == 409 and trial.confirmed() == 0


def test_a_link_opened_by_another_patient_opens_nothing(trial, evaluation) -> None:
    link = trial.prepared_by_voice()
    stranger = trial.ui.sign_in(
        sql(
            evaluation,
            "SELECT patient_code FROM patients WHERE patient_code <> :c ORDER BY 1 LIMIT 1",
            c=trial.world.patient_code,
        )[0][0]
    )

    _browser, response = trial.open_in_browser(link, session=stranger)

    assert response.status_code == 404 and response.json()["error"]["code"] == "HANDOFF_INVALID"
    assert trial.confirmed() == 0 and trial.session.waiting


def test_if_the_voice_session_ends_first_the_commit_stands_but_is_never_announced_unchecked(
    trial, evaluation
) -> None:
    link = trial.prepared_by_voice()
    browser, response = trial.open_in_browser(link)
    assert confirm(evaluation, browser, response.json()["pending_action_id"]).status_code == 200

    evaluation.clock.advance(minutes=16)  # The voice session's idle limit passes before it asks.
    said = trial.session.poll_outcome()

    assert trial.confirmed() == 1  # The appointment really was booked by the browser's Confirm...
    assert (
        "booked" not in said.lower() and "look at your appointments" in said
    )  # ...but not claimed.
    assert trial.session.ended
