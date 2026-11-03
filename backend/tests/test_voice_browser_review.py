"""Voice mode hands a prepared review to the browser and hears the verified outcome.

No microphone, model, database or browser here: the backend's handoff service is replaced by a
stand-in whose answers the tests choose, so what is asserted is what the voice side does with each
answer. The real database and HTTP flow is in tests/database/test_voice_browser_flow.py."""

import ast
import inspect
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_voice_agent import Screen, agent, transcripts

from app.core.errors import AuthenticationRequiredError, ResourceNotAvailableError
from app.evaluation.conversation import AgentReply
from app.models.appointment_actions import ActionSummary, AppointmentTime
from app.services import review_handoffs as handoffs
from app.voice import browser_review, voice_agent
from app.voice.browser_review import (
    NOT_COMMITTED,
    NOT_READY,
    UNAVAILABLE,
    UNVERIFIED,
    BrowserReviewSession,
)

TOKEN = "session-token-do-not-leak-0123456789"
CODE = "Ab3-xYz_Q9" * 4 + "Zz"  # 42 URL-safe characters
HANDOFF_ID = uuid.uuid4()
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
BOOKED_WORDS = ("booked", "rescheduled", "cancelled", "confirmed")


def appointment_time():
    return AppointmentTime(
        clinic="Lakeside Community Clinic",
        doctor="Dr. Rohan Desai",
        specialty="Internal Medicine",
        date="2026-11-03",
        time="8:00 AM",
        timezone="America/Chicago",
        starts_at_local=NOW,
        starts_at_utc=NOW,
    )


def status(state, result=None):
    summary = ActionSummary(operation="BOOK", description="Book", proposed=appointment_time())
    return handoffs.HandoffStatus(state, HANDOFF_ID, "BOOK", result, summary)


class Backend:
    """Stands in for the handoff service: records what was asked, answers what it is told to."""

    COMMITTED = handoffs.COMMITTED
    WAITING = handoffs.WAITING
    IN_BROWSER = handoffs.IN_BROWSER
    NOT_COMMITTED = handoffs.NOT_COMMITTED
    EXPIRED = handoffs.EXPIRED
    SUPERSEDED = handoffs.SUPERSEDED
    HANDOFF_MINUTES = handoffs.HANDOFF_MINUTES
    HandoffStatus = handoffs.HandoffStatus

    def __init__(self, *states):
        self.states = list(states)
        self.calls = []
        self.fail_create = None

    def create_handoff(self, engine, *, principal, request_id, now):
        self.calls.append("create")
        if self.fail_create:
            raise self.fail_create
        return handoffs.CreatedHandoff(HANDOFF_ID, CODE, NOW)

    def handoff_status(self, engine, *, principal, handoff_id, now):
        self.calls.append("status")
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        if isinstance(state, Exception):
            raise state
        return state

    def supersede_open_handoffs(self, engine, *, principal, request_id, now):
        self.calls.append("supersede")
        return 1


class Auth:
    def __init__(self):
        self.refused = False

    def authenticate(self, token):
        assert token == TOKEN
        if self.refused:
            raise AuthenticationRequiredError()
        return SimpleNamespace(session_id=uuid.uuid4())


class Conversations:
    def __init__(self, reply="Your appointment was booked, as confirmed in the app."):
        self.events, self.turns, self.reply = [], [], reply

    def turn(self, token, utterance):
        self.turns.append(utterance)
        return AgentReply("I prepared booking 8:00 AM.", True)

    def event(self, token, payload):
        self.events.append(payload)
        if isinstance(self.reply, Exception):
            raise self.reply
        return AgentReply(self.reply, False)


def make(monkeypatch, *states, opened=None, conversations=None, base="http://localhost:8000"):
    backend = Backend(*(states or [status(handoffs.WAITING)]))
    monkeypatch.setattr(browser_review, "handoffs", backend)
    auth = Auth()
    conversations = conversations or Conversations()
    session = BrowserReviewSession(
        conversations,
        TOKEN,
        engine=object(),
        auth=auth,
        patient_code="PT-1001",
        base_url=base,
        open_browser=opened,
        clock=lambda: NOW,
    )
    return session, backend, conversations, auth


# --- the link ---------------------------------------------------------------------------------


def test_the_link_carries_the_ticket_in_the_fragment_and_nothing_secret(monkeypatch) -> None:
    opened = []
    session, backend, _, _ = make(monkeypatch, opened=lambda link: opened.append(link) or True)

    notice = session.review_notice(AgentReply("I prepared it.", True))

    link = opened[0]
    assert link == f"http://localhost:8000/voice-review#h={CODE}&p=PT-1001"
    assert "?" not in link and link.index("#") < link.index(CODE)  # Fragment: never sent on.
    assert link in notice and "Opened in your browser" in notice
    assert TOKEN not in notice and TOKEN not in link  # The session token never appears.
    assert "Nothing is booked until you press Confirm in your browser" in notice
    assert backend.calls == ["create"] and session.waiting


def test_without_a_browser_the_link_is_shown_to_open_by_hand(monkeypatch) -> None:
    session, _, _, _ = make(monkeypatch, opened=None)

    notice = session.review_notice(AgentReply("I prepared it.", True))

    assert "Open this link" in notice and CODE in notice


def test_a_browser_that_cannot_be_opened_does_not_stop_the_review(monkeypatch) -> None:
    def broken(link):
        raise OSError("no browser")

    session, _, _, _ = make(monkeypatch, opened=broken)

    notice = session.review_notice(AgentReply("I prepared it.", True))

    assert "Open this link" in notice and session.waiting


def test_a_reply_with_no_review_has_no_link_and_asks_for_nothing(monkeypatch) -> None:
    session, backend, _, _ = make(monkeypatch)

    assert session.review_notice(AgentReply("Pine Clinic has openings.", False)) is None
    assert backend.calls == [] and not session.waiting


def test_when_the_backend_cannot_make_a_ticket_nothing_pretends_it_can(monkeypatch) -> None:
    session, backend, _, _ = make(monkeypatch)
    backend.fail_create = ResourceNotAvailableError()

    notice = session.review_notice(AgentReply("I prepared it.", True))

    assert notice == NOT_READY and not session.waiting and CODE not in notice


def test_the_link_and_code_are_never_logged(monkeypatch, caplog) -> None:
    session, backend, _, _ = make(monkeypatch)
    backend.fail_create = RuntimeError(f"boom {CODE} {TOKEN}")

    with caplog.at_level(logging.DEBUG):
        session.review_notice(AgentReply("I prepared it.", True))

    logged = " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert CODE not in logged and TOKEN not in logged and "RuntimeError" in logged


# --- the outcome ------------------------------------------------------------------------------


def test_nothing_is_said_while_the_review_waits_in_the_browser(monkeypatch) -> None:
    session, _, conversations, _ = make(
        monkeypatch, status(handoffs.WAITING), status(handoffs.IN_BROWSER)
    )
    session.review_notice(AgentReply("I prepared it.", True))

    assert session.poll_outcome() is None and session.poll_outcome() is None
    assert conversations.events == [] and session.waiting


def test_a_verified_commit_is_announced_and_the_conversation_is_told(monkeypatch) -> None:
    session, _, conversations, _ = make(monkeypatch, status(handoffs.COMMITTED, "BOOKED"))
    session.review_notice(AgentReply("I prepared it.", True))

    said = session.poll_outcome()

    assert said.startswith("Your appointment was booked, as confirmed in the app.")
    assert "Dr. Rohan Desai at Lakeside Community Clinic on Tuesday, November 3 at 8:00 AM" in said
    assert conversations.events == [{"type": "APPOINTMENT_COMMITTED", "result": "BOOKED"}]
    assert not session.waiting and session.poll_outcome() is None  # Said once.


def test_a_commit_is_still_announced_if_the_event_cannot_be_delivered(monkeypatch) -> None:
    conversations = Conversations(reply=RuntimeError("store down"))
    session, _, _, _ = make(
        monkeypatch, status(handoffs.COMMITTED, "RESCHEDULED"), conversations=conversations
    )
    session.review_notice(AgentReply("I prepared it.", True))

    said = session.poll_outcome()

    assert said.startswith("Your appointment was rescheduled, as confirmed in the app.")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (handoffs.NOT_COMMITTED, NOT_COMMITTED),
        (handoffs.EXPIRED, browser_review.EXPIRED),
        (handoffs.SUPERSEDED, UNAVAILABLE),
    ],
)
def test_every_outcome_but_a_verified_commit_says_nothing_was_changed(
    monkeypatch, state, expected
) -> None:
    session, _, conversations, _ = make(monkeypatch, status(state))
    session.review_notice(AgentReply("I prepared it.", True))

    said = session.poll_outcome()

    assert said == expected and "nothing was changed" in said.lower()
    assert not any(word in said.lower().replace("nothing was changed", "") for word in BOOKED_WORDS)
    assert conversations.events == []  # The conversation hears of a commit, and only of a commit.


def test_a_review_the_backend_does_not_know_is_reported_unavailable(monkeypatch) -> None:
    session, _, conversations, _ = make(monkeypatch, ResourceNotAvailableError())
    session.review_notice(AgentReply("I prepared it.", True))

    assert session.poll_outcome() == UNAVAILABLE and not session.waiting
    assert conversations.events == []


def test_a_session_that_ended_cannot_vouch_for_a_commit(monkeypatch) -> None:
    session, _, conversations, auth = make(monkeypatch, status(handoffs.COMMITTED, "BOOKED"))
    session.review_notice(AgentReply("I prepared it.", True))
    auth.refused = True

    said = session.poll_outcome()

    assert said == UNVERIFIED and session.ended and conversations.events == []
    assert "booked" not in said.lower()


def test_a_database_blink_while_waiting_is_not_an_outcome(monkeypatch) -> None:
    session, _, conversations, _ = make(monkeypatch, RuntimeError("connection reset"))
    session.review_notice(AgentReply("I prepared it.", True))

    assert session.poll_outcome() is None and session.waiting and conversations.events == []


# --- speaking again, and ending --------------------------------------------------------------


def test_speaking_again_retires_a_review_still_waiting_before_the_new_turn(monkeypatch) -> None:
    session, backend, conversations, _ = make(monkeypatch, status(handoffs.IN_BROWSER))
    session.review_notice(AgentReply("I prepared it.", True))

    reply = session.respond("actually, Thursday")

    assert backend.calls == ["create", "status", "supersede"]  # Retired before the turn ran.
    assert conversations.turns == ["actually, Thursday"] and not session.waiting
    assert reply.text == "I prepared booking 8:00 AM."


def test_a_commit_that_landed_just_before_the_next_turn_is_announced_first(monkeypatch) -> None:
    session, backend, conversations, _ = make(monkeypatch, status(handoffs.COMMITTED, "BOOKED"))
    session.review_notice(AgentReply("I prepared it.", True))

    reply = session.respond("thanks")

    assert reply.text.startswith("Your appointment was booked, as confirmed in the app.")
    assert "supersede" not in backend.calls and len(conversations.events) == 1


def test_ending_the_session_retires_whatever_the_browser_holds(monkeypatch) -> None:
    session, backend, _, _ = make(monkeypatch)
    session.review_notice(AgentReply("I prepared it.", True))

    session.end_review()

    assert backend.calls[-1] == "supersede" and not session.waiting


def test_a_failure_while_retiring_never_stops_a_sign_out(monkeypatch) -> None:
    session, backend, _, _ = make(monkeypatch)
    session.review_notice(AgentReply("I prepared it.", True))
    backend.supersede_open_handoffs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db"))

    session.end_review()  # Does not raise.

    assert not session.waiting


# --- the voice loop ---------------------------------------------------------------------------


class WaitingSession:
    """A session whose browser outcome arrives after a few checks."""

    def __init__(self, outcomes, reply=None):
        self.outcomes, self.polled, self.heard = list(outcomes), 0, []
        self.reply = reply or AgentReply("Done.", False)
        self.noticed = []

    @property
    def waiting(self):
        return bool(self.outcomes)

    def poll_outcome(self):
        self.polled += 1
        return self.outcomes.pop(0)

    def respond(self, utterance):
        self.heard.append(utterance)
        return self.reply

    def review_notice(self, reply):
        self.noticed.append(reply)
        return "(Open the link: http://localhost:8000/voice-review#h=x)"


def loop(session, screen, listen, *, key=lambda seconds: False):
    spoken = []
    speaker = SimpleNamespace(speaking=False, speak=lambda text: spoken.append(text) or 1.0)
    voice = agent(session, screen, listen, speaker)
    voice._key_pressed = key
    return voice, spoken


def test_the_loop_waits_for_the_browser_without_recording_then_says_the_outcome() -> None:
    session = WaitingSession([None, None, "Your appointment was booked, as confirmed in the app."])
    screen = Screen(["q"])
    recorded = []
    voice, spoken = loop(
        session,
        screen,
        lambda config: recorded.append(1) or transcripts("hello")(config),
    )
    lines = []
    voice._say = lines.append

    voice.run_turn()

    assert session.polled == 3 and spoken == [
        "Your appointment was booked, as confirmed in the app."
    ]
    assert any("Waiting for you to confirm or decline in the browser" in line for line in lines)
    assert recorded == [1]  # The microphone was opened only after the wait, never during it.


def test_pressing_enter_while_waiting_goes_back_to_speaking_without_an_announcement() -> None:
    session = WaitingSession([None, None, None])
    voice, spoken = loop(session, Screen([""]), transcripts("never mind"), key=lambda seconds: True)

    result = voice.run_turn()

    assert session.polled == 1 and spoken[:1] == ["Done."]
    assert session.heard == ["never mind"] and result.heard == "never mind"


def test_a_prepared_review_prints_the_sessions_link_and_the_legacy_notice_is_the_fallback() -> None:
    session = WaitingSession([], reply=AgentReply("I prepared booking 8:00 AM.", True))
    voice, _ = loop(session, Screen([""]), transcripts("book it"))
    lines = []
    voice._say = lines.append

    voice.run_turn()

    assert any("voice-review#h=x" in line for line in lines)
    assert not any(voice_agent.NEEDS_THE_APP in line for line in lines)

    plain = SimpleNamespace(respond=lambda text: AgentReply("Prepared.", True))
    voice2, _ = loop(plain, Screen([""]), transcripts("book it"))
    lines2 = []
    voice2._say = lines2.append
    voice2.run_turn()
    assert any(voice_agent.NEEDS_THE_APP in line for line in lines2)


def test_a_session_with_no_browser_support_is_untouched_by_the_wait() -> None:
    plain = SimpleNamespace(respond=lambda text: AgentReply("Hello.", False))
    voice, spoken = loop(plain, Screen([""]), transcripts("hi"))

    assert voice.run_turn().reply == "Hello." and spoken == ["Hello."]


# --- configuration and boundaries -------------------------------------------------------------


def test_the_review_page_must_be_on_an_origin_the_backend_allows() -> None:
    allowed = ["http://localhost:8000", "http://127.0.0.1:8000"]
    args = SimpleNamespace(review_url=None)

    assert voice_agent.review_base_url(args, allowed) == "http://localhost:8000"
    args.review_url = "http://127.0.0.1:8000/"
    assert voice_agent.review_base_url(args, allowed) == "http://127.0.0.1:8000"
    args.review_url = "http://evil.example"
    with pytest.raises(voice_agent.SpeechError, match="ALLOWED_ORIGINS"):
        voice_agent.review_base_url(args, allowed)


def test_the_voice_side_has_no_path_to_commit_or_to_the_confirmation_endpoint() -> None:
    source = Path(browser_review.__file__).read_text()
    imports = {
        name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom | ast.Import)
        for name in [getattr(node, "module", None) or "", *[a.name for a in node.names]]
    }

    assert not {"appointment_transactions", "confirm_appointment_action"} & imports
    assert "sqlalchemy" not in " ".join(imports) and "/confirm" not in source
    assert "confirm_appointment_action" not in inspect.getsource(handoffs)
    assert "appointment_transactions" not in inspect.getsource(handoffs).split('"""')[2]


def test_the_only_event_the_voice_side_sends_is_a_commit_and_only_after_the_database_says_so() -> (
    None
):
    tree = ast.parse(Path(browser_review.__file__).read_text())
    callers = [
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "event"
    ]
    poll = inspect.getsource(BrowserReviewSession.poll_outcome)

    assert callers == ["_announce_commit"]
    assert poll.index("handoffs.COMMITTED") < poll.index("_announce_commit")
    assert '"APPOINTMENT_COMMITTED"' in inspect.getsource(BrowserReviewSession._announce_commit)
