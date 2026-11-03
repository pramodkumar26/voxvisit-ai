"""LG-5F voice mode, stage 1, without a model, a microphone or a database: the adapter between the
voice loop and `GraphConversations`, the mode switch, and the boundaries the adapter must keep."""

import dataclasses
import inspect
import logging
from types import SimpleNamespace

import pytest
from test_voice_agent import Screen, agent, transcripts

from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.evaluation.conversation import AgentReply
from app.evaluation.llm import LlmConfig
from app.voice import lg5f_session, voice_agent
from app.voice.lg5f_session import (
    SESSION_ENDED_REPLY,
    TROUBLE_REPLY,
    GraphVoiceSession,
    voice_config,
)
from app.voice.stt import SpeechError

TOKEN = "tok-4f9c2a7e-session-secret"


class Conversations:
    """Stands in for GraphConversations: records what it is given, answers or raises."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def turn(self, token, utterance):
        self.calls.append((token, utterance))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def event(self, token, payload):  # Present so a test can prove it is never used.
        raise AssertionError("the voice session must never report a confirmation")


# --- the adapter ------------------------------------------------------------------------------


def test_a_turn_goes_to_the_orchestrator_with_the_session_token_and_returns_its_reply() -> None:
    reply = AgentReply("Pine Clinic has openings.", False)
    conversations = Conversations(reply)

    got = GraphVoiceSession(conversations, TOKEN).respond("what does Pine have")

    assert got is reply and conversations.calls == [(TOKEN, "what does Pine have")]


def test_the_session_token_is_not_shown_by_the_object() -> None:
    session = GraphVoiceSession(Conversations(), TOKEN)

    assert TOKEN not in repr(session) and TOKEN not in str(session)


@pytest.mark.parametrize("refusal", [AuthenticationRequiredError(), SessionExpiredError()])
def test_a_session_the_backend_refuses_ends_the_conversation_and_calls_nothing_again(
    refusal,
) -> None:
    conversations = Conversations(refusal)
    session = GraphVoiceSession(conversations, TOKEN)

    first = session.respond("book Pine on Monday")
    again = session.respond("yes")

    assert first.text == again.text == SESSION_ENDED_REPLY
    assert not first.awaiting_confirmation and session.ended
    assert len(conversations.calls) == 1  # The second turn never reached the backend or a model.
    assert "sign in again" in first.text and "Nothing was changed" in first.text


def test_any_other_failure_gets_a_plain_reply_and_the_session_carries_on(caplog) -> None:
    conversations = Conversations(
        RuntimeError(f"boom {TOKEN} book Dr. Rao"), AgentReply("Back again.", False)
    )
    session = GraphVoiceSession(conversations, TOKEN)

    with caplog.at_level(logging.DEBUG):
        failed = session.respond("book Dr. Rao")
        after = session.respond("hello")

    assert failed.text == TROUBLE_REPLY and not failed.awaiting_confirmation
    assert not session.ended and after.text == "Back again."
    logged = " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert "RuntimeError" in logged  # Only the kind of failure is recorded...
    assert TOKEN not in logged and "Dr. Rao" not in logged and "boom" not in logged


def test_the_loop_says_the_session_has_ended_once_and_then_stops_listening() -> None:
    session = GraphVoiceSession(Conversations(SessionExpiredError()), TOKEN)
    screen, spoken = Screen(["", ""]), []
    speaker = SimpleNamespace(speaking=False, speak=lambda text: spoken.append(text) or 1.0)
    voice = agent(session, screen, transcripts("book Pine Clinic", "never reached"), speaker)

    voice.run()

    assert spoken == [SESSION_ENDED_REPLY]
    assert len(screen.prompts) == 1  # The second recording was never asked for.


def test_the_adapter_has_no_path_to_confirm_or_to_the_commit_service() -> None:
    source = inspect.getsource(lg5f_session)
    imports = " ".join(
        line for line in source.splitlines() if line.lstrip().startswith(("from ", "import "))
    )

    assert ".event(" not in source and "confirmation_event" not in source
    assert "appointment_transactions" not in imports and "confirm_appointment_action" not in source
    assert "/confirm" not in source
    conversations = Conversations(AgentReply("I prepared the change.", True))
    GraphVoiceSession(conversations, TOKEN).respond("book it")  # event() would raise if used.


# --- configuration ----------------------------------------------------------------------------


def test_voice_changes_only_the_waiting_limits_of_the_validated_model_settings() -> None:
    assert voice_config() == dataclasses.replace(
        LlmConfig(), request_timeout_s=30.0, keep_alive="30m"
    )
    assert (voice_config().temperature, voice_config().seed) == (0.0, 7)


# --- the mode switch --------------------------------------------------------------------------


def run_main(monkeypatch, argv, environment=None):
    seen = {}

    def build(arguments):
        seen["agent"] = arguments.agent
        raise SpeechError("stop here")

    monkeypatch.setattr(voice_agent, "build", build)
    monkeypatch.setattr("app.voice.stt.WhisperCli.check", lambda self: "/bin/whisper-cli")
    monkeypatch.delenv("VOXVISIT_VOICE_AGENT", raising=False)
    for name, value in (environment or {}).items():
        monkeypatch.setenv(name, value)
    code = voice_agent.main(["--model", "m.bin", *argv])
    return code, seen


def test_lg5f_is_the_default(monkeypatch) -> None:
    code, seen = run_main(monkeypatch, ["--patient-code", "PT-1001"])

    assert code == 2 and seen == {"agent": "lg5f"}


def test_either_agent_can_be_selected_and_the_flag_wins(monkeypatch) -> None:
    assert run_main(monkeypatch, ["--agent", "lg5f"])[1] == {"agent": "lg5f"}
    assert run_main(monkeypatch, [], {"VOXVISIT_VOICE_AGENT": "legacy"})[1] == {"agent": "legacy"}
    assert run_main(monkeypatch, ["--agent", "legacy"], {"VOXVISIT_VOICE_AGENT": "lg5f"})[1] == {
        "agent": "legacy"
    }


def test_an_unknown_agent_is_refused_before_anything_starts(monkeypatch) -> None:
    with pytest.raises(SystemExit) as refused:
        run_main(monkeypatch, [], {"VOXVISIT_VOICE_AGENT": "experimental"})

    assert refused.value.code == 2


def test_lg5f_without_a_patient_code_stops_before_any_model_or_database_is_touched(
    monkeypatch, capsys
) -> None:
    def forbidden(*_a, **_k):
        raise AssertionError("must not be reached")

    monkeypatch.setattr("app.db.session.create_database_engine", forbidden)
    monkeypatch.setattr("app.evaluation.cli.served_model_digest", forbidden)
    monkeypatch.setattr("app.voice.stt.WhisperCli.check", lambda self: "/bin/whisper-cli")

    code = voice_agent.main(["--model", "m.bin"])

    assert code == 2 and "--patient-code" in capsys.readouterr().err


def test_start_up_failures_of_the_conversation_store_and_a_missing_package_are_named() -> None:
    import psycopg

    store = voice_agent.describe_startup_failure(psycopg.OperationalError("password=hunter2"))
    package = voice_agent.describe_startup_failure(ImportError("langgraph"))

    assert "conversation store could not be opened" in store and "hunter2" not in store
    assert "dev group" in package
