"""The voice front end of the text agent: checking what was heard, speaking replies, timing, and
the safety boundaries. No microphone, no speech, no model and no database."""

import inspect
import subprocess
from types import SimpleNamespace

import pytest

from app.evaluation.conversation import AgentReply, TextSession
from app.evaluation.model import Say
from app.voice import tts, voice_agent
from app.voice.stt import SpeechError, SttConfig, Transcript
from app.voice.tts import Speaker, TtsConfig
from app.voice.voice_agent import NEEDS_THE_APP, VoiceAgent, sign_in


class Session:
    def __init__(self, text="Done.", confirm=False):
        self.heard, self.text, self.confirm = [], text, confirm

    def respond(self, utterance):
        self.heard.append(utterance)
        return AgentReply(self.text, self.confirm)


class Screen:
    def __init__(self, answers=()):
        self.answers = list(answers)
        self.lines, self.prompts = [], []

    def say(self, text):
        self.lines.append(text)

    def read(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def transcripts(*texts):
    queue = [Transcript(t, 4.0, 0.5) for t in texts]
    return lambda config: queue.pop(0)


def agent(session, screen, listen, speaker=None, clock=None):
    speaker = speaker or SimpleNamespace(speaking=False, speak=lambda text: 1.5)
    ticks = iter(range(0, 1000))
    return VoiceAgent(
        session, SttConfig(), speaker, listen=listen, read=screen.read, say=screen.say,
        timer=clock or (lambda: float(next(ticks))),
    )  # fmt: skip


# --- checking what was heard -------------------------------------------------------------------


def test_nothing_is_sent_until_the_person_accepts_what_was_heard() -> None:
    session, screen = Session(), Screen([""])

    result = agent(session, screen, transcripts("Book me on November seventeenth")).run_turn()

    assert session.heard == ["Book me on November seventeenth"]
    assert result.heard == "Book me on November seventeenth"
    assert 'Heard: "Book me on November seventeenth"' in "\n".join(screen.lines)


def test_the_heard_text_is_sent_exactly_as_heard_never_corrected() -> None:
    session, screen = Session(), Screen([""])

    agent(session, screen, transcripts("with Dr. Nair on Nov 17th at ten")).run_turn()

    assert session.heard == ["with Dr. Nair on Nov 17th at ten"]


def test_names_dates_and_times_get_a_reminder_to_check() -> None:
    screen = Screen([""])
    agent(Session(), screen, transcripts("Cancel with Doctor Kim on Friday")).run_turn()
    plain = Screen([""])
    agent(Session(), plain, transcripts("Hello there")).run_turn()

    assert any("Check names, dates and times" in line for line in screen.lines)
    assert not any("Check names" in line for line in plain.lines)


def test_the_person_can_edit_the_text_and_the_edit_is_what_is_sent() -> None:
    session, screen = Session(), Screen(["e", "Book with Dr. Nair on November 17", ""])

    agent(session, screen, transcripts("Book with Dr. Nare on November 70")).run_turn()

    assert session.heard == ["Book with Dr. Nair on November 17"]
    assert 'Heard: "Book with Dr. Nair on November 17"' in "\n".join(screen.lines)  # Shown again.


def test_an_empty_edit_keeps_the_original_and_sends_nothing_by_itself() -> None:
    session, screen = Session(), Screen(["e", "", ""])

    agent(session, screen, transcripts("Hello")).run_turn()

    assert session.heard == ["Hello"]
    assert any("keeping the original" in line for line in screen.lines)


def test_the_person_can_record_again_and_only_the_new_text_is_sent() -> None:
    session, screen = Session(), Screen(["r", ""])

    agent(
        session, screen, transcripts("Book me on the sixth", "Book me on the sixteenth")
    ).run_turn()

    assert session.heard == ["Book me on the sixteenth"]


def test_the_person_can_type_instead_and_can_quit_at_the_review() -> None:
    session = Session()
    assert (
        agent(session, Screen(["t", "Cancel it"]), transcripts("garbled")).run_turn().heard
        == "Cancel it"
    )
    quit_session = Session()
    assert agent(quit_session, Screen(["q"]), transcripts("anything")).run_turn() is None
    assert quit_session.heard == []


def test_an_unknown_choice_asks_again_and_sends_nothing() -> None:
    session, screen = Session(), Screen(["x", ""])

    agent(session, screen, transcripts("Hello")).run_turn()

    assert session.heard == ["Hello"]
    assert any("choose one of the options" in line for line in screen.lines)


# --- when transcription fails ------------------------------------------------------------------


def test_a_failed_recording_falls_back_to_typing() -> None:
    session, screen = Session(), Screen(["Cancel my appointment"])

    def broken(config):
        raise SpeechError("No speech was heard (the recording is silent).")

    result = agent(session, screen, broken).run_turn()

    assert session.heard == ["Cancel my appointment"] and result.timings.stt_seconds == 0.0
    assert any("No speech was heard" in line for line in screen.lines)


def test_after_a_failure_enter_tries_the_microphone_again_and_q_quits() -> None:
    calls = []

    def flaky(config):
        calls.append(1)
        if len(calls) == 1:
            raise SpeechError("The audio contained no recognizable speech.")
        return Transcript("Hello", 2.0, 0.3)

    session = Session()
    assert agent(session, Screen(["", ""]), flaky).run_turn().heard == "Hello" and len(calls) == 2

    def always(config):
        raise SpeechError("broken")

    assert agent(Session(), Screen(["q"]), always).run_turn() is None


# --- speaking and the microphone ---------------------------------------------------------------


def test_the_reply_is_spoken_after_it_is_shown() -> None:
    spoken = []
    speaker = SimpleNamespace(speaking=False, speak=lambda text: spoken.append(text) or 2.0)
    screen = Screen([""])

    result = agent(Session("Your visit is at ten."), screen, transcripts("hi"), speaker).run_turn()

    assert spoken == ["Your visit is at ten."] and result.spoken
    assert "Assistant: Your visit is at ten." in "\n".join(screen.lines)


def test_a_speech_failure_leaves_the_text_reply_and_the_conversation_going() -> None:
    def fail(text):
        raise SpeechError("The text-to-speech command 'say' was not found.")

    screen = Screen([""])
    result = agent(
        Session("Hello."), screen, transcripts("hi"), SimpleNamespace(speaking=False, speak=fail)
    ).run_turn()

    assert not result.spoken and "Assistant: Hello." in "\n".join(screen.lines)
    assert any("Speech output unavailable" in line for line in screen.lines)


def test_the_microphone_is_never_opened_while_the_assistant_is_speaking() -> None:
    events = []

    class Voice:
        speaking = False

        def speak(self, text):
            self.speaking = True
            events.append("speak-start")
            self.speaking = False
            events.append("speak-end")
            return 1.0

    voice = Voice()

    def listen(config):
        assert not voice.speaking
        events.append("record")
        return Transcript("hi", 1.0, 0.2)

    machine = agent(Session(), Screen(["", "", "q"]), listen, voice)
    machine.run_turn()
    machine.run_turn()

    assert events == ["record", "speak-start", "speak-end", "record", "speak-start", "speak-end"]
    voice.speaking = True
    with pytest.raises(RuntimeError, match="during playback"):
        machine.run_turn()


def test_say_is_run_without_a_shell_with_the_text_on_standard_input(monkeypatch) -> None:
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/say")
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    speaker = Speaker(TtsConfig(voice="Samantha", rate=180), runner=run)
    text = "Dr. Rao; $(rm -rf ~) --voice x `ls`"

    speaker.speak(text)

    assert seen["command"] == ["/usr/bin/say", "-v", "Samantha", "-r", "180", "-f", "-"]
    assert seen["input"] == text and not seen.get("shell")  # Text never reaches the command line.
    assert text not in seen["command"] and speaker.speaking is False


@pytest.mark.parametrize("voice", ["Samantha; rm -rf /", "-v x", "$(id)", "a" * 80, ""])
def test_an_unsafe_voice_name_is_refused(voice) -> None:
    with pytest.raises(SpeechError, match="voice name"):
        Speaker(TtsConfig(voice=voice))


def test_a_bad_rate_a_failure_and_a_timeout_are_clear_errors(monkeypatch) -> None:
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/say")
    with pytest.raises(SpeechError, match="rate"):
        Speaker(TtsConfig(rate=5))

    def failing(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "secret")

    with pytest.raises(SpeechError, match="exit code 1") as error:
        Speaker(TtsConfig(), runner=failing).speak("hello")
    assert "secret" not in str(error.value)

    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired("say", 1)

    speaker = Speaker(TtsConfig(), runner=slow)
    with pytest.raises(SpeechError, match="too long"):
        speaker.speak("hello")
    assert speaker.speaking is False  # Not left stuck on after a failure.


def test_a_missing_say_command_is_explained(monkeypatch) -> None:
    monkeypatch.setattr(tts.shutil, "which", lambda name: None)

    with pytest.raises(SpeechError, match="need macOS"):
        Speaker(TtsConfig()).speak("hello")


def test_nothing_is_said_for_empty_text_and_long_text_is_cut_at_a_word() -> None:
    speaker = Speaker(TtsConfig(), runner=lambda *a, **k: pytest.fail("said nothing"))
    assert speaker.speak("   ") == 0.0
    long = "word " * 1000
    assert len(tts.shorten(long)) <= tts.MAX_SPOKEN_CHARS and tts.shorten(long).endswith("word")


# --- timing ------------------------------------------------------------------------------------


def test_each_stage_is_timed_separately_and_the_person_s_own_time_is_kept_out() -> None:
    clock = iter([10.0, 25.0, 25.0, 28.0])  # review starts/ends, agent starts/ends
    spoken = SimpleNamespace(speaking=False, speak=lambda text: 4.0)

    result = agent(
        Session(), Screen([""]), transcripts("hi"), spoken, clock=lambda: next(clock)
    ).run_turn()

    t = result.timings
    assert t.recording_seconds == 4.0 and t.stt_seconds == 0.5
    assert t.review_seconds == 15.0  # Reading and approving: not machine time.
    assert t.agent_seconds == 3.0 and t.speech_seconds == 4.0
    assert t.time_to_first_word == 3.5 and t.machine_total == 7.5


# --- safety ------------------------------------------------------------------------------------


def test_a_prepared_change_is_reported_as_needing_the_app_and_is_never_confirmed() -> None:
    session = Session("I prepared booking. Please confirm in the app.", confirm=True)
    screen = Screen([""])
    session.confirmation_event = lambda payload: pytest.fail("the terminal confirmed something")

    result = agent(session, screen, transcripts("Book me tomorrow")).run_turn()

    assert result.awaiting_confirmation and NEEDS_THE_APP in "\n".join(screen.lines)


def test_saying_yes_is_just_text_for_the_agent_and_changes_nothing_here() -> None:
    class Model:
        name = "stub"
        state = SimpleNamespace(prepared=object())

        def decide(self, transcript):
            return Say(frozenset(), "I can't confirm for you. Please press Confirm in the app.")

    real = TextSession(Model(), runtime=None, principal=None)
    screen = Screen([""])

    result = agent(real, screen, transcripts("Yes, confirm it")).run_turn()

    assert "can't confirm" in result.reply and result.awaiting_confirmation


def test_the_same_session_carries_the_conversation_across_voice_turns() -> None:
    session = Session()
    machine = agent(session, Screen(["", "", ""]), transcripts("Book me at Pine", "Tomorrow"))

    machine.run_turn()
    machine.run_turn()

    assert session.heard == ["Book me at Pine", "Tomorrow"]  # One session object, in order.


def test_the_module_has_no_path_to_confirmation_or_to_the_database() -> None:
    source = inspect.getsource(voice_agent)
    imports = [line for line in source.splitlines() if line.startswith(("from ", "import "))]
    text = " ".join(imports)
    assert "appointment_transactions" not in text and "confirm_appointment_action" not in source
    assert "confirmation_event" not in source and "/confirm" not in source
    assert "sqlalchemy" not in text


def test_the_pin_goes_to_the_login_service_only_and_never_to_the_agent() -> None:
    calls = []

    class Auth:
        def login(self, kind, code, pin, **kwargs):
            calls.append((kind, code, pin))
            return SimpleNamespace(
                issued=SimpleNamespace(principal="p", token="t"), rate_limited=False
            )

    issued = sign_in(Auth(), "PT-1001", "123456")
    session = Session()
    agent(session, Screen([""]), transcripts("Book me tomorrow")).run_turn()

    assert calls == [("PATIENT", "PT-1001", "123456")] and issued.token == "t"
    assert all("123456" not in heard for heard in session.heard)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SimpleNamespace(issued=None, rate_limited=False), "not accepted"),
        (SimpleNamespace(issued=None, rate_limited=True, retry_after_seconds=90), "90 s"),
    ],
)
def test_a_refused_sign_in_is_a_clear_error_that_repeats_no_secret(result, message) -> None:
    auth = SimpleNamespace(login=lambda *a, **k: result)

    with pytest.raises(SpeechError, match=message) as error:
        sign_in(auth, "PT-1001", "654321")

    assert "654321" not in str(error.value)


def test_nothing_is_logged(caplog) -> None:
    with caplog.at_level("DEBUG"):
        agent(Session("Private reply."), Screen([""]), transcripts("Private words.")).run_turn()

    assert "Private" not in caplog.text


# --- start-up failures -------------------------------------------------------------------------


def test_sign_in_asks_the_login_service_for_the_patient_actor_by_its_own_constant() -> None:
    """The first real start-up failed with a KeyError: the service looks actors up by "PATIENT",
    and the voice agent had passed "patient". A stand-in login service hid it."""
    from app.auth.service import ACTORS, PATIENT

    asked = []

    def login(kind, *args, **kwargs):
        asked.append(kind)
        return SimpleNamespace(issued=SimpleNamespace(), rate_limited=False)

    sign_in(SimpleNamespace(login=login), "PT-1001", "000000")

    assert asked == [PATIENT] and asked[0] in ACTORS


def test_each_kind_of_start_up_failure_is_named_without_secrets_or_values() -> None:
    from pydantic import SecretStr, ValidationError
    from sqlalchemy.exc import OperationalError

    from app.core.config import Settings

    try:
        Settings(_env_file=None, database_url=SecretStr("not-a-url"), database_pool_size="many")
    except ValidationError as invalid:
        configuration = voice_agent.describe_startup_failure(invalid)
    database = voice_agent.describe_startup_failure(
        OperationalError("SELECT secret", {"password": "hunter2"}, Exception("hunter2"))
    )
    refused = voice_agent.describe_startup_failure(SpeechError("Sign-in failed: not accepted."))
    unexpected = voice_agent.describe_startup_failure(KeyError("PATIENT-SECRET-VALUE"))

    assert "configuration is invalid" in configuration and "many" not in configuration
    assert "database could not be used" in database and "hunter2" not in database
    assert refused == "Sign-in failed: not accepted."
    assert "unexpected failure" in unexpected and "KeyError" in unexpected
    assert "SECRET-VALUE" not in unexpected


def test_a_start_up_failure_prints_one_line_and_exits_with_code_2(monkeypatch, capsys) -> None:
    def explode(arguments):
        raise KeyError("do-not-print-this")

    monkeypatch.setattr(voice_agent, "build", explode)
    monkeypatch.setattr("app.voice.stt.WhisperCli.check", lambda self: "/bin/whisper-cli")

    code = voice_agent.main(["--model", "m.bin", "--patient-code", "PT-1001"])

    shown = capsys.readouterr()
    assert code == 2 and "unexpected failure at start-up (KeyError)" in shown.err
    assert "do-not-print-this" not in shown.err + shown.out and "Traceback" not in shown.err
