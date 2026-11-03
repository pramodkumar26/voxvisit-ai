"""The local voice agent: speak, check what was heard, send it to the text agent, speak the reply.

    PYTHONPATH=backend uv run --group voice python -m app.voice.voice_agent \
        --model ~/models/ggml-base.en.bin --patient-code PT-1001

This is only a front end for `TextSession`. The conversation, the tools, the authorization and the
database are exactly those of the text agent; this module adds a microphone, a speaker, a check on
what was heard, and timing. It can read availability and prepare a change, and it can do nothing
else: confirming an appointment change is a separate authenticated request that the browser
interface will make, and nothing here, spoken or typed, can make it.

Two agents sit behind this loop. The default is the LG-5F LangGraph orchestrator, through
`app.voice.lg5f_session`; it needs a signed-in patient. `--agent legacy` (or
VOXVISIT_VOICE_AGENT=legacy) keeps the original text agent explicitly available for compatibility.
Everything below the loop is the same.

Secrets: a PIN is read with `getpass` (not echoed), goes only to the login service, and is never
given to the agent, a log, or a file. The session token stays in memory. Transcripts are printed
and sent to the agent, never logged or saved; audio is deleted by the speech-to-text layer.
"""

import argparse
import getpass
import os
import re
import select
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.auth.service import PATIENT
from app.voice.stt import SpeechError, SttConfig, Transcript, listen_once
from app.voice.tts import Speaker, TtsConfig

REVIEW_PROMPT = "[Enter] send   [e] edit   [r] record again   [t] type instead   [q] quit: "
# A reminder when the transcript holds what Whisper most often gets wrong.
CRITICAL = re.compile(
    r"\b(dr\.?|doctor|\d|january|february|march|april|may|june|july|august|september|october|"
    r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|"
    r"morning|afternoon|noon|o'clock|quarter|half|a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
WAIT_SECONDS = 1.5  # How often the browser's outcome is checked while the patient is away.
WAITING_NOTICE = (
    "Waiting for you to confirm or decline in the browser. "
    "Press Enter to cancel that review and speak instead."
)
NEEDS_THE_APP = (
    "A change is ready for review. Confirming it needs the authenticated confirmation screen; "
    "it cannot be done by voice or from this terminal."
)


class Session(Protocol):
    """What the loop needs. A session may also offer `review_notice(reply)`, `waiting` and
    `poll_outcome()`: the way a review is handed to a browser and its verified outcome heard."""

    def respond(self, utterance: str) -> Any: ...


@dataclass
class TurnTimings:
    recording_seconds: float = 0.0
    stt_seconds: float = 0.0  # Transcription only.
    review_seconds: float = 0.0  # You reading and approving the text: not machine time.
    agent_seconds: float = 0.0
    speech_seconds: float = 0.0  # Playback of the reply.

    @property
    def time_to_first_word(self) -> float:
        """From the end of your recording to the start of the spoken reply."""
        return self.stt_seconds + self.agent_seconds

    @property
    def machine_total(self) -> float:
        return self.stt_seconds + self.agent_seconds + self.speech_seconds


@dataclass
class TurnResult:
    heard: str
    reply: str
    awaiting_confirmation: bool
    timings: TurnTimings
    spoken: bool


class VoiceAgent:
    def __init__(
        self,
        session: Session,
        stt_config: SttConfig,
        speaker: Speaker,
        *,
        listen: Callable[[SttConfig], Transcript] = listen_once,
        read: Callable[[str], str] = input,
        say: Callable[[str], None] = print,
        timer: Callable[[], float] = time.perf_counter,
        key_pressed: Callable[[float], bool] | None = None,
    ) -> None:
        self.session, self.stt_config, self.speaker = session, stt_config, speaker
        self._listen, self._read, self._say, self._timer = listen, read, say, timer
        self._key_pressed = key_pressed or _stdin_ready

    # -- one turn ------------------------------------------------------------------------------

    def run_turn(self) -> TurnResult | None:
        """One spoken exchange; None when the person quits."""
        timings = TurnTimings()
        self._await_browser()
        accepted = self._get_text(timings)
        if accepted is None:
            return None
        started = self._timer()
        reply = self.session.respond(accepted)  # The text agent, unchanged.
        timings.agent_seconds = self._timer() - started
        self._say(f"\nAssistant: {reply.text}")
        if reply.awaiting_confirmation:
            self._say(self._review_notice(reply))
        spoken = self._speak(reply.text, timings)
        self._say(self._summary(timings))
        return TurnResult(accepted, reply.text, reply.awaiting_confirmation, timings, spoken)

    def run(self) -> None:
        # A session the backend has ended (`ended`) is told to the person once, then the loop stops.
        while self.run_turn() is not None and not getattr(self.session, "ended", False):
            pass

    # -- a review waiting in the browser -------------------------------------------------------

    def _review_notice(self, reply: Any) -> str:
        """Where to confirm. A session that hands reviews to a browser says how; any other says
        only that the app is needed."""
        notice = getattr(self.session, "review_notice", None)
        return (notice(reply) if callable(notice) else None) or f"({NEEDS_THE_APP})"

    def _await_browser(self) -> None:
        """While a review waits in the browser, listen for its outcome, not for the microphone.

        Returns when the outcome is known (it is said aloud) or when the patient presses Enter to
        speak instead (which retires the review, on the next turn). Nothing is recorded here."""
        shown = False
        while getattr(self.session, "waiting", False):
            outcome = self.session.poll_outcome()
            if outcome:
                self._announce(outcome)
                return
            if not shown:
                self._say(f"\n{WAITING_NOTICE}")
                shown = True
            if self._key_pressed(WAIT_SECONDS):
                return

    def _announce(self, text: str) -> None:
        self._say(f"\nAssistant: {text}")
        self._speak(text, TurnTimings())

    # -- getting the person's words, checked ---------------------------------------------------

    def _get_text(self, timings: TurnTimings) -> str | None:
        while True:
            if self.speaker.speaking:  # Never record while the assistant is talking.
                raise RuntimeError("the microphone was opened during playback")
            try:
                heard = self._listen(self.stt_config)
            except SpeechError as error:
                self._say(f"\nCould not use the recording: {error}")
                typed = self._typed()
                if typed is not _RETRY:
                    return typed
                continue
            timings.recording_seconds = heard.recording_seconds
            timings.stt_seconds = heard.processing_seconds
            outcome = self._review(heard.text, timings)
            if outcome is not _RETRY:
                return outcome

    def _review(self, text: str, timings: TurnTimings) -> Any:
        """Show what was heard. Nothing is sent until the person accepts, edits, or types."""
        waited = self._timer()
        try:
            while True:
                self._say(f'\nHeard: "{text}"')
                if CRITICAL.search(text):
                    self._say("Check names, dates and times carefully: speech recognition can err.")
                choice = self._read(REVIEW_PROMPT).strip().lower()
                if choice == "":
                    return text
                if choice == "e":
                    edited = self._read("Type the corrected text: ").strip()
                    if edited:
                        text = edited
                        continue
                    self._say("Nothing typed; keeping the original.")
                elif choice == "r":
                    return _RETRY
                elif choice == "t":
                    return self._typed()
                elif choice == "q":
                    return None
                else:
                    self._say("Please choose one of the options.")
        finally:
            timings.review_seconds += self._timer() - waited

    def _typed(self) -> Any:
        typed = self._read("Type your message ([Enter] to record again, q to quit): ").strip()
        if typed.lower() == "q":
            return None
        if not typed:
            return _RETRY
        return typed

    # -- speaking ------------------------------------------------------------------------------

    def _speak(self, text: str, timings: TurnTimings) -> bool:
        try:
            timings.speech_seconds = self.speaker.speak(text)
        except SpeechError as error:
            self._say(f"(Speech output unavailable: {error})")
            return False
        return True

    @staticmethod
    def _summary(t: TurnTimings) -> str:
        return (
            f"[recorded {t.recording_seconds:.1f} s | you reviewed {t.review_seconds:.1f} s | "
            f"STT {t.stt_seconds:.2f} s | agent {t.agent_seconds:.2f} s | "
            f"speech {t.speech_seconds:.1f} s | to first word {t.time_to_first_word:.2f} s | "
            f"machine total {t.machine_total:.2f} s]"
        )


_RETRY = object()


def _stdin_ready(timeout: float) -> bool:
    """Whether a line is waiting on the terminal, without reading it (the recording step will)."""
    try:
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except (OSError, ValueError):  # No usable terminal: wait the interval and carry on.
        time.sleep(timeout)
        return False


# --- signing in, and the real wiring ----------------------------------------------------------


def sign_in(auth: Any, patient_code: str, pin: str) -> Any:
    """Verify the patient with the same login service the web interface uses.

    The PIN goes to the login service and nowhere else. Returns the issued session.
    """
    result = auth.login(PATIENT, patient_code, pin, network_address=None, request_id=uuid.uuid4())
    if result.rate_limited:
        raise SpeechError(
            f"Too many sign-in attempts; try again in {result.retry_after_seconds} s."
        )
    if result.issued is None:
        raise SpeechError("Sign-in failed: the patient code or PIN was not accepted.")
    return result.issued


AGENTS = ("legacy", "lg5f")


def build(arguments: argparse.Namespace):
    """Wire the real text agent. Imported late: it needs the database and the local model."""
    import httpx

    from app.agent_tools import ToolRuntime
    from app.auth.passwords import create_hasher
    from app.auth.service import AuthService
    from app.core.config import get_settings
    from app.db.session import create_database_engine
    from app.evaluation.cli import served_model_digest
    from app.evaluation.conversation import TextSession
    from app.evaluation.hybrid import HybridControllerModel, StructuredIntentExtractor
    from app.evaluation.llm import LlmConfig

    agent = getattr(arguments, "agent", "lg5f")
    if agent == "lg5f" and not arguments.patient_code:
        raise SpeechError("LG-5F voice mode talks only to a signed-in patient: use --patient-code.")
    config = LlmConfig()
    try:
        if served_model_digest(config) is None:
            raise SpeechError(f"The local model server has no model '{config.model}'.")
    except httpx.HTTPError:
        raise SpeechError("The local model server (Ollama) cannot be reached.") from None
    settings = get_settings()
    engine = create_database_engine(settings)
    secret = (
        settings.auth_secret_key.get_secret_value().encode()
        if settings.auth_secret_key
        else b"x" * 32
    )
    auth = AuthService(engine, secret_key=secret, hasher=create_hasher())
    issued = None
    if arguments.patient_code:
        pin = getpass.getpass("PIN (not shown, never sent to the assistant): ")
        issued = sign_in(auth, arguments.patient_code, pin)
        del pin
    principal = issued.principal if issued else None
    release = None
    try:
        if agent == "lg5f":
            from app.voice.browser_review import BrowserReviewSession, system_browser
            from app.voice.lg5f_session import open_graph_voice

            base = review_base_url(arguments, settings.allowed_origins)

            def with_browser_review(conversations, token):
                return BrowserReviewSession(
                    conversations,
                    token,
                    engine=engine,
                    auth=auth,
                    patient_code=arguments.patient_code,
                    base_url=base,
                    open_browser=None if arguments.no_browser else system_browser,
                )

            session, release = open_graph_voice(
                engine, auth, issued, session_factory=with_browser_review
            )
        else:
            model = HybridControllerModel(StructuredIntentExtractor(config))
            session = TextSession(model, ToolRuntime(engine), principal)
    except BaseException:
        if principal is not None:
            auth.logout(principal, request_id=uuid.uuid4())  # Nothing is left signed in.
        engine.dispose()
        raise

    def close() -> None:
        try:
            end_review = getattr(session, "end_review", None)
            if end_review is not None:
                end_review()  # Whatever was handed to a browser stops working first.
            if principal is not None:
                auth.logout(principal, request_id=uuid.uuid4())  # Also retires any open review.
        finally:
            try:
                if release is not None:
                    release()  # After sign-out: the conversation's saved state goes with it.
            finally:
                engine.dispose()

    return session, close, issued is not None


def review_base_url(arguments: argparse.Namespace, allowed_origins: Sequence[str]) -> str:
    """Where the review page is served. It must be an origin the backend already allows, or the
    page's own requests would be refused (the Origin check) and no review could be confirmed."""
    base = (getattr(arguments, "review_url", None) or "http://localhost:8000").rstrip("/")
    if base.lower() not in allowed_origins:
        raise SpeechError(
            f"The review page address {base} is not one of the backend's ALLOWED_ORIGINS; "
            "use --review-url with an allowed origin."
        )
    return base


def describe_startup_failure(error: BaseException) -> str:
    """Say which kind of start-up failure this is, without a secret, a value or a traceback."""
    from pydantic import ValidationError
    from sqlalchemy.exc import SQLAlchemyError

    if isinstance(error, SpeechError):
        return str(error)  # Already written for the person: a missing file, a refused sign-in.
    if isinstance(error, ValidationError):
        fields = ", ".join(sorted({str(e["loc"][0]) for e in error.errors() if e["loc"]}))
        return (
            f"configuration is invalid or incomplete (check .env: {fields or 'unknown setting'})."
        )
    if isinstance(error, SQLAlchemyError):
        return "the database could not be used; check that PostgreSQL is running and DATABASE_URL."
    if isinstance(error, ImportError):
        return "a required package is missing; LG-5F voice mode needs the project's dev group."
    if type(error).__module__.split(".")[0] in ("psycopg", "psycopg_pool"):
        return "the conversation store could not be opened; check that PostgreSQL is running."
    return f"unexpected failure at start-up ({type(error).__name__}); this is a bug, not a setting."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local voice agent (whisper.cpp + macOS say)")
    parser.add_argument("--whisper-cli")
    parser.add_argument("--model", help="ggml model file (or VOXVISIT_WHISPER_MODEL)")
    parser.add_argument("--patient-code", help="sign in as this synthetic patient (PIN is asked)")
    parser.add_argument("--voice", help="a `say -v ?` voice name")
    parser.add_argument("--rate", type=int, help="speaking rate, words per minute")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--review-url",
        default=os.environ.get("VOXVISIT_REVIEW_BASE_URL"),
        help="where the backend serves the review page (default http://localhost:8000); "
        "must be one of its ALLOWED_ORIGINS. Also VOXVISIT_REVIEW_BASE_URL.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="show the review link but do not open the browser for you (lg5f mode)",
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default=os.environ.get("VOXVISIT_VOICE_AGENT", "lg5f"),
        help="lg5f: the default LangGraph orchestrator (needs --patient-code). "
        "legacy: the original text agent for compatibility. Also VOXVISIT_VOICE_AGENT.",
    )
    arguments = parser.parse_args(argv)
    if arguments.agent not in AGENTS:
        parser.error(f"VOXVISIT_VOICE_AGENT must be one of: {', '.join(AGENTS)}")
    try:
        stt_config = SttConfig.from_environment(
            whisper_cli=arguments.whisper_cli,
            model_path=arguments.model,
            max_record_seconds=arguments.max_seconds,
        )
        speaker = Speaker(TtsConfig(voice=arguments.voice, rate=arguments.rate))
        from app.voice.stt import WhisperCli

        WhisperCli(stt_config).check()
        session, close, signed_in = build(arguments)
    except Exception as error:
        print(f"error: {describe_startup_failure(error)}", file=sys.stderr)
        return 2
    if arguments.agent == "lg5f":
        print("Agent: LG-5F orchestrator.")
    else:
        print("Agent: legacy text agent.")
    print(
        "Signed in. "
        if signed_in
        else "Not signed in: you can ask about availability, but changes need a patient sign-in "
        "(--patient-code). "
    )
    print("Speak after pressing Enter. Nothing is confirmed by voice. Ctrl-C quits.\n")
    try:
        VoiceAgent(session, stt_config, speaker).run()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
