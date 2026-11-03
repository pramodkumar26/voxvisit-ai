"""Spoken output with the macOS `say` command.

The text is passed on standard input, never on the command line, so nothing a reply contains can be
read as an option or a shell command; the voice name is checked against a short safe pattern; no
shell is involved. `speak` returns only when playback has finished, which is what keeps the
microphone from recording the assistant's own voice. Nothing is logged or stored.
"""

import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.voice.stt import SpeechError

VOICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ()._-]{0,59}$")
MAX_SPOKEN_CHARS = 1500  # Longer text is cut at a word, not spoken for minutes.


@dataclass(frozen=True)
class TtsConfig:
    voice: str | None = None  # A `say -v ?` voice name; None is the system default voice.
    rate: int | None = None  # Words per minute; None is the system default.
    command: str = "say"
    timeout_seconds: float = 120.0


class Speaker:
    """Speaks text and blocks until it has been said."""

    def __init__(
        self,
        config: TtsConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config.voice is not None and not VOICE_NAME.match(config.voice):
            raise SpeechError("The voice name contains characters that are not allowed.")
        if config.rate is not None and not 50 <= config.rate <= 600:
            raise SpeechError("The speaking rate must be between 50 and 600 words per minute.")
        self.config = config
        self._run = runner or (lambda *args, **kwargs: subprocess.run(*args, **kwargs))
        self._timer = timer
        self.speaking = False  # True only while playback is running.

    def check(self) -> str:
        found = shutil.which(self.config.command)
        if found is None:
            raise SpeechError(
                f"The text-to-speech command '{self.config.command}' was not found. "
                "Spoken replies need macOS."
            )
        return found

    def speak(self, text: str) -> float:
        """Say the text; return the seconds playback took (0.0 for nothing to say)."""
        text = shorten(text)
        if not text.strip():
            return 0.0
        executable = self.check()
        command = [executable]
        if self.config.voice:
            command += ["-v", self.config.voice]
        if self.config.rate:
            command += ["-r", str(self.config.rate)]
        command += ["-f", "-"]  # The text arrives on standard input.
        self.speaking = True
        started = self._timer()
        try:
            done = self._run(
                command,
                input=text,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SpeechError("Speaking took too long and was stopped.") from None
        except OSError as error:
            raise SpeechError(f"'say' could not be started ({type(error).__name__}).") from None
        finally:
            self.speaking = False
        if done.returncode != 0:
            raise SpeechError(
                f"'say' failed (exit code {done.returncode}); check the voice name with `say -v ?`."
            )
        return self._timer() - started


def shorten(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_SPOKEN_CHARS:
        return text
    return text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0]
