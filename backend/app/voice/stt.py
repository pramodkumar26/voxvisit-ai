"""Push-to-talk capture and local transcription with whisper.cpp.

Audio is captured at 16 kHz, mono, 16-bit PCM (what whisper.cpp reads), written to a temporary
directory outside the repository, transcribed by the `whisper-cli` executable, and deleted in every
case, including on failure. Recordings and transcripts are never logged or stored: the transcript
exists only as the return value.

`sounddevice` is imported only when a recording starts, so importing this module (and testing it)
needs neither a microphone nor PortAudio.
"""

import array
import contextlib
import math
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # Bytes: 16-bit PCM.
# Text whisper.cpp prints for audio it hears nothing in; not speech.
NON_SPEECH_MARKERS = ("[BLANK_AUDIO]", "[SILENCE]", "(silence)", "[ Silence ]", "[MUSIC]")
REPO_ROOT = Path(__file__).resolve().parents[3]


class SpeechError(Exception):
    """A problem the person can act on. The message is safe to show: it holds no audio or text."""


@dataclass(frozen=True)
class SttConfig:
    whisper_cli: str | None = None  # Path or name of the whisper-cli executable.
    model_path: str | None = None  # Path of a ggml model file, for example ggml-base.en.bin.
    language: str = "en"
    max_record_seconds: float = 30.0
    min_record_seconds: float = 0.3
    silence_rms: float = 100.0  # Below this (of 32768) the recording is treated as silence.
    transcribe_timeout_seconds: float = 60.0

    @classmethod
    def from_environment(cls, **overrides: object) -> "SttConfig":
        base = cls(
            whisper_cli=os.environ.get("VOXVISIT_WHISPER_CLI"),
            model_path=os.environ.get("VOXVISIT_WHISPER_MODEL"),
        )
        values = {k: v for k, v in overrides.items() if v is not None}
        return SttConfig(**{**base.__dict__, **values})  # type: ignore[arg-type]


@dataclass(frozen=True)
class Transcript:
    text: str
    recording_seconds: float
    processing_seconds: float  # Transcription only: the subprocess, not the recording.

    @property
    def real_time_factor(self) -> float:
        """Processing time per second of audio; below 1.0 is faster than real time."""
        return self.processing_seconds / self.recording_seconds if self.recording_seconds else 0.0


# --- audio ------------------------------------------------------------------------------------


def write_wav(path: Path, pcm: bytes) -> None:
    """Write 16 kHz mono 16-bit PCM samples as a WAV file."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def duration_seconds(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)


def rms(pcm: bytes) -> float:
    """Root-mean-square level of 16-bit little-endian samples; 0.0 for no samples."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


# --- transcription ----------------------------------------------------------------------------


class WhisperCli:
    """Runs the `whisper-cli` executable of whisper.cpp on a WAV file."""

    def __init__(
        self, config: SttConfig, runner: Callable[..., subprocess.CompletedProcess] | None = None
    ):
        self.config = config
        self._run = runner or (lambda *args, **kwargs: subprocess.run(*args, **kwargs))

    def check(self) -> str:
        """Resolve the executable and the model, or say exactly what is missing."""
        wanted = self.config.whisper_cli or "whisper-cli"
        executable = shutil.which(wanted)
        if executable is None:
            raise SpeechError(
                f"The whisper.cpp executable '{wanted}' was not found. Install it "
                "(brew install whisper-cpp) or pass its path with --whisper-cli or "
                "VOXVISIT_WHISPER_CLI."
            )
        if not self.config.model_path:
            raise SpeechError(
                "No Whisper model is set. Download ggml-base.en.bin and pass its path with "
                "--model or VOXVISIT_WHISPER_MODEL."
            )
        if not Path(self.config.model_path).expanduser().is_file():
            raise SpeechError(
                f"The Whisper model file was not found: {self.config.model_path}. "
                "Download ggml-base.en.bin and check the path."
            )
        return executable

    def transcribe(self, wav_path: Path) -> str:
        """The recognized text; an empty string if the audio holds no speech."""
        executable = self.check()
        command = [
            executable,
            "-m", str(Path(self.config.model_path or "").expanduser()),
            "-f", str(wav_path),
            "-l", self.config.language,
            "-nt",  # No timestamps in the text.
            "-np",  # Print only the result.
        ]  # fmt: skip
        try:
            done = self._run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.transcribe_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SpeechError(
                f"Transcription took longer than {self.config.transcribe_timeout_seconds:g} s "
                "and was stopped."
            ) from None
        except OSError as error:
            raise SpeechError(
                f"whisper-cli could not be started ({type(error).__name__})."
            ) from None
        if done.returncode != 0:
            # The program's own message can echo paths; only the exit code is reported.
            raise SpeechError(f"whisper-cli failed (exit code {done.returncode}).")
        return clean_text(done.stdout)


def clean_text(output: str) -> str:
    text = " ".join(line.strip() for line in output.splitlines() if line.strip())
    for marker in NON_SPEECH_MARKERS:
        text = text.replace(marker, "")
    return " ".join(text.split())


# --- push to talk -----------------------------------------------------------------------------


def record_push_to_talk(
    config: SttConfig,
    *,
    wait_for_enter: Callable[[float | None], bool] | None = None,
    announce: Callable[[str], None] = print,
    open_stream: Callable[..., object] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Record from the default microphone: Enter starts, Enter again stops (or the limit does).

    Returns raw 16 kHz mono 16-bit PCM. `wait_for_enter(timeout)` returns True when Enter was
    pressed within the timeout; the defaults use the terminal.
    """
    wait = wait_for_enter or _terminal_enter
    announce("Press Enter to start recording.")
    wait(None)
    chunks: list[bytes] = []
    stream = _open_microphone(chunks, open_stream)
    started = clock()
    try:
        announce(f"Recording... press Enter to stop (limit {config.max_record_seconds:g} s).")
        while clock() - started < config.max_record_seconds:
            if wait(0.1):
                break
        else:
            announce("Recording limit reached; stopping.")
    finally:
        _close(stream)
    return b"".join(chunks)


def _open_microphone(chunks: list[bytes], open_stream: Callable[..., object] | None):
    def callback(indata, frames, time_info, status):
        chunks.append(bytes(indata))

    if open_stream is None:
        try:
            import sounddevice
        except (ImportError, OSError):
            raise SpeechError(
                "The 'sounddevice' package (or PortAudio) is not available. "
                "Install it with: uv sync --group voice"
            ) from None
        open_stream = sounddevice.RawInputStream
        errors: tuple[type[BaseException], ...] = (sounddevice.PortAudioError,)
    else:
        errors = (OSError,)
    stream = None
    try:
        stream = open_stream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=callback
        )
        stream.start()  # type: ignore[attr-defined]
        return stream
    except errors as error:
        if stream is not None:
            _close(stream)
        raise SpeechError(
            "The microphone could not be opened. Allow microphone access for your terminal app "
            "in System Settings > Privacy & Security > Microphone, and check that an input "
            f"device is connected ({type(error).__name__})."
        ) from None


def _close(stream: object) -> None:
    for step in ("stop", "close"):
        with contextlib.suppress(Exception):  # Closing must never hide the original problem.
            getattr(stream, step)()


def _terminal_enter(timeout: float | None) -> bool:
    if timeout is None:
        sys.stdin.readline()
        return True
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        sys.stdin.readline()
        return True
    return False


# --- the whole thing --------------------------------------------------------------------------


@dataclass
class Listener:
    config: SttConfig
    whisper: WhisperCli = field(init=False)

    def __post_init__(self) -> None:
        self.whisper = WhisperCli(self.config)

    def transcribe_pcm(
        self, pcm: bytes, *, timer: Callable[[], float] = time.perf_counter
    ) -> Transcript:
        seconds = duration_seconds(pcm)
        if seconds < self.config.min_record_seconds:
            raise SpeechError(
                "The recording was too short. Hold the pause between the two Enters longer."
            )
        if seconds > self.config.max_record_seconds + 1:
            raise SpeechError("The recording is longer than the allowed limit.")
        if rms(pcm) < self.config.silence_rms:
            raise SpeechError(
                "No speech was heard (the recording is silent). Check the microphone, its "
                "input level, and that your terminal app has microphone access."
            )
        with _private_temp_directory() as directory:
            wav = directory / "utterance.wav"
            write_wav(wav, pcm)
            started = timer()
            text = self.whisper.transcribe(wav)
            processing = timer() - started
        if not text:
            raise SpeechError("The audio contained no recognizable speech.")
        return Transcript(text, seconds, processing)


class _private_temp_directory:
    """A temporary directory outside the repository, removed on exit however the block ends."""

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory(prefix="voxvisit-stt-")
        path = Path(self._dir.name).resolve()
        if REPO_ROOT in (path, *path.parents):
            self._dir.cleanup()
            raise SpeechError(
                "The temporary directory is inside the repository; set TMPDIR elsewhere."
            )
        return path

    def __exit__(self, *exc: object) -> None:
        self._dir.cleanup()


def listen_once(config: SttConfig, **recording: object) -> Transcript:
    """Record one utterance with push-to-talk and transcribe it. Raises `SpeechError`."""
    listener = Listener(config)
    listener.whisper.check()  # Fail before the person speaks, not after.
    pcm = record_push_to_talk(config, **recording)  # type: ignore[arg-type]
    return listener.transcribe_pcm(pcm)
