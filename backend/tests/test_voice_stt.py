"""Local speech-to-text, with no microphone and no whisper.cpp: synthetic audio and stand-ins."""

import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import pytest

from app.voice import stt
from app.voice.stt import (
    Listener,
    SpeechError,
    SttConfig,
    WhisperCli,
    clean_text,
    duration_seconds,
    listen_once,
    record_push_to_talk,
    rms,
    write_wav,
)


def tone(seconds: float, amplitude: int = 8000, rate: int = stt.SAMPLE_RATE) -> bytes:
    count = int(seconds * rate)
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(count)
    )


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * stt.SAMPLE_RATE)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A config whose executable and model 'exist'."""
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(stt.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    return SttConfig(whisper_cli="whisper-cli", model_path=str(model))


class Runner:
    """Stands in for subprocess.run and remembers what it was given."""

    def __init__(self, stdout="Hello there.\n", returncode=0, error=None):
        self.stdout, self.returncode, self.error = stdout, returncode, error
        self.command = None
        self.wav_existed = None
        self.wav_format = None

    def __call__(self, command, **kwargs):
        self.command = command
        wav = Path(command[command.index("-f") + 1])
        self.wav_existed = wav.exists()
        with wave.open(str(wav)) as handle:
            self.wav_format = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        self.wav_path = wav
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(
            command, self.returncode, self.stdout, "SECRET PATH noise"
        )


# --- audio -------------------------------------------------------------------------------------


def test_a_recording_is_written_as_16khz_mono_16_bit_wav(tmp_path) -> None:
    path = tmp_path / "a.wav"

    write_wav(path, tone(0.5))

    with wave.open(str(path)) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (
            1,
            2,
            16000,
        )
        assert handle.getnframes() == 8000


def test_levels_and_durations_are_measured_from_the_samples() -> None:
    assert duration_seconds(tone(1.5)) == pytest.approx(1.5)
    assert rms(silence(1)) == 0.0 and rms(b"") == 0.0
    assert rms(tone(1, 8000)) == pytest.approx(8000 / math.sqrt(2), rel=0.01)


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ("  Hello there.\n", "Hello there."),
        ("[BLANK_AUDIO]\n", ""),
        (" Book me\n at ten. [BLANK_AUDIO]\n", "Book me at ten."),
        ("\n\n", ""),
    ],
)
def test_the_recognized_text_is_cleaned(raw, text) -> None:
    assert clean_text(raw) == text


# --- whisper.cpp -------------------------------------------------------------------------------


def test_a_missing_executable_says_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setattr(stt.shutil, "which", lambda name: None)

    with pytest.raises(SpeechError, match=r"whisper\.cpp executable .* not found.*brew install"):
        WhisperCli(SttConfig(model_path="x")).check()


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (None, "No Whisper model is set"),
        ("/nonexistent/ggml-base.en.bin", "model file was not found"),
    ],
)
def test_a_missing_model_is_reported_before_anything_is_recorded(
    model, message, monkeypatch
) -> None:
    monkeypatch.setattr(stt.shutil, "which", lambda name: "/bin/whisper-cli")
    opened = []

    with pytest.raises(SpeechError, match=message):
        listen_once(SttConfig(model_path=model), open_stream=lambda **_: opened.append(1))

    assert opened == []  # The microphone was never touched.


def test_transcription_runs_the_cli_on_a_wav_and_returns_the_text(installed) -> None:
    runner = Runner()
    cli = WhisperCli(installed, runner=runner)
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "u.wav"
        write_wav(wav, tone(0.5))

        assert cli.transcribe(wav) == "Hello there."

    command = runner.command
    assert command[0] == "/usr/local/bin/whisper-cli"
    assert command[command.index("-m") + 1] == installed.model_path
    assert command[command.index("-l") + 1] == "en" and "-nt" in command and "-np" in command
    assert runner.wav_format == (1, 2, 16000)
    assert runner.kwargs["timeout"] == installed.transcribe_timeout_seconds


def test_a_failing_cli_reports_the_exit_code_and_not_its_output(installed, tmp_path) -> None:
    wav = tmp_path / "u.wav"
    write_wav(wav, tone(0.5))

    with pytest.raises(SpeechError) as error:
        WhisperCli(installed, runner=Runner(returncode=3)).transcribe(wav)

    assert "exit code 3" in str(error.value) and "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (subprocess.TimeoutExpired("x", 1), "took longer"),
        (PermissionError("denied"), "could not be started"),
    ],
)
def test_a_timeout_or_an_unstartable_cli_is_a_clear_error(
    installed, tmp_path, raised, message
) -> None:
    wav = tmp_path / "u.wav"
    write_wav(wav, tone(0.5))

    with pytest.raises(SpeechError, match=message):
        WhisperCli(installed, runner=Runner(error=raised)).transcribe(wav)


# --- from audio to text ------------------------------------------------------------------------


def listener(installed, runner) -> Listener:
    result = Listener(installed)
    result.whisper = WhisperCli(installed, runner=runner)
    return result


def test_speech_is_transcribed_and_the_processing_time_is_measured_apart_from_the_recording(
    installed,
) -> None:
    ticks = iter([100.0, 100.8])
    runner = Runner(stdout=" Cancel my appointment.\n")

    result = listener(installed, runner).transcribe_pcm(tone(2.0), timer=lambda: next(ticks))

    assert result.text == "Cancel my appointment."
    assert result.recording_seconds == pytest.approx(2.0)
    assert result.processing_seconds == pytest.approx(0.8)
    assert result.real_time_factor == pytest.approx(0.4)


def test_the_temporary_audio_is_outside_the_repository_and_deleted_afterwards(installed) -> None:
    runner = Runner()

    listener(installed, runner).transcribe_pcm(tone(1.0))

    assert runner.wav_existed is True
    assert not runner.wav_path.exists() and not runner.wav_path.parent.exists()
    assert stt.REPO_ROOT not in runner.wav_path.resolve().parents


@pytest.mark.parametrize(
    "failure", [Runner(returncode=1), Runner(error=subprocess.TimeoutExpired("x", 1))]
)
def test_the_temporary_audio_is_deleted_when_transcription_fails(installed, failure) -> None:
    with pytest.raises(SpeechError):
        listener(installed, failure).transcribe_pcm(tone(1.0))

    assert not failure.wav_path.exists() and not failure.wav_path.parent.exists()


def test_a_temporary_directory_inside_the_repository_is_refused(installed, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(stt.REPO_ROOT / "backend"))
    runner = Runner()

    with pytest.raises(SpeechError, match="inside the repository"):
        listener(installed, runner).transcribe_pcm(tone(1.0))

    assert runner.command is None  # Nothing was written or run.
    assert not list((stt.REPO_ROOT / "backend").glob("voxvisit-stt-*"))


@pytest.mark.parametrize(
    ("pcm", "message"),
    [
        (silence(1.0), "No speech was heard"),
        (tone(0.1), "too short"),
        (b"", "too short"),
        (tone(0.5, amplitude=20), "No speech was heard"),
        (tone(40.0), "longer than the allowed limit"),
    ],
)
def test_silence_short_and_overlong_recordings_are_refused_without_running_whisper(
    installed, pcm, message
) -> None:
    runner = Runner()

    with pytest.raises(SpeechError, match=message):
        listener(installed, runner).transcribe_pcm(pcm)

    assert runner.command is None


def test_audio_with_no_recognized_words_is_reported(installed) -> None:
    with pytest.raises(SpeechError, match="no recognizable speech"):
        listener(installed, Runner(stdout="[BLANK_AUDIO]\n")).transcribe_pcm(tone(1.0))


def test_the_transcript_is_not_logged(installed, caplog) -> None:
    with caplog.at_level("DEBUG"):
        listener(installed, Runner(stdout="Private words.\n")).transcribe_pcm(tone(1.0))

    assert "Private words" not in caplog.text


# --- recording ---------------------------------------------------------------------------------


class FakeStream:
    def __init__(self, chunks_to_deliver, **kwargs):
        self.callback = kwargs["callback"]
        self.kwargs = kwargs
        self.chunks_to_deliver = chunks_to_deliver
        self.events = []

    def start(self):
        self.events.append("start")
        for chunk in self.chunks_to_deliver:
            self.callback(chunk, len(chunk) // 2, None, None)

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


def test_push_to_talk_records_between_two_presses_of_enter_in_the_right_format() -> None:
    made = []

    def open_stream(**kwargs):
        made.append(FakeStream([b"\x01\x00" * 100, b"\x02\x00" * 100], **kwargs))
        return made[0]

    presses = iter([True, True])  # Start; then stop on the first poll.
    said = []

    pcm = record_push_to_talk(
        SttConfig(),
        wait_for_enter=lambda timeout: next(presses),
        announce=said.append,
        open_stream=open_stream,
    )

    assert len(pcm) == 400
    stream = made[0]
    assert stream.kwargs["samplerate"] == 16000 and stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.events == ["start", "stop", "close"]
    assert "Press Enter to start recording." in said[0]


def test_the_recording_limit_stops_a_recording_nobody_stopped() -> None:
    now = iter([0.0, 0.0, 1.0, 2.5, 3.5])
    streams = []
    said = []

    def open_stream(**kwargs):
        streams.append(FakeStream([b"\x01\x00" * 10], **kwargs))
        return streams[0]

    pcm = record_push_to_talk(
        SttConfig(max_record_seconds=3.0),
        wait_for_enter=lambda timeout: timeout is None,  # Start pressed; never stopped.
        announce=said.append,
        open_stream=open_stream,
        clock=lambda: next(now),
    )

    assert pcm and streams[0].events == ["start", "stop", "close"]
    assert any("limit reached" in line for line in said)


def test_a_microphone_that_cannot_be_opened_explains_permissions() -> None:
    def refuse(**_):
        raise OSError("no input device")

    with pytest.raises(SpeechError, match="System Settings > Privacy & Security > Microphone"):
        record_push_to_talk(
            SttConfig(), wait_for_enter=lambda t: True, announce=lambda _: None, open_stream=refuse
        )


def test_a_stream_that_fails_to_start_is_closed_and_explained() -> None:
    class Denied(FakeStream):
        def start(self):
            raise OSError("permission")

    streams = []

    def open_stream(**kwargs):
        streams.append(Denied([], **kwargs))
        return streams[0]

    with pytest.raises(SpeechError, match="microphone could not be opened"):
        record_push_to_talk(
            SttConfig(),
            wait_for_enter=lambda t: True,
            announce=lambda _: None,
            open_stream=open_stream,
        )

    assert "close" in streams[0].events


def test_a_missing_sounddevice_package_says_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # Makes the import fail.

    with pytest.raises(SpeechError, match="uv sync --group voice"):
        record_push_to_talk(SttConfig(), wait_for_enter=lambda t: True, announce=lambda _: None)


def test_one_utterance_from_microphone_to_text(installed, monkeypatch) -> None:
    runner = Runner(stdout="Book me for Tuesday.\n")
    monkeypatch.setattr(stt.subprocess, "run", runner)
    presses = iter([True, True])

    result = listen_once(
        installed,
        wait_for_enter=lambda timeout: next(presses),
        announce=lambda _: None,
        open_stream=lambda **kwargs: FakeStream([tone(1.0)], **kwargs),
    )

    assert result.text == "Book me for Tuesday."
    assert result.recording_seconds == pytest.approx(1.0)
    assert result.processing_seconds >= 0 and not runner.wav_path.exists()
