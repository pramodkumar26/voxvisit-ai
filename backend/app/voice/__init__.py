"""Local voice input. Milestone 3.1: microphone capture and speech-to-text only.

Nothing here knows about the scheduling agent, the database or the conversation layer. The one
function a later milestone needs is `listen_once`, which returns text and timings.
"""

from app.voice.stt import (
    SpeechError,
    SttConfig,
    Transcript,
    WhisperCli,
    listen_once,
    write_wav,
)

__all__ = ["SpeechError", "SttConfig", "Transcript", "WhisperCli", "listen_once", "write_wav"]
