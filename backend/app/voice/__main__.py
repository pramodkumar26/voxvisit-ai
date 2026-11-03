"""Try local speech-to-text from the terminal.

    uv run python -m app.voice --model ~/models/ggml-base.en.bin

Press Enter to start, Enter again to stop. The recognized text is printed here and nowhere else.
"""

import argparse
import sys
from collections.abc import Sequence

from app.voice.stt import SpeechError, SttConfig, listen_once


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local push-to-talk speech-to-text (whisper.cpp)")
    parser.add_argument(
        "--whisper-cli", help="whisper-cli executable (default: VOXVISIT_WHISPER_CLI or PATH)"
    )
    parser.add_argument(
        "--model", help="ggml model file, e.g. ggml-base.en.bin (or VOXVISIT_WHISPER_MODEL)"
    )
    parser.add_argument("--max-seconds", type=float, help="recording limit in seconds (default 30)")
    parser.add_argument("--once", action="store_true", help="transcribe one utterance and exit")
    arguments = parser.parse_args(argv)
    config = SttConfig.from_environment(
        whisper_cli=arguments.whisper_cli,
        model_path=arguments.model,
        max_record_seconds=arguments.max_seconds,
    )
    while True:
        try:
            result = listen_once(config)
        except SpeechError as error:
            print(f"error: {error}", file=sys.stderr)
            if arguments.once:
                return 1
        else:
            print(f"\nHeard: {result.text}")
            print(
                f"[recorded {result.recording_seconds:.1f} s | transcribed in "
                f"{result.processing_seconds:.2f} s | {result.real_time_factor:.2f}x real time]"
            )
            if arguments.once:
                return 0
        print("\n(Ctrl-C to quit)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(130)
