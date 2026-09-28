"""Generate a reusable two-speaker synthetic call for STT testing.

Edit ``demo/dialogue.txt`` and run this script again. The generated file is written
to ``demo/generated_call.mp3`` and can be uploaded to the Streamlit application.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import re
import shutil
import tempfile

import edge_tts
from pydub import AudioSegment


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEMO_DIR / "dialogue.txt"
DEFAULT_OUTPUT = DEMO_DIR / "generated_call.mp3"
VOICE_BY_SPEAKER = {
    "agent": "en-US-GuyNeural",
    "consumer": "en-US-JennyNeural",
}
SPEAKER_LINE = re.compile(r"^(?P<speaker>[^:]{1,40})\s*:\s*(?P<text>.+?)\s*$")


def parse_dialogue(path: Path) -> list[tuple[str, str]]:
    """Read ``Speaker: text`` lines from the editable dialogue file."""

    if not path.exists():
        raise FileNotFoundError(f"Dialogue file not found: {path}")

    turns: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = SPEAKER_LINE.match(line)
        if not match:
            raise ValueError(
                f"Line {line_number} must use the format 'Agent: text' or 'Consumer: text'."
            )

        speaker = match.group("speaker").strip().lower()
        text = match.group("text").strip()
        if speaker not in VOICE_BY_SPEAKER:
            supported = ", ".join(sorted(VOICE_BY_SPEAKER))
            raise ValueError(f"Line {line_number} has unsupported speaker '{speaker}'. Use: {supported}.")
        if not text:
            raise ValueError(f"Line {line_number} has no dialogue text.")
        turns.append((speaker, text))

    if not turns:
        raise ValueError(f"No dialogue turns found in {path}.")
    return turns


async def synthesize_turn(speaker: str, text: str, output_path: Path, rate: str) -> None:
    communicate = edge_tts.Communicate(
        text,
        VOICE_BY_SPEAKER[speaker],
        rate=rate,
    )
    await communicate.save(str(output_path))


async def generate_audio(
    turns: list[tuple[str, str]],
    output_path: Path,
    *,
    pause_ms: int = 350,
    rate: str = "-5%",
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to combine speaker audio files but was not found on PATH.")
    AudioSegment.converter = ffmpeg
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        AudioSegment.ffprobe = ffprobe

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="astra-demo-audio-") as temporary_directory:
        temporary_dir = Path(temporary_directory)
        combined = AudioSegment.empty()
        for index, (speaker, text) in enumerate(turns, 1):
            segment_path = temporary_dir / f"segment-{index:03d}.mp3"
            await synthesize_turn(speaker, text, segment_path, rate)
            combined += AudioSegment.from_file(segment_path, format="mp3")
            if index < len(turns):
                combined += AudioSegment.silent(duration=pause_ms)

        combined.export(output_path, format="mp3", bitrate="128k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Dialogue text file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated MP3 path.")
    parser.add_argument("--pause-ms", type=int, default=350, help="Silence between turns.")
    parser.add_argument("--rate", default="-5%", help="Speech rate, for example -5%% or +10%%.")
    args = parser.parse_args()

    if args.pause_ms < 0:
        parser.error("--pause-ms cannot be negative.")

    turns = parse_dialogue(args.input)
    asyncio.run(generate_audio(turns, args.output, pause_ms=args.pause_ms, rate=args.rate))
    print(f"Created {args.output} from {len(turns)} dialogue turns.")


if __name__ == "__main__":
    main()
