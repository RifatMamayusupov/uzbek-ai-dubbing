"""Command-line interface: ``python -m dubbing VIDEO [options]``."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dubbing.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dubbing",
        description="Automatically dub an English video into Uzbek.",
    )
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="working/output directory (default: outputs/<video name>)",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="exact number of speakers (default: auto-detect)",
    )
    parser.add_argument("--source-lang", default="en", help="Whisper source language code (default: en)")
    parser.add_argument("--target-lang", default="Uzbek", help="translation target language (default: Uzbek)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--cleanup", action="store_true", help="delete temp files when finished")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    config = Config(
        num_speakers=args.num_speakers,
        source_language=args.source_lang,
        target_language=args.target_lang,
    )
    if args.device:
        config.device = args.device

    # Imported lazily so `--help` works without loading torch models.
    from dubbing.pipeline import DubbingPipeline

    output_dir = args.output_dir or Path("outputs") / args.video.stem
    pipeline = DubbingPipeline(output_dir, config)
    output_video, _ = pipeline.run(str(args.video), cleanup_temp=args.cleanup)
    print(output_video)


if __name__ == "__main__":
    main()
