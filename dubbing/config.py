"""Runtime configuration.

Secrets are read from the environment (or a local ``.env`` file) and are never
hard-coded. Everything else has a sensible default and can be overridden
from the CLI or by constructing :class:`Config` directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # --- Credentials -------------------------------------------------------
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # --- Hardware ----------------------------------------------------------
    device: str = field(default_factory=_default_device)

    # --- Models ------------------------------------------------------------
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    gender_model: str = "prithivMLmods/Common-Voice-Geneder-Detection"
    whisper_model: str = "large-v2"
    translation_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    tts_config_path: Path = PROJECT_ROOT / "configs" / "tts.yaml"

    # --- Languages ---------------------------------------------------------
    source_language: str = "en"
    target_language: str = "Uzbek"

    # --- Audio -------------------------------------------------------------
    audio_extraction_sr: int = 44100

    # --- Diarization (num_speakers=None -> auto-detect within min/max) ----
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 10

    # --- Segment post-processing ------------------------------------------
    merge_gap_threshold: float = 0.5   # seconds between same-speaker segments to merge
    min_segment_duration: float = 0.3  # drop segments shorter than this
    overlap_threshold: float = 0.3     # >30% overlap with another segment -> skip dubbing

    # --- Translation -------------------------------------------------------
    translation_batch_size: int = 10
    translation_temperature: float = 0.7

    # --- Alignment ---------------------------------------------------------
    max_stretch_rate: float = 1.5
    min_stretch_rate: float = 0.6
    stretch_tolerance: float = 0.05    # don't stretch if within 5%
    expected_speech_ratio: float = 0.7  # share of a segment that is active speech
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100

    # --- Mixing ------------------------------------------------------------
    background_reduction_db: float = 0.0
    voice_boost_db: float = 2.0
    demucs_timeout_s: int = 300

    # --- Output video ------------------------------------------------------
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_bitrate: str = "5000k"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    def validate(self) -> None:
        """Fail fast when required credentials are missing."""
        missing = [
            name
            for name, value in (("HF_TOKEN", self.hf_token), ("OPENAI_API_KEY", self.openai_api_key))
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
