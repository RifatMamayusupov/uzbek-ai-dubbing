"""Uzbek voice-cloning TTS (F5-TTS style DiT + Vocos vocoder).

The model is loaded once and each reference voice is preprocessed once and
cached, instead of on every call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml

from dubbing.config import PROJECT_ROOT
from dubbing.tts.uvc.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from dubbing.tts.uvc.model import DiT
from dubbing.tts.uvc.text.splitter import remove_silence, split_sentence

log = logging.getLogger(__name__)

DIT_CONFIG = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


class Synthesizer:
    def __init__(self, config_path: Path, device: str = "cuda"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.device = device
        self.cross_fade_duration = float(self.cfg.get("cross_fade_duration", 0.15))
        self.voices: Dict[str, dict] = self.cfg["voices"]
        self._prepared: Dict[str, Tuple[str, str]] = {}

        log.info("Loading TTS model: %s", self.cfg["ckpt_file"])
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
        self.model = load_model(
            DiT,
            DIT_CONFIG,
            _resolve(self.cfg["ckpt_file"]),
            mel_spec_type="vocos",
            vocab_file=_resolve(self.cfg["vocab_file"]),
            device=device,
        )

    def _reference(self, gender: str) -> Tuple[str, str, float]:
        if gender not in self.voices:
            log.warning("No voice configured for gender %r, using 'male'", gender)
            gender = "male"
        voice = self.voices[gender]
        if gender not in self._prepared:
            self._prepared[gender] = preprocess_ref_audio_text(
                _resolve(voice["ref_audio"]), voice["ref_text"], device=self.device
            )
        ref_audio, ref_text = self._prepared[gender]
        return ref_audio, ref_text, float(voice.get("speed", 1.0))

    def synthesize(self, text: str, wave_path: str, gender: str) -> Optional[str]:
        """Generate speech for ``text`` with the voice for ``gender``.

        Returns ``wave_path`` on success, ``None`` if nothing was generated.
        """
        ref_audio, ref_text, speed = self._reference(gender)
        pieces = []
        sample_rate = None
        for chunk in split_sentence(text):
            audio, sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                chunk,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                cross_fade_duration=self.cross_fade_duration,
                speed=speed,
                device=self.device,
            )
            pieces.append(audio)

        if not pieces:
            return None
        sf.write(wave_path, np.concatenate(pieces), sample_rate)
        remove_silence(wave_path, wave_path)
        return wave_path
