"""Fit synthesized speech into the original segment's time window.

The stretch factor is computed from *active* speech (Silero VAD) rather than
raw file length, so leading/trailing silence in the TTS output does not
distort the timing.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from dubbing.config import Config

log = logging.getLogger(__name__)

VAD_SR = 16000  # Silero VAD supports 8 kHz / 16 kHz only
TARGET_RMS = 0.1
MAX_GAIN = 3.0
FADE_S = 0.01


class AudioAligner:
    def __init__(self, config: Config):
        self.config = config
        self.vad_model = load_silero_vad()
        self.vad_model.eval()

    def active_speech(self, audio: np.ndarray, sr: int) -> Tuple[List[dict], float]:
        """Return speech regions (seconds) and their total duration."""
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=VAD_SR) if sr != VAD_SR else audio
        timestamps = get_speech_timestamps(
            torch.FloatTensor(audio_16k),
            self.vad_model,
            sampling_rate=VAD_SR,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.vad_min_speech_ms,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
        )
        regions = [{"start": t["start"] / VAD_SR, "end": t["end"] / VAD_SR} for t in timestamps]
        return regions, sum(r["end"] - r["start"] for r in regions)

    def _time_stretch(self, audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        try:
            return pyrb.time_stretch(audio, sr, rate, rbargs={"--fine": None, "--formant": None})
        except Exception as exc:  # rubberband CLI missing or failed
            log.debug("pyrubberband failed (%s); falling back to librosa", exc)
            return librosa.effects.time_stretch(audio, rate=rate)

    def align(self, tts_path: str, target_duration: float, output_path: str) -> bool:
        """Stretch, trim/pad, normalize and fade ``tts_path`` into ``output_path``."""
        cfg = self.config
        try:
            audio, sr = librosa.load(tts_path, sr=None)
            _, active = self.active_speech(audio, sr)

            if active > 0.1:
                raw_rate = active / (target_duration * cfg.expected_speech_ratio)
            else:
                raw_rate = (len(audio) / sr) / target_duration
            rate = float(np.clip(raw_rate, cfg.min_stretch_rate, cfg.max_stretch_rate))

            if abs(rate - 1.0) > cfg.stretch_tolerance:
                audio = self._time_stretch(audio, sr, rate)

            target_samples = int(target_duration * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)))

            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            if rms > 0.001:
                audio = audio * min(TARGET_RMS / rms, MAX_GAIN)
            audio = np.tanh(audio * 0.95)  # soft clip

            fade = min(int(FADE_S * sr), len(audio) // 20)
            if fade > 0:
                audio[:fade] *= np.linspace(0, 1, fade)
                audio[-fade:] *= np.linspace(1, 0, fade)

            sf.write(output_path, audio, sr)
            log.debug("Aligned %s: rate=%.3f, %.2fs", tts_path, rate, len(audio) / sr)
            return True
        except Exception:
            log.exception("Alignment failed for %s", tts_path)
            return False
