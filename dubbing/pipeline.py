"""End-to-end dubbing pipeline.

    video ─► audio ─► speaker analysis ─► translation ─► TTS ─► alignment
                 └──► Demucs background ─────────────────────────┐   │
                                                                 ▼   ▼
                                                      mix ─► mux ─► dubbed video
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dubbing import media
from dubbing.alignment import AudioAligner
from dubbing.config import Config
from dubbing.speaker_analysis import SpeakerAnalyzer
from dubbing.translation import Translator
from dubbing.tts.synthesizer import Synthesizer

log = logging.getLogger(__name__)

SENTENCE_END = (".", "!", "?")


class DubbingPipeline:
    def __init__(self, output_dir: str | Path, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.validate()

        root = Path(output_dir)
        self.dirs = {name: root / name for name in ("temp", "chunks", "tts", "aligned", "output")}
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        log.info("Initializing models on %s", self.config.device)
        self.analyzer = SpeakerAnalyzer(self.config)
        self.translator = Translator(self.config)
        self.synthesizer = Synthesizer(self.config.tts_config_path, device=self.config.device)
        self.aligner = AudioAligner(self.config)

    # ------------------------------------------------------------------ steps

    def _synthesize(self, segments: List[Dict]) -> None:
        for seg in segments:
            seg["tts_path"] = None
            if seg.get("overlap"):
                log.info("[%03d] skipped (overlapping speech)", seg["id"])
                continue

            text = seg["text"].strip()
            if not text.endswith(SENTENCE_END):
                text += " ."
            path = self.dirs["tts"] / f"{seg['id']:03d}_{seg['speaker']}_{seg['gender']}.wav"
            try:
                generated = self.synthesizer.synthesize(f" {text} ", str(path), seg["gender"])
                if generated and Path(generated).exists():
                    seg["tts_path"] = generated
                else:
                    log.warning("[%03d] TTS produced no audio", seg["id"])
            except Exception:
                log.exception("[%03d] TTS failed", seg["id"])

        done = sum(1 for s in segments if s["tts_path"])
        log.info("TTS: %d/%d segments", done, len(segments))

    def _align(self, segments: List[Dict]) -> None:
        for seg in segments:
            seg["aligned_path"] = None
            if not seg.get("tts_path"):
                continue
            path = self.dirs["aligned"] / f"{seg['id']:03d}_aligned.wav"
            if self.aligner.align(seg["tts_path"], seg["end"] - seg["start"], str(path)):
                seg["aligned_path"] = str(path)

        done = sum(1 for s in segments if s["aligned_path"])
        log.info("Alignment: %d/%d segments", done, len(segments))

    def _save_metadata(self, segments: List[Dict], video_path: str) -> Path:
        path = self.dirs["output"] / "metadata.json"
        data = {
            "video": Path(video_path).name,
            "total_segments": len(segments),
            "speakers": sorted({s["speaker"] for s in segments}),
            "segments": segments,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # --------------------------------------------------------------- pipeline

    def run(self, video_path: str, cleanup_temp: bool = False) -> Tuple[Path, List[Dict]]:
        """Dub ``video_path``; return the output video path and segment metadata."""
        cfg = self.config
        log.info("Input: %s -> %s", Path(video_path).name, cfg.target_language)

        audio_path = media.extract_audio(
            video_path, self.dirs["temp"] / "original_audio.wav", cfg.audio_extraction_sr
        )
        info = media.probe_video(video_path)

        segments = self.analyzer.analyze(str(audio_path))
        media.save_segment_chunks(audio_path, segments, self.dirs["chunks"])

        segments = self.translator.translate(segments)
        self._synthesize(segments)
        self._align(segments)

        voice = media.build_voice_track(segments, info["duration"], cfg.voice_boost_db)
        voice.export(str(self.dirs["temp"] / "dubbed_voice.wav"), format="wav")

        background = media.separate_background(audio_path, self.dirs["temp"], cfg.demucs_timeout_s)
        final_audio = media.mix(voice, background, info["duration"], cfg.background_reduction_db)
        final_audio_path = self.dirs["output"] / "final_audio.wav"
        final_audio.export(str(final_audio_path), format="wav")

        output_video = media.mux(
            video_path,
            final_audio_path,
            self.dirs["output"] / f"DUBBED_{Path(video_path).stem}.mp4",
            cfg,
            fps=info["fps"],
            temp_dir=self.dirs["temp"],
        )
        self._save_metadata(segments, video_path)

        if cleanup_temp:
            shutil.rmtree(self.dirs["temp"], ignore_errors=True)

        skipped = sum(1 for s in segments if s.get("overlap"))
        log.info("Done: %s (%d segments, %d skipped for overlap)", output_video, len(segments), skipped)
        return output_video, segments
