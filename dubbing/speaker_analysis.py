"""Speaker diarization, gender detection and speech-to-text.

Produces the segment list that drives the rest of the pipeline::

    [{"id": 1, "speaker": "SPEAKER_00", "gender": "male",
      "start": 0.5, "end": 3.2, "text": "Hello world", "overlap": False}, ...]
"""
from __future__ import annotations

import logging
from typing import Dict, List

import librosa
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from dubbing import cuda_libs
from dubbing.config import Config

log = logging.getLogger(__name__)

Segment = Dict[str, object]

GENDER_SR = 16000
MIN_GENDER_SAMPLES = 1600  # 0.1s at 16 kHz


class SpeakerAnalyzer:
    """Wraps pyannote (who), wav2vec2 (gender) and Whisper (what)."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        log.info("Loading diarization model: %s", config.diarization_model)
        self.diarization = Pipeline.from_pretrained(
            config.diarization_model, token=config.hf_token
        ).to(self.device)

        log.info("Loading gender model: %s", config.gender_model)
        self.gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            config.gender_model
        ).to(self.device)
        self.gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.gender_model)

        log.info("Loading Whisper model: %s", config.whisper_model)
        if config.device == "cuda":
            cuda_libs.preload()
        compute_type = "float16" if config.device == "cuda" else "int8"
        self.whisper = WhisperModel(config.whisper_model, device=config.device, compute_type=compute_type)

    # ------------------------------------------------------------------ steps

    def diarize(self, audio_path: str) -> List[Segment]:
        """Return speaker turns: ``[{"start", "end", "speaker"}, ...]``."""
        cfg = self.config
        if cfg.num_speakers:
            speaker_args = {"num_speakers": cfg.num_speakers}
        else:
            speaker_args = {"min_speakers": cfg.min_speakers, "max_speakers": cfg.max_speakers}

        with ProgressHook() as hook:
            result = self.diarization(audio_path, hook=hook, **speaker_args)

        timeline = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in result.speaker_diarization
        ]
        log.info(
            "Diarization: %d turns, %d speakers",
            len(timeline), len({t["speaker"] for t in timeline}),
        )
        return timeline

    def detect_genders(self, audio_path: str, timeline: List[Segment]) -> Dict[str, str]:
        """Classify each speaker using their longest turn."""
        audio, sr = librosa.load(audio_path, sr=GENDER_SR)
        gender_map: Dict[str, str] = {}

        for speaker in sorted({t["speaker"] for t in timeline}):
            turns = [t for t in timeline if t["speaker"] == speaker]
            longest = max(turns, key=lambda t: t["end"] - t["start"])
            chunk = audio[int(longest["start"] * sr): int(longest["end"] * sr)]

            if len(chunk) < MIN_GENDER_SAMPLES:
                gender_map[speaker] = "male"
                log.warning("%s: turn too short for gender detection, defaulting to male", speaker)
                continue

            inputs = self.gender_processor(
                chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.gender_model(**inputs).logits, dim=1).squeeze()

            # Label order for this model: 0 = female, 1 = male
            gender_id = int(probs.argmax())
            gender_map[speaker] = "female" if gender_id == 0 else "male"
            log.info("%s: %s (%.0f%%)", speaker, gender_map[speaker], 100 * float(probs[gender_id]))

        return gender_map

    def transcribe(self, audio_path: str) -> List[Segment]:
        """Whisper STT: ``[{"start", "end", "text"}, ...]``."""
        segments, _ = self.whisper.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.config.vad_min_silence_ms},
            language=self.config.source_language,
        )
        result = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]
        log.info("Transcription: %d segments", len(result))
        return result

    def align_and_merge(
        self,
        transcripts: List[Segment],
        timeline: List[Segment],
        gender_map: Dict[str, str],
    ) -> List[Segment]:
        """Assign each transcript to the speaker with the largest overlap,
        merge adjacent same-speaker segments and drop very short ones."""
        aligned = []
        for trans in transcripts:
            best_speaker, best_overlap = "SPEAKER_UNKNOWN", 0.0
            for turn in timeline:
                overlap = min(trans["end"], turn["end"]) - max(trans["start"], turn["start"])
                if overlap > best_overlap:
                    best_speaker, best_overlap = turn["speaker"], overlap
            aligned.append({
                **trans,
                "speaker": best_speaker,
                "gender": gender_map.get(best_speaker, "male"),
            })

        if not aligned:
            return []

        merged = [aligned[0].copy()]
        for seg in aligned[1:]:
            current = merged[-1]
            if (
                seg["speaker"] == current["speaker"]
                and seg["start"] - current["end"] < self.config.merge_gap_threshold
            ):
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"]
            else:
                merged.append(seg.copy())

        final = [s for s in merged if s["end"] - s["start"] >= self.config.min_segment_duration]
        for idx, seg in enumerate(final, start=1):
            seg["id"] = idx

        log.info("Merged %d -> %d segments", len(aligned), len(final))
        return final

    def mark_overlaps(self, segments: List[Segment]) -> List[Segment]:
        """Flag segments that overlap another one by more than the threshold."""
        for seg in segments:
            seg["overlap"] = False
            duration = seg["end"] - seg["start"]
            if duration <= 0:
                continue
            for other in segments:
                if other is seg:
                    continue
                overlap = max(0.0, min(seg["end"], other["end"]) - max(seg["start"], other["start"]))
                if overlap / duration > self.config.overlap_threshold:
                    seg["overlap"] = True
                    log.warning("Segment %d overlaps segment %d", seg["id"], other["id"])
                    break
        return segments

    # --------------------------------------------------------------- pipeline

    def analyze(self, audio_path: str) -> List[Segment]:
        """Diarization -> gender -> STT -> alignment -> overlap detection."""
        timeline = self.diarize(audio_path)
        gender_map = self.detect_genders(audio_path, timeline)
        transcripts = self.transcribe(audio_path)
        segments = self.align_and_merge(transcripts, timeline, gender_map)
        return self.mark_overlaps(segments)
