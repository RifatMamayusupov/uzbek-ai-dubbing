"""Video/audio I/O helpers: extraction, source separation, mixing, muxing."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import soundfile as sf
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment, effects

from dubbing.config import Config

log = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: Path, sample_rate: int) -> Path:
    """Write the video's audio track to ``output_path`` (cached if present)."""
    if not output_path.exists():
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(str(output_path), fps=sample_rate, logger=None)
    info = sf.info(str(output_path))
    log.info("Audio: %.2fs @ %d Hz", info.duration, info.samplerate)
    return output_path


def probe_video(video_path: str) -> Dict[str, float]:
    with VideoFileClip(video_path) as video:
        return {"duration": video.duration, "fps": video.fps}


def save_segment_chunks(audio_path: Path, segments: List[Dict], output_dir: Path) -> None:
    """Save the original audio of every segment (useful for QA/debugging)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(str(audio_path), sr=None)
    for seg in segments:
        chunk = audio[int(seg["start"] * sr): int(seg["end"] * sr)]
        sf.write(str(output_dir / f"{seg['id']:03d}_{seg['speaker']}.wav"), chunk, sr)


def separate_background(audio_path: Path, output_dir: Path, timeout_s: int) -> Optional[Path]:
    """Run Demucs and return the ``no_vocals`` stem, or ``None`` on failure."""
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(output_dir), str(audio_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.error("Demucs failed: %s", exc)
        return None
    if result.returncode != 0:
        log.warning("Demucs exited with %d: %s", result.returncode, result.stderr[-500:])

    stem = output_dir / "htdemucs" / audio_path.stem / "no_vocals.wav"
    if stem.exists():
        return stem
    log.warning("Background stem not found at %s", stem)
    return None


def _fit(audio: AudioSegment, duration_ms: int) -> AudioSegment:
    if len(audio) > duration_ms:
        return audio[:duration_ms]
    return audio + AudioSegment.silent(duration=duration_ms - len(audio))


def build_voice_track(segments: List[Dict], duration_s: float, boost_db: float) -> AudioSegment:
    """Overlay every aligned segment at its original start time."""
    track = AudioSegment.silent(duration=int(duration_s * 1000))
    for seg in segments:
        path = seg.get("aligned_path")
        if not path:
            continue
        try:
            clip = effects.normalize(AudioSegment.from_wav(path) + boost_db, headroom=1.0)
            track = track.overlay(clip, position=int(seg["start"] * 1000))
        except Exception:
            log.exception("Overlay failed for segment %s", seg["id"])
    return track


def mix(voice: AudioSegment, background: Optional[Path], duration_s: float, reduction_db: float) -> AudioSegment:
    duration_ms = int(duration_s * 1000)
    voice = _fit(voice, duration_ms)
    if background and background.exists():
        bg = _fit(AudioSegment.from_wav(str(background)) - reduction_db, duration_ms)
        return bg.overlay(voice)
    log.warning("No background stem; output will contain the dubbed voice only")
    return voice


def get_fps(video_path: str, default: float = 25.0) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "json", video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        num, den = map(int, json.loads(out)["streams"][0]["r_frame_rate"].split("/"))
        return num / den if den else default
    except Exception:
        return default


def mux(video_path: str, audio_path: Path, output_path: Path, config: Config, fps: Optional[float], temp_dir: Path) -> Path:
    """Replace the video's audio track with ``audio_path``."""
    fps = fps or get_fps(video_path)
    with VideoFileClip(video_path) as video, AudioFileClip(str(audio_path)) as audio:
        video.set_audio(audio).write_videofile(
            str(output_path),
            codec=config.video_codec,
            audio_codec=config.audio_codec,
            fps=fps,
            preset=config.video_preset,
            bitrate=config.video_bitrate,
            audio_bitrate=config.audio_bitrate,
            temp_audiofile=str(temp_dir / "temp_audio.m4a"),
            remove_temp=True,
            logger=None,
        )
    return output_path
