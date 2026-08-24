import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from app.schemas import AudioQuality

TARGET_SAMPLE_RATE = 16_000
MAX_DECODE_SECONDS = 30
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_DURATION_S = 0.5
SILENCE_RMS = 0.01
CLIP_ABS_THRESHOLD = 0.99
CLIP_RATIO_DEGRADED = 0.01
FLATNESS_DEGRADED = 0.45
FRAME_MS = 20
FRAME_HOP_MS = 10


class AudioDecodeError(Exception):
    """Raised when bytes cannot be decoded to PCM."""


@dataclass(frozen=True)
class Waveform:
    samples: np.ndarray  # float32 in [-1, 1], mono
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)


def _ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        return None


def decode_to_pcm16k_mono(audio_bytes: bytes) -> Waveform:
    if not audio_bytes:
        raise AudioDecodeError("empty audio payload")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise AudioDecodeError("audio payload exceeds size limit")

    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise AudioDecodeError("ffmpeg is not installed on PATH")

    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-fflags",
        "+discardcorrupt",
        "-t",
        str(MAX_DECODE_SECONDS),
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=audio_bytes,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("audio decode timed out") from exc

    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(detail or "ffmpeg could not decode audio")

    pcm = np.frombuffer(completed.stdout, dtype="<i2")
    if pcm.size == 0:
        raise AudioDecodeError("decoded audio was empty")

    samples = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return Waveform(samples=samples, sample_rate=TARGET_SAMPLE_RATE)


def assess_quality(waveform: Waveform) -> AudioQuality:
    if waveform.duration_s < MIN_DURATION_S:
        return AudioQuality.insufficient

    rms = float(np.sqrt(np.mean(np.square(waveform.samples))))
    if rms < SILENCE_RMS:
        return AudioQuality.insufficient

    clip_ratio = float(np.mean(np.abs(waveform.samples) >= CLIP_ABS_THRESHOLD))
    if clip_ratio >= CLIP_RATIO_DEGRADED:
        return AudioQuality.degraded

    # Broadband noise is spectrally flat; speech and tones are peaky.
    if _mean_spectral_flatness(waveform.samples, waveform.sample_rate) >= FLATNESS_DEGRADED:
        return AudioQuality.degraded
    return AudioQuality.good


def _mean_spectral_flatness(samples: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(sample_rate * FRAME_MS / 1000))
    hop = max(1, int(sample_rate * FRAME_HOP_MS / 1000))
    if len(samples) < frame:
        return 1.0

    window = np.hanning(frame).astype(np.float64)
    flats: list[float] = []
    for start in range(0, len(samples) - frame + 1, hop):
        chunk = samples[start : start + frame].astype(np.float64) * window
        power = np.square(np.abs(np.fft.rfft(chunk)))
        power = np.maximum(power, 1e-20)
        geometric = float(np.exp(np.mean(np.log(power))))
        arithmetic = float(np.mean(power))
        flats.append(geometric / arithmetic)
    return float(np.mean(flats))
