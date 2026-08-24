import io
import math
import wave
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import AudioDecodeError, Waveform, assess_quality, decode_to_pcm16k_mono
from app.main import app
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def stub_attributes():
    """Keep ingest tests off Hugging Face weights."""
    stub = (
        GenderPrediction(prediction=GenderLabel.male, confidence=0.9),
        AgeBracketPrediction(prediction=AgeBracketLabel.age_18_30, confidence=0.8),
    )
    with patch("app.pipeline.predict_attributes", return_value=stub):
        yield stub


def _pcm_wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sine(duration_s: float, freq: float = 220.0, amplitude: float = 0.3, sr: int = 16_000) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def test_short_audio_is_insufficient():
    samples = _sine(0.2)
    quality = assess_quality(Waveform(samples=samples))
    assert quality == AudioQuality.insufficient


def test_silence_is_insufficient():
    samples = np.zeros(16_000, dtype=np.float32)
    quality = assess_quality(Waveform(samples=samples))
    assert quality == AudioQuality.insufficient


def test_clean_tone_is_good():
    samples = _sine(1.5, amplitude=0.3)
    quality = assess_quality(Waveform(samples=samples))
    assert quality == AudioQuality.good


def test_clipped_audio_is_degraded():
    samples = _sine(1.5, amplitude=1.0)
    quality = assess_quality(Waveform(samples=samples))
    assert quality == AudioQuality.degraded


def test_white_noise_is_degraded():
    rng = np.random.default_rng(0)
    samples = (rng.uniform(-0.4, 0.4, 16_000 * 2)).astype(np.float32)
    quality = assess_quality(Waveform(samples=samples))
    assert quality == AudioQuality.degraded


def test_decode_resamples_to_16k_mono():
    wav = _pcm_wav_bytes(_sine(1.0, sr=8_000), sample_rate=8_000)
    waveform = decode_to_pcm16k_mono(wav)
    assert waveform.sample_rate == 16_000
    assert abs(waveform.duration_s - 1.0) < 0.05


def test_decode_rejects_garbage():
    with pytest.raises(AudioDecodeError):
        decode_to_pcm16k_mono(b"this is not audio")


def test_analyze_multipart_reports_good_quality():
    wav = _pcm_wav_bytes(_sine(1.2, amplitude=0.3))
    response = client.post("/analyze", files={"file": ("clip.wav", wav, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["audio_quality"] == "good"
    assert body["gender"]["prediction"] == "male"
    assert body["age_bracket"]["prediction"] == "18-30"


def test_analyze_raw_body_stream():
    wav = _pcm_wav_bytes(_sine(1.2, amplitude=0.3))
    response = client.post(
        "/analyze",
        content=wav,
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    assert response.json()["audio_quality"] == "good"


def test_analyze_raw_body_rejects_garbage():
    response = client.post(
        "/analyze",
        content=b"not-audio",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400


def test_analyze_empty_payload_is_rejected():
    response = client.post(
        "/analyze",
        content=b"",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
