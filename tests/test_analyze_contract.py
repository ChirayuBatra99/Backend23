import io
import wave
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AgeBracketLabel, AudioQuality, GenderLabel

client = TestClient(app)


def _silent_wav_bytes(duration_ms: int = 100, sample_rate: int = 8000) -> bytes:
    n_frames = sample_rate * duration_ms // 1000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_contract_shape():
    wav = _silent_wav_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()

    UUID(body["contact_id"])
    assert body["gender"]["prediction"] in {label.value for label in GenderLabel}
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert body["age_bracket"]["prediction"] in {
        label.value for label in AgeBracketLabel
    }
    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0
    assert isinstance(body["processing_ms"], int)
    assert body["processing_ms"] >= 0
    assert body["audio_quality"] in {label.value for label in AudioQuality}


def test_analyze_uses_provided_contact_id():
    contact_id = "550e8400-e29b-41d4-a716-446655440000"
    wav = _silent_wav_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"contact_id": contact_id},
    )
    assert response.status_code == 200
    assert response.json()["contact_id"] == contact_id


def test_analyze_rejects_invalid_contact_id():
    wav = _silent_wav_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"contact_id": "not-a-uuid"},
    )
    assert response.status_code == 422
