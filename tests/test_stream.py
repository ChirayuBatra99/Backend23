import math
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import TARGET_SAMPLE_RATE
from app.main import app
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AnalyzeResponse,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
    LanguagePrediction,
)

client = TestClient(app)


def _pcm_sine(duration_s: float = 1.0, amplitude: float = 0.3) -> bytes:
    n = int(TARGET_SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float64) / TARGET_SAMPLE_RATE
    samples = (amplitude * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _stub_result() -> AnalyzeResponse:
    return AnalyzeResponse(
        contact_id=uuid4(),
        gender=GenderPrediction(prediction=GenderLabel.female, confidence=0.8),
        age_bracket=AgeBracketPrediction(
            prediction=AgeBracketLabel.age_31_45, confidence=0.7
        ),
        processing_ms=0,
        audio_quality=AudioQuality.good,
        language=LanguagePrediction(prediction="en", confidence=0.6),
    )


def test_websocket_emits_partial_then_final():
    stub = _stub_result()
    with patch("app.stream.analyze_waveform", return_value=stub):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_bytes(_pcm_sine(1.0))
            partial = ws.receive_json()
            ws.send_text('{"event": "end"}')
            final = ws.receive_json()

    assert partial["partial"] is True
    assert partial["gender"]["prediction"] == "female"
    assert partial["buffered_ms"] >= 500
    assert final["partial"] is False
    assert final["audio_quality"] == "good"


def test_websocket_rejects_invalid_contact_id():
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/analyze?contact_id=not-a-uuid"):
            pass
