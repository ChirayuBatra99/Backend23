from uuid import uuid4
from unittest.mock import patch

import numpy as np

from app.audio import Waveform
from app.pipeline import analyze
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
)

_STUB = (
    GenderPrediction(prediction=GenderLabel.female, confidence=0.88),
    AgeBracketPrediction(prediction=AgeBracketLabel.age_31_45, confidence=0.7),
)


def test_insufficient_audio_skips_model_and_returns_unknown():
    waveform = Waveform(samples=np.zeros(800, dtype=np.float32))
    with (
        patch("app.pipeline.decode_to_pcm16k_mono", return_value=waveform),
        patch("app.pipeline.assess_quality", return_value=AudioQuality.insufficient),
        patch("app.pipeline.predict_attributes") as predict,
    ):
        result = analyze(b"audio", str(uuid4()))

    predict.assert_not_called()
    assert result.gender.prediction is GenderLabel.unknown
    assert result.age_bracket.prediction is AgeBracketLabel.unknown
    assert result.audio_quality is AudioQuality.insufficient


def test_good_audio_uses_model_predictions():
    waveform = Waveform(samples=np.ones(16_000, dtype=np.float32) * 0.2)
    with (
        patch("app.pipeline.decode_to_pcm16k_mono", return_value=waveform),
        patch("app.pipeline.assess_quality", return_value=AudioQuality.good),
        patch("app.pipeline.predict_attributes", return_value=_STUB) as predict,
    ):
        result = analyze(b"audio", str(uuid4()))

    predict.assert_called_once()
    assert predict.call_args.kwargs["degraded"] is False
    assert result.gender.prediction is GenderLabel.female
    assert result.age_bracket.prediction is AgeBracketLabel.age_31_45
    assert result.audio_quality is AudioQuality.good


def test_degraded_audio_still_infers_with_degraded_flag():
    waveform = Waveform(samples=np.ones(16_000, dtype=np.float32) * 0.2)
    with (
        patch("app.pipeline.decode_to_pcm16k_mono", return_value=waveform),
        patch("app.pipeline.assess_quality", return_value=AudioQuality.degraded),
        patch("app.pipeline.predict_attributes", return_value=_STUB) as predict,
    ):
        result = analyze(b"audio", str(uuid4()))

    predict.assert_called_once()
    assert predict.call_args.kwargs["degraded"] is True
    assert result.gender.prediction is GenderLabel.female
    assert result.audio_quality is AudioQuality.degraded
