from app.audio import assess_quality, decode_to_pcm16k_mono
from app.inference import predict_attributes
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AnalyzeResponse,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
)

_UNKNOWN_GENDER = GenderPrediction(prediction=GenderLabel.unknown, confidence=0.0)
_UNKNOWN_AGE = AgeBracketPrediction(prediction=AgeBracketLabel.unknown, confidence=0.0)


def analyze(audio_bytes: bytes, contact_id: str) -> AnalyzeResponse:
    waveform = decode_to_pcm16k_mono(audio_bytes)
    quality = assess_quality(waveform)
    if quality is AudioQuality.insufficient:
        gender, age = _UNKNOWN_GENDER, _UNKNOWN_AGE
    else:
        gender, age = predict_attributes(
            waveform, degraded=quality is AudioQuality.degraded
        )
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=gender,
        age_bracket=age,
        processing_ms=0,
        audio_quality=quality,
    )
