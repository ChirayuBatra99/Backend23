from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AnalyzeResponse,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
)


def analyze(audio_bytes: bytes, contact_id: str) -> AnalyzeResponse:
    """Stub inference. Set B/C will decode audio and fill real predictions."""
    _ = audio_bytes
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderPrediction(prediction=GenderLabel.unknown, confidence=0.0),
        age_bracket=AgeBracketPrediction(
            prediction=AgeBracketLabel.unknown, confidence=0.0
        ),
        processing_ms=0,
        audio_quality=AudioQuality.insufficient,
    )
