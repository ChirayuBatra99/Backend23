# Short note: The main pipeline with starting point as analyze function, it receives raw audio bytes and contact id and returns an AnalyzeResponse.

from app.audio import assess_quality, decode_to_pcm16k_mono
from app.inference import predict_attributes
from app.language import detect_language
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    AnalyzeResponse,
    AudioQuality,
    GenderLabel,
    GenderPrediction,
    LanguagePrediction,
)

_UNKNOWN_GENDER = GenderPrediction(prediction=GenderLabel.unknown, confidence=0.0)
_UNKNOWN_AGE = AgeBracketPrediction(prediction=AgeBracketLabel.unknown, confidence=0.0)
_UNKNOWN_LANGUAGE = LanguagePrediction(prediction="unknown", confidence=0.0)


def analyze(audio_bytes: bytes, contact_id: str) -> AnalyzeResponse:
    waveform = decode_to_pcm16k_mono(audio_bytes)
    return analyze_waveform(waveform, contact_id)


def analyze_waveform(waveform, contact_id: str) -> AnalyzeResponse:
    quality = assess_quality(waveform)
    if quality is AudioQuality.insufficient:
        gender, age, language = _UNKNOWN_GENDER, _UNKNOWN_AGE, _UNKNOWN_LANGUAGE
    else:
        gender, age = predict_attributes(
            waveform, degraded=quality is AudioQuality.degraded
        )
        language = detect_language(waveform)
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=gender,
        age_bracket=age,
        processing_ms=0,
        audio_quality=quality,
        language=language,
    )
