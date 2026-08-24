from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class GenderLabel(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"


class AgeBracketLabel(str, Enum):
    age_18_30 = "18-30"
    age_31_45 = "31-45"
    age_46_60 = "46-60"
    age_60_plus = "60+"
    unknown = "unknown"


class AudioQuality(str, Enum):
    good = "good"
    degraded = "degraded"
    insufficient = "insufficient"


class LabeledPrediction(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class GenderPrediction(LabeledPrediction):
    prediction: GenderLabel


class AgeBracketPrediction(LabeledPrediction):
    prediction: AgeBracketLabel


class LanguagePrediction(LabeledPrediction):
    prediction: str = "unknown"


class AnalyzeResponse(BaseModel):
    contact_id: UUID
    gender: GenderPrediction
    age_bracket: AgeBracketPrediction
    processing_ms: int = Field(ge=0)
    audio_quality: AudioQuality
    language: LanguagePrediction = Field(
        default_factory=lambda: LanguagePrediction(prediction="unknown", confidence=0.0)
    )


class StreamUpdate(AnalyzeResponse):
    partial: bool
    buffered_ms: int
