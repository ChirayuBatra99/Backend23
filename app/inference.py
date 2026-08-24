"""Age/gender inference from 16 kHz mono PCM.

Uses audEERING's 6-layer wav2vec2 model (public weights on Hugging Face):
https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender

The 6-layer variant is chosen over the 24-layer one to stay closer to the
500ms budget on a 5s chunk. Weights are CC-BY-NC-SA; swap for a commercially
licensed model before production.
"""

from __future__ import annotations

import threading

import numpy as np

from app.audio import Waveform
from app.schemas import (
    AgeBracketLabel,
    AgeBracketPrediction,
    GenderLabel,
    GenderPrediction,
)

MODEL_ID = "audeering/wav2vec2-large-robust-6-ft-age-gender"
MAX_INFER_SECONDS = 5.0
MIN_GENDER_CONFIDENCE = 0.5
CHILD_DOMINANCE = 0.5
DEGRADED_CONFIDENCE_SCALE = 0.75

# 6-layer model head order: child, female, male (see model card).
_CHILD, _FEMALE, _MALE = 0, 1, 2

_AGE_BRACKETS: tuple[tuple[float, float, AgeBracketLabel], ...] = (
    (18.0, 31.0, AgeBracketLabel.age_18_30),
    (31.0, 46.0, AgeBracketLabel.age_31_45),
    (46.0, 60.0, AgeBracketLabel.age_46_60),
    (60.0, 101.0, AgeBracketLabel.age_60_plus),
)

_lock = threading.Lock()
_processor = None
_model = None


def predict_attributes(
    waveform: Waveform, *, degraded: bool = False
) -> tuple[GenderPrediction, AgeBracketPrediction]:
    age_norm, child, female, male = _raw_scores(waveform)
    gender = map_gender(child=child, female=female, male=male)
    age = map_age_years(float(age_norm) * 100.0)
    if degraded:
        gender = _scale_confidence(gender, DEGRADED_CONFIDENCE_SCALE)
        age = _scale_confidence(age, DEGRADED_CONFIDENCE_SCALE)
    return gender, age


def map_gender(*, child: float, female: float, male: float) -> GenderPrediction:
    if child >= CHILD_DOMINANCE and child >= female and child >= male:
        return GenderPrediction(
            prediction=GenderLabel.unknown, confidence=_clip_conf(child)
        )
    if male >= female:
        label, confidence = GenderLabel.male, male
    else:
        label, confidence = GenderLabel.female, female
    if confidence < MIN_GENDER_CONFIDENCE:
        return GenderPrediction(
            prediction=GenderLabel.unknown, confidence=_clip_conf(confidence)
        )
    return GenderPrediction(prediction=label, confidence=_clip_conf(confidence))


def map_age_years(years: float) -> AgeBracketPrediction:
    if years < 18.0:
        return AgeBracketPrediction(prediction=AgeBracketLabel.unknown, confidence=0.0)
    for low, high, label in _AGE_BRACKETS:
        if low <= years < high:
            return AgeBracketPrediction(
                prediction=label, confidence=_interval_confidence(years, low, high)
            )
    return AgeBracketPrediction(prediction=AgeBracketLabel.age_60_plus, confidence=0.55)


def _interval_confidence(years: float, low: float, high: float) -> float:
    midpoint = (low + high) / 2.0
    half = max((high - low) / 2.0, 1e-6)
    # 1.0 at bin center, ~0.55 at the edges.
    dist = min(abs(years - midpoint) / half, 1.0)
    return _clip_conf(1.0 - 0.45 * dist)


def _scale_confidence(pred, scale: float):
    return pred.model_copy(update={"confidence": _clip_conf(pred.confidence * scale)})


def _clip_conf(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _raw_scores(waveform: Waveform) -> tuple[float, float, float, float]:
    processor, model = _get_model()
    import torch

    max_samples = int(MAX_INFER_SECONDS * waveform.sample_rate)
    samples = np.asarray(waveform.samples[:max_samples], dtype=np.float32)
    inputs = processor(samples, sampling_rate=waveform.sample_rate, return_tensors="pt")
    input_values = inputs["input_values"]
    with torch.inference_mode():
        _, age_logits, gender_probs = model(input_values)
    age_norm = float(age_logits.squeeze().cpu().numpy())
    gender = gender_probs.squeeze().cpu().numpy()
    return (
        age_norm,
        float(gender[_CHILD]),
        float(gender[_FEMALE]),
        float(gender[_MALE]),
    )


def _get_model():
    global _processor, _model
    if _model is not None and _processor is not None:
        return _processor, _model
    with _lock:
        if _model is not None and _processor is not None:
            return _processor, _model
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels: int):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values):
                hidden_states = self.wav2vec2(input_values)[0]
                pooled = torch.mean(hidden_states, dim=1)
                logits_age = self.age(pooled)
                logits_gender = torch.softmax(self.gender(pooled), dim=1)
                return pooled, logits_age, logits_gender

        _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _model = AgeGenderModel.from_pretrained(MODEL_ID)
        _model.eval()
        return _processor, _model
