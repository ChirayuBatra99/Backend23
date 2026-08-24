# Short note: The detect_language is the main function here, it receives some raw waveform and makes language predictions.


"""Best-effort spoken language ID via Whisper tiny (transformers).

Language is a bonus signal, not on the critical path for gender/age. If the
weights are missing or inference fails, we return unknown rather than failing
the request. Accent is not predicted: that needs a dedicated dialect corpus.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from app.audio import Waveform
from app.schemas import LanguagePrediction

logger = logging.getLogger(__name__)

WHISPER_ID = "openai/whisper-tiny"
MAX_LANGUAGE_SECONDS = 3.0

_NON_LANG_TOKENS = {
    "<|startoftranscript|>",
    "<|endoftext|>",
    "<|notimestamps|>",
    "<|transcribe|>",
    "<|translate|>",
    "<|nospeech|>",
}

_lock = threading.Lock()
_processor = None
_model = None


def detect_language(waveform: Waveform) -> LanguagePrediction:
    try:
        processor, model = _get_whisper()
        import torch

        max_samples = int(MAX_LANGUAGE_SECONDS * waveform.sample_rate)
        samples = np.asarray(waveform.samples[:max_samples], dtype=np.float32)
        inputs = processor(
            samples, sampling_rate=waveform.sample_rate, return_tensors="pt"
        )
        with torch.inference_mode():
            generated = model.generate(inputs.input_features, max_new_tokens=5)
        tokens = processor.tokenizer.convert_ids_to_tokens(generated[0].tolist())
        lang = language_from_whisper_tokens(tokens)
        if lang == "unknown":
            return LanguagePrediction(prediction="unknown", confidence=0.0)
        return LanguagePrediction(prediction=lang, confidence=0.6)
    except Exception:
        logger.warning("language_detect_failed", exc_info=True)
        return LanguagePrediction(prediction="unknown", confidence=0.0)


def language_from_whisper_tokens(tokens: list[str]) -> str:
    for token in tokens:
        if not token.startswith("<|") or not token.endswith("|>"):
            continue
        if token in _NON_LANG_TOKENS:
            continue
        code = token[2:-2]
        if code and code.isalpha() and 2 <= len(code) <= 3:
            return code.lower()
    return "unknown"


def _get_whisper():
    global _processor, _model
    if _model is not None and _processor is not None:
        return _processor, _model
    with _lock:
        if _model is not None and _processor is not None:
            return _processor, _model
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        _processor = WhisperProcessor.from_pretrained(WHISPER_ID)
        _model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID)
        _model.eval()
        return _processor, _model
