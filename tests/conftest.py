from unittest.mock import patch

import pytest

from app.pipeline import detect_language
from app.schemas import LanguagePrediction


@pytest.fixture(autouse=True)
def stub_language_detect():
    """Keep tests off Whisper weights."""
    stub = LanguagePrediction(prediction="unknown", confidence=0.0)
    with patch("app.pipeline.detect_language", return_value=stub):
        yield
