from unittest.mock import patch

import numpy as np

from app.audio import Waveform
from app.inference import map_age_years, map_gender, predict_attributes
from app.schemas import AgeBracketLabel, GenderLabel


def test_map_gender_male():
    result = map_gender(child=0.05, female=0.10, male=0.85)
    assert result.prediction is GenderLabel.male
    assert result.confidence == 0.85


def test_map_gender_female():
    result = map_gender(child=0.02, female=0.81, male=0.17)
    assert result.prediction is GenderLabel.female
    assert result.confidence == 0.81


def test_map_gender_child_is_unknown():
    result = map_gender(child=0.72, female=0.15, male=0.13)
    assert result.prediction is GenderLabel.unknown


def test_map_gender_low_confidence_is_unknown():
    result = map_gender(child=0.2, female=0.41, male=0.39)
    assert result.prediction is GenderLabel.unknown
    assert result.confidence == 0.41


def test_map_age_brackets():
    assert map_age_years(24).prediction is AgeBracketLabel.age_18_30
    assert map_age_years(31).prediction is AgeBracketLabel.age_31_45
    assert map_age_years(50).prediction is AgeBracketLabel.age_46_60
    assert map_age_years(72).prediction is AgeBracketLabel.age_60_plus


def test_map_age_under_18_is_unknown():
    result = map_age_years(12.0)
    assert result.prediction is AgeBracketLabel.unknown
    assert result.confidence == 0.0


def test_map_age_confidence_higher_near_center():
    center = map_age_years(24.5)
    edge = map_age_years(18.2)
    assert center.prediction is AgeBracketLabel.age_18_30
    assert edge.prediction is AgeBracketLabel.age_18_30
    assert center.confidence > edge.confidence


def test_degraded_flag_scales_confidence():
    waveform = Waveform(samples=np.zeros(1600, dtype=np.float32))
    scores = (0.24, 0.05, 0.10, 0.85)
    with patch("app.inference._raw_scores", return_value=scores):
        gender, age = predict_attributes(waveform, degraded=False)
        gender_d, age_d = predict_attributes(waveform, degraded=True)
    assert gender.prediction is GenderLabel.male
    assert gender_d.prediction is GenderLabel.male
    assert gender_d.confidence == round(gender.confidence * 0.75, 4)
    assert age_d.confidence == round(age.confidence * 0.75, 4)
