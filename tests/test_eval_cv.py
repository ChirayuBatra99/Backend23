import pytest

from app.eval_cv import (
    accuracy,
    expected_calibration_error,
    map_cv_age,
    map_cv_gender,
)
from app.schemas import AgeBracketLabel, GenderLabel


def test_map_cv_gender():
    assert map_cv_gender("female") is GenderLabel.female
    assert map_cv_gender("male_masculine") is GenderLabel.male
    assert map_cv_gender("other") is None


def test_map_cv_age_decades():
    assert map_cv_age("twenties") is AgeBracketLabel.age_18_30
    assert map_cv_age("fourties") is AgeBracketLabel.age_31_45
    assert map_cv_age("fifties") is AgeBracketLabel.age_46_60
    assert map_cv_age("sixties") is AgeBracketLabel.age_60_plus
    assert map_cv_age("teens") is None


def test_accuracy_and_ece():
    assert accuracy([True, True, False]) == pytest.approx(2 / 3)
    ece = expected_calibration_error([0.9, 0.9, 0.1], [True, True, False])
    assert 0.0 <= ece <= 1.0
