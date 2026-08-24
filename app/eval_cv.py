"""Common Voice label mapping and calibration metrics for the eval harness."""

from __future__ import annotations

from app.schemas import AgeBracketLabel, GenderLabel

# Common Voice `age` is a decade word, not an integer.
CV_AGE_TO_BRACKET: dict[str, AgeBracketLabel] = {
    "twenties": AgeBracketLabel.age_18_30,
    "thirties": AgeBracketLabel.age_31_45,
    "fourties": AgeBracketLabel.age_31_45,
    "forties": AgeBracketLabel.age_31_45,
    "fifties": AgeBracketLabel.age_46_60,
    "sixties": AgeBracketLabel.age_60_plus,
    "seventies": AgeBracketLabel.age_60_plus,
    "eighties": AgeBracketLabel.age_60_plus,
    "nineties": AgeBracketLabel.age_60_plus,
}

CV_GENDER_TO_LABEL: dict[str, GenderLabel] = {
    "male": GenderLabel.male,
    "female": GenderLabel.female,
    "male_masculine": GenderLabel.male,
    "female_feminine": GenderLabel.female,
}


def map_cv_age(raw: str | None) -> AgeBracketLabel | None:
    if not raw:
        return None
    return CV_AGE_TO_BRACKET.get(raw.strip().lower())


def map_cv_gender(raw: str | None) -> GenderLabel | None:
    if not raw:
        return None
    return CV_GENDER_TO_LABEL.get(raw.strip().lower())


def accuracy(correct: list[bool]) -> float:
    if not correct:
        return 0.0
    return sum(correct) / len(correct)


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """ECE over equal-width confidence bins (lower is better)."""
    if not confidences or len(confidences) != len(correct):
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for conf, is_correct in zip(confidences, correct, strict=True):
        idx = min(n_bins - 1, max(0, int(conf * n_bins)))
        bins[idx].append((conf, is_correct))
    ece = 0.0
    n = len(confidences)
    for bucket in bins:
        if not bucket:
            continue
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - mean_conf)
    return ece
