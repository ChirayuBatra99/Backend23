#!/usr/bin/env python3
"""Evaluate gender/age predictions against a local Common Voice export.

Does not download Common Voice (the license must be accepted in a browser).
Point this at a validated.tsv + clips directory from a Common Voice release.

Example:
  python scripts/eval_common_voice.py --tsv /data/cv/en/validated.tsv \\
      --clips /data/cv/en/clips --max-samples 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval_cv import (
    accuracy,
    expected_calibration_error,
    map_cv_age,
    map_cv_gender,
)
from app.pipeline import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    gender_ok: list[bool] = []
    gender_conf: list[float] = []
    age_ok: list[bool] = []
    age_conf: list[float] = []
    skipped = 0

    with args.tsv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if len(gender_ok) + len(age_ok) >= args.max_samples * 2 and (
                len(gender_ok) >= args.max_samples or len(age_ok) >= args.max_samples
            ):
                if len(gender_ok) >= args.max_samples and len(age_ok) >= args.max_samples:
                    break
            gold_gender = map_cv_gender(row.get("gender"))
            gold_age = map_cv_age(row.get("age"))
            if gold_gender is None and gold_age is None:
                skipped += 1
                continue
            clip = args.clips / row["path"]
            if not clip.is_file():
                skipped += 1
                continue
            try:
                result = analyze(clip.read_bytes(), "00000000-0000-0000-0000-000000000000")
            except Exception as exc:
                print(f"skip {clip.name}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            if gold_gender is not None and len(gender_ok) < args.max_samples:
                pred = result.gender.prediction
                gender_ok.append(pred == gold_gender)
                gender_conf.append(result.gender.confidence)
            if gold_age is not None and len(age_ok) < args.max_samples:
                pred_age = result.age_bracket.prediction
                age_ok.append(pred_age == gold_age)
                age_conf.append(result.age_bracket.confidence)

    print(f"skipped={skipped}")
    print(
        "gender "
        f"n={len(gender_ok)} accuracy={accuracy(gender_ok):.3f} "
        f"ece={expected_calibration_error(gender_conf, gender_ok):.3f}"
    )
    print(
        "age "
        f"n={len(age_ok)} accuracy={accuracy(age_ok):.3f} "
        f"ece={expected_calibration_error(age_conf, age_ok):.3f}"
    )


if __name__ == "__main__":
    main()
