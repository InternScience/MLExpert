#!/usr/bin/env python3
"""Grade one submission using the metric and hidden answers in this task."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Submission CSV to evaluate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        spec = importlib.util.spec_from_file_location("task_metric", root / "metric.py")
        metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)
        submission = pd.read_csv(args.submission)
        answer = pd.read_csv(root / "data/private/test_answer.csv")
        if hasattr(metric, "validate_submission"):
            metric.validate_submission(submission, answer)
        score, higher_is_better = metric.grade(submission, answer)
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Metric returned a non-finite score")
        print(json.dumps({"score": score, "higher_is_better": bool(higher_is_better)}))
    except Exception as error:
        parser.exit(1, f"Grading failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
