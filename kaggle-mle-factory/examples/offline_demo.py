#!/usr/bin/env python3
"""Exercise analysis, preparation, validation and grading without network calls.

All rows and leaderboards are synthetic. The prepare and metric scripts are
supplied fixtures; this example does not test LLM generation or Kaggle access.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parent.parent
SLUG = "synthetic-regression"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=SOURCE_ROOT / "demo-workdir")
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    raw = root / "raw" / SLUG
    workspace = root / "workspace" / SLUG
    private = root / "output" / SLUG / "data" / "private"
    for directory in (raw, workspace, private):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "id": [f"row_{i:04d}" for i in range(256)],
        "x": rng.normal(size=256), "z": rng.normal(size=256),
    })
    data["target"] = 2 * data.x + 0.5 * data.z
    data.to_csv(raw / "train.csv", index=False)
    (raw / "description.txt").write_text(
        "# Overview\n## Description\nSynthetic regression demo with generated rows.\n"
        "## Evaluation\nRoot mean squared error (RMSE); lower is better.\n"
        "## Submission\nCSV columns: id,target.\n"
        "# Data\nFeatures x and z; target = 2*x + 0.5*z.\n",
        encoding="utf-8",
    )
    # Seed local synthetic leaderboards so preparation needs no Kaggle request.
    leaderboard = pd.DataFrame({"score": np.linspace(0.1, 4.0, 64)})
    for name in ("private_leaderboard.csv", "public_leaderboard.csv"):
        leaderboard.to_csv(private / name, index=False)
    for name in ("prepare.py", "metric.py"):
        shutil.copyfile(SOURCE_ROOT / "examples" / name, workspace / name)

    # Use the example defaults even if the user has a private config.yaml.
    env = dict(os.environ, MLE_FACTORY_ROOT=str(root),
               MLE_FACTORY_CONFIG=str(SOURCE_ROOT / "config.example.yaml"))
    command = [sys.executable, str(SOURCE_ROOT / "scripts" / "run_pipeline.py"), "--slug", SLUG]
    subprocess.run(command + ["--step", "analyze"], env=env, check=True)
    subprocess.run(command + ["--from", "prepare"], env=env, check=True)

    spec = importlib.util.spec_from_file_location("demo_metric", workspace / "metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    public = root / "output" / SLUG / "data" / "public"
    test = pd.read_csv(public / "test.csv")
    submission = test[["id"]].assign(target=2 * test.x + 0.5 * test.z)
    answer = pd.read_csv(private / "test_answer.csv")
    score, higher_is_better = metric.grade(submission.sample(frac=1, random_state=7), answer)
    perturbed, _ = metric.grade(submission.assign(target=submission.target + 1), answer)
    if higher_is_better or score > 1e-10 or abs(perturbed - 1.0) > 1e-10:
        raise RuntimeError("Synthetic metric did not produce the expected scores")
    submission.to_csv(root / "submission.csv", index=False)
    # The exported task can grade independently of the Factory source tree.
    subprocess.run([sys.executable, str(root / "output" / SLUG / "grade.py"),
                    str(root / "submission.csv")], env=env, check=True)
    print(json.dumps({"demo": SLUG, "rmse": score, "perturbed_rmse": perturbed,
                      "work_dir": str(root), "synthetic_only": True}, indent=2))


if __name__ == "__main__":
    main()
