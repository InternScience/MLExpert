"""Run generated scoring code in a timed subprocess and check its behavior."""
import importlib.util
import ast
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def find_submission(public_dir):
    matches = sorted(p for p in Path(public_dir).iterdir()
                     if p.is_file() and "sample" in p.name.lower() and "submission" in p.name.lower())
    if not matches:
        raise ValueError("No sample submission found")
    return matches[0]


def read_submission(public_dir):
    return pd.read_csv(find_submission(public_dir), float_precision="round_trip")


def _score(module, submission, answer):
    result = module.grade(submission.copy(), answer.copy())
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("grade must return (score, higher_is_better)")
    score, direction = result
    if not isinstance(direction, (bool, np.bool_)):
        raise ValueError("higher_is_better must be a boolean")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("grade returned a non-finite score")
    return score, bool(direction)


def _oracle_submission(submission, answer):
    """Construct an answer-derived probe only for unambiguous tabular formats."""
    id_col = submission.columns[0]
    if not answer.iloc[:, 0].is_unique:
        return None
    truth = answer.set_index(answer.columns[0]).reindex(submission[id_col])
    columns = list(submission.columns[1:])
    oracle = submission.copy()
    if columns == list(answer.columns[1:]):
        # File paths in answer tables are not necessarily valid predictions.
        if any(truth[c].astype(str).str.contains(r"[/\\]|\.(?:png|npy|jpg)$", regex=True).any()
               for c in columns if not pd.api.types.is_numeric_dtype(truth[c])):
            return None
        oracle[columns] = truth[columns].to_numpy()
        return oracle
    if len(columns) > 1 and truth.shape[1] == 1:
        labels = truth.iloc[:, 0].astype(str)
        if set(labels) <= set(columns):
            for column in columns:
                oracle[column] = np.where(labels.to_numpy() == column, 0.999, 0.001 / (len(columns) - 1))
            return oracle
    return None


def _invariant_score(module, submission, answer):
    score, direction = _score(module, submission, answer)
    for shuffled_sub, shuffled_answer in (
        (submission.sample(frac=1, random_state=11), answer),
        (submission, answer.sample(frac=1, random_state=17)),
    ):
        value, changed_direction = _score(module, shuffled_sub, shuffled_answer)
        if changed_direction != direction or not math.isclose(value, score, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Score changes when row order changes; align submission and answer by ID")
    return score, direction


def inspect_behavior(metric_path, submission, answer):
    spec = importlib.util.spec_from_file_location("task_metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(metric_path).parent))
    exec(compile(Path(metric_path).read_text(encoding="utf-8"), str(metric_path), "exec"), module.__dict__)
    if not callable(getattr(module, "grade", None)) or not callable(getattr(module, "validate_submission", None)):
        raise ValueError("Metric must define grade and validate_submission")
    if submission.empty or answer.empty or len(submission.columns) < 2:
        raise ValueError("Submission and answer must contain rows, IDs and predictions")
    module.validate_submission(submission.copy(), answer.copy())
    baseline, direction = _invariant_score(module, submission, answer)
    scores = [baseline]

    invalid = [submission.iloc[:0], submission.drop(columns=submission.columns[-1])]
    wrong_ids = submission.copy()
    wrong_ids[wrong_ids.columns[0]] = "__factory_invalid_id__"
    invalid.append(wrong_ids)
    for bad in invalid:
        try:
            module.validate_submission(bad, answer.copy())
        except Exception:
            continue
        raise ValueError("validate_submission accepted an invalid submission (IDs, rows or required columns)")

    oracle = _oracle_submission(submission, answer)
    if oracle is not None:
        module.validate_submission(oracle.copy(), answer.copy())
        oracle_score, oracle_direction = _invariant_score(module, oracle, answer)
        if oracle_direction != direction:
            raise ValueError("Metric direction changes between submissions")
        scores.append(oracle_score)
        if len(oracle) > 1:
            for seed in (23, 29, 31):
                probe = oracle.copy()
                probe.iloc[:, 1:] = oracle.iloc[:, 1:].sample(frac=1, random_state=seed).to_numpy()
                try:
                    module.validate_submission(probe.copy(), answer.copy())
                    value, probe_direction = _score(module, probe, answer)
                except Exception:
                    continue
                if probe_direction != direction:
                    raise ValueError("Metric direction changes between submissions")
                scores.append(value)
        if all(math.isclose(value, scores[0], rel_tol=1e-9, abs_tol=1e-12) for value in scores):
            raise ValueError("Score is constant across answer-derived and changed-prediction probes")
        # An answer-derived prediction must not lose to a deliberately mismatched
        # one under the direction declared by the generated metric.
        if any((value > oracle_score + 1e-8 if direction else value < oracle_score - 1e-8)
               for value in scores):
            raise ValueError("Answer-derived predictions score worse than mismatched predictions; check metric direction")
    warnings = [] if oracle is not None else [
        "Warning: Prediction-response check needs a task-specific valid reference submission for this format"
    ]
    return {"errors": [], "warnings": warnings, "score": baseline,
            "higher_is_better": direction, "response_checked": oracle is not None}


def metric_semantic_errors(metric_path, description):
    """Reject a known transform mismatch when the official rule is explicit."""
    match = re.search(r"(?ims)^#{1,6}\s*Evaluation\b(.*?)(?=^#{1,2}\s+\S|\Z)", description or "")
    evaluation = match.group(1).lower() if match else ""
    plain_log_rmse = bool(re.search(r"(?:rmse|root.mean.squared.error).*?between.*?logarithm.*?predicted.*?logarithm.*?observed", evaluation, re.S))
    plus_one = bool(re.search(r"log1p|log\s*\(\s*1\s*\+|plus one|plus 1|rmsle", evaluation))
    if not plain_log_rmse or plus_one:
        return []
    try:
        tree = ast.parse(Path(metric_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []  # The executable check reports missing or malformed code.
    aliases = {alias.asname or alias.name: alias.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) for alias in node.names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            aliases.get(node.func.id, node.func.id) if isinstance(node.func, ast.Name) else "")
        if name in ("mean_squared_log_error", "log1p"):
            return ["Official RMSE between logarithms requires log(x), but the metric calls "
                    f"{name}, which uses log(1+x). Implement the exact stated transform."]
    return []


def check_metric(metric_path, submission, answer, timeout=120, max_rows=1000, description=None):
    """Return diagnostics without running generated metric code in this process."""
    semantics = metric_semantic_errors(metric_path, description)
    if semantics:
        return {"errors": semantics, "warnings": []}
    if max_rows and len(submission) > max_rows:
        submission = submission.sample(n=max_rows, random_state=42)
        # Match IDs rather than independently truncating differently ordered data.
        answer = answer[answer.iloc[:, 0].isin(submission.iloc[:, 0])]
    with tempfile.TemporaryDirectory(prefix="mle-factory-grading-") as temporary:
        root = Path(temporary)
        submission.to_csv(root / "submission.csv", index=False)
        answer.to_csv(root / "answer.csv", index=False)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(Path(metric_path).resolve()), str(root)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            return {"errors": [f"Grading timed out after {timeout}s"], "warnings": []}
        result = root / "result.json"
        if process.returncode != 0 or not result.exists():
            return {"errors": [f"Grading subprocess failed: {(stderr or stdout)[-2000:]}"], "warnings": []}
        return json.loads(result.read_text(encoding="utf-8"))


if __name__ == "__main__":
    metric_path, root = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        report = inspect_behavior(metric_path,
                                  pd.read_csv(root / "submission.csv", float_precision="round_trip"),
                                  pd.read_csv(root / "answer.csv", float_precision="round_trip"))
    except Exception as error:
        report = {"errors": [f"{type(error).__name__}: {error}"], "warnings": []}
    (root / "result.json").write_text(json.dumps(report), encoding="utf-8")
