"""Repair preparation and scoring failures using validation feedback."""

import logging
import shutil
from pathlib import Path

import pandas as pd
from src.utils import get_output_dir, get_workspace_dir, get_raw_dir, load_config

logger = logging.getLogger("kaggle_task.auto_fix")

MAX_METRIC_RETRIES = 3


def auto_fix_all(slug: str, errors=None) -> dict:
    """Repair validation failures while preserving the original report interface."""
    from src.repair import repair_task
    return repair_task(slug, errors)


def auto_copy_auxiliary(slug: str) -> dict:
    """Copy only explicitly configured feature files, never arbitrary raw files."""
    result = {"copied": [], "skipped": []}
    extracted = get_raw_dir(slug) / "extracted"
    public = get_output_dir(slug) / "data/public"
    if not extracted.is_dir():
        return result
    for relative in load_config().get("prepare", {}).get("auxiliary_files", []):
        source = (extracted / relative).resolve()
        destination = (public / relative).resolve()
        if not source.is_relative_to(extracted.resolve()) or not destination.is_relative_to(public.resolve()):
            result["skipped"].append(f"Unsafe auxiliary path: {relative}")
            continue
        if any(token in source.name.lower() for token in ("answer", "solution", "test_label", "ground_truth")):
            result["skipped"].append(f"Answer-like auxiliary file: {relative}")
            continue
        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            result["copied"].append(str(relative))
    return result


def fix_sample_submission(slug: str) -> str:
    """Repair a row-count mismatch in compatible submission templates."""
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    private_dir = output_dir / "data" / "private"

    from src.grading import find_submission
    try:
        sample_path = find_submission(public_dir)
    except (OSError, ValueError):
        return ""
    answer_path = private_dir / "test_answer.csv"

    if not sample_path.exists() or not answer_path.exists():
        return ""

    try:
        sample = pd.read_csv(sample_path)
        answer = pd.read_csv(answer_path)

        if len(sample) == len(answer):
            return ""

        if list(sample.columns) != list(answer.columns):
            return ""

        new_sample = answer.copy()
        target_cols = answer.columns[1:]

        for col in target_cols:
            if pd.api.types.is_numeric_dtype(answer[col]):
                new_sample[col] = 0.0
            else:
                new_sample[col] = answer[col].mode().iloc[0] if len(answer[col].mode()) > 0 else ""

        new_sample.to_csv(sample_path, index=False)
        msg = f"sample_submission fixed: {len(sample)} → {len(new_sample)} rows"
        logger.info(f"[{slug}] {msg}")
        return msg

    except Exception:
        return ""


def auto_fix_metric(slug: str) -> dict:
    """Repair scoring behavior using the same checks as final validation."""
    from src.grading import read_submission
    from src.context import task_description
    metric_path = get_workspace_dir(slug) / "metric.py"
    try:
        sample = read_submission(get_output_dir(slug) / "data/public")
        answer = pd.read_csv(get_output_dir(slug) / "data/private/test_answer.csv", float_precision="round_trip")
    except Exception as error:
        return {"status": "manual", "message": f"Missing scoring inputs: {error}"}
    description = task_description(slug)
    error = _try_run_metric(metric_path, sample, answer, description) if metric_path.exists() else "Missing metric.py"
    if error is None:
        return {"status": "ok", "message": "Scoring behavior checks passed"}
    for attempt in range(1, MAX_METRIC_RETRIES + 1):
        code = _regenerate_metric(slug, error, sample, answer)
        if code is None:
            continue
        metric_path.write_text(code, encoding="utf-8")
        error = _try_run_metric(metric_path, sample, answer, description)
        if error is None:
            return {"status": "fixed", "message": f"Fixed on attempt {attempt}"}
    return {"status": "failed", "message": str(error)}


def _try_run_metric(metric_path: Path, sample_sub, test_answer, description=None) -> str:
    from src.grading import check_metric
    settings = load_config().get("validation", {})
    report = check_metric(metric_path, sample_sub, test_answer,
                          timeout=settings.get("grading_timeout", 120),
                          max_rows=settings.get("max_grading_rows", 1000), description=description)
    return "\n".join(report["errors"]) or None


def _regenerate_metric(slug: str, prev_error: str, sample_sub, test_answer):
    """Regenerate a metric from complete rules, current code and behavioral errors."""
    from src.context import task_description
    from src.generator import _call_llm, _extract_code, _validate_entrypoints
    path = get_workspace_dir(slug) / "metric.py"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    prompt = f"""Repair the metric for Kaggle competition {slug}.

## Complete official task description
{task_description(slug)}

## Submission schema and sample rows
{sample_sub.head(5).to_json(orient="records")}
Columns: {list(sample_sub.columns)}

## Hidden-answer schema and sample rows
{test_answer.head(5).to_json(orient="records")}
Columns: {list(test_answer.columns)}

## Current metric.py
```python
{current}
```

## Failed executable checks
{prev_error}

Return a complete Python module defining validate_submission(submission_df, answer_df)
and grade(submission_df, answer_df). grade returns (finite_score, higher_is_better_boolean).
Implement the OFFICIAL metric, align rows by ID, reject missing/duplicate/mismatched IDs,
wrong row counts and missing prediction columns. Submission columns may differ from
answer columns: multiclass submissions contain probabilities and answers contain labels.
Handle valid changed predictions and shuffled rows. Score must depend on predictions
and true answers, and the declared optimization direction must match the rules.
Use pandas, numpy and sklearn.metrics. If the official metric cannot be implemented,
raise an explicit error; never substitute a different metric or a constant score.
Output only complete Python code.
"""
    try:
        code = _extract_code(_call_llm(prompt, max_tokens=16384))
        valid, error = _validate_entrypoints(code, "grade", "validate_submission")
        if not valid:
            raise ValueError(error)
        return code
    except Exception as error:
        logger.error("Metric regeneration failed: %s", error)
        return None
