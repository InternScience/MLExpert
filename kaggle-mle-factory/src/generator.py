"""Generate task preparation and scoring code from source evidence."""

import json
import ast
import logging
import re
from pathlib import Path
from typing import Optional

from src.utils import get_workspace_dir, get_raw_dir, get_llm_config
from src.context import task_description, runtime_context

logger = logging.getLogger("kaggle_task.generator")


METRIC_PROMPT_TEMPLATE = """You are implementing a scoring metric for a Kaggle competition.

## Competition: {slug}

## Description (focus on the Evaluation section):
{description}

## Data format:
- Columns observed in the original sample submission: {columns}
- First few rows of the original sample submission: {head_rows}
- Likely ID column: {id_column}
- Likely target column(s): {target_columns}
- NOTE: the submission file may have DIFFERENT columns than test_answer (e.g. one probability column per class). Read the Evaluation / Submission File section of the description to learn the real submission format.

## Original data profiles (including labeled training data):
{data_profiles}

## Task:
Generate a Python file `metric.py` with a grading function.

## Requirements:
The file must contain these two functions:

```python
def validate_submission(submission_df, answer_df):
    '''Validate submission format. Raise ValueError if invalid. Return "Valid" if OK.'''
    ...

def grade(submission_df, answer_df):
    '''Score submission against ground truth. Return a single float.
    Also return whether higher_is_better as a second return value.'''
    ...
```

## Rules:
- grade returns (score: float, higher_is_better: bool)
- NEVER return a hardcoded/placeholder score. `return 0.0, True` or `score = 0.0` as a stand-in
  for an unimplemented metric is FORBIDDEN and will be rejected. If the metric is genuinely too
  complex to implement faithfully here (e.g. instance-segmentation mAP needing pycocotools),
  `raise NotImplementedError("metric too complex for standalone impl")` so the task fails loudly
  — a stub that always returns the same number is worse than an honest failure.
- grade MUST actually read the ground-truth in answer_df and compute from the submission content
  (the score must change when the submission changes).
- Align submission and answer BY ID (merge or set_index on the id column) — do NOT assume rows are in the same order
- Match the competition's evaluation metric EXACTLY as described in the Evaluation section
- Preserve the specified transform: RMSE between logarithms means numpy.log(x), not log1p(x). sklearn.mean_squared_log_error uses log1p and is not equivalent to RMSE of log(x). Use log1p only when the rules explicitly specify adding one.
- IMPORTANT: submission and answer may have DIFFERENT columns. The submission follows the competition's submission format; the answer stores the ground-truth label(s). validate_submission must check the SUBMISSION against the official format (required columns present, IDs match, row count matches) — do NOT require `submission.columns == answer.columns`.
- Multi-class log loss where the submission is a probability matrix (one column per class) and the answer holds a single true-label column: build the probability matrix in `sorted(class_names)` order and call `log_loss(y_true, proba, labels=sorted(class_names))`. sklearn sorts `labels` lexicographically and assumes the proba columns follow that sorted order — mismatching this SILENTLY produces a huge wrong loss. ALSO row-normalize the probability matrix to sum to 1 before calling log_loss (Kaggle rescales each row prior to scoring): `rs = proba.sum(axis=1, keepdims=True); rs[rs==0]=1.0; proba = proba/rs`. This both matches Kaggle and silences the "y_prob does not sum to one" warning.
- Multi-label macro-F1: split the space-separated labels, binarize over a fixed class set, then `f1_score(..., average='macro')`.
- Use the correct sklearn metric: roc_auc_score, mean_squared_error (RMSE = numpy.sqrt(mean_squared_error(...))), mean_squared_log_error, accuracy_score, balanced_accuracy_score, log_loss, f1_score, cohen_kappa_score (weights='quadratic' for QWK), average_precision_score, etc. Read the description carefully — e.g. "balanced accuracy" is balanced_accuracy_score, not accuracy_score.
- Only use: pandas, numpy, sklearn.metrics

Output ONLY Python code wrapped in ```python ... ```. No explanations outside the code block."""


def _call_llm(prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
    """Request code from the configured Chat Completions service."""
    llm_cfg = get_llm_config()
    prompt = runtime_context() + "\n" + prompt
    limit = llm_cfg.get("max_prompt_chars", 0)
    if limit and len(prompt) > limit:
        raise ValueError(
            f"Prompt has {len(prompt)} characters, exceeding configured max_prompt_chars={limit}. "
            "Use a larger-context model or increase the explicit limit; task evidence was not truncated."
        )
    from openai import OpenAI
    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["base_url"],
                    timeout=llm_cfg.get("timeout", 300), max_retries=llm_cfg.get("max_retries", 2))
    use_model = model or llm_cfg["model"]

    response = client.chat.completions.create(
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens or llm_cfg.get("max_tokens", 8192),
        temperature=llm_cfg.get("temperature", 1.0),
    )
    return response.choices[0].message.content or ""


def _extract_code(response: str) -> str:
    """Extract Python code from a plain or Markdown-fenced response."""
    text = response.strip()

    for pat in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[0].strip()

    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("```")]
    return "\n".join(lines).strip()


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Check whether generated Python code compiles."""
    try:
        compile(code, "<generated>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _validate_entrypoints(code: str, *required: str) -> tuple[bool, str]:
    valid, error = _validate_syntax(code)
    if not valid:
        return valid, error
    names = {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
    missing = set(required) - names
    return (False, f"Missing functions: {', '.join(sorted(missing))}") if missing else (True, "")


def generate_prepare_script(slug: str, analysis: dict) -> tuple[str, bool]:
    """Generate a preparation function from the complete task evidence."""
    import os
    description = task_description(slug, analysis)
    analysis_subset = {
        "files": analysis.get("files", [])[:20],
        "csv_profiles": analysis.get("csv_profiles", {}),
        "data_type": analysis.get("data_type", "tabular"),
        "split_suggestion": analysis.get("split_suggestion", {}),
    }

    raw_dir = get_raw_dir(slug) / "extracted"
    if not raw_dir.exists():
        raw_dir = get_raw_dir(slug)
    file_list = []
    if raw_dir.exists():
        for root, dirs, fnames in os.walk(raw_dir):
            for f in fnames:
                fp = Path(root) / f
                rel = fp.relative_to(raw_dir)
                file_list.append(f"{rel} ({fp.stat().st_size / 1024:.0f}KB)")
                if len(file_list) > 25:
                    break
            if len(file_list) > 25:
                break

    prompt = f"""You are a data scientist preparing a Kaggle competition dataset for an ML benchmark.

## Competition: {slug}

## Description:
{description}

## Data Analysis:
{json.dumps(analysis_subset, indent=2, ensure_ascii=False)}

## Files in raw directory:
{chr(10).join(file_list[:25])}

## CRITICAL FORMAT — YOU MUST FOLLOW THIS EXACTLY:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import os

def prepare(raw_dir, public_dir, private_dir):
    raw_dir = Path(raw_dir)
    public_dir = Path(public_dir)
    private_dir = Path(private_dir)
    # ... your code ...
```

RULES:
- DO NOT use sys.argv or argparse
- DO NOT write if __name__ == "__main__"
- The function signature MUST be: def prepare(raw_dir, public_dir, private_dir):
- raw_dir, public_dir, private_dir are Path objects passed as arguments

## GROUND TRUTH — THE SINGLE MOST IMPORTANT RULE (read carefully)

The held-out test set's answers (test_answer.csv) MUST be REAL ground-truth labels.

1. Build the held-out test ONLY from data that CONTAINS the true labels — i.e. the
   competition's TRAINING files (train.csv / train_*.tsv / labeled train records).
   NEVER build the test split by sampling the competition's OFFICIAL test file
   (test.csv / test_stage_*.tsv / the provided sample-submission ids). For a FINISHED
   competition those official test files have NO labels, so any answer derived from them
   is fake.
2. NEVER fill test_answer's target with a constant / default / placeholder when the real
   label is not available (e.g. do NOT write `target='NEITHER'`, `label=0`, or copy the
   same string to every row). If you cannot obtain the true label for the held-out rows,
   `raise RuntimeError("no ground-truth labels available for <slug>")` — a broken task
   MUST fail loudly, not ship a fake answer key.
3. MANDATORY final check inside prepare(), right before returning:
   ```python
   _ans = pd.read_csv(private_dir / "test_answer.csv")
   _tgt = _ans.columns[-1]
   if _ans[_tgt].nunique(dropna=True) <= 1:
       raise RuntimeError(f"degenerate test_answer: target '{{_tgt}}' has <=1 unique value")
   ```
   (Skip this assert only for file-path-based answers like segmentation masks.)

## KEY PRINCIPLE: Preserve Original File Structure

The goal is to give the ML agent the SAME file structure that Kaggle contestants saw.
DO NOT merge multiple files into a single train.csv. Keep original filenames.

### What to do:
1. Determine which rows go to train (90%) and which to test (10%)
2. For EACH data file that contains sample-level data:
   - Split it into train/test portions using the same row IDs
   - Keep the original filename pattern (e.g. train_transaction.csv → keep as train_transaction.csv for train rows, test_transaction.csv for test rows)
3. Copy necessary feature/metadata files to public_dir only after checking that they do not expose held-out labels:
   - Mapping tables (breed_labels.csv, color_labels.csv, etc.)
   - Metadata files (metadata.csv, taxonomy.csv, etc.)
   - Any file that does NOT contain the target/label column
   - Large parquet/hdf5 files that contain features
4. For image/audio directories: split files into train/ and test/ subdirectories

### Output structure:
- public_dir/ — ALL files the agent needs (train data with labels, test data without labels, ALL auxiliary files, sample_submission)
- public_dir/sample_submission.csv — a valid DUMMY submission in the EXACT format the competition requires (see the format rules below)
- private_dir/test_answer.csv — ground truth for the test portion (ID column + the TRUE label(s)). This does NOT need the same columns as sample_submission.

### sample_submission format — READ THE "Evaluation" / "Submission File" SECTION OF THE DESCRIPTION
The submission format is defined BY THE COMPETITION, not by the train label column. Do NOT just copy the train target column. Match the described format exactly:
- Single-label classification / regression / binary-probability → ID column + ONE prediction column (use the official column name). Dummy value: 0, 0.5, or the most frequent class.
- For regression evaluated in log space, use a strictly positive dummy prediction such as 1.0.
- Multi-class log loss ("submit a probability for each class") → ID column + ONE PROBABILITY COLUMN PER CLASS, each named after the class. Dummy: uniform 1/num_classes. (e.g. `image,class_a,class_b,...`)
- Multi-label ("multiple labels can be predicted", space-separated) → ID column + ONE label column with the OFFICIAL name (often `Predicted`); values are space-separated label ids. Dummy: "0".
- Ranking / MAP@k / retrieval → ID column + a column of space-separated ranked ids.
- If the description shows an example submission table, REPRODUCE ITS HEADER EXACTLY (same column names, same order).

### test_answer.csv
- Stores the TRUE answer for the test portion. For probability-matrix tasks, store just ID + the true class label (a SINGLE column) — the metric expands it. For multi-label, store the true space-separated labels. test_answer's columns are whatever the metric needs and are INDEPENDENT of sample_submission's columns.

### Split strategy:
- Default: train_test_split(test_size=0.1, random_state=42)
- Time-series (has date column): split by time (last 10%)
- Grouped data (patient_id, essay_id, molecule_name): split by group
- Classification: stratify on target

### Critical checks:
- test_answer's target must be REAL labels with >1 unique value (see GROUND TRUTH rule) — never a constant/placeholder
- sample_submission columns MUST match the official competition submission format (from the description) — NOT necessarily test_answer's columns
- sample_submission row count == test_answer row count, and their ID sets are identical
- Train and test must NOT share any IDs
- Test data files (public test.csv) must NOT contain the target/label column
- Copy necessary auxiliary features; do NOT expose held-out labels, original labeled training archives containing held-out rows, or answer-reconstructing calibration files

Only use: pandas, numpy, sklearn, pathlib, shutil, os, zipfile, json.

Output ONLY Python code. No markdown fences. No explanations."""

    logger.info(f"Generating prepare.py for {slug} (prompt: {len(prompt)} chars)...")
    response = _call_llm(prompt)
    code = _extract_code(response)

    if "def prepare(" not in code:
        logger.warning("No def prepare() found, retrying with higher max_tokens...")
        response = _call_llm(prompt + "\n\nCRITICAL: Your previous response did NOT contain 'def prepare('. You MUST define this function and return the COMPLETE script.", max_tokens=16384)
        code = _extract_code(response)

    valid, err = _validate_entrypoints(code, "prepare")
    if not valid:
        logger.warning(f"Generated prepare.py has syntax error: {err}")
        logger.info("Retrying generation with higher max_tokens...")
        retry_prompt = prompt + f"\n\nPrevious attempt had a syntax error (possibly truncated): {err}\nPlease fix and regenerate the COMPLETE script."
        response = _call_llm(retry_prompt, max_tokens=16384)
        code = _extract_code(response)
        valid, err = _validate_entrypoints(code, "prepare")

    return code, valid


def generate_metric_script(slug: str, analysis: dict) -> tuple[str, bool]:
    """Generate the task-specific submission validator and metric."""
    description = task_description(slug, analysis)

    csv_profiles = analysis.get("csv_profiles", {})
    columns = []
    head_rows = []
    id_column = "id"
    target_columns = []

    for name, profile in csv_profiles.items():
        if "submission" in name.lower() or "sample" in name.lower():
            columns = profile.get("columns", [])
            head_rows = profile.get("head", [])
            if columns:
                id_column = columns[0]
                target_columns = columns[1:]
            break

    if not columns:
        for name, profile in csv_profiles.items():
            if "train" in name.lower():
                all_cols = profile.get("columns", [])
                if all_cols:
                    id_column = all_cols[0]
                    target_columns = all_cols[-1:]
                    columns = [id_column] + target_columns
                break

    prompt = METRIC_PROMPT_TEMPLATE.format(
        slug=slug,
        description=description,
        data_profiles=json.dumps(csv_profiles, ensure_ascii=False, indent=2),
        columns=columns,
        head_rows=json.dumps(head_rows[:3], ensure_ascii=False),
        id_column=id_column,
        target_columns=target_columns,
    )

    logger.info(f"Generating metric.py for {slug}...")
    response = _call_llm(prompt)
    code = _extract_code(response)

    def _metric_valid(c):
        return _validate_entrypoints(c, "grade", "validate_submission")

    valid, err = _metric_valid(code)

    retry_tokens = [16384, 16384]
    for attempt, mt in enumerate(retry_tokens):
        if valid:
            break
        logger.warning(f"Generated metric.py invalid (attempt {attempt+1}): {err}; retrying with max_tokens={mt}")
        retry_prompt = prompt + (
            f"\n\nPrevious attempt was invalid: {err}\n"
            "Return the COMPLETE script including a full def grade(...). "
            "Keep it concise enough to finish; no markdown fences, no commentary."
        )
        response = _call_llm(retry_prompt, max_tokens=mt)
        code = _extract_code(response)
        valid, err = _metric_valid(code)

    return code, valid


def generate_scripts(slug: str, analysis: dict) -> dict:
    """Generate and save the preparation and scoring modules."""
    ws = get_workspace_dir(slug)
    ws.mkdir(parents=True, exist_ok=True)

    results = {}

    prepare_code, prepare_valid = generate_prepare_script(slug, analysis)
    prepare_path = ws / "prepare.py"
    prepare_path.write_text(prepare_code, encoding="utf-8")
    results["prepare"] = {"path": str(prepare_path), "valid_syntax": prepare_valid}

    metric_code, metric_valid = generate_metric_script(slug, analysis)
    metric_path = ws / "metric.py"
    metric_path.write_text(metric_code, encoding="utf-8")
    results["metric"] = {"path": str(metric_path), "valid_syntax": metric_valid}

    logger.info(f"Scripts generated: prepare={'✓' if prepare_valid else '✗'}, metric={'✓' if metric_valid else '✗'}")
    return results


if __name__ == "__main__":
    import sys
    from src.utils import setup_logging
    setup_logging()
    slug = sys.argv[1] if len(sys.argv) > 1 else "spaceship-titanic"
    analysis_path = get_workspace_dir(slug) / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
    else:
        print(f"No analysis found at {analysis_path}. Run analyzer first.")
        sys.exit(1)
    results = generate_scripts(slug, analysis)
    for k, v in results.items():
        print(f"  {k}: {v}")
