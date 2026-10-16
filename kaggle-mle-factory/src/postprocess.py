"""Update task descriptions and report artifact quality."""
import re
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import get_output_dir, get_workspace_dir, get_raw_dir

logger = logging.getLogger("kaggle_task.postprocess")


# Part 1: task descriptions


def fix_description(slug: str) -> dict:
    """Update the description to match the prepared public files."""
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    desc_path = public_dir / "description.txt"

    result = {"success": False, "changes_made": [], "warnings": []}

    if not desc_path.exists():
        result["warnings"].append("No description.txt found")
        return result

    content = desc_path.read_text(encoding="utf-8")
    original_content = content

    # Generate actual files section
    actual_files = _build_files_section(public_dir)

    # Replace ### Files section
    pattern = r"### Files\s*\n(.*?)(?=\n##|\n###|\Z)"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, actual_files + "\n", content, count=1, flags=re.DOTALL)
        result["changes_made"].append("Replaced ### Files section")
    else:
        content = content.rstrip() + "\n\n" + actual_files + "\n"
        result["changes_made"].append("Added ### Files section (was missing)")

    # Remove old Note on Data if present
    content = re.sub(r"\n### Note on Data\n.*?(?=\n##|\n###|\Z)", "", content, flags=re.DOTALL)

    # Add Note on Data
    note = _build_data_note(slug)
    content = content.rstrip() + "\n" + note
    result["changes_made"].append("Added/updated Note on Data")

    # Write back
    desc_path.write_text(content, encoding="utf-8")
    result["success"] = True

    if content != original_content:
        logger.info(f"[{slug}] Description fixed: {len(original_content)} → {len(content)} chars")

    return result


def _build_files_section(public_dir: Path) -> str:
    """Describe files present in the public task directory."""
    lines = ["### Files", ""]

    for f in sorted(public_dir.iterdir()):
        if f.name == "description.txt":
            continue

        if f.is_dir():
            file_count = sum(1 for _ in f.rglob("*") if _.is_file())
            lines.append(f"- **{f.name}/** — directory containing {file_count} files")
            continue

        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f}MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size}B"

        # Read columns for CSV files
        col_desc = ""
        if f.suffix == ".csv" and size > 0:
            try:
                df = pd.read_csv(f, nrows=0)
                cols = list(df.columns)
                if len(cols) <= 8:
                    col_desc = f" Columns: `{'`, `'.join(cols)}`."
                else:
                    col_desc = f" Columns: `{'`, `'.join(cols[:5])}`, ... ({len(cols)} total)."
            except Exception:
                pass

        # File-specific descriptions
        name_lower = f.name.lower()
        if "train" in name_lower and f.suffix == ".csv":
            desc = "Training data (with labels)."
        elif "test" in name_lower and "answer" not in name_lower and f.suffix == ".csv":
            desc = "Test data (predict target for these rows)."
        elif "sample" in name_lower and "submission" in name_lower:
            desc = "Example submission file in the required format."
        elif "answer" in name_lower:
            desc = "Ground truth labels."
        else:
            desc = ""

        line = f"- **{f.name}** ({size_str})"
        if desc:
            line += f" — {desc}"
        if col_desc:
            line += col_desc

        lines.append(line)

    return "\n".join(lines)


def _build_data_note(slug: str) -> str:
    """Build Note on Data section with target column info."""
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"

    target_info = ""
    try:
        from src.grading import find_submission
        submission_path = find_submission(public_dir)
        submission_name = submission_path.name
        cols = list(pd.read_csv(submission_path, nrows=0).columns)
        if len(cols) <= 30:
            target_info = f"Submission columns: `{'`, `'.join(cols)}`.\n"
    except Exception:
        submission_name = "the sample submission"

    return f"""
### Note on Data

The data has been re-split from the original training set for evaluation purposes.
The held-out answers are stored separately from the agent-visible inputs.
{target_info}
Your submission file must match the format of {submission_name} exactly.
"""


# Part 2: artifact quality


def generate_quality_report(slug: str) -> dict:
    """Report missing artifacts and task-specific review items."""
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    private_dir = output_dir / "data" / "private"
    ws = get_workspace_dir(slug)
    raw_dir = get_raw_dir(slug)

    report = {
        "slug": slug,
        "issues": [],
        "warnings": [],
        "info": [],
    }

    # --- 1. File completeness ---
    if not public_dir.exists():
        report["issues"].append("output/data/public/ does not exist")
        report["status"] = "fail"
        return report

    has_train = any(public_dir.glob("train*")) or any(public_dir.glob("*train*")) or any(public_dir.glob("validation*")) or any(public_dir.glob("de_train*"))
    has_test = any(public_dir.glob("test*")) or any(public_dir.glob("*test*")) or any(public_dir.glob("comments_to_score*")) or any(public_dir.glob("de_test*"))
    has_sample = any("sample" in p.name.lower() and "submission" in p.name.lower()
                     for p in public_dir.iterdir() if p.is_file())
    has_answer = any(private_dir.glob("test_answer*")) if private_dir.exists() else False
    has_desc = (public_dir / "description.txt").exists()

    if not has_train: report["issues"].append("No training data file found")
    if not has_test: report["issues"].append("No test data file found")
    if not has_sample: report["issues"].append("No sample_submission.csv found")
    if not has_answer: report["issues"].append("No test_answer.csv found")
    if not has_desc: report["issues"].append("No description.txt found")

    # --- 2. Description file reference check ---
    if has_desc:
        desc_content = (public_dir / "description.txt").read_text(encoding="utf-8")
        actual_files = set(f.name for f in public_dir.iterdir() if f.name != "description.txt")

        # Find file references in description body (before ### Files section)
        files_section_start = desc_content.find("### Files")
        body = desc_content[:files_section_start] if files_section_start > 0 else desc_content

        # Look for data file references in the description body.
        referenced_files = set(re.findall(r'[\w\-]+\.(?:csv|zip|parquet|jsonl|tsv)', body))
        for ref_file in referenced_files:
            base_name = ref_file.replace('.zip', '')
            if ref_file not in actual_files and base_name not in actual_files:
                # Check if it's a known pattern (train.csv for train_transaction.csv etc)
                if not any(ref_file.split('.')[0] in af for af in actual_files):
                    report["warnings"].append(f"Description references '{ref_file}' but file not in output")

        # Check for specific counts in description that may be wrong
        count_patterns = re.findall(r'(\d[\d,]+)\s+(?:images|rows|samples|records|documents|entries|observations)', body)
        if count_patterns:
            report["info"].append(f"Description mentions specific counts: {count_patterns} — verify against actual data")

    # --- 3. Metric check ---
    metric_path = ws / "metric.py"
    if metric_path.exists():
        try:
            code = metric_path.read_text(encoding="utf-8")
            compile(code, "metric.py", "exec")
            if "def grade" not in code:
                report["issues"].append("metric.py has no grade() function")
            else:
                report["info"].append("metric.py: syntax OK, has grade()")
        except SyntaxError as e:
            report["issues"].append(f"metric.py syntax error: {e}")
    else:
        report["issues"].append("No metric.py found in workspace")

    # --- 4. ID leakage check ---
    if has_answer and has_train:
        try:
            answer = pd.read_csv(private_dir / "test_answer.csv", nrows=0)
            id_col = answer.columns[0]

            # Find train file
            train_file = None
            for pattern in ["train.csv", "train_*.csv", "*train*.csv", "validation.csv", "de_train.csv"]:
                matches = list(public_dir.glob(pattern))
                if matches:
                    train_file = matches[0]
                    break

            test_file = None
            for pattern in ["test.csv", "test_*.csv", "*test*.csv", "comments_to_score.csv", "de_test.csv"]:
                matches = list(public_dir.glob(pattern))
                if matches:
                    test_file = matches[0]
                    break

            if train_file and test_file and id_col in pd.read_csv(train_file, nrows=0).columns and id_col in pd.read_csv(test_file, nrows=0).columns:
                train_ids = set(pd.read_csv(train_file, usecols=[id_col])[id_col])
                test_ids = set(pd.read_csv(test_file, usecols=[id_col])[id_col])
                overlap = train_ids & test_ids
                if overlap:
                    report["issues"].append(f"ID LEAKAGE: {len(overlap)} overlapping IDs between train and test (id_col={id_col})")
                else:
                    report["info"].append(f"ID check passed: 0 overlap (id_col={id_col})")
        except Exception as e:
            report["warnings"].append(f"Could not check ID leakage: {e}")

    # --- 5. Structure check (multi-file merge detection) ---
    extracted_dir = raw_dir / "extracted"
    if extracted_dir.exists():
        raw_csvs = [f.name for f in extracted_dir.iterdir()
                    if f.is_file() and f.suffix == ".csv"
                    and not f.name.startswith("test")
                    and f.name != "sample_submission.csv"]
        output_csvs = [f.name for f in public_dir.iterdir()
                       if f.is_file() and f.suffix in (".csv", ".parquet")]

        if len(raw_csvs) > 2 and len(output_csvs) <= 3:
            report["warnings"].append(
                f"Possible over-simplification: raw has {len(raw_csvs)} train-related CSVs "
                f"({raw_csvs[:4]}) but output only has {len(output_csvs)} files"
            )

    # Summarize automated artifact checks.
    report["status"] = "pass" if not report["issues"] else "fail"
    return report


def save_quality_report(slug: str, report: dict, output_path: Optional[Path] = None) -> Path:
    """Save quality report to workspace."""
    if output_path is None:
        ws = get_workspace_dir(slug)
        output_path = ws / "quality_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def print_quality_report(report: dict) -> None:
    """Print quality report in human-readable format."""
    slug = report["slug"]
    status = report["status"]
    icon = "✓" if status == "pass" else "✗"

    print(f"\n{icon} {slug}")
    for issue in report["issues"]:
        print(f"  ✗ ISSUE: {issue}")
    for warning in report["warnings"]:
        print(f"  ⚠ WARNING: {warning}")
    for info in report["info"]:
        print(f"  ℹ {info}")
