"""Validate task data, submission formats, and executable scoring behavior."""

import logging
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import get_output_dir, get_workspace_dir, get_raw_dir, read_csv_safe

logger = logging.getLogger("kaggle_task.validator")


def validate_structure(slug: str) -> list[str]:
    """Check required task files and usable leaderboard scores."""
    errors = []
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    private_dir = output_dir / "data" / "private"

    required_private = ["test_answer.csv"]

    # Preserve competition-specific sample submission filenames.
    ss_hits = [f for f in public_dir.iterdir()
               if 'sample' in f.name.lower() and 'submission' in f.name.lower()]
    if not ss_hits:
        errors.append("Missing: public/sample_submission.csv")
    elif all(f.stat().st_size == 0 for f in ss_hits):
        errors.append("Empty: public/sample_submission.csv")

    for fname in required_private:
        fpath = private_dir / fname
        if not fpath.exists():
            errors.append(f"Missing: private/{fname}")
        elif fpath.stat().st_size == 0:
            errors.append(f"Empty: private/{fname}")

    train_keys = ('train', 'validation', 'de_train')

    def _nested_names(root: Path) -> list[str]:
        """Inspect one directory level and archive member names."""
        names = []
        for f in root.iterdir():
            if f.is_dir():
                try:
                    names.extend(c.name for c in f.iterdir())
                except OSError:
                    pass
            elif f.suffix.lower() == '.zip':
                try:
                    with zipfile.ZipFile(f) as z:
                        names.extend(n.split('/')[-1] for n in z.namelist()[:2000])
                except Exception:
                    pass
        return names

    has_train = any(
        any(k in f.name.lower() for k in train_keys)
        for f in public_dir.iterdir()
        if f.name != 'description.txt' and 'sample' not in f.name.lower()
    )

    if not has_train:
        has_train = any(f.suffix in ('.zarr',) or f.name.lower() in ('scenes', 'train', 'images')
                        for f in public_dir.iterdir() if f.is_dir())

    if not has_train:
        has_train = any(any(k in n.lower() for k in train_keys)
                        for n in _nested_names(public_dir))
    if not has_train:
        errors.append("Missing: no train data found in public/")

    test_keys = ('test', 'comments_to_score', 'de_test')
    has_test = any(
        any(k in f.name.lower() for k in test_keys)
        for f in public_dir.iterdir()
        if f.name != 'description.txt'
        and 'answer' not in f.name.lower() and 'sample' not in f.name.lower()
    )

    if not has_test:
        has_test = any('sample' in f.name.lower() and 'submission' in f.name.lower()
                       for f in public_dir.iterdir())
    if not has_test:
        errors.append("Missing: no test data found in public/")

    def _lb_rows(path):
        if not path.exists():
            return 0
        try:
            lb = pd.read_csv(path)
        except Exception:
            return 0
        if "score" not in lb.columns:
            return 0
        return int(lb["score"].dropna().shape[0])

    priv_rows = _lb_rows(private_dir / "private_leaderboard.csv")
    pub_rows = _lb_rows(private_dir / "public_leaderboard.csv")
    if priv_rows < 5 and pub_rows < 5:
        errors.append(
            "private/private_leaderboard.csv unusable and no public leaderboard to "
            "fall back on — beat_ratio cannot be computed, trajectories will be "
            "misjudged as unscored"
        )
    elif priv_rows < 5:
        errors.append(
            f"Warning: private_leaderboard.csv has {priv_rows} usable rows; "
            f"scoring will fall back to public_leaderboard.csv ({pub_rows} rows), "
            f"whose score distribution differs"
        )

    if not (public_dir / "description.txt").exists():
        errors.append("public/description.txt missing — model cannot understand the task without it")

    return errors


def validate_format(slug: str) -> list[str]:
    """Check submission IDs and row counts against hidden answers."""
    errors = []
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    private_dir = output_dir / "data" / "private"

    # find sample_submission by pattern (original filename preserved from competition)
    ss_hits = sorted([f for f in public_dir.iterdir()
                      if 'sample' in f.name.lower() and 'submission' in f.name.lower()])
    sample_path = ss_hits[0] if ss_hits else None
    sample_zip  = public_dir / "sample_submission.csv.zip"

    if sample_path is not None and sample_path.suffix.lower() == '.zip':
        sample_zip = sample_path; sample_path = None

    if sample_path is not None and sample_path.exists():
        sample_sub = read_csv_safe(sample_path)
    elif sample_zip.exists():
        try:
            with zipfile.ZipFile(sample_zip) as zf:
                inner = [n for n in zf.namelist() if n.endswith('.csv')][0]
                with zf.open(inner) as f:
                    sample_sub = pd.read_csv(f)
        except Exception:
            sample_sub = pd.DataFrame()
    else:
        return ["sample_submission.csv not found"]

    test_answer = read_csv_safe(private_dir / "test_answer.csv")

    if sample_sub.empty:
        errors.append("sample_submission.csv is empty or unreadable")
        return errors
    if test_answer.empty:
        errors.append("test_answer.csv is empty or unreadable")
        return errors

    # Prediction columns can differ from answer columns, as in multiclass
    # probabilities. validate_grading checks the task-specific submission format.
    sub_id = sample_sub.columns[0]
    ans_id = test_answer.columns[0]
    if sub_id != ans_id:
        errors.append(
            f"ID column mismatch: sample_submission first column is '{sub_id}', "
            f"test_answer first column is '{ans_id}'"
        )
    elif set(sample_sub[sub_id].astype(str)) != set(test_answer[ans_id].astype(str)):
        errors.append("ID values in sample_submission do not match test_answer")

    if len(sample_sub) != len(test_answer):
        errors.append(
            f"Row count mismatch: sample_submission has {len(sample_sub)}, "
            f"test_answer has {len(test_answer)}"
        )

    return errors


def validate_no_leakage(slug: str) -> list[str]:
    """Check row-level ID overlap for supported table formats."""
    errors = []
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"
    private_dir = output_dir / "data" / "private"

    test_answer_path = private_dir / "test_answer.csv"
    if not test_answer_path.exists():
        return errors

    test_df = read_csv_safe(test_answer_path)
    if test_df.empty:
        return errors

    id_col = test_df.columns[0]

    if not test_df[id_col].is_unique:
        return ["Warning: Repeated answer IDs need a task-specific composite-ID leakage check"]

    train_file = None
    for pattern in ["train.csv", "train_*.csv", "*train*.csv", "validation.csv", "de_train.csv"]:
        matches = [f for f in public_dir.glob(pattern)
                   if f.is_file() and 'sample' not in f.name.lower()]
        if matches:
            train_file = matches[0]
            break

    if train_file is None:
        for pattern in ["train.csv.zip", "train_*.csv.zip"]:
            matches = list(public_dir.glob(pattern))
            if matches:
                try:
                    with zipfile.ZipFile(matches[0]) as zf:
                        csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
                        with zf.open(csv_name) as f:
                            pd.read_csv(f, usecols=[id_col] if id_col in pd.read_csv(f, nrows=0).columns else None, nrows=0)
                except Exception:
                    pass
                return errors

    if train_file is None:
        return errors

    try:
        train_cols = pd.read_csv(train_file, nrows=0).columns.tolist()
        if id_col not in train_cols:
            return errors

        train_ids = set(pd.read_csv(train_file, usecols=[id_col])[id_col].astype(str))
        test_ids = set(test_df[id_col].astype(str))
        overlap = train_ids & test_ids
        if overlap:
            errors.append(f"DATA LEAKAGE: {len(overlap)} IDs overlap between train and test (id_col={id_col})")
    except Exception as e:
        errors.append(f"Warning: Could not check ID leakage: {e}")

    return errors


def validate_usability(slug: str) -> list[str]:
    """Check that the task contains usable feature data."""
    errors = []
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"

    has_substantial_data = False

    for f in public_dir.iterdir():
        if f.name == 'description.txt':
            continue

        if f.is_dir() and sum(1 for _ in f.iterdir()) > 0:
            has_substantial_data = True
            break

        if f.suffix.lower() in ('.zip', '.parquet', '.hdf5', '.h5', '.jsonl',
                                '.feather', '.npy', '.npz', '.7z', '.json') and f.stat().st_size > 10000:
            has_substantial_data = True
            break

        if (f.suffix.lower() in ('.csv', '.tsv', '.txt', '.data', '.libsvm', '.svm', '.arff')
                and f.stat().st_size > 50000 and 'sample' not in f.name.lower()):
            has_substantial_data = True
            break

    if not has_substantial_data:
        for f in public_dir.iterdir():
            if (f.is_file() and f.suffix.lower() in ('.csv', '.tsv')
                    and 'sample' not in f.name.lower() and 'test' not in f.name.lower()
                    and ('train' in f.name.lower() or f.name.lower() in ('data.csv',))):
                try:
                    head = pd.read_csv(f, nrows=60)
                    if head.shape[1] >= 3 and len(head) >= 50:
                        has_substantial_data = True
                        break
                except Exception:
                    pass

    if not has_substantial_data:
        errors.append("UNUSABLE: No substantial feature data found (no large CSV/TSV/parquet/media directory, no data files)")

    train_files = list(public_dir.glob("train*.csv")) + list(public_dir.glob("*train*.csv"))
    train_files = [f for f in train_files if 'sample' not in f.name.lower() and f.is_file()]

    if train_files and not has_substantial_data:
        try:
            df = pd.read_csv(train_files[0], nrows=2)
            if len(df.columns) <= 2:

                has_media = any(d.is_dir() for d in public_dir.iterdir()
                               if d.name in ['train', 'test', 'train_images', 'test_images',
                                             'train_audio', 'test_audio', 'images'])
                if not has_media:
                    errors.append(
                        f"UNUSABLE: train file only has {len(df.columns)} columns ({list(df.columns)}) "
                        f"and no media directory found — agent has no features to work with"
                    )
        except Exception:
            pass

    return errors


def validate_completeness(slug: str) -> list[str]:
    """Report source feature files absent from the public workspace."""
    errors = []
    raw_dir = get_raw_dir(slug)
    extracted_dir = raw_dir / "extracted"
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"

    if not extracted_dir.exists():
        return errors

    output_files = set()
    for f in public_dir.iterdir():
        if f.name != 'description.txt':
            output_files.add(f.name)
            if f.is_dir():
                output_files.add(f.name + "/")

    # Check first-level files in extracted/
    for f in extracted_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name

        if name.startswith('test') and 'train' not in name:
            continue

        if 'sample_submission' in name.lower() or 'samplesubmission' in name.lower():
            continue

        if any(p in name.lower() for p in ['solution', 'test_label', 'gold_submission', 'ground_truth']):
            continue

        base = name.replace('.zip', '').replace('.csv', '')
        found = (name in output_files or
                 name + '.zip' in output_files or
                 base in output_files or
                 base + '.csv' in output_files or
                 base + '.csv.zip' in output_files or
                 base + '/' in output_files)

        if not found:
            size_kb = f.stat().st_size / 1024
            if size_kb > 10:
                errors.append(f"Warning: auxiliary file '{name}' ({size_kb:.0f}KB) in raw but not in output")

    return errors


def validate_format_preservation(slug: str) -> list[str]:
    """Report changes to source table compression formats."""
    errors = []
    raw_dir = get_raw_dir(slug)
    extracted_dir = raw_dir / "extracted"
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"

    if not extracted_dir.exists():
        return errors

    raw_zips = [f.name for f in extracted_dir.iterdir()
                if f.is_file() and f.name.endswith('.csv.zip')]

    if raw_zips:
        output_names = set(f.name for f in public_dir.iterdir())
        for zip_name in raw_zips:
            csv_name = zip_name.replace('.zip', '')

            if csv_name in output_names and zip_name not in output_names:
                errors.append(f"Warning: original format is '{zip_name}' but output has unzipped '{csv_name}'")

    return errors


def detect_public_test_labels(slug: str) -> list[str]:
    """Report source files that may contain original test labels."""
    info = []
    raw_dir = get_raw_dir(slug)
    extracted_dir = raw_dir / "extracted"

    if not extracted_dir.exists():
        return info

    label_patterns = ['test_labels.csv', 'solution.csv', 'gold_submission.csv',
                      'test_answer.csv', 'ground_truth.csv']

    for pattern in label_patterns:
        fpath = extracted_dir / pattern
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, nrows=2)
                info.append(
                    f"INFO: Found public test labels '{pattern}' (cols={list(df.columns)[:4]}) — "
                    f"consider using Kaggle original split instead of re-splitting from train"
                )
            except Exception:
                pass

    return info


def validate_grading(slug: str) -> list[str]:
    """Check generated grading behavior in an isolated, timed subprocess."""
    from src.grading import check_metric, read_submission
    from src.context import task_description
    from src.utils import load_config
    metric = get_workspace_dir(slug) / "metric.py"
    if not metric.is_file():
        return ["metric.py not found in workspace"]
    try:
        public = get_output_dir(slug) / "data/public"
        submission = read_submission(public)
        answer = pd.read_csv(get_output_dir(slug) / "data/private/test_answer.csv",
                             float_precision="round_trip")
        cfg = load_config().get("validation", {})
        report = check_metric(metric, submission, answer,
                              timeout=cfg.get("grading_timeout", 120),
                              max_rows=cfg.get("max_grading_rows", 1000),
                              description=task_description(slug))
        return ["Grading test failed: " + error for error in report["errors"]] + report["warnings"]
    except Exception as error:
        return [f"Grading test failed: {type(error).__name__}: {error}"]


def validate_answer_nondegenerate(slug: str) -> list[str]:
    """Reject empty or uninformative held-out answers."""
    errors = []
    private_dir = get_output_dir(slug) / "data" / "private"
    answer_path = private_dir / "test_answer.csv"
    if not answer_path.exists():
        return errors
    try:
        ans = read_csv_safe(answer_path)
    except Exception:
        return errors
    if ans.empty or ans.shape[1] < 2:
        return errors

    n = len(ans)

    target_cols = list(ans.columns[1:])
    informative = False
    reasons = []
    for c in target_cols:
        col = ans[c]
        nonnull = col.dropna()
        nonempty = nonnull[nonnull.astype(str).str.strip() != ""]
        nuniq = nonempty.nunique()
        if len(nonempty) == 0:
            reasons.append(f"'{c}' is empty")
        elif nuniq <= 1 and n >= 20:
            reasons.append(f"'{c}' has a single value")
        else:
            informative = True
    if not informative:
        errors.append(
            f"test_answer has no valid ground truth ({'; '.join(reasons)}); "
            f"held-out labels are missing or degenerate")
    return errors


def validate_all(slug: str) -> tuple[bool, list[str]]:
    """Run task checks and return the decision and diagnostics."""
    all_errors = []

    logger.info(f"Validating {slug}...")

    for directory in (get_output_dir(slug) / "data" / "public",
                      get_output_dir(slug) / "data" / "private"):
        if not directory.is_dir():
            return False, [f"Missing data directory: {directory}"]

    all_errors.extend(validate_structure(slug))
    all_errors.extend(validate_format(slug))

    all_errors.extend(validate_answer_nondegenerate(slug))

    all_errors.extend(validate_no_leakage(slug))

    all_errors.extend(validate_usability(slug))

    all_errors.extend(validate_completeness(slug))

    all_errors.extend(validate_format_preservation(slug))

    all_errors.extend(detect_public_test_labels(slug))

    all_errors.extend(validate_grading(slug))

    critical_errors = [e for e in all_errors
                       if not e.startswith("Warning:") and not e.startswith("INFO:")]
    passed = len(critical_errors) == 0

    if passed:
        logger.info(f"✓ {slug} passed all validation checks")
    else:
        logger.warning(f"✗ {slug} failed validation: {len(critical_errors)} errors")
        for err in critical_errors:
            logger.warning(f"  - {err}")

    return passed, all_errors


if __name__ == "__main__":
    import sys as _sys
    from src.utils import setup_logging
    setup_logging()

    if len(_sys.argv) > 1:
        slugs = _sys.argv[1:]
    else:
        slugs = [d.name for d in get_output_dir("").parent.iterdir() if d.is_dir()]

    total_pass = 0
    total_fail = 0
    for slug in sorted(slugs):
        passed, errors = validate_all(slug)
        if passed:
            total_pass += 1
        else:
            total_fail += 1
            print(f"\n✗ {slug}")
            for e in errors:
                if not e.startswith("Warning:") and not e.startswith("INFO:"):
                    print(f"  ✗ {e}")
            for e in errors:
                if e.startswith("Warning:"):
                    print(f"  ⚠ {e}")
            for e in errors:
                if e.startswith("INFO:"):
                    print(f"  ℹ {e}")

    print(f"\n{'='*50}")
    print(f"Total: {total_pass} passed, {total_fail} failed")
