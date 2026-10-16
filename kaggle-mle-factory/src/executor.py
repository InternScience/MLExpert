"""Execute generated preparation scripts and repair execution errors."""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils import get_raw_dir, get_workspace_dir, get_output_dir, extract_zip, load_config

logger = logging.getLogger("kaggle_task.executor")

MAX_RETRIES = 3


def execute_prepare(slug: str, timeout: int = None) -> tuple[bool, str]:
    """Run preparation with bounded error-feedback repair attempts."""
    if timeout is None:
        timeout = load_config().get("prepare", {}).get("timeout", 5400)
    ws = get_workspace_dir(slug)
    prepare_path = ws / "prepare.py"
    if not prepare_path.exists():
        return False, f"prepare.py not found at {prepare_path}"

    raw_dir = get_raw_dir(slug)
    extracted_dir = raw_dir / "extracted"
    # Retry extraction after an empty directory was left by a failed attempt.
    needs_extract = (not extracted_dir.exists()) or (not any(extracted_dir.iterdir()))
    if needs_extract:
        from src.utils import extract_nested_7z
        zip_files = list(raw_dir.glob("*.zip"))
        seven_files = list(raw_dir.glob("*.7z"))
        if zip_files:
            extracted_dir.mkdir(parents=True, exist_ok=True)
            for zf in zip_files:
                extract_zip(zf, extracted_dir)

            extract_nested_7z(extracted_dir)
        elif seven_files:

            extracted_dir.mkdir(parents=True, exist_ok=True)
            for sz in seven_files:
                from src.utils import extract_7z
                extract_7z(sz, extracted_dir)
            extract_nested_7z(extracted_dir)
        else:
            extracted_dir = raw_dir

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"[{slug}] prepare attempt {attempt}/{MAX_RETRIES}")

        # Clean output before each attempt
        output_dir = get_output_dir(slug)
        public_dir = output_dir / "data" / "public"
        private_dir = output_dir / "data" / "private"

        # Preserve downloaded leaderboards when clearing generated data.
        stashed = {}
        for name in ("private_leaderboard.csv", "public_leaderboard.csv",
                     "leaderboard_scores.csv"):
            f = private_dir / name
            if f.exists() and f.stat().st_size > 0:
                stashed[name] = f.read_bytes()

        for d in [public_dir, private_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        for name, blob in stashed.items():
            (private_dir / name).write_bytes(blob)
        if stashed:
            logger.info(f"[{slug}] preserved {len(stashed)} leaderboard file(s) across clean")

        ok, error_msg = _run_prepare(slug, ws, prepare_path, extracted_dir, public_dir, private_dir, timeout)
        if ok:
            return True, "Success"

        last_error = error_msg
        logger.warning(f"[{slug}] attempt {attempt} failed: {error_msg[:200]}")

        # If not last attempt, regenerate prepare.py with error feedback
        if attempt < MAX_RETRIES:
            logger.info(f"[{slug}] Regenerating prepare.py with error feedback...")
            regenerated = _regenerate_prepare(slug, ws, extracted_dir, error_msg)
            if not regenerated:
                logger.error(f"[{slug}] Failed to regenerate prepare.py")
                break

    return False, last_error


def _run_prepare(slug, ws, prepare_path, extracted_dir, public_dir, private_dir, timeout) -> tuple[bool, str]:
    """Execute one preparation attempt in a timed subprocess."""
    runner_code = f"""
import sys
sys.path.insert(0, {str(ws)!r})
from pathlib import Path

raw_dir = Path({str(extracted_dir)!r})
public_dir = Path({str(public_dir)!r})
private_dir = Path({str(private_dir)!r})

exec(open({str(prepare_path)!r}, encoding='utf-8').read())
prepare(raw_dir, public_dir, private_dir)
print("PREPARE_SUCCESS")
"""
    runner_path = ws / "_runner.py"
    runner_path.write_text(runner_code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ws),
        )

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode == 0 and "PREPARE_SUCCESS" in stdout:
            logger.info(f"prepare.py executed successfully for {slug}")
            return True, "Success"
        else:
            error_msg = stderr[-2000:] if stderr else stdout[-2000:]
            logger.error(f"prepare.py failed for {slug}: {error_msg[:500]}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, f"Execution timed out ({timeout}s)"
    except Exception as e:
        return False, f"Execution error: {e}"


def _regenerate_prepare(slug: str, ws: Path, extracted_dir: Path, prev_error: str) -> bool:
    """Regenerate preparation code from execution or validation errors."""
    from src.generator import _call_llm, _extract_code, _validate_entrypoints
    from src.context import task_description

    files = []
    for root, dirs, fnames in os.walk(extracted_dir):
        for f in fnames:
            fp = Path(root) / f
            rel = fp.relative_to(extracted_dir)
            files.append(f"{rel} ({fp.stat().st_size / 1024:.0f}KB)")
            if len(files) > 30:
                break
        if len(files) > 30:
            break

    analysis_info = ""
    analysis_path = ws / "analysis.json"
    if analysis_path.exists():
        try:
            with open(analysis_path) as f:
                a = json.load(f)
            for name, p in a.get("csv_profiles", {}).items():
                analysis_info += f"  {name}: columns={p['columns']}, shape={p['shape']}\n"
                if p.get("head"):
                    analysis_info += f"    head: {p['head'][:2]}\n"
        except Exception:
            pass

    desc = task_description(slug)

    current_code = ""
    prepare_path = ws / "prepare.py"
    if prepare_path.exists():
        current_code = prepare_path.read_text(encoding="utf-8")

    prompt = f"""Fix this prepare.py that failed with an error.

## Competition: {slug}

## Description:
{desc}

## Files in raw directory:
{chr(10).join(files[:25])}

## CSV profiles:
{analysis_info}

## Previous code that FAILED:
```python
{current_code}
```

## Error message:
{prev_error}

## STRICT REQUIREMENTS:
1. Function signature MUST be: def prepare(raw_dir, public_dir, private_dir):
2. DO NOT use sys.argv, argparse, or if __name__ == "__main__"
3. raw_dir, public_dir, private_dir are Path objects (already created)
4. PRESERVE ORIGINAL FILE STRUCTURE — do NOT merge multiple CSVs into a single train.csv
5. Split the main training data into train/test (90/10), keeping original filenames
6. Copy only necessary feature/metadata files; never copy held-out labels, answer archives, or files that reveal them
7. Output to private_dir: test_answer.csv (ID + ground truth labels)
8. Output sample_submission.csv in the OFFICIAL submission format with dummy predictions. Its prediction columns may differ from the hidden answer columns (e.g. multiclass probabilities).
9. Match submission and answer IDs; keep held-out labels out of all public files. Use real training labels; never fabricate labels to satisfy validation.
10. Use a deterministic 90/10 split, stratifying classification, splitting groups together, and respecting time order when appropriate
11. Only use: pandas, numpy, sklearn, pathlib, shutil, os, zipfile, json
12. Handle the specific error above — fix the root cause

Output ONLY the corrected Python code. No markdown fences. No explanation."""

    try:
        code = _extract_code(_call_llm(prompt, max_tokens=16384))

        if len(code) < 100:
            logger.warning(f"Regenerated code too short: {len(code)} chars")
            return False

        if "def prepare(" not in code:
            logger.warning("Regenerated code missing def prepare()")
            return False

        valid, error = _validate_entrypoints(code, "prepare")
        if not valid:
            raise ValueError(error)
        prepare_path.write_text(code, encoding="utf-8")
        logger.info(f"Regenerated prepare.py: {len(code)} chars")
        return True

    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        return False


def copy_description(slug: str) -> None:
    """Restore the official description in the public task directory."""
    raw_dir = get_raw_dir(slug)
    output_dir = get_output_dir(slug)
    public_dir = output_dir / "data" / "public"

    desc_src = raw_dir / "description.txt"
    desc_dst = public_dir / "description.txt"

    if desc_src.exists() and not desc_dst.exists():
        shutil.copy2(desc_src, desc_dst)
        logger.info(f"Copied description to {desc_dst}")


MIN_LEADERBOARD_ROWS = 5


def leaderboard_rows(path) -> int:
    """Count usable scores; return zero for missing or invalid leaderboards."""
    if not path.exists():
        return 0
    try:
        import pandas as pd
        lb = pd.read_csv(path)
    except Exception:
        return 0
    if "score" not in lb.columns:
        return 0
    return int(lb["score"].dropna().shape[0])


def copy_leaderboard(slug: str) -> None:
    """Restore available leaderboard scores without creating empty substitutes."""
    from src.downloader import download_leaderboard, download_public_leaderboard

    output_dir = get_output_dir(slug)
    private_dir = output_dir / "data" / "private"
    lb_path = private_dir / "private_leaderboard.csv"
    pub_lb_path = private_dir / "public_leaderboard.csv"

    # Private leaderboard
    if leaderboard_rows(lb_path) < MIN_LEADERBOARD_ROWS:
        ok, msg = download_leaderboard(slug)
        if not ok:

            raw_dir = get_raw_dir(slug)
            lb_files = list(raw_dir.glob("*[Ll]eaderboard*"))
            for lb in lb_files:
                dst = private_dir / lb.name
                if not dst.exists():
                    shutil.copy2(lb, dst)
                    logger.info(f"Copied leaderboard from raw: {lb.name}")
            if leaderboard_rows(lb_path) < MIN_LEADERBOARD_ROWS:

                if lb_path.exists():
                    lb_path.unlink()
                logger.warning(
                    f"[{slug}] No private leaderboard available (API failed: {msg}); "
                    f"leaving it missing so validation fails loudly")

    # Public leaderboard
    if leaderboard_rows(pub_lb_path) < MIN_LEADERBOARD_ROWS:
        ok, msg = download_public_leaderboard(slug)
        if not ok:
            logger.warning(f"[{slug}] Public leaderboard download failed: {msg}")


if __name__ == "__main__":
    import sys as _sys
    from src.utils import setup_logging
    setup_logging()
    slug = _sys.argv[1] if len(_sys.argv) > 1 else "spaceship-titanic"
    ok, msg = execute_prepare(slug)
    print(f"{'✓' if ok else '✗'} {msg}")
    if ok:
        copy_description(slug)
        copy_leaderboard(slug)
