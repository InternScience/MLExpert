"""Profile competition files and suggest a data split."""

import json
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import get_raw_dir

logger = logging.getLogger("kaggle_task.analyzer")


def analyze_competition(slug: str) -> dict:
    """Inspect source files and build structured data profiles."""
    raw_dir = get_raw_dir(slug)
    if not raw_dir.exists():
        return {"error": f"Raw directory not found: {raw_dir}"}

    zip_files = list(raw_dir.glob("*.zip"))
    seven_files = list(raw_dir.glob("*.7z"))
    extract_dir = raw_dir / "extracted"
    if zip_files or seven_files:
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract_dir.mkdir(parents=True, exist_ok=True)
            for zf in zip_files:
                try:
                    with zipfile.ZipFile(zf, "r") as z:
                        z.extractall(extract_dir)
                except Exception as e:
                    logger.warning(f"Failed to extract {zf}: {e}")
            # standalone .7z (e.g. kkbox *.csv.7z) + nested .tsv.7z/.csv.7z
            from src.utils import extract_7z, extract_nested_7z
            for sz in seven_files:
                extract_7z(sz, extract_dir)
            extract_nested_7z(extract_dir)
        data_dir = extract_dir
    else:
        data_dir = raw_dir

    analysis = {
        "slug": slug,
        "data_dir": str(data_dir),
        "files": [],
        "csv_profiles": {},
        "data_type": "unknown",
        "description": "",
    }

    desc_path = raw_dir / "description.txt"
    if desc_path.exists():
        analysis["description"] = desc_path.read_text(encoding="utf-8")

    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            fp = Path(root) / f
            rel = fp.relative_to(data_dir)
            size = fp.stat().st_size
            all_files.append({"path": str(rel), "size": size, "ext": fp.suffix.lower()})

    all_files.sort(key=lambda x: x["size"], reverse=True)
    analysis["files"] = all_files[:50]

    csv_files = [f for f in all_files if f["ext"] == ".csv"]
    for finfo in csv_files[:10]:
        fpath = data_dir / finfo["path"]
        try:
            df = pd.read_csv(fpath, nrows=5, float_precision="round_trip")
            with open(fpath, encoding="utf-8") as source:
                first_line_size = len(source.readline())
            full_shape_est = int(finfo["size"] / max(1, first_line_size)) if finfo["size"] < 500_000_000 else -1
            profile = {
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "shape": [full_shape_est, len(df.columns)],
                "head": df.head(3).to_dict(orient="records"),
                "path": finfo["path"],
            }
            analysis["csv_profiles"][finfo["path"]] = profile
        except Exception as e:
            logger.debug(f"Failed to profile {fpath}: {e}")

    analysis["data_type"] = _detect_data_type(analysis)
    analysis["split_suggestion"] = _suggest_split_strategy(analysis)

    return analysis


def _detect_data_type(analysis: dict) -> str:
    """Infer the data modality from main tables and file types."""
    files = analysis.get("files", [])
    exts = [f["ext"] for f in files]

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
    img_count = sum(1 for e in exts if e in img_exts)
    if img_count > 10:
        return "image"

    # Inspect temporal structure before auxiliary text fields.
    # Inspect main tables for text features.
    for profile_name, profile in analysis.get("csv_profiles", {}).items():

        is_main = any(m in profile_name.lower() for m in ["train", "transaction"])
        if not is_main:
            continue
        cols = [c.lower() for c in profile.get("columns", [])]

        if any("date" in c or "time" in c or "timestamp" in c for c in cols):
            head = profile.get("head", [])
            for row in head:
                for key, val in row.items():
                    if any(kw in key.lower() for kw in ["date", "time"]):
                        val_str = str(val)

                        import re
                        if re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", val_str):
                            return "timeseries"

    # Inspect main tables for text features.
    for profile_name, profile in analysis.get("csv_profiles", {}).items():
        is_main = any(m in profile_name.lower() for m in ["train", "test"])
        if not is_main:
            continue
        cols = [c.lower() for c in profile.get("columns", [])]
        nlp_cols = {"text", "comment", "sentence", "premise", "hypothesis",
                    "question", "answer", "review", "title", "body", "abstract"}
        if any(c in nlp_cols for c in cols):
            return "nlp"

    return "tabular"


def _suggest_split_strategy(analysis: dict) -> dict:
    """Suggest a split strategy for the detected modality."""
    dtype = analysis.get("data_type", "tabular")

    if dtype == "timeseries":
        return {
            "method": "temporal",
            "note": "Split by time column - last N% as test",
        }
    elif dtype == "image":
        return {
            "method": "stratified_random",
            "note": "Random split on labels CSV, copy corresponding image files",
        }
    elif dtype == "nlp":
        return {
            "method": "random",
            "note": "Standard random split on text data",
        }
    else:
        return {
            "method": "random",
            "note": "Standard train_test_split with stratify if classification",
        }


def save_analysis(slug: str, analysis: dict) -> Path:
    """Write the analysis to the task workspace."""
    from src.utils import get_workspace_dir
    ws = get_workspace_dir(slug)
    ws.mkdir(parents=True, exist_ok=True)
    out_path = ws / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Analysis saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    from src.utils import setup_logging
    setup_logging()
    slug = sys.argv[1] if len(sys.argv) > 1 else "spaceship-titanic"
    analysis = analyze_competition(slug)
    save_analysis(slug, analysis)
    print(f"\nData type: {analysis['data_type']}")
    print(f"Files: {len(analysis['files'])}")
    print(f"CSV profiles: {list(analysis['csv_profiles'].keys())}")
    print(f"Split suggestion: {analysis['split_suggestion']}")
