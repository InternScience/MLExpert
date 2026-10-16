"""Discover and filter candidate Kaggle competitions."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from src.utils import load_config, load_exclusion_list, get_project_root

logger = logging.getLogger("kaggle_task.discovery")


def kaggle_api_list_competitions(kaggle_bin: str, page: int = 1) -> list[dict]:
    """Parse competition metadata returned by the Kaggle CLI."""
    cmd = [kaggle_bin, "competitions", "list", "-v", "--page", str(page), "--sort-by", "numberOfTeams"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"kaggle list page {page} failed: {result.stderr[:200]}")
            return []
        # CSV fields can contain quoted commas.
        import csv, io

        # Skip CLI notices before the CSV header.
        out = result.stdout.strip()
        lines = out.split("\n")
        start = next((i for i, l in enumerate(lines) if l.startswith("ref,")), 0)
        rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
        if len(rows) < 2:
            return []
        headers = [h.strip() for h in rows[0]]
        competitions = []
        for fields in rows[1:]:
            if len(fields) >= len(headers):
                comp = dict(zip(headers, [f.strip() for f in fields[:len(headers)]]))
                competitions.append(comp)
        return competitions
    except Exception as e:
        logger.error(f"Failed to list competitions: {e}")
        return []


def filter_competition(comp: dict, exclusion: set, cfg: dict) -> tuple[bool, str]:
    """Check metadata filters and return the decision and reason."""
    from datetime import datetime

    slug = comp.get("ref", "").split("/")[-1] if "/" in comp.get("ref", "") else comp.get("ref", "")

    if not slug:
        return False, "no slug"

    if slug in exclusion:
        return False, "excluded"

    if cfg["filtering"].get("ended_only", True):
        deadline = comp.get("deadline", "")
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                if deadline_dt > datetime.now(deadline_dt.tzinfo):
                    return False, f"not ended yet (deadline={deadline})"
            except (ValueError, TypeError):
                pass

    team_count = 0
    try:
        team_count = int(comp.get("teamCount", 0))
    except (ValueError, TypeError):
        pass
    min_teams = cfg["filtering"].get("min_teams", 50)
    if team_count < min_teams:
        return False, f"too few teams ({team_count} < {min_teams})"

    max_size_bytes = cfg["filtering"].get("max_size_gb", 5) * 1024 * 1024 * 1024
    total_bytes = comp.get("totalBytes", 0) or 0
    if total_bytes > max_size_bytes:
        size_gb = total_bytes / (1024**3)
        return False, f"data too large ({size_gb:.1f}GB > {cfg['filtering']['max_size_gb']}GB)"

    if cfg["filtering"].get("require_csv_submission", True):

        tags = [t.get("name", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in comp.get("tags", [])]
        non_csv_types = {"reinforcement learning", "simulations", "games", "optimization"}
        if any(tag in non_csv_types for tag in tags):

            ml_types = {"tabular", "classification", "regression", "binary classification",
                        "multiclass classification", "nlp", "image", "time series analysis"}
            if not any(tag in ml_types for tag in tags):
                return False, f"non-CSV submission type (tags: {tags})"

    return True, "passed"


def discover_candidates(max_pages: int = 10) -> list[dict]:
    """Collect candidates that pass the configured filters."""
    cfg = load_config()
    kaggle_bin = cfg["kaggle"]["bin"]
    exclusion = load_exclusion_list()

    logger.info(f"Exclusion list has {len(exclusion)} competitions")

    all_candidates = []
    for page in range(1, max_pages + 1):
        comps = kaggle_api_list_competitions(kaggle_bin, page=page)
        if not comps:
            logger.info(f"No more results at page {page}, stopping")
            break

        for comp in comps:
            passed, reason = filter_competition(comp, exclusion, cfg)
            if passed:
                slug = comp.get("ref", "").split("/")[-1] if "/" in comp.get("ref", "") else comp.get("ref", "")
                all_candidates.append({
                    "slug": slug,
                    "title": comp.get("title", ""),
                    "team_count": int(comp.get("teamCount", 0)),
                    "deadline": comp.get("deadline", ""),
                    "category": comp.get("category", ""),
                    "reward": comp.get("reward", ""),
                })
            else:
                slug = comp.get("ref", "")
                logger.debug(f"Filtered out {slug}: {reason}")

        logger.info(f"Page {page}: {len(comps)} total, {len(all_candidates)} candidates so far")

    logger.info(f"Discovery complete: {len(all_candidates)} candidates found")
    return all_candidates


def save_candidates(candidates: list[dict], output_path: Optional[Path] = None):
    """Write candidate metadata as JSON."""
    if output_path is None:
        output_path = get_project_root() / "data" / "candidates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    candidates = discover_candidates()
    save_candidates(candidates)
    print(f"\nFound {len(candidates)} candidate competitions")
    for c in candidates[:10]:
        print(f"  {c['slug']} ({c['team_count']} teams)")
