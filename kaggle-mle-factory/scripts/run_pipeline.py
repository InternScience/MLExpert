#!/usr/bin/env python3
"""Build executable MLE tasks from supplied Kaggle competition slugs."""

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logging, get_project_root, get_workspace_dir, get_output_dir, get_raw_dir, validate_slug

logger = logging.getLogger("kaggle_task.pipeline")


def step_download(slug: str) -> dict:
    """Download competition data and leaderboard scores."""
    from src.downloader import download_all
    dl_result = download_all(slug)
    if not dl_result["data"]["success"]:
        return {"success": False, "error": dl_result["data"]["message"]}
    return {"success": True, "detail": dl_result}


def step_scrape(slug: str) -> dict:
    """Retrieve the official description and report missing sections."""
    from src.downloader import fetch_competition_description

    raw_desc = get_raw_dir(slug) / "description.txt"
    out_desc = get_output_dir(slug) / "data" / "public" / "description.txt"
    out_desc.parent.mkdir(parents=True, exist_ok=True)

    method = "none"
    chars = 0
    warnings = []

    ok, msg = fetch_competition_description(slug)
    if ok and raw_desc.exists() and raw_desc.stat().st_size >= 500:
        method = "playwright_or_requests"
        chars = raw_desc.stat().st_size
        shutil.copy2(raw_desc, out_desc)
    else:
        warnings.append(f"Scraping failed: {msg}")
        return {"success": False, "error": f"Description scraping failed (no LLM fallback): {msg}", "warnings": warnings}

    content = raw_desc.read_text(encoding="utf-8").lower()
    missing_sections = []
    for section in ["evaluation", "submission"]:
        if section not in content:
            missing_sections.append(section)
    if "description" not in content and "overview" not in content:
        missing_sections.append("description")

    if missing_sections:
        warnings.append(f"Missing sections: {missing_sections}")

    if chars < 800:
        warnings.append(f"Content may be incomplete ({chars} chars, expected 1000+)")

    return {
        "success": True,
        "method": method,
        "chars": chars,
        "warnings": warnings,
    }


def step_analyze(slug: str) -> dict:
    """Profile the downloaded competition data."""
    from src.analyzer import analyze_competition, save_analysis
    analysis = analyze_competition(slug)
    if "error" in analysis:
        return {"success": False, "error": analysis["error"]}
    save_analysis(slug, analysis)
    return {
        "success": True,
        "data_type": analysis.get("data_type"),
        "csv_count": len(analysis.get("csv_profiles", {})),
    }


def step_generate(slug: str) -> dict:
    """Generate preparation and scoring modules."""
    from src.generator import generate_scripts

    ws = get_workspace_dir(slug)
    analysis_path = ws / "analysis.json"
    if not analysis_path.exists():
        return {"success": False, "error": "No analysis.json. Run analyze first."}

    with open(analysis_path) as f:
        analysis = json.load(f)

    results = generate_scripts(slug, analysis)
    prepare_ok = results.get("prepare", {}).get("valid_syntax", False)
    metric_ok = results.get("metric", {}).get("valid_syntax", False)

    if not prepare_ok:
        return {"success": False, "error": "prepare.py has syntax errors"}
    if not metric_ok:
        return {"success": False, "error": "metric.py is invalid"}
    return {"success": True, "prepare_ok": prepare_ok, "metric_ok": metric_ok}


def step_prepare(slug: str) -> dict:
    """Prepare the task data and restore source metadata."""
    from src.executor import execute_prepare, copy_description, copy_leaderboard

    ok, msg = execute_prepare(slug)
    if not ok:
        return {"success": False, "error": msg[:300]}
    copy_description(slug)
    copy_leaderboard(slug)
    return {"success": True}


def step_validate(slug: str) -> dict:
    """Validate artifacts and retry repairable failures."""
    from src.validator import validate_all
    from src.auto_fix import auto_fix_all

    passed, errors = validate_all(slug)
    if passed:
        return {"success": True, "warnings": [e for e in errors if e.startswith("Warning:")]}

    fix_report = auto_fix_all(slug, errors)
    errors = fix_report.get("remaining_errors", errors)

    if fix_report["success"]:

        passed, errors = validate_all(slug)
        if passed:
            return {
                "success": True,
                "auto_fixed": fix_report["fixes_applied"],
                "warnings": [e for e in errors if e.startswith("Warning:")],
            }

    return {
        "success": False,
        "errors": errors,
        "auto_fixed": fix_report["fixes_applied"],
        "fixes_failed": fix_report["fixes_failed"],
        "manual_needed": fix_report["manual_needed"],
    }


def step_postprocess(slug: str) -> dict:
    """Check task quality and export a standalone task package."""
    from src.postprocess import fix_description, generate_quality_report, save_quality_report, print_quality_report
    from src.validator import validate_all

    passed, errors = validate_all(slug)
    if not passed:
        return {"success": False, "errors": errors}

    # Update the description to match the prepared files.
    fix_result = fix_description(slug)

    # Generate and save the artifact quality report.
    report = generate_quality_report(slug)
    save_quality_report(slug, report)
    print_quality_report(report)

    success = fix_result["success"] and report.get("status") == "pass"
    if success:
        from src.exporter import export_task
        export_task(slug)

    return {
        "success": success,
        "errors": report["issues"] or fix_result["warnings"],
        "description_fixed": fix_result["success"],
        "quality_status": report["status"],
        "issues": report["issues"],
        "warnings": report["warnings"],
    }


STEP_FUNCTIONS = {
    "download": step_download,
    "scrape": step_scrape,
    "analyze": step_analyze,
    "generate": step_generate,
    "prepare": step_prepare,
    "validate": step_validate,
    "postprocess": step_postprocess,
}

ALL_STEPS = ["download", "scrape", "analyze", "generate", "prepare", "validate", "postprocess"]


def run_single(slug: str, steps: list[str]) -> dict:
    """Run a competition through the selected stages, stopping on failure."""
    result = {
        "slug": slug,
        "status": "success",
        "steps": {},
        "errors": [],
        "total_time": 0,
    }
    start_total = time.time()

    for step_name in steps:
        step_fn = STEP_FUNCTIONS[step_name]
        start_step = time.time()

        logger.info(f"[{slug}] {step_name}...")
        try:
            step_result = step_fn(slug)
            elapsed = time.time() - start_step
            step_result["time"] = round(elapsed, 1)
            result["steps"][step_name] = step_result

            if not step_result.get("success"):
                result["status"] = "failed"
                error_msg = step_result.get("error") or (step_result.get("errors") or ["Unknown"])[0]
                result["errors"].append(f"[{step_name}] {error_msg}")
                break

            logger.info(f"[{slug}] {step_name} ✓ ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - start_step
            result["steps"][step_name] = {"success": False, "error": str(e), "time": round(elapsed, 1)}
            result["status"] = "failed"
            result["errors"].append(f"[{step_name}] Exception: {type(e).__name__}: {e}")
            logger.error(f"[{slug}] {step_name} crashed ({elapsed:.0f}s): {e}")
            break

    result["total_time"] = round(time.time() - start_total, 1)
    return result


def run_batch_parallel_download(slugs: list[str]) -> dict[str, dict]:
    """Download competition assets concurrently."""
    from src.downloader import download_all

    results = {}
    logger.info(f"Parallel downloading {len(slugs)} competitions...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_all, slug): slug for slug in slugs}
        for future in as_completed(futures):
            slug = futures[future]
            try:
                dl_result = future.result()
                results[slug] = dl_result
                status = "✓" if dl_result["data"]["success"] else "✗"
                logger.info(f"  {status} {slug} downloaded")
            except Exception as e:
                results[slug] = {"data": {"success": False, "message": str(e)}}
                logger.error(f"  ✗ {slug} download error: {e}")

    return results


def run_batch(slugs: list[str], steps: list[str]) -> list[dict]:
    """Download concurrently, then construct each task sequentially."""
    all_results = []

    if "download" in steps:
        logger.info(f"\n{'='*60}\nPhase 1: Parallel Download ({len(slugs)} competitions)\n{'='*60}")
        dl_results = run_batch_parallel_download(slugs)

        download_ok = []
        for slug in slugs:
            dl = dl_results.get(slug, {})
            if dl.get("data", {}).get("success"):
                download_ok.append(slug)
            else:
                all_results.append({
                    "slug": slug,
                    "status": "failed",
                    "steps": {"download": {"success": False, "error": dl.get("data", {}).get("message", "Unknown")}},
                    "errors": [f"[download] {dl.get('data', {}).get('message', 'Failed')}"],
                    "total_time": 0,
                })
        remaining_steps = [s for s in steps if s != "download"]
    else:
        download_ok = slugs
        remaining_steps = steps

    if remaining_steps:
        logger.info(f"\n{'='*60}\nPhase 2: Sequential Processing ({len(download_ok)} competitions)\n{'='*60}")
        for i, slug in enumerate(download_ok):
            logger.info(f"\n[{i+1}/{len(download_ok)}] {slug}")
            result = run_single(slug, remaining_steps)

            if "download" in steps:
                result["steps"] = {"download": {"success": True}} | result["steps"]
            all_results.append(result)
    else:
        all_results.extend({
            "slug": slug, "status": "success", "steps": {"download": {"success": True}},
            "errors": [], "total_time": 0,
        } for slug in download_ok)

    return all_results


def print_report(results: list[dict]):
    """Summarize completed stages and exported task packages."""
    print(f"\n{'='*70}")
    print("PIPELINE RESULTS")
    print(f"{'='*70}")

    success_list = []
    failed_list = []
    warning_list = []

    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        time_str = f"{r['total_time']:.0f}s" if r.get('total_time') else ""
        steps_summary = " → ".join(
            f"{k}{'✓' if v.get('success') else '✗'}"
            for k, v in r.get("steps", {}).items()
        )
        print(f"  {icon} {r['slug']:50s} {time_str:>6s}  {steps_summary}")

        if r["status"] == "success":
            success_list.append(r)

            scrape_info = r.get("steps", {}).get("scrape", {})
            if scrape_info.get("warnings"):
                warning_list.append(r)
        else:
            failed_list.append(r)

    print(f"\n{'─'*70}")
    print(f"Summary: {len(success_list)} succeeded ✓ | {len(failed_list)} failed ✗ | {len(results)} total")

    if warning_list:
        print(f"\n⚠ SCRAPE WARNINGS ({len(warning_list)}) — description may be incomplete:")
        for r in warning_list:
            scrape = r["steps"]["scrape"]
            method = scrape.get("method", "?")
            chars = scrape.get("chars", 0)
            for w in scrape.get("warnings", []):
                print(f"  ⚠ {r['slug']}: {w} (method={method}, {chars} chars)")

    if failed_list:
        print(f"\n✗ FAILED ({len(failed_list)}):")
        for r in failed_list:
            for err in r["errors"][:2]:
                print(f"  ✗ {r['slug']}: {err[:120]}")

    ready = [r for r in success_list if r.get("steps", {}).get("postprocess", {}).get("success")]
    if ready:
        print(f"\n✓ Ready to use ({len(ready)}):")
        for r in ready:
            print(f"  {get_output_dir(r['slug'])}")


# CLI


def main():
    parser = argparse.ArgumentParser(description="Build runnable MLE tasks from supplied Kaggle competition slugs")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--slug", type=validate_slug, help="Competition slug")
    inputs.add_argument("--slugs", nargs="+", type=validate_slug, help="Explicit competition slugs")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all"] + ALL_STEPS,
                        help="Run one stage (default: all)")
    parser.add_argument("--from", dest="from_step", type=str, default=None,
                        choices=ALL_STEPS,
                        help="Resume from this stage through packaging; overrides --step")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Use an existing raw/<slug>/description.txt; do not scrape")
    args = parser.parse_args()

    setup_logging()

    if args.from_step:
        steps = ALL_STEPS[ALL_STEPS.index(args.from_step):]
    else:
        steps = list(ALL_STEPS) if args.step == "all" else [args.step]
    if args.skip_scrape and "scrape" in steps:
        steps.remove("scrape")

    slugs = [args.slug] if args.slug else args.slugs

    print(f"Processing {len(slugs)} competition(s): steps={steps}\n")

    if len(slugs) == 1:
        result = run_single(slugs[0], steps)
        results = [result]
    else:
        results = run_batch(slugs, steps)

    print_report(results)

    log_path = get_project_root() / "data" / "processing_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nLog saved: {log_path}")
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
