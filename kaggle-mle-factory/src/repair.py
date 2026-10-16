"""Route validation feedback to the code that produced the invalid artifact."""
import json

from src.utils import get_raw_dir, get_workspace_dir, load_config


def repair_task(slug, initial_errors=None):
    from src.auto_fix import auto_copy_auxiliary, fix_sample_submission, auto_fix_metric
    from src.executor import _regenerate_prepare, execute_prepare, copy_description
    from src.validator import validate_all
    report = {"slug": slug, "fixes_applied": [], "fixes_failed": [], "manual_needed": [], "attempts": []}
    workspace = get_workspace_dir(slug)
    workspace.mkdir(parents=True, exist_ok=True)
    errors = initial_errors
    rounds = load_config().get("validation", {}).get("repair_rounds", 2)
    for attempt in range(1, rounds + 1):
        if errors is None:
            passed, errors = validate_all(slug)
            if passed:
                break
        critical = [e for e in errors if not e.startswith(("Warning:", "INFO:"))]
        if not critical:
            break
        record = {"round": attempt, "errors_before": critical, "actions": []}
        report["attempts"].append(record)
        try:
            # Restore source descriptions, but do not fabricate unavailable assets.
            if any("description.txt" in e for e in critical):
                copy_description(slug)
                record["actions"].append("restore_description")
            copied = auto_copy_auxiliary(slug)
            report["manual_needed"].extend(copied["skipped"])
            if copied["copied"]:
                report["fixes_applied"].append(f"Copied approved auxiliary files: {copied['copied']}")
            sample_fix = fix_sample_submission(slug)
            if sample_fix:
                report["fixes_applied"].append(sample_fix)
            passed, errors = validate_all(slug)
            if passed:
                break
            critical = [e for e in errors if not e.startswith(("Warning:", "INFO:"))]
            unavailable = [e for e in critical if "leaderboard" in e or "description.txt" in e]
            if unavailable:
                report["manual_needed"].extend(unavailable)
                break
            data_errors = [e for e in critical if not e.startswith(("Grading test failed:", "metric.py"))]
            if data_errors:
                record["actions"].append("regenerate_prepare")
                raw = get_raw_dir(slug)
                extracted = raw / "extracted" if (raw / "extracted").is_dir() else raw
                if not _regenerate_prepare(slug, workspace, extracted, "\n".join(data_errors)):
                    report["fixes_failed"].append("prepare regeneration failed")
                    continue
                ok, message = execute_prepare(slug)
                if not ok:
                    report["fixes_failed"].append(f"prepare execution failed: {message}")
                    continue
                copy_description(slug)
                report["fixes_applied"].append("Regenerated and executed prepare.py from validation feedback")
                passed, errors = validate_all(slug)
                if passed:
                    break
            # Preserve the metric when the remaining failure is a data/split issue.
            if any(e.startswith(("Grading test failed:", "metric.py")) for e in errors):
                record["actions"].append("regenerate_metric")
                result = auto_fix_metric(slug)
                if result["status"] == "fixed":
                    report["fixes_applied"].append(result["message"])
                elif result["status"] != "ok":
                    report["fixes_failed"].append(result["message"])
        except Exception as error:
            report["fixes_failed"].append(f"{type(error).__name__}: {error}")
        errors = None  # Revalidate the actual new artifacts before another round.
    passed, final_errors = validate_all(slug)
    report["success"] = passed
    report["remaining_errors"] = final_errors
    (workspace / "repair_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
