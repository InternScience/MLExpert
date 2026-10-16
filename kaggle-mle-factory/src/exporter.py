"""Assemble a task with data and a standalone submission grading command."""
import shutil

from src.utils import SOURCE_ROOT, get_output_dir, get_workspace_dir


def export_task(slug: str) -> None:
    from src.grading import find_submission
    output = get_output_dir(slug)
    workspace = get_workspace_dir(slug)
    sample_name = find_submission(output / "data/public").name
    shutil.copyfile(workspace / "metric.py", output / "metric.py")
    shutil.copyfile(SOURCE_ROOT / "templates" / "grade.py", output / "grade.py")
    shutil.copyfile(workspace / "prepare.py", output / "prepare.py")
    shutil.copyfile(workspace / "quality_report.json", output / "quality_report.json")
    (output / "requirements.txt").write_text(
        "pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\n", encoding="utf-8",
    )
    (output / "README.md").write_text(
        f"# {slug}\n\n"
        "Give the solving agent access to `data/public/` and the task description "
        "at `data/public/description.txt`. Predictions must follow "
        f"`data/public/{sample_name}`.\n\n"
        "Keep `data/private/`, `prepare.py`, `metric.py` and `grade.py` with the "
        "evaluator. Directory names alone do not enforce access restrictions.\n\n"
        "Install `requirements.txt`, then grade a submission from this directory:\n\n"
        "```bash\npython grade.py /path/to/submission.csv\n```\n\n"
        "The command prints the score and its optimization direction as JSON.\n\n"
        "## Important notes\n\n"
        "Review the generated data split and scoring code against the official "
        "competition rules before training or evaluation. For difficult tasks, "
        "the model may simplify a custom metric or omit parts of its protocol "
        "while producing runnable code. `quality_report.json` records automated "
        "checks; correct any scoring or data-split errors found during review.\n\n"
        "Reuse and redistribution of competition data and descriptions remain "
        "subject to their source rules and licenses.\n",
        encoding="utf-8",
    )
