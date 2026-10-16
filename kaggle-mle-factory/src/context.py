"""Load complete task evidence for generation and error-feedback repair."""
from importlib.metadata import PackageNotFoundError, version
import sys
from src.utils import get_raw_dir, get_output_dir


def task_description(slug: str, analysis=None) -> str:
    # Reload the source file even when resuming from an old, truncated analysis.
    for path in (get_raw_dir(slug) / "description.txt",
                 get_output_dir(slug) / "data/public/description.txt"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    text = (analysis or {}).get("description", "").strip()
    if not text:
        raise ValueError("No task description available. Run the scrape stage first.")
    return text


def runtime_context() -> str:
    versions = {"Python": sys.version.split()[0]}
    for package in ("pandas", "numpy", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return "Execution environment: " + ", ".join(f"{name}={value}" for name, value in versions.items()) + (
        ". Use APIs supported by these installed versions.\n"
    )
