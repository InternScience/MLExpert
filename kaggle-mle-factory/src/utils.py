"""Configuration, paths, archives, and CSV helpers."""

import json
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger("kaggle_task")

SOURCE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("MLE_FACTORY_ROOT", SOURCE_ROOT)).expanduser().resolve()


def load_config() -> dict:
    """Load local settings, falling back to the distributable defaults.

    Runtime data can live outside the source tree via MLE_FACTORY_ROOT.
    Credentials remain in environment variables or an ignored config.yaml.
    """
    custom = os.environ.get("MLE_FACTORY_CONFIG")
    config_path = Path(custom).expanduser() if custom else SOURCE_ROOT / "config.yaml"
    if not custom and not config_path.exists():
        config_path = SOURCE_ROOT / "config.example.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = cfg.setdefault("llm", {})
    for key, env_name in (("api_key", "MLE_FACTORY_LLM_API_KEY"),
                          ("base_url", "MLE_FACTORY_LLM_BASE_URL"),
                          ("model", "MLE_FACTORY_LLM_MODEL")):
        if env_name in os.environ:
            llm[key] = os.environ[env_name]
    return cfg


def get_llm_config() -> dict:
    cfg = load_config()["llm"]
    if not all(cfg.get(name) for name in ("api_key", "base_url", "model")):
        raise ValueError(
            "Set MLE_FACTORY_LLM_API_KEY, MLE_FACTORY_LLM_BASE_URL and "
            "MLE_FACTORY_LLM_MODEL, or fill the llm settings in config.yaml. "
            "A local model server can use a non-secret API key such as 'local'."
        )
    return cfg


def load_kaggle_credentials() -> dict:
    """Read standard Kaggle environment variables or a user credential file."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return {"username": username, "key": key}
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    credential_path = (
        Path(config_dir).expanduser() / "kaggle.json" if config_dir
        else Path(load_config()["kaggle"].get("credentials", "~/.kaggle/kaggle.json")).expanduser()
    )
    with open(credential_path, encoding="utf-8") as f:
        credentials = json.load(f)
    if not credentials.get("username") or not credentials.get("key"):
        raise ValueError("Kaggle credentials must contain username and key")
    return credentials


def get_project_root() -> Path:
    return PROJECT_ROOT


def validate_slug(slug: str) -> str:
    if slug and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug):
        raise ValueError("Competition slug must contain only letters, digits, '-' and '_'")
    return slug


def get_raw_dir(slug: str) -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg["paths"]["raw_dir"] / validate_slug(slug)


def get_workspace_dir(slug: str) -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg["paths"]["workspace_dir"] / validate_slug(slug)


def get_output_dir(slug: str) -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg["paths"]["output_dir"] / validate_slug(slug)


def get_proxy_url(cfg=None) -> str:
    for name in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
        if os.environ.get(name):
            return os.environ[name]
    cfg = cfg if cfg is not None else load_config()
    return cfg.get("kaggle", {}).get("proxy") or cfg.get("proxy", {}).get("url", "")


def load_exclusion_list() -> set:
    """Optional discovery exclusions; explicit construction never consults them."""
    configured = load_config().get("paths", {}).get("exclusion_list", "data/exclusion_list.txt")
    path = PROJECT_ROOT / configured
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def extract_zip(zip_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)
    logger.info(f"Extracted {zip_path} -> {dst}")


def extract_7z(sz_path: Path, dst: Path) -> bool:
    """Extract a single .7z archive (needs py7zr). Returns True on success."""
    try:
        import py7zr
    except ImportError:
        logger.warning("py7zr not installed; cannot extract .7z")
        return False
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        with py7zr.SevenZipFile(str(sz_path), "r") as z:
            z.extractall(path=str(dst))
        logger.info(f"Extracted(7z) {sz_path} -> {dst}")
        return True
    except Exception as e:
        logger.warning(f"7z extract failed for {sz_path}: {e}")
        return False


def extract_nested_7z(root: Path, remove: bool = True, max_gb: float = 10.0) -> int:
    """Recursively extract every .7z under `root` (e.g. Kaggle ships data as
    per-table .tsv.7z/.csv.7z, sometimes inside an outer zip). Skips archives
    larger than max_gb compressed (avoids pulling giant side files like a full
    database dump). Returns the count extracted."""
    try:
        __import__("py7zr")
    except ImportError:
        logger.warning("py7zr not installed; cannot extract nested .7z")
        return 0
    root = Path(root)
    n = 0
    for f in list(root.rglob("*.7z")):
        if max_gb and f.stat().st_size > max_gb * (1024 ** 3):
            logger.warning(f"Skipping large .7z ({f.stat().st_size/1024**3:.1f}G): {f.name}")
            continue
        if extract_7z(f, f.parent):
            n += 1
            if remove:
                try:
                    f.unlink()
                except OSError:
                    pass
    return n


def read_csv_safe(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    try:
        kwargs = {"float_precision": "round_trip"}
        if nrows:
            kwargs["nrows"] = nrows
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return pd.DataFrame()


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
