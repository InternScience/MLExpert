"""Download competition data, leaderboards, and official descriptions."""

import logging
import re
import subprocess

import requests
from bs4 import BeautifulSoup

from src.utils import load_config, get_raw_dir, get_output_dir, get_proxy_url, load_kaggle_credentials

logger = logging.getLogger("kaggle_task.downloader")


def download_competition_data(slug: str) -> tuple[bool, str]:
    """Download competition files to the raw-data directory."""
    cfg = load_config()
    kaggle_bin = cfg["kaggle"]["bin"]
    raw_dir = get_raw_dir(slug)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [kaggle_bin, "competitions", "download", "-c", slug, "-p", str(raw_dir)]
    logger.info(f"Downloading data for {slug}...")

    dl_timeout = cfg.get("kaggle", {}).get("download_timeout", 7200)

    import sys as _sys, os as _os
    interactive = _sys.stderr.isatty()
    stderr_dst = None if interactive else subprocess.PIPE

    # Explicitly inject both upper- and lower-case proxy vars so the kaggle CLI
    # always finds them regardless of how the parent shell was launched.
    proxy_url = get_proxy_url(cfg)
    env = _os.environ.copy()
    if proxy_url:
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "KAGGLE_PROXY"):
            env.setdefault(k, proxy_url)

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr_dst,
                                text=True, timeout=dl_timeout, env=env)
        if result.returncode != 0:

            detail = (result.stdout or "").strip() or ((result.stderr or "").strip() if not interactive else "")
            return False, f"Download failed: {detail[:300]}"
        zip_files = list(raw_dir.glob("*.zip"))
        if zip_files:
            logger.info(f"Downloaded: {[f.name for f in zip_files]}")
            return True, str(zip_files[0])
        csv_files = list(raw_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Downloaded CSV directly: {[f.name for f in csv_files]}")
            return True, "direct_csv"
        # no error code but no files → surface any CLI message (e.g. 403 on stdout)
        detail = (result.stdout or "").strip() or (result.stderr or "").strip()
        return False, f"No files downloaded. {detail[:200]}"
    except subprocess.TimeoutExpired:
        return False, f"Download timed out ({dl_timeout}s)"
    except Exception as e:
        return False, str(e)


def download_leaderboard(slug: str) -> tuple[bool, str]:
    """Retrieve paginated private leaderboard scores."""
    import base64
    import pandas as pd

    cfg = load_config()
    output_dir = get_output_dir(slug)
    lb_path = output_dir / "data" / "private" / "private_leaderboard.csv"
    lb_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        creds = load_kaggle_credentials()
    except (OSError, ValueError) as e:
        return False, f"Kaggle credentials unavailable: {type(e).__name__}"
    username, key = creds["username"], creds["key"]

    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{username}:{key}".encode()).decode()
    }
    proxy_url = get_proxy_url(cfg)
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    logger.info(f"Downloading leaderboard for {slug} via API (paginated)...")

    try:
        all_entries = []
        page_token = None

        for page in range(200):  # max 200 pages (20 per page = 4000 teams)
            url = f"https://www.kaggle.com/api/v1/competitions/{slug}/leaderboard/view"
            params = {}
            if page_token:
                params["pageToken"] = page_token

            r = None
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=headers, proxies=proxies,
                                     params=params, timeout=60)
                    break
                except requests.RequestException as e:
                    logger.warning(
                        f"[{slug}] leaderboard page {page + 1} attempt {attempt + 1}/3 "
                        f"failed: {type(e).__name__}")
                    r = None
            if r is None:
                if all_entries:
                    logger.warning(
                        f"[{slug}] stopping at page {page + 1} after repeated timeouts; "
                        f"keeping {len(all_entries)} entries already fetched")
                    break
                return False, "Leaderboard request timed out before any page was fetched"

            if r.status_code != 200:
                if all_entries:
                    break
                return False, f"API returned {r.status_code}: {r.text[:200]}"

            data = r.json()
            entries = data.get("submissions", [])
            all_entries.extend(entries)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        if not all_entries:
            return False, "No leaderboard entries returned"

        lb_df = pd.DataFrame(all_entries)
        lb_df = lb_df.drop(columns=["teamNameNullable", "teamName"], errors="ignore")
        lb_df.to_csv(lb_path, index=False)

        if "score" in lb_df.columns:
            scores = lb_df["score"].astype(float)
            scores.to_frame().to_csv(lb_path.parent / "leaderboard_scores.csv", index=False)

        logger.info(f"Leaderboard saved: {len(lb_df)} teams → {lb_path}")
        return True, f"{len(lb_df)} teams"

    except Exception as e:
        return False, f"Leaderboard download failed: {e}"


def download_public_leaderboard(slug: str) -> tuple[bool, str]:
    """Retrieve the downloadable public leaderboard."""
    import base64
    import io
    import zipfile
    import pandas as pd

    cfg = load_config()
    output_dir = get_output_dir(slug)
    pub_lb_path = output_dir / "data" / "private" / "public_leaderboard.csv"
    pub_lb_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        creds = load_kaggle_credentials()
    except (OSError, ValueError) as e:
        return False, f"Kaggle credentials unavailable: {type(e).__name__}"

    headers = {
        "Authorization": "Basic " + base64.b64encode(
            f"{creds['username']}:{creds['key']}".encode()
        ).decode()
    }
    proxy_url = get_proxy_url(cfg)
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    url = f"https://www.kaggle.com/api/v1/competitions/{slug}/leaderboard/download"

    try:
        r = requests.get(url, headers=headers, proxies=proxies, timeout=60)
        if r.status_code != 200 or len(r.content) < 100:
            return False, f"API returned {r.status_code}"

        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)

        if "Score" in df.columns:
            lb = df[["Score"]].rename(columns={"Score": "score"})
        elif "score" in df.columns:
            lb = df[["score"]]
        else:
            return False, f"No score column: {df.columns.tolist()}"

        lb.to_csv(pub_lb_path, index=False)
        logger.info(f"Public leaderboard saved: {len(lb)} teams → {pub_lb_path}")
        return True, f"{len(lb)} teams"

    except Exception as e:
        return False, f"Public leaderboard download failed: {e}"


def fetch_competition_description(slug: str) -> tuple[bool, str]:
    """Retrieve the official overview, evaluation, and data text."""
    raw_dir = get_raw_dir(slug)
    raw_dir.mkdir(parents=True, exist_ok=True)
    desc_path = raw_dir / "description.txt"

    try:
        from src.scraper import fetch_and_save_description
        success, result = fetch_and_save_description(slug, output_path=desc_path)
        if success:
            content = desc_path.read_text()
            if len(content) > 200:
                return True, result
            logger.warning(f"Playwright got too little content ({len(content)} chars), trying requests fallback")
    except Exception as e:
        logger.warning(f"Playwright scraping failed: {e}, trying requests fallback")

    description_parts = []
    proxy_url = get_proxy_url()
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}
    url = f"https://www.kaggle.com/competitions/{slug}/overview"
    headers = {"User-Agent": "Kaggle-MLE-Factory"}
    try:
        resp = requests.get(url, headers=headers, proxies=proxies, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            content_div = soup.find("div", {"class": re.compile(".*competition-overview.*|.*markdown.*")})
            if content_div:
                description_parts.append(content_div.get_text(separator="\n", strip=True))
            else:
                body = soup.find("body")
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        description_parts.append(text[:5000])
    except Exception as e:
        logger.warning(f"Requests fallback failed for {slug}: {e}")

    if not description_parts:

        return False, f"All scraping failed for {slug} (Playwright + requests, no LLM fallback)"

    full_desc = "\n\n".join(description_parts)
    desc_path.write_text(full_desc, encoding="utf-8")
    logger.info(f"Description saved via requests fallback ({len(full_desc)} chars)")
    return True, str(desc_path)


def download_all(slug: str) -> dict:
    """Download data and leaderboard; description retrieval is the scrape stage."""
    results = {}

    ok, msg = download_competition_data(slug)
    results["data"] = {"success": ok, "message": msg}

    ok, msg = download_leaderboard(slug)
    results["leaderboard"] = {"success": ok, "message": msg}

    return results


if __name__ == "__main__":
    import sys
    from src.utils import setup_logging
    setup_logging()
    slug = sys.argv[1] if len(sys.argv) > 1 else "spaceship-titanic"
    results = download_all(slug)
    for k, v in results.items():
        status = "✓" if v["success"] else "✗"
        print(f"  {status} {k}: {v['message'][:100]}")
