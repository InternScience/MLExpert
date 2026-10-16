"""Retrieve official Kaggle descriptions with Playwright and HTTP fallback."""

import base64
import logging
import re
import time
from pathlib import Path
from typing import Optional
from src.utils import get_proxy_url, load_kaggle_credentials

logger = logging.getLogger("kaggle_task.scraper")

SKIP_SECTIONS = {
    "start here if",
    "practice skills",
    "acknowledgements",
    "kaggle learn",
    "frequently asked",
    "what is a getting started",
    "how do i create",
    "what are notebooks",
    "what’s the difference",
    "why are there",
    "why did my team",
    "how do i contact",
}

EXTRACT_SECTIONS_JS = """
() => {
    const sections = [];
    for (const h3 of document.querySelectorAll('h3')) {
        const name = (h3.innerText || '').trim();
        if (!name) continue;
        const parts = [];
        let el = h3.nextElementSibling;
        while (el && el.tagName !== 'H3' && el.tagName !== 'H2') {
            const text = (el.innerText || '').trim();
            if (text) parts.push(text);
            el = el.nextElementSibling;
        }
        sections.push({ name, text: parts.join('\\n\\n') });
    }
    return sections;
}
"""


def _get_proxy() -> Optional[dict]:
    proxy_url = get_proxy_url()
    return {"server": proxy_url} if proxy_url else None


def _get_kaggle_auth_headers() -> dict:
    cred = load_kaggle_credentials()
    token = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _fetch_competition_files(slug: str) -> list[dict]:
    import requests

    proxies = {}
    proxy_url = get_proxy_url()
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}
    try:
        resp = requests.get(
            f"https://www.kaggle.com/api/v1/competitions/data/list/{slug}",
            headers=_get_kaggle_auth_headers(),
            proxies=proxies,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except Exception as e:
        logger.warning(f"Failed to fetch file list for {slug}: {e}")
    return []


def _should_skip_section(name: str) -> bool:
    lowered = name.lower().strip()
    return any(key in lowered for key in SKIP_SECTIONS)


def _pick_section(sections: dict[str, str], *names: str) -> str:
    for name in names:
        for key, value in sections.items():
            if name.lower() in key.lower() and value.strip():
                return value.strip()
    return ""


def _clean_text(text: str) -> str:
    for noise in (
        "link\nkeyboard_arrow_up",
        "keyboard_arrow_up",
        "get_app fullscreen chevron_right",
        "text_snippet",
        "Metadata",
    ):
        text = text.replace(noise, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_data_description(main_text: str) -> str:
    if "Dataset Description" not in main_text:
        return ""
    text = main_text.split("Dataset Description", 1)[1]
    for stop in (
        "\nUsability",
        "\nLicense",
        "\nExpected update frequency",
        "\nFiles\n",
        "\nExpand\n",
    ):
        idx = text.find(stop)
        if idx > 100:
            text = text[:idx]
            break

    text = re.sub(r"\nFiles\s*\n[\s\S]{0,200}$", "", text)
    return _clean_text(text)


def _format_submission_block(text: str) -> str:
    """Format a submission example as a code block."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    example_lines = []
    for ln in lines:
        if re.match(r"^[A-Za-z].*,", ln) or re.match(r"^\d", ln):
            example_lines.append(ln)
        elif ln.lower().startswith("imageid") or ln.lower().startswith("id,"):
            example_lines.append(ln)
    if not example_lines:
        return text
    intro = text.split(example_lines[0])[0].strip()
    body = "\n".join(example_lines)
    if intro:
        return f"{intro}\n\n```\n{body}\n```"
    return f"```\n{body}\n```"


def format_mlebench_description(
    title: str,
    overview_sections: dict[str, str],
    data_description: str,
    files: list[dict],
) -> str:
    """Organize official text into the MLE task description layout."""
    parts: list[str] = []

    overview_parts: list[str] = []

    description = _pick_section(
        overview_sections,
        "competition description",
        "description",
    )
    if description:
        overview_parts.append(f"## Description\n\n{description}")

    evaluation = _pick_section(overview_sections, "evaluation", "goal")
    metric = _pick_section(overview_sections, "metric")
    eval_text = evaluation
    if metric:
        eval_text = f"{evaluation}\n\n{metric}".strip() if evaluation else metric
    if eval_text:
        overview_parts.append(f"## Evaluation\n\n{eval_text}")

    submission = _pick_section(
        overview_sections,
        "submission file format",
        "submission file",
    )
    if submission:
        overview_parts.append(f"## Submission File\n\n{_format_submission_block(submission)}")

    if overview_parts:
        parts.append("# Overview\n\n" + "\n\n".join(overview_parts))

    data_parts: list[str] = []

    if data_description:
        data_parts.append(f"## Dataset Description\n\n{data_description}")

    if files:
        file_lines = []
        for f in files:
            name = f.get("name", "")
            size = f.get("totalBytes") or f.get("size", "")
            if isinstance(size, int):
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.0f}KB"
                else:
                    size_str = f"{size}B"
            else:
                size_str = str(size) if size else ""
            desc = f.get("description", "").strip()
            line = f"- **{name}**"
            if size_str:
                line += f" ({size_str})"
            if desc:
                line += f" - {desc}"
            file_lines.append(line)
        data_parts.append("### Files\n\n" + "\n".join(file_lines))

    if data_parts:
        parts.append("# Data\n\n" + "\n\n".join(data_parts))

    if not parts:
        return f"# Overview\n\n## Description\n\nKaggle Competition: {title}\n\nDescription could not be fetched.\n"

    return "\n\n".join(parts) + "\n"


def _load_page(page, url: str, timeout: int = 120000, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        logger.info(f"Loading ({attempt}/{retries}): {url}")
        try:
            page.goto(url, timeout=timeout, wait_until="networkidle")
            time.sleep(2)
            return True
        except Exception:
            try:
                page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                for _ in range(20):
                    main = page.query_selector('div[role="main"]')
                    text = main.inner_text() if main else ""
                    if len(text) > 800:
                        return True
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {url}: {e}")
    return False


def _extract_overview_sections(page) -> tuple[str, dict[str, str]]:
    title_el = page.query_selector("h1")
    title = title_el.inner_text().strip() if title_el else ""
    raw_sections = page.evaluate(EXTRACT_SECTIONS_JS)
    sections: dict[str, str] = {}
    for item in raw_sections:
        name = item.get("name", "").strip()
        text = _clean_text(item.get("text", ""))
        if not name or not text or _should_skip_section(name):
            continue
        sections[name] = text

    # Fallback: if h3 extraction missed key sections (Evaluation, Submission),
    # parse them from the content container's full text using keyword splitting.
    needed = {"evaluation", "submission"}
    found = {k.lower() for k in sections}
    if not needed.issubset(found):

        content_el = (
            page.query_selector('[data-testid="competition-detail-render-tid"]')
            or page.query_selector('div[role="main"]')
        )
        if content_el:
            full_text = content_el.inner_text()
            full_text = _clean_text(full_text)
            _parse_overview_from_text(full_text, sections)

    return title, sections


def _parse_overview_from_text(text: str, sections: dict[str, str]):
    """Extract description, evaluation, and submission sections."""
    markers = [
        ("Description", ["Your Goal:", "Competition Description", "Description\n"]),
        ("Evaluation", ["Evaluation\n", "Evaluation\nlink"]),
        ("Submission File", ["Submission File\n", "Submission File Format\n"]),
    ]
    for section_name, keywords in markers:
        if section_name.lower() in {k.lower() for k in sections}:
            continue
        for kw in keywords:
            idx = text.find(kw)
            if idx == -1:
                continue
            chunk = text[idx + len(kw):]

            end_markers = ["Timeline", "About the", "Citation", "Prizes", "Code Requirements",
                           "Evaluation\n", "Submission File\n", "Dataset Description"]
            end = len(chunk)
            for em in end_markers:
                if em.lower() == kw.strip().lower():
                    continue
                em_idx = chunk.find(em)
                if 50 < em_idx < end:
                    end = em_idx
            content = _clean_text(chunk[:end])

            content = _remove_noise_lines(content)
            if len(content) > 10:
                sections[section_name] = content
                break


def _remove_noise_lines(text: str) -> str:
    """Remove recognized Kaggle UI text while preserving source content."""
    noise_patterns = [
        r"^\d+ days? (ago|to go)$",
        r"^more_horiz$",
        r"^expand_all$",
        r"^content_copy$",
        r"^all_inclusive$",
        r"^keyboard_arrow_up$",
        r"^keyboard_arrow_down$",
        r"^get_app fullscreen chevron_right$",
    ]
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        skip = False
        for pat in noise_patterns:
            if re.match(pat, stripped, re.IGNORECASE):
                skip = True
                break
        if not skip:
            filtered.append(line)
    return "\n".join(filtered)


def _extract_data_page(page) -> str:
    content_el = (
        page.query_selector('[data-testid="competition-detail-render-tid"]')
        or page.query_selector('div[role="main"]')
    )
    if not content_el:
        return ""
    text = content_el.inner_text()
    for noise in (
        "Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.",
        "OK, Got it.",
        "Learn more",
    ):
        text = text.replace(noise, "")
    return _extract_data_description(text)


def scrape_competition_description(slug: str) -> str:
    """Retrieve and format the official competition description."""
    from playwright.sync_api import sync_playwright

    proxy = _get_proxy()
    logger.info("Using configured proxy" if proxy else "Using a direct connection")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            proxy=proxy,
            args=["--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page()

        overview_url = f"https://www.kaggle.com/competitions/{slug}/overview"
        if not _load_page(page, overview_url):
            browser.close()
            return f"## Description\n\nKaggle Competition: {slug}\n\nDescription could not be fetched.\n"

        title, overview_sections = _extract_overview_sections(page)
        if not title:
            title = slug.replace("-", " ").title()

        data_description = ""
        data_url = f"https://www.kaggle.com/competitions/{slug}/data"
        if _load_page(page, data_url):
            data_description = _extract_data_page(page)
        else:
            logger.warning("Data page failed to load")

        browser.close()

    files = _fetch_competition_files(slug)
    return format_mlebench_description(title, overview_sections, data_description, files)


def fetch_and_save_description(slug: str, output_path: Optional[Path] = None) -> tuple[bool, str]:
    if output_path is None:
        from src.utils import get_raw_dir
        output_path = get_raw_dir(slug) / "description.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        description = scrape_competition_description(slug)
        if len(description) < 500 or "could not be fetched" in description.lower():
            return False, f"Scraped content too short ({len(description)} chars)"
        output_path.write_text(description, encoding="utf-8")
        logger.info(f"Description saved: {len(description)} chars -> {output_path}")
        return True, str(output_path)
    except Exception as e:
        error_msg = f"Scraping failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        return False, error_msg


if __name__ == "__main__":
    import sys
    from src.utils import setup_logging

    setup_logging()
    slug = sys.argv[1] if len(sys.argv) > 1 else "digit-recognizer"
    print(f"Scraping description for: {slug}")

    success, result = fetch_and_save_description(slug)
    if success:
        desc = Path(result).read_text()
        print(f"\n✓ Playwright succeeded ({len(desc)} chars)")
        print(f"Saved to: {result}")
        print(f"\n{desc}")
        sys.exit(0)

    print(f"\n✗ Playwright failed: {result}")
    sys.exit(1)
