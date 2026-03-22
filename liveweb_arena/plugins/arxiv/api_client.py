"""ArXiv API client with rate limiting.

Fetches the HTML listing page (arxiv.org/list/<category>/new) and parses
paper metadata (title, authors, primary category) directly from the page
HTML.  This guarantees the ground-truth data matches what the agent sees.

Rate limit: ArXiv requests max 1 request per 3 seconds.
"""

import asyncio
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

import aiohttp

from liveweb_arena.plugins.base_client import APIFetchError, BaseAPIClient, RateLimiter

logger = logging.getLogger(__name__)

CACHE_SOURCE = "arxiv"

# Shared session for connection reuse
_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            headers={"User-Agent": "LiveWebArena/1.0"},
        )
    return _session


async def close_session():
    """Close the shared session. Call during shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
    _session = None


# ---------------------------------------------------------------------------
# HTML listing page parsing
# ---------------------------------------------------------------------------

# Matches arxiv IDs like "2603.17021" in <dt> blocks
_DT_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")

# Extracts title text after the "Title:" descriptor span
_TITLE_RE = re.compile(
    r"<div\s+class=['\"]list-title\s+mathjax['\"]>"
    r"\s*<span\s+class=['\"]descriptor['\"]>Title:</span>\s*(.*?)\s*</div>",
    re.DOTALL,
)

# Extracts author names from <a> tags inside list-authors div
_AUTHORS_DIV_RE = re.compile(
    r"<div\s+class=['\"]list-authors['\"]>(.*?)</div>", re.DOTALL
)
_AUTHOR_NAME_RE = re.compile(r"<a[^>]*>([^<]+)</a>")

# Extracts primary subject: "Artificial Intelligence (cs.AI)" → "cs.AI"
_PRIMARY_SUBJECT_RE = re.compile(
    r'<span\s+class=["\']primary-subject["\']>[^(]*\(([^)]+)\)</span>'
)


def parse_listing_html(html_text: str) -> List[Dict[str, Any]]:
    """Parse an ArXiv listing page into a list of new-submission paper dicts.

    Only papers in the "New submissions" section are included — cross-listings
    and replacements are excluded.  Paper order matches what the agent sees.

    Args:
        html_text: Raw HTML of an arxiv.org/list/<cat>/new page.

    Returns:
        List of paper dicts in page order.
    """
    # Isolate the "New submissions" section (before cross-lists/replacements)
    cross_idx = html_text.find("<h3>Cross submissions")
    repl_idx = html_text.find("<h3>Replacement submissions")
    # Use the earliest section boundary that exists
    boundaries = [b for b in (cross_idx, repl_idx) if b > 0]
    if boundaries:
        new_section = html_text[: min(boundaries)]
    else:
        new_section = html_text

    # Split into <dt>/<dd> pairs
    dt_blocks = re.split(r"<dt>", new_section)[1:]  # skip before first <dt>

    papers: List[Dict[str, Any]] = []
    for block in dt_blocks:
        # Each block contains the <dt> content and the following <dd>
        # Split at <dd> to separate ID block from metadata block
        parts = block.split("<dd>", 1)
        if len(parts) < 2:
            continue

        dt_part, dd_part = parts

        # Extract arxiv ID from <dt>
        id_match = _DT_ID_RE.search(dt_part)
        if not id_match:
            continue
        arxiv_id = id_match.group(1)

        # Extract title
        title_match = _TITLE_RE.search(dd_part)
        title = ""
        if title_match:
            raw = title_match.group(1)
            # Strip any remaining HTML tags and normalize whitespace
            raw = re.sub(r"<[^>]+>", "", raw)
            title = " ".join(raw.split())

        # Extract authors
        authors: List[str] = []
        authors_div_match = _AUTHORS_DIV_RE.search(dd_part)
        if authors_div_match:
            authors = _AUTHOR_NAME_RE.findall(authors_div_match.group(1))

        # Extract primary category code from primary-subject span
        primary_category = ""
        primary_match = _PRIMARY_SUBJECT_RE.search(dd_part)
        if primary_match:
            primary_category = primary_match.group(1).strip()

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "primary_category": primary_category,
            "categories": [primary_category] if primary_category else [],
            "published": "",
            "summary": "",
        })

    return papers


class ArxivClient(BaseAPIClient):
    """
    ArXiv API client with rate limiting.

    Fetches HTML listing pages and parses paper data directly from the
    page structure, guaranteeing GT matches the agent's view.

    Rate limit: 1 request per 3 seconds (ArXiv policy).
    """

    _rate_limiter: ClassVar[RateLimiter] = RateLimiter(min_interval=3.0)

    MAX_RETRIES = 3

    @classmethod
    async def fetch_listing(
        cls,
        category: str,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Fetch and parse the HTML listing page for a category.

        Returns new-submission papers in page order.
        """
        url = f"https://arxiv.org/list/{category}/new"
        session = await _get_session()
        req_timeout = aiohttp.ClientTimeout(total=timeout)

        for attempt in range(cls.MAX_RETRIES):
            await cls._rate_limit()
            try:
                async with session.get(url, timeout=req_timeout) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return parse_listing_html(text)
                    if resp.status >= 500 and attempt < cls.MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        logger.info(f"ArXiv listing {resp.status}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    logger.warning(f"ArXiv listing error: status={resp.status}")
                    return []
            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.info(f"ArXiv listing failed: {e}, retry in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"ArXiv listing request failed: {e}")
                return []
        return []


async def fetch_listing_api_data(category: str) -> Dict[str, Any]:
    """
    Fetch data for a category listing page (e.g., /list/cs.AI/new).

    Fetches and parses the same HTML page the agent sees, so the
    ground-truth paper list, order, titles, and author counts are
    guaranteed to match the page content.

    Returns:
    {
        "category": "cs.AI",
        "paper_count": <int>,
        "papers": {
            "<arxiv_id>": {
                "rank": <1-based>,
                "arxiv_id": "...",
                "title": "...",
                "authors": [...],
                "primary_category": "...",
                "categories": [...],
                "published": "...",
                "summary": "...",
            },
            ...
        }
    }
    """
    papers_list = await ArxivClient.fetch_listing(category)

    if not papers_list:
        raise APIFetchError(
            f"No new papers on listing page for category '{category}'",
            source="arxiv",
        )

    papers = {}
    for rank, paper in enumerate(papers_list, start=1):
        arxiv_id = paper["arxiv_id"]
        papers[arxiv_id] = {**paper, "rank": rank}

    return {
        "category": category,
        "paper_count": len(papers),
        "papers": papers,
    }
