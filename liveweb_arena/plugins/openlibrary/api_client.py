"""Open Library API client with rate limiting."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from liveweb_arena.plugins.base_client import APIFetchError, BaseAPIClient, RateLimiter

logger = logging.getLogger(__name__)

# Shared session for connection reuse across requests
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

CACHE_SOURCE = "openlibrary"

OL_API_BASE = "https://openlibrary.org"

# Fields requested from the search API for ground truth
SEARCH_FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year",
    "edition_count", "has_fulltext", "ratings_average", "ratings_count",
    "want_to_read_count", "currently_reading_count", "already_read_count",
    "number_of_pages_median",
])


class OpenLibraryClient(BaseAPIClient):
    """
    Open Library API client.

    Uses the public Open Library APIs:
    - /search.json - search works
    - /subjects/{subject}.json - works by subject
    - /works/{id}.json - work details
    - /authors/{id}.json - author details
    """

    # Rate limit: 1.5s between requests (Open Library asks for politeness)
    _rate_limiter = RateLimiter(min_interval=1.5)

    MAX_RETRIES = 3

    @classmethod
    async def get(
        cls,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Optional[Any]:
        url = f"{OL_API_BASE}{endpoint}"
        session = await _get_session()
        req_timeout = aiohttp.ClientTimeout(total=timeout)

        for attempt in range(cls.MAX_RETRIES):
            await cls._rate_limit()
            try:
                async with session.get(url, params=params, timeout=req_timeout) as response:
                    if response.status == 200:
                        return await response.json(content_type=None)
                    if response.status >= 500 and attempt < cls.MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        logger.info(f"OL API {response.status} for {endpoint}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    logger.warning(f"OL API error: status={response.status} for {endpoint}")
                    return None
            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.info(f"OL API failed for {endpoint}: {e}, retry in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"OL API request failed for {endpoint}: {e}")
                return None
        return None

    @classmethod
    async def search_works(
        cls,
        query: str,
        limit: int = 20,
        sort: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for works.

        Args:
            query: Search query string
            limit: Maximum results
            sort: Sort order (e.g., "rating", "editions", "new")
            mode: Search mode (e.g., "everything") to match HTML page behavior

        Returns:
            List of work dicts with stats fields
        """
        params: Dict[str, Any] = {
            "q": query,
            "limit": limit,
            "fields": SEARCH_FIELDS,
        }
        if sort:
            params["sort"] = sort
        if mode:
            params["mode"] = mode

        data = await cls.get("/search.json", params=params)
        if data and isinstance(data, dict):
            return data.get("docs", [])
        return []

    @classmethod
    async def get_subject_works(
        cls,
        subject: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get works for a subject.

        Args:
            subject: Subject slug (e.g., "science_fiction")
            limit: Maximum works to return

        Returns:
            Subject data with works list
        """
        data = await cls.get(
            f"/subjects/{subject}.json",
            params={"limit": limit},
        )
        if data and isinstance(data, dict):
            return data
        return {}

    @classmethod
    async def get_work_stats(cls, work_key: str) -> Optional[Dict[str, Any]]:
        """
        Get stats for a specific work via search API.

        The search API returns richer stats (ratings, read counts)
        than the works API endpoint.

        Args:
            work_key: Work key (e.g., "/works/OL103123W")

        Returns:
            Work stats dict or None
        """
        docs = await cls.search_works(f"key:{work_key}", limit=1)
        if docs:
            return docs[0]
        return None

    @classmethod
    async def get_work(cls, work_id: str) -> Optional[Dict[str, Any]]:
        """
        Get work details from works API.

        Args:
            work_id: Work ID (e.g., "OL103123W")

        Returns:
            Work data dict or None
        """
        data = await cls.get(f"/works/{work_id}.json")
        if data and isinstance(data, dict):
            return data
        return None


async def fetch_subject_api_data(subject: str, limit: int = 20) -> Dict[str, Any]:
    """
    Fetch API data for a subject page.

    Returns:
    {
        "subject": "<subject>",
        "work_count": <int>,
        "works": {
            "<work_key>": {
                "key": "<work_key>",
                "rank": <1-based>,
                "title": "...",
                "edition_count": <int>,
                "authors": [...],
                "first_publish_year": <int>,
                ...
            },
            ...
        }
    }
    """
    data = await OpenLibraryClient.get_subject_works(subject, limit=limit)
    if not data or "works" not in data:
        raise APIFetchError(f"Failed to fetch subject '{subject}'", source="openlibrary")

    works = {}
    for rank, work in enumerate(data["works"], start=1):
        key = work.get("key", "")
        works[key] = {
            "key": key,
            "rank": rank,
            "title": work.get("title", ""),
            "edition_count": work.get("edition_count"),
            "authors": [a.get("name", "") for a in work.get("authors", [])],
            "first_publish_year": work.get("first_publish_year"),
            "has_fulltext": work.get("has_fulltext"),
        }

    return {
        "subject": subject,
        "work_count": data.get("work_count"),
        "works": works,
    }


async def fetch_search_api_data(
    query: str, limit: int = 20, sort: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch API data for a search results page.

    Returns:
    {
        "query": "<query>",
        "num_found": <int>,
        "works": {
            "<work_key>": {
                "key": "<work_key>",
                "rank": <1-based>,
                "title": "...",
                "edition_count": <int>,
                "ratings_average": <float>,
                "ratings_count": <int>,
                "want_to_read_count": <int>,
                ...
            },
            ...
        }
    }
    """
    docs = await OpenLibraryClient.search_works(query, limit=limit, sort=sort, mode=mode)
    if not docs:
        raise APIFetchError(f"Search returned no results for '{query}'", source="openlibrary")

    works = {}
    for rank, doc in enumerate(docs, start=1):
        key = doc.get("key", "")
        works[key] = {
            "key": key,
            "rank": rank,
            "title": doc.get("title", ""),
            "author_name": doc.get("author_name", []),
            "first_publish_year": doc.get("first_publish_year"),
            "edition_count": doc.get("edition_count"),
            "has_fulltext": doc.get("has_fulltext"),
            "ratings_average": doc.get("ratings_average"),
            "ratings_count": doc.get("ratings_count"),
            "want_to_read_count": doc.get("want_to_read_count"),
            "currently_reading_count": doc.get("currently_reading_count"),
            "already_read_count": doc.get("already_read_count"),
            "number_of_pages_median": doc.get("number_of_pages_median"),
        }

    return {
        "query": query,
        "sort": sort,
        "num_found": len(docs),
        "works": works,
    }


async def fetch_work_api_data(work_key: str) -> Dict[str, Any]:
    """
    Fetch API data for a single work detail page.

    Args:
        work_key: Work key (e.g., "/works/OL103123W")

    Returns:
        Work stats dict

    Raises:
        APIFetchError: If work not found
    """
    stats = await OpenLibraryClient.get_work_stats(work_key)
    if not stats:
        raise APIFetchError(f"Work {work_key} not found", source="openlibrary")

    return stats
