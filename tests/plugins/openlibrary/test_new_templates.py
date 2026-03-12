import asyncio
from typing import Any, Dict

from liveweb_arena.core.gt_collector import set_current_gt_collector
from liveweb_arena.core.task_registry import TaskRegistry
from liveweb_arena.plugins.openlibrary.templates.author_editions import (
    OpenLibraryAuthorEditionsTemplate,
)
from liveweb_arena.plugins.openlibrary.templates.book_comparison import (
    OpenLibraryBookComparisonTemplate,
)
from liveweb_arena.plugins.openlibrary.templates.search_ranking import (
    OpenLibrarySearchRankingTemplate,
)


class _DummyCollector:
    def __init__(self, data: Dict[str, Dict[str, Any]]):
        self._data = data

    def get_collected_api_data(self) -> Dict[str, Dict[str, Any]]:
        return self._data


def _run_with_collected(data: Dict[str, Dict[str, Any]], coro):
    set_current_gt_collector(_DummyCollector(data))
    try:
        return asyncio.run(coro)
    finally:
        set_current_gt_collector(None)


def _make_search_entry(query: str, sort: str, works: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query": query,
        "sort": sort,
        "works": {work["key"]: work for work in works},
    }


def test_book_comparison_picks_higher_metric():
    template = OpenLibraryBookComparisonTemplate()
    collected = {
        "ol:search:a": _make_search_entry(
            "poetry",
            "editions",
            [
                {"key": "/works/OL1W", "rank": 1, "title": "Pride and Prejudice", "ratings_count": 1200},
                {"key": "/works/OL2W", "rank": 2, "title": "Jane Eyre", "ratings_count": 900},
            ],
        ),
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth(
            {
                "metric": "ratings_count",
                "book_a": "Pride and Prejudice",
                "book_b": "Jane Eyre",
            }
        ),
    )
    assert result.success is True
    assert result.value == "Pride and Prejudice"


def test_book_comparison_tie_breaks_alphabetically():
    template = OpenLibraryBookComparisonTemplate()
    collected = {
        "ol:search:a": _make_search_entry(
            "classics",
            "editions",
            [
                {"key": "/works/OL3W", "rank": 1, "title": "Pride and Prejudice", "edition_count": 1000},
                {"key": "/works/OL4W", "rank": 2, "title": "Jane Eyre", "edition_count": 1000},
            ],
        ),
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth(
            {
                "metric": "edition_count",
                "book_a": "Pride and Prejudice",
                "book_b": "Jane Eyre",
            }
        ),
    )
    assert result.success is True
    assert result.value == "Jane Eyre"


def test_search_ranking_uses_matching_sort_entry():
    template = OpenLibrarySearchRankingTemplate()
    collected = {
        "ol:search:unsorted": {
            "query": "poetry",
            "sort": None,
            "works": {
                "/works/OLA": {"key": "/works/OLA", "rank": 1, "title": "Wrong A"},
                "/works/OLB": {"key": "/works/OLB", "rank": 2, "title": "Wrong B"},
            },
        },
        "ol:search:sorted": {
            "query": "poetry",
            "sort": "editions",
            "works": {
                "/works/OLC": {"key": "/works/OLC", "rank": 1, "title": "Right One"},
                "/works/OLD": {"key": "/works/OLD", "rank": 2, "title": "Right Two"},
            },
        },
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth({"query": "poetry", "sort": "editions", "rank": 2}),
    )
    assert result.success is True
    assert result.value == "Right Two"


def test_search_ranking_reports_not_collected_when_sort_missing():
    template = OpenLibrarySearchRankingTemplate()
    collected = {
        "ol:search:rating": _make_search_entry(
            "poetry",
            "rating",
            [
                {"key": "/works/OL1", "rank": 1, "title": "A"},
                {"key": "/works/OL2", "rank": 2, "title": "B"},
            ],
        )
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth({"query": "poetry", "sort": "editions", "rank": 1}),
    )
    assert result.success is False
    assert result.is_data_not_collected()


def test_author_editions_sums_first_n_results():
    template = OpenLibraryAuthorEditionsTemplate()
    collected = {
        "ol:search:dickens": _make_search_entry(
            "Charles Dickens books",
            "editions",
            [
                {"key": "/works/OL10W", "rank": 1, "title": "A Tale of Two Cities", "edition_count": 100},
                {"key": "/works/OL11W", "rank": 2, "title": "Oliver Twist", "edition_count": 200},
                {"key": "/works/OL12W", "rank": 3, "title": "Great Expectations", "edition_count": 300},
            ],
        )
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth(
            {
                "query": "Charles Dickens books",
                "author": "Charles Dickens",
                "sort": "editions",
                "work_count": 2,
            }
        ),
    )
    assert result.success is True
    assert result.value == "300"


def test_author_editions_accepts_author_query_without_books_suffix():
    template = OpenLibraryAuthorEditionsTemplate()
    collected = {
        "ol:search:twain": _make_search_entry(
            "Mark Twain",
            "editions",
            [
                {"key": "/works/OL20W", "rank": 1, "title": "Huckleberry Finn", "edition_count": 500},
                {"key": "/works/OL21W", "rank": 2, "title": "Tom Sawyer", "edition_count": 300},
                {"key": "/works/OL22W", "rank": 3, "title": "A Connecticut Yankee", "edition_count": 200},
            ],
        )
    }

    result = _run_with_collected(
        collected,
        template.get_ground_truth(
            {
                "query": "Mark Twain books",
                "author": "Mark Twain",
                "sort": "editions",
                "work_count": 2,
            }
        ),
    )
    assert result.success is True
    assert result.value == "800"


def test_task_registry_includes_new_openlibrary_templates():
    assert TaskRegistry.TEMPLATES[82] == ("openlibrary", "openlibrary_book_comparison")
    assert TaskRegistry.TEMPLATES[83] == ("openlibrary", "openlibrary_search_ranking")
    assert TaskRegistry.TEMPLATES[84] == ("openlibrary", "openlibrary_author_editions")
    assert 82 in TaskRegistry.TEMPLATE_VERSIONS[-1]
    assert 83 in TaskRegistry.TEMPLATE_VERSIONS[-1]
    assert 84 in TaskRegistry.TEMPLATE_VERSIONS[-1]
