"""Author editions aggregation template for Open Library - MEDIUM DIFFICULTY."""

import random
from typing import Any, Dict, Optional

from liveweb_arena.core.ground_truth_trigger import (
    GroundTruthResult,
    TriggerConfig,
    UrlPatternTrigger,
)
from liveweb_arena.core.gt_collector import GTSourceType
from liveweb_arena.core.validators.base import (
    GeneratedQuestion,
    QuestionTemplate,
    ValidationResult,
    register_template,
)
from .common import get_collected_data, parse_numeric

AUTHOR_POOL = [
    ("Charles Dickens", "charles dickens"),
    ("Jane Austen", "jane austen"),
    ("William Shakespeare", "william shakespeare"),
    ("Mark Twain", "mark twain"),
    ("Leo Tolstoy", "leo tolstoy"),
    ("Fyodor Dostoevsky", "fyodor dostoevsky"),
    ("Virginia Woolf", "virginia woolf"),
    ("George Orwell", "george orwell"),
    ("Agatha Christie", "agatha christie"),
    ("Ernest Hemingway", "ernest hemingway"),
    ("Jules Verne", "jules verne"),
    ("H.G. Wells", "h g wells"),
    ("Arthur Conan Doyle", "arthur conan doyle"),
    ("Mary Shelley", "mary shelley"),
    ("Franz Kafka", "franz kafka"),
    ("Herman Melville", "herman melville"),
    ("Victor Hugo", "victor hugo"),
    ("Emily Bronte", "emily bronte"),
    ("Miguel de Cervantes", "miguel de cervantes"),
    ("Alexandre Dumas", "alexandre dumas"),
]

RESULT_COUNTS = [3, 5, 7]
PATTERNS = [
    (
        "Search Open Library for books by \"{author}\" sorted by most editions. "
        "What is the total number of editions across the first {n} results?"
    ),
    (
        "On Open Library, look up books by \"{author}\" and sort by most editions. "
        "Sum the edition counts of the top {n} books."
    ),
    (
        "Find books by \"{author}\" on Open Library (sort: most editions). "
        "Among the first {n} results, what is the combined editions total?"
    ),
]


@register_template("openlibrary_author_editions")
class OpenLibraryAuthorEditionsTemplate(QuestionTemplate):
    """Aggregate edition counts across top author search results."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openlibrary_author_editions")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        author_name, author_query = rng.choice(AUTHOR_POOL)
        count = RESULT_COUNTS[variant % len(RESULT_COUNTS)] if variant is not None else rng.choice(RESULT_COUNTS)

        pattern = rng.choice(PATTERNS)
        question_text = pattern.format(author=author_name, n=count)
        query_encoded = author_query.replace(" ", "+")
        start_url = f"https://openlibrary.org/search?q={query_encoded}&sort=editions"

        return GeneratedQuestion(
            question_text=question_text,
            start_url=start_url,
            variables={
                "author": author_name,
                "work_count": count,
            },
            validation_info={
                "author_name": author_name,
                "author_query": author_query,
                "sort": "editions",
                "work_count": count,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        author = validation_info.get("author_name", "")
        count = validation_info.get("work_count", "")
        return f"""Task-Specific Rules (Open Library Author Editions):
- Author query: "{author}"
- Sum target: first {count} results sorted by editions
- Score 1.0: Exact summed edition count
- Score 0.5: Within ±1 of correct total
- Score 0.0: Wrong total or no answer"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        collected = get_collected_data()
        if not collected:
            return GroundTruthResult.fail("No Open Library data collected")

        author_name = validation_info.get("author_name")
        author_query = validation_info.get("author_query")
        sort = validation_info.get("sort")
        work_count = validation_info.get("work_count")
        if (
            not isinstance(author_name, str)
            or not isinstance(author_query, str)
            or not isinstance(sort, str)
            or not isinstance(work_count, int)
        ):
            return GroundTruthResult.fail("Missing or invalid author aggregation inputs")
        if work_count <= 0:
            return GroundTruthResult.fail(f"Invalid work_count: {work_count}")

        data = self._find_author_search_entry(
            collected, author_query=author_query, sort=sort,
        )
        if data is None:
            ol_keys = [k for k in collected if k.startswith("ol:")][:5]
            return GroundTruthResult.not_collected(
                f"Did not collect search data for author '{author_name}' "
                f"sorted by '{sort}'. Collected OL keys: {ol_keys}"
            )

        works_dict = data.get("works")
        if not isinstance(works_dict, dict):
            return GroundTruthResult.fail("Collected search data missing works dictionary")
        if len(works_dict) < work_count:
            return GroundTruthResult.fail(
                f"Only {len(works_dict)} works collected for '{author_query}', need {work_count}"
            )

        ranked_works = []
        for work in works_dict.values():
            rank = work.get("rank")
            if not isinstance(rank, int):
                return GroundTruthResult.fail("Encountered work without integer rank")
            ranked_works.append(work)
        ranked_works.sort(key=lambda work: work["rank"])
        ranked_works = ranked_works[:work_count]

        total_editions = 0
        for work in ranked_works:
            title = work.get("title", "<unknown>")
            edition_count = parse_numeric(work.get("edition_count"))
            if edition_count is None:
                return GroundTruthResult.fail(f"Missing edition_count for work '{title}'")
            total_editions += int(edition_count)

        return GroundTruthResult.ok(str(total_editions))

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Strip punctuation from a token for robust matching."""
        return "".join(ch for ch in token.lower() if ch.isalnum())

    @classmethod
    def _find_author_search_entry(
        cls,
        collected: Dict[str, Dict[str, Any]],
        *,
        author_query: str,
        sort: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find search data for an author while tolerating natural query variants.

        Agents may search for:
        - "mark twain"
        - "Mark Twain books"
        - "\"Mark Twain\" books"
        - "h g wells" (normalized from "H.G. Wells")
        """
        target_normalized = author_query.strip().lower()
        author_tokens = {
            cls._normalize_token(t)
            for t in author_query.lower().split()
            if cls._normalize_token(t)
        }
        matched_entry: Optional[Dict[str, Any]] = None

        for key, entry in collected.items():
            if not key.startswith("ol:") or not isinstance(entry, dict):
                continue
            works = entry.get("works")
            if not isinstance(works, dict):
                continue
            if entry.get("sort") != sort:
                continue

            entry_query = str(entry.get("query", "")).strip().lower().replace('"', "")
            if not entry_query:
                continue

            # Exact match on the pre-normalized query
            if entry_query == target_normalized:
                matched_entry = entry
                continue

            # Flexible token match: strip punctuation, ignore filler words
            entry_tokens = {
                cls._normalize_token(t)
                for t in entry_query.split()
                if cls._normalize_token(t) not in {"books", "book", "by"}
            }
            if author_tokens and author_tokens.issubset(entry_tokens):
                matched_entry = entry

        return matched_entry

    async def validate_answer(
        self,
        answer: str,
        validation_info: Dict[str, Any],
    ) -> ValidationResult:
        return ValidationResult(
            score=0.0,
            is_correct=False,
            expected=None,
            actual=answer,
            details="Use LLM validation",
        )

    def get_ground_truth_trigger(self, validation_info: dict) -> TriggerConfig:
        trigger = UrlPatternTrigger(domains=["openlibrary.org"])
        return TriggerConfig(trigger=trigger)

    @classmethod
    def get_cache_source(cls) -> str:
        return "openlibrary"

    def get_gt_source(self) -> GTSourceType:
        return self.GT_SOURCE
