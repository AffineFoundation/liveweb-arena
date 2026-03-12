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

AUTHORS = [
    "Charles Dickens",
    "Jane Austen",
    "William Shakespeare",
    "Mark Twain",
    "Leo Tolstoy",
    "Fyodor Dostoevsky",
    "Virginia Woolf",
    "George Orwell",
    "Agatha Christie",
    "Ernest Hemingway",
    "Jules Verne",
    "H. G. Wells",
    "Arthur Conan Doyle",
    "Mary Shelley",
    "Franz Kafka",
    "Herman Melville",
    "Victor Hugo",
    "Emily Bronte",
    "Miguel de Cervantes",
    "Alexandre Dumas",
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
        author = rng.choice(AUTHORS)
        count = RESULT_COUNTS[variant % len(RESULT_COUNTS)] if variant is not None else rng.choice(RESULT_COUNTS)
        search_query = f"{author} books"

        pattern = rng.choice(PATTERNS)
        question_text = pattern.format(author=author, n=count)
        query_encoded = search_query.replace(" ", "+")
        start_url = f"https://openlibrary.org/search?q={query_encoded}&sort=editions"

        return GeneratedQuestion(
            question_text=question_text,
            start_url=start_url,
            variables={
                "author": author,
                "work_count": count,
            },
            validation_info={
                "query": search_query,
                "sort": "editions",
                "author": author,
                "work_count": count,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        author = validation_info.get("author", "")
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

        query = validation_info.get("query")
        author = validation_info.get("author")
        sort = validation_info.get("sort")
        work_count = validation_info.get("work_count")
        if (
            not isinstance(query, str)
            or not isinstance(author, str)
            or not isinstance(sort, str)
            or not isinstance(work_count, int)
        ):
            return GroundTruthResult.fail("Missing or invalid author aggregation inputs")
        if work_count <= 0:
            return GroundTruthResult.fail(f"Invalid work_count: {work_count}")

        data = self._find_author_search_entry(collected, author=author, fallback_query=query, sort=sort)
        if data is None:
            return GroundTruthResult.not_collected(
                f"Did not collect search data for author '{author}' sorted by '{sort}'"
            )

        works_dict = data.get("works")
        if not isinstance(works_dict, dict):
            return GroundTruthResult.fail("Collected search data missing works dictionary")
        if len(works_dict) < work_count:
            return GroundTruthResult.fail(
                f"Only {len(works_dict)} works collected for '{query}', need {work_count}"
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
    def _find_author_search_entry(
        collected: Dict[str, Dict[str, Any]],
        *,
        author: str,
        fallback_query: str,
        sort: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find search data for an author while tolerating natural query variants.

        Agents may search for:
        - "Mark Twain"
        - "Mark Twain books"
        - "\"Mark Twain\" books"
        """
        author_tokens = {token for token in author.lower().split() if token}
        fallback_normalized = fallback_query.strip().lower()
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

            if entry_query == fallback_normalized:
                matched_entry = entry
                continue

            tokens = {token for token in entry_query.split() if token not in {"books", "book", "by"}}
            if author_tokens and author_tokens.issubset(tokens):
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
