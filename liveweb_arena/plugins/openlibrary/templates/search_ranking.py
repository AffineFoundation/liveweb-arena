"""Search ranking template for Open Library - MEDIUM DIFFICULTY."""

import random
from enum import Enum
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
from .common import find_search_entry, get_collected_data
from .subject_multi_condition import SUBJECTS


class RankingSort(Enum):
    """Supported Open Library sort order for ranking tasks."""

    EDITIONS = ("editions", "most editions")
    RATING = ("rating", "highest rating")


RANKS = [3, 4, 5, 6, 7, 8]
PATTERNS = [
    (
        "Search Open Library for \"{query}\" books sorted by {sort_label}. "
        "What is the title of the {rank_ordinal} result?"
    ),
    (
        "On Open Library, look up \"{query}\" and sort by {sort_label}. "
        "Which title appears at rank {rank}?"
    ),
    (
        "Find \"{query}\" books on Open Library ordered by {sort_label}. "
        "Answer with the title in position {rank}."
    ),
]


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


@register_template("openlibrary_search_ranking")
class OpenLibrarySearchRankingTemplate(QuestionTemplate):
    """Identify the Nth title from a sorted Open Library search result list."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openlibrary_search_ranking")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        sort_options = list(RankingSort)
        sort_option = (
            sort_options[variant % len(sort_options)]
            if variant is not None
            else rng.choice(sort_options)
        )

        subject_slug = rng.choice(SUBJECTS)
        query = subject_slug.replace("_", " ")
        rank = rng.choice(RANKS)
        pattern = rng.choice(PATTERNS)
        question_text = pattern.format(
            query=query,
            sort_label=sort_option.value[1],
            rank=rank,
            rank_ordinal=_ordinal(rank),
        )

        query_encoded = query.replace(" ", "+")
        start_url = f"https://openlibrary.org/search?q={query_encoded}&sort={sort_option.value[0]}"

        return GeneratedQuestion(
            question_text=question_text,
            start_url=start_url,
            variables={
                "query": query,
                "sort": sort_option.value[0],
                "rank": rank,
            },
            validation_info={
                "query": query,
                "sort": sort_option.value[0],
                "sort_label": sort_option.value[1],
                "rank": rank,
            },
            template_name=self.name,
            expected_steps=6,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        query = validation_info.get("query", "")
        rank = validation_info.get("rank", "")
        sort_label = validation_info.get("sort_label", "")
        return f"""Task-Specific Rules (Open Library Search Ranking):
- Query: "{query}"
- Sort: {sort_label}
- Target rank: {rank}
- Score 1.0: Correct title at target rank
- Score 0.5: N/A
- Score 0.0: Wrong title or no answer"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        collected = get_collected_data()
        if not collected:
            return GroundTruthResult.fail("No Open Library data collected")

        query = validation_info.get("query")
        sort = validation_info.get("sort")
        rank = validation_info.get("rank")
        if not isinstance(query, str) or not isinstance(sort, str) or not isinstance(rank, int):
            return GroundTruthResult.fail("Missing or invalid ranking inputs")
        if rank <= 0:
            return GroundTruthResult.fail(f"Invalid rank requested: {rank}")

        data = find_search_entry(collected, query=query, sort=sort)
        if data is None:
            ol_keys = [k for k in collected if k.startswith("ol:")][:5]
            return GroundTruthResult.not_collected(
                f"Did not collect search data for query '{query}' with sort '{sort}'. "
                f"Collected OL keys: {ol_keys}"
            )

        works_dict = data.get("works")
        if not isinstance(works_dict, dict):
            return GroundTruthResult.fail("Collected search data missing works dictionary")
        if len(works_dict) < rank:
            return GroundTruthResult.fail(
                f"Only {len(works_dict)} results collected for '{query}', need rank {rank}"
            )

        ranked_works = []
        for work in works_dict.values():
            work_rank = work.get("rank")
            if not isinstance(work_rank, int):
                return GroundTruthResult.fail("Encountered work without integer rank")
            ranked_works.append(work)
        ranked_works.sort(key=lambda work: work["rank"])

        target_work = ranked_works[rank - 1]
        title = target_work.get("title")
        if not isinstance(title, str) or not title.strip():
            return GroundTruthResult.fail(f"Missing title for rank {rank} in query '{query}'")
        return GroundTruthResult.ok(title)

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
