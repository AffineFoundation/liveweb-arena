"""Multi-author filter template for ArXiv - MEDIUM DIFFICULTY.

Asks how many of the top-N newest papers in a category have more than K
authors.  The agent must scan multiple entries, inspect each author list,
and count those exceeding the threshold.

Dynamic data: paper pool rotates daily.
Computation required: filter + count, not just read a single value.
41 categories × 3 top-N × 3 thresholds × 5 patterns = 1845 question variants.
"""

import random
from typing import Any, Dict, Optional

from liveweb_arena.core.validators.base import (
    QuestionTemplate, GeneratedQuestion, ValidationResult, register_template,
)
from liveweb_arena.core.ground_truth_trigger import (
    UrlPatternTrigger, TriggerConfig, GroundTruthResult,
)
from liveweb_arena.core.gt_collector import GTSourceType

from .common import get_collected_listing_data, get_papers_from_listing
from .variables import CATEGORIES, TOP_N_CHOICES

# Author-count thresholds: "more than K authors"
AUTHOR_THRESHOLDS = [1, 2, 3]

PATTERNS = [
    "Among the {n} most recent papers in today's new submissions for {category} on ArXiv, how many have more than {k} author(s)?",
    "On ArXiv, look at today's new submissions in {category}. Among the top {n}, how many have more than {k} author(s)?",
    "In today's new {category} submissions on ArXiv, check the first {n} papers. How many list more than {k} author(s)?",
    "Check today's new submissions for {category} on ArXiv. Of the first {n} papers, how many have more than {k} author(s)?",
    "Among the first {n} papers in today's new ArXiv submissions for {category}, how many have more than {k} author(s)?",
]


@register_template("arxiv_multi_author_filter")
class ArxivMultiAuthorFilterTemplate(QuestionTemplate):
    """
    MEDIUM: Count papers exceeding an author-count threshold among the top N.

    Requires inspecting each paper's author list and filtering.
    41 categories × 3 top-N × 3 thresholds × 5 patterns = 1845 question variants.
    """

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("arxiv_multi_author_filter")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        top_n = rng.choice(TOP_N_CHOICES)
        threshold = rng.choice(AUTHOR_THRESHOLDS)
        category = rng.choice(CATEGORIES)
        pattern = rng.choice(PATTERNS)
        question_text = pattern.format(n=top_n, category=category.name, k=threshold)

        return GeneratedQuestion(
            question_text=question_text,
            start_url=category.listing_url,
            variables={"category": category.code, "top_n": top_n, "threshold": threshold},
            validation_info={
                "category_code": category.code,
                "category_name": category.name,
                "top_n": top_n,
                "threshold": threshold,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        cat_name = validation_info["category_name"]
        top_n = validation_info["top_n"]
        threshold = validation_info["threshold"]
        return f"""Task-Specific Rules (ArXiv Multi-Author Filter):
- Category: {cat_name}
- Looking for: count of papers with MORE THAN {threshold} authors among the top {top_n} newest
- Score 1.0: Exact count matches
- Score 0.5: Off by 1
- Score 0.0: Wrong count or no answer
- The answer must be a number
- "More than {threshold}" means strictly greater than {threshold} (not equal)
- Data source: ArXiv new submissions listing"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        category_code = validation_info["category_code"]
        top_n = validation_info["top_n"]
        threshold = validation_info["threshold"]

        data, failure = get_collected_listing_data(category_code)
        if failure is not None:
            return failure

        papers, paper_failure = get_papers_from_listing(data)
        if paper_failure is not None:
            return paper_failure

        # System error if listing has fewer papers than requested
        if len(papers) < top_n:
            return GroundTruthResult.system_error(
                f"Category {category_code} has only {len(papers)} papers, "
                f"need {top_n} for this question"
            )
        subset = papers[:top_n]

        count = sum(1 for p in subset if len(p["authors"]) > threshold)
        return GroundTruthResult.ok(str(count))

    async def validate_answer(
        self, answer: str, validation_info: Dict[str, Any]
    ) -> ValidationResult:
        """Not used — the pipeline uses LLM-based validation via get_validation_rules()."""
        return ValidationResult(
            score=0.0, is_correct=False, expected=None, actual=answer,
            details="Use LLM validation",
        )

    def get_ground_truth_trigger(self, validation_info: dict) -> TriggerConfig:
        trigger = UrlPatternTrigger(domains=["arxiv.org"])
        return TriggerConfig(trigger=trigger)

    @classmethod
    def get_cache_source(cls) -> str:
        return "arxiv"

    def get_gt_source(self) -> GTSourceType:
        return self.GT_SOURCE
