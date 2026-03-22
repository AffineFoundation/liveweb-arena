"""Title length extrema template for ArXiv - MEDIUM DIFFICULTY.

Asks which paper among the top-N newest submissions in a category has the
longest (or shortest) title.  The agent must scan multiple paper entries
and compare title lengths.

Dynamic data: paper pool rotates daily.
Computation required: agent must compare across papers, not read a single value.
41 categories × 3 top-N × 2 extrema × 5 patterns = 1230 question variants.
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

PATTERNS_LONGEST = [
    "Among the {n} most recent papers in today's new submissions for {category} on ArXiv, which one has the longest title? Give its title.",
    "On ArXiv, look at today's new submissions in {category}. Among the top {n}, which paper has the longest title? Report the title.",
    "In today's new {category} submissions on ArXiv, find the paper with the longest title among the top {n}. What is it?",
    "Check today's new submissions for {category} on ArXiv. Of the first {n} papers, which has the longest title? Give the title.",
    "Among the first {n} papers in today's new ArXiv submissions for {category}, which one has the longest title? Report it.",
]

PATTERNS_SHORTEST = [
    "Among the {n} most recent papers in today's new submissions for {category} on ArXiv, which one has the shortest title? Give its title.",
    "On ArXiv, look at today's new submissions in {category}. Among the top {n}, which paper has the shortest title? Report the title.",
    "In today's new {category} submissions on ArXiv, find the paper with the shortest title among the top {n}. What is it?",
    "Check today's new submissions for {category} on ArXiv. Of the first {n} papers, which has the shortest title? Give the title.",
    "Among the first {n} papers in today's new ArXiv submissions for {category}, which one has the shortest title? Report it.",
]


@register_template("arxiv_title_length_extrema")
class ArxivTitleLengthExtremaTemplate(QuestionTemplate):
    """
    MEDIUM: Find the paper with the longest or shortest title among the top N.

    Requires scanning multiple paper entries and comparing title lengths.
    41 categories × 3 top-N × 2 extrema × 5 patterns = 1230 question variants.
    """

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("arxiv_title_length_extrema")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        is_longest = (variant % 2 == 0) if variant is not None else rng.choice([True, False])
        top_n = rng.choice(TOP_N_CHOICES)

        category = rng.choice(CATEGORIES)
        patterns = PATTERNS_LONGEST if is_longest else PATTERNS_SHORTEST
        question_text = rng.choice(patterns).format(n=top_n, category=category.name)

        return GeneratedQuestion(
            question_text=question_text,
            start_url=category.listing_url,
            variables={"category": category.code, "is_longest": is_longest, "top_n": top_n},
            validation_info={
                "category_code": category.code,
                "category_name": category.name,
                "is_longest": is_longest,
                "top_n": top_n,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        cat_name = validation_info["category_name"]
        is_longest = validation_info["is_longest"]
        top_n = validation_info["top_n"]
        extrema = "longest" if is_longest else "shortest"
        return f"""Task-Specific Rules (ArXiv Title Length Extrema):
- Category: {cat_name}
- Looking for: paper with the {extrema} title among the top {top_n} newest
- Score 1.0: Title matches the correct paper (allow minor formatting differences)
- Score 0.5: Title partially matches or identifies correct paper with slight error
- Score 0.0: Wrong paper or no answer
- If there is a tie in title length, any of the tied papers is acceptable
- Data source: ArXiv new submissions listing"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        category_code = validation_info["category_code"]
        is_longest = validation_info["is_longest"]
        top_n = validation_info["top_n"]

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

        if is_longest:
            target = max(subset, key=lambda p: len(p["title"]))
        else:
            target = min(subset, key=lambda p: len(p["title"]))

        title = target["title"]
        char_count = len(title)

        return GroundTruthResult.ok(f"{title} ({char_count} characters)")

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
