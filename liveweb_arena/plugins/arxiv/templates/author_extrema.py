"""Author extrema template for ArXiv - MEDIUM DIFFICULTY.

Asks which paper among the top-N newest submissions in a category has the
most (or fewest) authors.  The agent must scan multiple paper entries on
the listing page and compare author counts.

Dynamic data: paper pool rotates daily.
Computation required: agent must compare across papers, not read a single value.
41 categories × 3 top-N × 2 extrema × 5 patterns = 1230 variants.
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

PATTERNS_MOST = [
    "Among the {n} most recent papers in today's new submissions for {category} on ArXiv, which one has the most authors? Give its title.",
    "On ArXiv, look at today's new submissions in {category}. Among the top {n}, which paper has the largest number of authors? Report the title.",
    "In today's new {category} submissions on ArXiv, find the paper with the most authors among the top {n}. What is its title?",
    "Check today's new submissions for {category} on ArXiv. Of the first {n} papers, which has the most authors? Give the title.",
    "Among the first {n} papers in today's new ArXiv submissions for {category}, which one lists the most authors? Report its title.",
]

PATTERNS_FEWEST = [
    "Among the {n} most recent papers in today's new submissions for {category} on ArXiv, which one has the fewest authors? Give its title.",
    "On ArXiv, look at today's new submissions in {category}. Among the top {n}, which paper has the smallest number of authors? Report the title.",
    "In today's new {category} submissions on ArXiv, find the paper with the fewest authors among the top {n}. What is its title?",
    "Check today's new submissions for {category} on ArXiv. Of the first {n} papers, which has the fewest authors? Give the title.",
    "Among the first {n} papers in today's new ArXiv submissions for {category}, which one lists the fewest authors? Report its title.",
]


@register_template("arxiv_author_extrema")
class ArxivAuthorExtremaTemplate(QuestionTemplate):
    """
    MEDIUM: Find the paper with the most or fewest authors among the top N.

    Requires scanning multiple paper entries and comparing author counts.
    41 categories × 3 top-N × 2 extrema × 5 patterns = 1230 question variants.
    """

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("arxiv_author_extrema")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        is_most = (variant % 2 == 0) if variant is not None else rng.choice([True, False])
        top_n = rng.choice(TOP_N_CHOICES)

        category = rng.choice(CATEGORIES)
        patterns = PATTERNS_MOST if is_most else PATTERNS_FEWEST
        question_text = rng.choice(patterns).format(n=top_n, category=category.name)

        return GeneratedQuestion(
            question_text=question_text,
            start_url=category.listing_url,
            variables={"category": category.code, "is_most": is_most, "top_n": top_n},
            validation_info={
                "category_code": category.code,
                "category_name": category.name,
                "is_most": is_most,
                "top_n": top_n,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        cat_name = validation_info["category_name"]
        is_most = validation_info["is_most"]
        top_n = validation_info["top_n"]
        extrema = "most" if is_most else "fewest"
        return f"""Task-Specific Rules (ArXiv Author Extrema):
- Category: {cat_name}
- Looking for: paper with the {extrema} authors among the top {top_n} newest
- Score 1.0: Title matches the correct paper (allow minor formatting differences)
- Score 0.5: Title partially matches or identifies correct paper with slight error
- Score 0.0: Wrong paper or no answer
- If there is a tie, any of the tied papers is acceptable
- Data source: ArXiv new submissions listing"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        category_code = validation_info["category_code"]
        is_most = validation_info["is_most"]
        top_n = validation_info["top_n"]

        data, failure = get_collected_listing_data(category_code)
        if failure is not None:
            return failure

        papers, paper_failure = get_papers_from_listing(data)
        if paper_failure is not None:
            return paper_failure

        # Take only the top N papers — system error if listing has fewer than
        # requested (not the agent's fault; live data has insufficient volume)
        if len(papers) < top_n:
            return GroundTruthResult.system_error(
                f"Category {category_code} has only {len(papers)} papers, "
                f"need {top_n} for this question"
            )
        subset = papers[:top_n]

        if is_most:
            target = max(subset, key=lambda p: len(p["authors"]))
        else:
            target = min(subset, key=lambda p: len(p["authors"]))

        title = target["title"]
        author_count = len(target["authors"])

        return GroundTruthResult.ok(f"{title} ({author_count} authors)")

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
