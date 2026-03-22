"""Category comparison template for ArXiv - HARD DIFFICULTY.

Compares the total number of authors across the N most recent submissions
in two categories from different subject groups.  The agent must visit two
separate listing pages, count authors on each paper, sum them per category,
and compute the numeric difference.

Dynamic data: paper pool rotates daily.
Multi-page + computation: agent visits two pages and sums + subtracts.
Answer is a numeric difference — random baseline ≈ 0%.
605 cross-group pairs × 6 patterns × 2 orderings = 7260 variants.
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
from .variables import CATEGORY_PAIRS, TOP_N

PATTERNS = [
    "On ArXiv, look at today's new submissions in {cat1} and {cat2}. Sum the author counts for the top {n} papers in each category. What is the difference ({cat1} total minus {cat2} total)?",
    "Sum up the author counts for the {n} newest papers in today's new submissions for {cat1} and {cat2} on ArXiv. What is the difference ({cat1} total minus {cat2} total)?",
    "Compare today's new submissions in {cat1} and {cat2} on ArXiv. Count all authors across the top {n} papers in each and report the difference ({cat1} minus {cat2}).",
    "Using ArXiv, add up the number of authors across the top {n} papers in today's new {cat1} and {cat2} submissions. How many more total authors does {cat1} have? (Give a signed number.)",
    "In today's new ArXiv submissions, sum the authors for the first {n} papers in {cat1} and the first {n} in {cat2}. Report the difference ({cat1} minus {cat2}).",
    "Check today's new submissions for {cat1} and {cat2} on ArXiv. For each category, sum the author counts of the top {n} papers. What is {cat1}'s total minus {cat2}'s total?",
]


@register_template("arxiv_category_comparison")
class ArxivCategoryComparisonTemplate(QuestionTemplate):
    """
    HARD: Compare total author counts across the top-N papers in two categories.

    Requires visiting two different listing pages, counting authors on each
    paper, summing per category, and computing the numeric difference.
    605 cross-group pairs × 6 patterns × 2 orderings = 7260 question variants.
    """

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("arxiv_category_comparison")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        pair = rng.choice(CATEGORY_PAIRS)
        cat1, cat2 = pair

        # Randomly swap order so both orderings are represented
        if rng.random() > 0.5:
            cat1, cat2 = cat2, cat1

        pattern = rng.choice(PATTERNS)
        question_text = pattern.format(
            n=TOP_N,
            cat1=cat1.name,
            cat2=cat2.name,
        )

        return GeneratedQuestion(
            question_text=question_text,
            start_url=cat1.listing_url,
            variables={"cat1": cat1.code, "cat2": cat2.code},
            validation_info={
                "cat1_code": cat1.code,
                "cat1_name": cat1.name,
                "cat2_code": cat2.code,
                "cat2_name": cat2.name,
                "cat2_url": cat2.listing_url,
                "top_n": TOP_N,
            },
            template_name=self.name,
            expected_steps=9,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        cat1 = validation_info["cat1_name"]
        cat2 = validation_info["cat2_name"]
        top_n = validation_info["top_n"]
        return f"""Task-Specific Rules (ArXiv Category Comparison):
- Answer is the total-author-count difference across the {top_n} most recent papers: {cat1} minus {cat2}
- Positive means {cat1} has more total authors
- Score 1.0: Difference within ±3 of ground truth
- Score 0.5: Difference within ±8 of ground truth
- Score 0.0: Difference off by more than 8, or no numeric answer
- Accept formats: "12", "+12", "-3", "7 more"
- Do NOT accept answers that only name a category without the numeric difference"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        cat1_code = validation_info["cat1_code"]
        cat2_code = validation_info["cat2_code"]
        cat1_name = validation_info["cat1_name"]
        cat2_name = validation_info["cat2_name"]
        top_n = validation_info["top_n"]

        data1, fail1 = get_collected_listing_data(cat1_code)
        if fail1 is not None:
            return fail1
        data2, fail2 = get_collected_listing_data(cat2_code)
        if fail2 is not None:
            return fail2

        papers1, pf1 = get_papers_from_listing(data1)
        if pf1 is not None:
            return pf1
        papers2, pf2 = get_papers_from_listing(data2)
        if pf2 is not None:
            return pf2

        # System error if listing has fewer papers than requested —
        # not the agent's fault; live data has insufficient volume.
        if len(papers1) < top_n:
            return GroundTruthResult.system_error(
                f"{cat1_code} has only {len(papers1)} papers, need {top_n}"
            )
        if len(papers2) < top_n:
            return GroundTruthResult.system_error(
                f"{cat2_code} has only {len(papers2)} papers, need {top_n}"
            )

        total1 = sum(len(p["authors"]) for p in papers1[:top_n])
        total2 = sum(len(p["authors"]) for p in papers2[:top_n])
        diff = total1 - total2

        return GroundTruthResult.ok(
            f"{diff} ({cat1_name}: {total1} authors, {cat2_name}: {total2} authors)"
        )

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
