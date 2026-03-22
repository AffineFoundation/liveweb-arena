"""Paper info template for ArXiv - EASY DIFFICULTY.

Asks for a single fact about the Nth newest paper in a category:
author count or title. The agent navigates to the new-submissions
listing and reads data from the specified paper entry.

Dynamic data: new papers are posted daily, rotating the answer.
Large entity pool: 41 categories × 2 metrics × 5 patterns × 3 ranks = 1230 variants.
"""

import random
from enum import Enum
from typing import Any, Dict, Optional

from liveweb_arena.core.validators.base import (
    QuestionTemplate, GeneratedQuestion, ValidationResult, register_template,
)
from liveweb_arena.core.ground_truth_trigger import (
    UrlPatternTrigger, TriggerConfig, GroundTruthResult,
)
from liveweb_arena.core.gt_collector import GTSourceType

from .common import get_collected_listing_data, get_papers_from_listing
from .variables import CATEGORIES, RANK_CHOICES, RANK_LABELS


class PaperMetric(Enum):
    """Metrics extractable from a single paper entry."""
    AUTHOR_COUNT = ("author_count", "number of authors")
    TITLE = ("title", "title")

    @property
    def api_field(self) -> str:
        return self.value[0]

    @property
    def display_name(self) -> str:
        return self.value[1]


PATTERNS = {
    PaperMetric.AUTHOR_COUNT: [
        "How many authors does the {rank} paper in today's new submissions for {category} on ArXiv have?",
        "On ArXiv, look at today's new submissions in {category}. How many authors does the {rank} paper have?",
        "In today's new submissions for {category} on ArXiv, report the author count for the {rank} paper.",
        "Find the {rank} paper in today's new {category} submissions on ArXiv. How many authors are listed?",
        "What is the author count of the {rank} paper among today's new ArXiv submissions in {category}?",
    ],
    PaperMetric.TITLE: [
        "What is the title of the {rank} paper in today's new submissions for {category} on ArXiv?",
        "On ArXiv, find today's new submissions in {category}. What is the title of the {rank} paper?",
        "In today's new submissions for {category} on ArXiv, what is the {rank} paper's title?",
        "Find the {rank} paper in today's new {category} submissions on ArXiv. What is its title?",
        "Report the title of the {rank} paper among today's new ArXiv submissions in {category}.",
    ],
}


@register_template("arxiv_paper_info")
class ArxivPaperInfoTemplate(QuestionTemplate):
    """
    EASY: Navigate to a category listing and read one fact about the Nth paper.

    RL value:
    - Category navigation: must find and open the correct listing page
    - Dynamic data: paper pool rotates daily
    - 41 categories × 2 metrics × 5 patterns × 3 ranks = 1230 question variants
    """

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("arxiv_paper_info")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        metrics = list(PaperMetric)
        metric = metrics[variant % len(metrics)] if variant is not None else rng.choice(metrics)

        rank = rng.choice(RANK_CHOICES)
        category = rng.choice(CATEGORIES)
        pattern = rng.choice(PATTERNS[metric])
        question_text = pattern.format(
            category=category.name,
            rank=RANK_LABELS[rank],
        )

        return GeneratedQuestion(
            question_text=question_text,
            start_url=category.listing_url,
            variables={"category": category.code, "metric": metric.name, "rank": rank},
            validation_info={
                "category_code": category.code,
                "category_name": category.name,
                "metric_field": metric.api_field,
                "metric_label": metric.display_name,
                "rank": rank,
            },
            template_name=self.name,
            expected_steps=5,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        cat_name = validation_info["category_name"]
        label = validation_info["metric_label"]
        metric_field = validation_info["metric_field"]
        rank = validation_info["rank"]
        rank_label = RANK_LABELS[rank]
        if metric_field == "author_count":
            return f"""Task-Specific Rules (ArXiv Paper Info):
- Category: {cat_name}
- Metric: {label} of the {rank_label} paper
- Score 1.0: Exact author count matches
- Score 0.5: Off by 1 (co-authors may appear differently on page vs API)
- Score 0.0: Wrong count or no answer
- The answer must be a number
- Data source: ArXiv new submissions listing"""
        return f"""Task-Specific Rules (ArXiv Paper Info):
- Category: {cat_name}
- Metric: {label} of the {rank_label} paper
- Score 1.0: Title matches exactly (minor whitespace/punctuation differences OK)
- Score 0.5: Title is substantially correct with minor omissions or additions
- Score 0.0: Wrong paper title or no answer
- Data source: ArXiv new submissions listing"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        category_code = validation_info["category_code"]
        metric_field = validation_info["metric_field"]
        rank = validation_info["rank"]

        data, failure = get_collected_listing_data(category_code)
        if failure is not None:
            return failure

        papers, paper_failure = get_papers_from_listing(data)
        if paper_failure is not None:
            return paper_failure

        if len(papers) < rank:
            return GroundTruthResult.system_error(
                f"Category {category_code} has only {len(papers)} papers, "
                f"need at least {rank} for this question"
            )

        target_paper = papers[rank - 1]

        if metric_field == "author_count":
            return GroundTruthResult.ok(str(len(target_paper["authors"])))

        if metric_field == "title":
            return GroundTruthResult.ok(target_paper["title"])

        return GroundTruthResult.fail(f"Unknown metric: {metric_field}")

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
