"""Subject multi-condition filter template for Open Library - HARD DIFFICULTY

RL-friendly design:
- Requires browsing a subject page and checking multiple books
- Requires evaluating TWO conditions simultaneously (cannot sort single column)
- Large exploration space: subject × condition × thresholds
- Dynamic data: edition counts and publish years shift as new editions appear
- Combinatorial question space prevents memorization
- All data visible on subject page: edition counts, publish years, fulltext
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
from liveweb_arena.plugins.openlibrary.api_client import fetch_subject_api_data


class ConditionType(Enum):
    """Types of multi-condition filters for Open Library subject works."""
    # Many editions AND old (both conditions needed)
    HIGH_EDITIONS_OLD = ("high_editions_old", "editions > {t1} AND published before {t2}")
    # Editions in a specific range (two-threshold, cannot sort-and-scan)
    EDITION_RANGE = ("edition_range", "{t1} < editions < {t2}")
    # Old with full text available
    OLD_WITH_FULLTEXT = ("old_with_fulltext", "published before {t1} AND full text available")


THRESHOLDS = {
    ConditionType.HIGH_EDITIONS_OLD: [
        (1000, 1850),
        (500, 1880),
        (1500, 1900),
        (700, 1870),
    ],
    ConditionType.EDITION_RANGE: [
        (300, 1200),
        (500, 2000),
        (800, 2500),
        (1000, 3000),
    ],
    ConditionType.OLD_WITH_FULLTEXT: [
        (1800,),
        (1850,),
        (1880,),
        (1750,),
    ],
}

# Subjects with rich, dynamic book collections
SUBJECTS = [
    "science_fiction",
    "fantasy",
    "mystery",
    "romance",
    "horror",
    "thriller",
    "biography",
    "history",
    "philosophy",
    "poetry",
    "adventure",
    "children",
    "classic_literature",
    "drama",
    "psychology",
]

STORY_COUNTS = [10, 15, 20]

PATTERNS = {
    ConditionType.HIGH_EDITIONS_OLD: [
        "On the Open Library \"{subject}\" subject page, among the first {n} books listed, how many have more than {t1} editions AND were first published before {t2}?",
        "Browse the \"{subject}\" subject on Open Library. Of the first {n} books shown, count how many have {t1}+ editions and a first publish year before {t2}.",
        "On Open Library, look at the \"{subject}\" subject listing. How many of the first {n} books are classic works (before {t2}) with more than {t1} editions?",
    ],
    ConditionType.EDITION_RANGE: [
        "On the Open Library \"{subject}\" subject page, among the first {n} books listed, how many have between {t1} and {t2} editions?",
        "Browse the \"{subject}\" subject on Open Library. Of the first {n} books shown, count how many have edition counts between {t1} and {t2}.",
        "On Open Library, look at the \"{subject}\" subject listing. How many of the first {n} books have a number of editions in the range {t1}\u2013{t2}?",
    ],
    ConditionType.OLD_WITH_FULLTEXT: [
        "On the Open Library \"{subject}\" subject page, among the first {n} books listed, how many were first published before {t1} and have full text available (borrowable)?",
        "Browse the \"{subject}\" subject on Open Library. Of the first {n} books shown, count how many are older works (pre-{t1}) with full text available.",
        "On Open Library, look at the \"{subject}\" subject listing. How many of the first {n} classic works (before {t1}) have their full text available for borrowing?",
    ],
}


@register_template("openlibrary_subject_multi_condition")
class OpenLibrarySubjectMultiConditionTemplate(QuestionTemplate):
    """
    Template for multi-condition count queries on Open Library subject pages.

    HARD difficulty: Requires checking multiple conditions across books
    on a subject page. Cannot be solved by sorting a single column.

    RL value:
    - Exploration space: Must scan subject listing and evaluate each book
    - Delayed reward: Must check multiple books before answering
    - Strategy: Can skip obviously non-matching books
    - Uncertainty: Cannot predict count without checking each book
    - All data visible: edition counts, publish years, fulltext on subject page
    """

    GT_SOURCE = GTSourceType.API_ONLY

    def __init__(self):
        super().__init__("openlibrary_subject_multi_condition")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)

        conditions = list(ConditionType)
        if variant is not None:
            condition = conditions[variant % len(conditions)]
        else:
            condition = rng.choice(conditions)

        n = rng.choice(STORY_COUNTS)
        subject = rng.choice(SUBJECTS)
        display_subject = subject.replace("_", " ")

        thresholds = THRESHOLDS[condition]
        threshold_tuple = rng.choice(thresholds)
        t1 = threshold_tuple[0]
        t2 = threshold_tuple[1] if len(threshold_tuple) > 1 else 0

        patterns = PATTERNS[condition]
        pattern = rng.choice(patterns)
        question_text = pattern.format(n=n, subject=display_subject, t1=t1, t2=t2)

        start_url = f"https://openlibrary.org/subjects/{subject}"

        validation_info = {
            "condition_type": condition.value[0],
            "work_count": n,
            "threshold1": t1,
            "threshold2": t2,
            "subject": subject,
        }

        return GeneratedQuestion(
            question_text=question_text,
            start_url=start_url,
            variables={"condition": condition, "n": n, "subject": subject, "t1": t1, "t2": t2},
            validation_info=validation_info,
            template_name=self.name,
            expected_steps=12,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        condition = validation_info.get("condition_type", "")
        n = validation_info.get("work_count", 10)
        t1 = validation_info.get("threshold1", 0)
        t2 = validation_info.get("threshold2", 0)
        subject = validation_info.get("subject", "")
        return f"""Task-Specific Rules (Open Library Subject Multi-Condition):
- Subject: {subject}
- Condition: {condition} with thresholds {t1}, {t2}
- Must check top {n} books in the subject category
- Score 1.0: Exact count match
- Score 0.5: Count within ±1 of correct answer
- Score 0.0: Wrong count or no answer
- Data is on the subject page on Open Library
- Edition counts, publish years, and borrowability visible on the subject listing"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        condition_type = validation_info.get("condition_type", "")
        n = validation_info.get("work_count", 10)
        t1 = validation_info.get("threshold1", 0)
        t2 = validation_info.get("threshold2", 0)
        subject = validation_info.get("subject", "")

        # Use /subjects/{subject}.json API — returns the exact same data
        # as the HTML subjects page at /subjects/{subject}.
        # This ensures GT and page data share the same source (CLAUDE.md).
        try:
            data = await fetch_subject_api_data(subject, limit=n)
        except Exception as e:
            return GroundTruthResult.fail(f"API call failed: {e}")

        works_dict = data.get("works", {})
        if len(works_dict) < n:
            return GroundTruthResult.fail(
                f"Only {len(works_dict)} works returned for subject '{subject}' "
                f"(need {n}). Subject may not have enough results."
            )

        # Sort by rank and take top n (rank = order on the subject page)
        works = sorted(works_dict.values(), key=lambda x: x.get("rank", 999))[:n]

        # Count matching works
        count = 0
        for work in works:
            edition_count = work.get("edition_count")
            first_year = work.get("first_publish_year")
            has_fulltext = work.get("has_fulltext")

            # Required fields must be present — no defaults (per CLAUDE.md)
            if edition_count is None:
                return GroundTruthResult.fail(
                    f"Missing edition_count for work '{work.get('title', '?')}'"
                )

            match = False
            if condition_type == "high_editions_old":
                if first_year is None:
                    continue
                match = edition_count > t1 and first_year < t2
            elif condition_type == "edition_range":
                match = t1 < edition_count < t2
            elif condition_type == "old_with_fulltext":
                if first_year is None or has_fulltext is None:
                    continue
                match = first_year < t1 and has_fulltext

            if match:
                count += 1

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
        trigger = UrlPatternTrigger(domains=["openlibrary.org"])
        return TriggerConfig(trigger=trigger)

    @classmethod
    def get_cache_source(cls) -> str:
        return "openlibrary"

    def get_gt_source(self) -> GTSourceType:
        return self.GT_SOURCE
