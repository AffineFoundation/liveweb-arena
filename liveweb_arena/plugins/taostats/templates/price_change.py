"""Price change query template for Taostats"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from liveweb_arena.core.validators.base import (
    QuestionTemplate, GeneratedQuestion, ValidationResult, register_template,
    Variable, VariableType
)


class PriceChangeType(Enum):
    """Types of price change queries"""
    TAO_24H = "tao_24h"  # TAO's 24h price change on homepage
    SUBNET_24H = "subnet_24h"  # Specific subnet's 24h change


# Subnets likely to be visible on the first page
TOP_SUBNETS = [
    (1, "Apex"),
    (19, "ItsAI"),
    (51, "lium.io"),
    (64, "Chutes"),
    (9, "Pretrain"),
]


@dataclass
class PriceChangeSpec:
    """Specification for a price change query"""
    change_type: PriceChangeType
    display_name: str


class PriceChangeVariable(Variable):
    """Variable for price change type selection"""

    TYPES: Dict[PriceChangeType, PriceChangeSpec] = {
        PriceChangeType.TAO_24H: PriceChangeSpec(
            PriceChangeType.TAO_24H, "TAO 24h price change"
        ),
        PriceChangeType.SUBNET_24H: PriceChangeSpec(
            PriceChangeType.SUBNET_24H, "subnet 24h price change"
        ),
    }

    def __init__(self):
        super().__init__("price_change_type", VariableType.TEXT)

    def sample(self, rng: random.Random) -> PriceChangeSpec:
        change_type = rng.choice(list(self.TYPES.keys()))
        return self.TYPES[change_type]

    def get_display_value(self, value: PriceChangeSpec) -> str:
        return value.display_name

    def get_api_value(self, value: PriceChangeSpec) -> str:
        return value.change_type.value


@register_template("taostats_price_change")
class PriceChangeTemplate(QuestionTemplate):
    """
    Template for price change queries on Taostats.

    Generates questions about TAO or subnet price changes over time.
    - TAO 24h change is shown on the homepage
    - Subnet 24h changes are shown in the 24H column on /subnets
    """

    TAO_PATTERNS = [
        "Go to taostats.io and find TAO's 24-hour price change percentage.",
        "On taostats.io, what is the 24h price change for TAO?",
        "Visit taostats.io homepage and report the TAO 24h price change.",
    ]

    SUBNET_PATTERNS = [
        "Go to taostats.io/subnets and find the 24h price change for {subnet}.",
        "On taostats.io/subnets, what is {subnet}'s 24-hour price change?",
        "Visit taostats.io/subnets and report the 24H change for {subnet} subnet.",
    ]

    def __init__(self):
        super().__init__("taostats_price_change")
        self.register_variable(PriceChangeVariable())

    def generate(self, seed: int) -> GeneratedQuestion:
        rng = random.Random(seed)

        spec: PriceChangeSpec = self._variables["price_change_type"].sample(rng)

        if spec.change_type == PriceChangeType.TAO_24H:
            pattern = rng.choice(self.TAO_PATTERNS)
            question_text = pattern
            start_url = "https://taostats.io"
            validation_info = {
                "change_type": spec.change_type.value,
                "target": "TAO",
            }
        else:
            # SUBNET_24H
            subnet = rng.choice(TOP_SUBNETS)
            pattern = rng.choice(self.SUBNET_PATTERNS)
            question_text = pattern.format(subnet=subnet[1])
            start_url = "https://taostats.io/subnets"
            validation_info = {
                "change_type": spec.change_type.value,
                "target": subnet[1],
                "subnet_id": subnet[0],
            }

        return GeneratedQuestion(
            question_text=question_text,
            start_url=start_url,
            variables={"price_change": spec},
            validation_info=validation_info,
            template_name=self.name,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        target = validation_info.get("target", "TAO")
        return f"""Task-Specific Rules (24h Price Change for {target}):
- Score 1.0: Agent provides a specific percentage with sign (e.g., "-2.26%", "+5.3%", "down 2.26%")
- Score 0.5: Agent provides percentage without sign or approximate value
- Score 0.0: No percentage or clearly wrong format"""

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> Optional[Any]:
        """Price changes are dynamic, use LLM validation"""
        return None

    async def validate_answer(
        self, answer: str, validation_info: Dict[str, Any]
    ) -> ValidationResult:
        """Validate answer - delegates to LLM validation"""
        return ValidationResult(
            score=0.0,
            is_correct=False,
            expected=None,
            actual=answer,
            details="Price change validation requires LLM",
        )
