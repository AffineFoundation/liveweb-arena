"""
Question Template Module.

Defines the interface for question templates and the registry system.

Usage:
    @register_template("coingecko/price")
    class PriceTemplate(QuestionTemplate):
        plugin_name = "coingecko"
        expected_steps = 3

        def generate(self, rng: Random) -> GeneratedQuestion:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Dict, List, Optional

from liveweb_arena.core.cache import CachedPage, PageRequirement


@dataclass
class GeneratedQuestion:
    """
    A generated question instance.

    Attributes:
        intent: The question text
        required_pages: List of pages that need to be cached
        answer_extractor: Function to extract answer from cached data
        expected_steps: Expected number of browser interaction steps
        metadata: Additional metadata for the question
    """
    intent: str
    required_pages: List[PageRequirement]
    answer_extractor: Callable[[Dict[str, CachedPage]], str]
    expected_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuestionTemplate(ABC):
    """
    Base class for question templates.

    Subclasses must:
    1. Define class attributes: plugin_name, expected_steps
    2. Implement generate() method

    Example:
        @register_template("coingecko/price")
        class PriceTemplate(QuestionTemplate):
            plugin_name = "coingecko"
            expected_steps = 3
            difficulty = "easy"

            def generate(self, rng: Random) -> GeneratedQuestion:
                coin = rng.choice(COINS)
                url = f"https://www.coingecko.com/en/coins/{coin.id}"

                return GeneratedQuestion(
                    intent=f"What is the current price of {coin.name}?",
                    required_pages=[PageRequirement.data(url)],
                    answer_extractor=lambda data: str(data[url].api_data["current_price"]),
                    expected_steps=self.expected_steps,
                )
    """

    # Must be defined by subclasses
    plugin_name: str
    expected_steps: int
    difficulty: str = "medium"

    @abstractmethod
    def generate(self, rng: Random) -> GeneratedQuestion:
        """
        Generate a question instance using the random number generator.

        Args:
            rng: Random number generator for reproducible generation

        Returns:
            GeneratedQuestion with intent, required_pages, and answer_extractor
        """
        pass

    @classmethod
    def get_cache_source(cls) -> str:
        """Return the cache source name (for backward compatibility)."""
        return cls.plugin_name


# Template registry
_templates: Dict[str, type] = {}


def register_template(name: str):
    """
    Decorator to register a question template.

    Args:
        name: Template name in format "plugin/template_name"

    Example:
        @register_template("coingecko/price")
        class PriceTemplate(QuestionTemplate):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, QuestionTemplate):
            raise TypeError(f"{cls.__name__} must be a subclass of QuestionTemplate")
        _templates[name] = cls
        return cls
    return decorator


def get_template(name: str) -> Optional[type]:
    """Get a registered template by name."""
    return _templates.get(name)


def get_all_templates() -> Dict[str, type]:
    """Get all registered templates."""
    return dict(_templates)


def get_templates_for_plugin(plugin_name: str) -> Dict[str, type]:
    """Get all templates for a specific plugin."""
    result = {}
    for name, cls in _templates.items():
        if hasattr(cls, 'plugin_name') and cls.plugin_name == plugin_name:
            result[name] = cls
    return result


def clear_templates():
    """Clear all registered templates (for testing)."""
    _templates.clear()
