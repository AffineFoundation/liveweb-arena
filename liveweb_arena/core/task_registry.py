"""
Task ID Registry - Deterministic task generation for reproducible evaluations.

This module provides a stable mapping from task_id to question configurations.
Adding new templates only appends new combinations, never affecting existing task_ids.

Usage:
    # With task_id (deterministic)
    config = TaskRegistry.parse_task_id(12345)

    # Without task_id (random, backward compatible)
    config = TaskRegistry.random_config(seed=100, num_tasks=3)

Adding new templates:
    1. Add template to TEMPLATES dict with a new ID
    2. Call TaskRegistry.rebuild_combinations()
    3. New combinations are appended, old task_ids unchanged
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any
import hashlib


class TaskRegistry:
    """Registry for deterministic task_id to question configuration mapping."""

    # Task IDs allocated per combination
    TASK_IDS_PER_COMBO = 10000

    # Maximum templates in a combination (1, 2, or 3)
    MAX_COMBO_SIZE = 3

    # Template registry: ID -> (plugin_name, template_name)
    # IDs are permanent, only append new ones
    TEMPLATES: Dict[int, Tuple[str, str]] = {
        # Weather templates
        1: ("weather", "location_name"),
        2: ("weather", "time_of_day"),
        3: ("weather", "multi_day"),
        4: ("weather", "current_weather"),
        5: ("weather", "astronomy"),

        # Stooq templates
        10: ("stooq", "stooq_price"),
        11: ("stooq", "stooq_comparison"),
        12: ("stooq", "stooq_ranking"),
        13: ("stooq", "stooq_sector_analysis"),
        14: ("stooq", "stooq_52week"),
        15: ("stooq", "stooq_currency"),

        # Taostats templates
        20: ("taostats", "taostats_subnet_info"),
        21: ("taostats", "taostats_network"),
        22: ("taostats", "taostats_price"),

        # CoinGecko templates
        30: ("coingecko", "coingecko_price"),

        # Add new templates here with new IDs...
    }

    # Combination registry: list of template ID tuples
    # Order is permanent, only append new combinations
    _combinations: List[Tuple[int, ...]] = []
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls):
        """Ensure combinations are initialized."""
        if not cls._initialized:
            cls.rebuild_combinations()

    @classmethod
    def rebuild_combinations(cls):
        """
        Build all template combinations.

        This generates combinations in a deterministic order:
        1. All 1-template combinations (sorted by ID)
        2. All 2-template combinations (sorted)
        3. All 3-template combinations (sorted)

        When new templates are added, their combinations are appended
        at the end of each size group, maintaining existing indices.
        """
        template_ids = sorted(cls.TEMPLATES.keys())

        new_combinations = []
        for size in range(1, cls.MAX_COMBO_SIZE + 1):
            for combo in combinations(template_ids, size):
                new_combinations.append(combo)

        cls._combinations = new_combinations
        cls._initialized = True

    @classmethod
    def get_combinations(cls) -> List[Tuple[int, ...]]:
        """Get all registered combinations."""
        cls._ensure_initialized()
        return cls._combinations.copy()

    @classmethod
    def max_task_id(cls) -> int:
        """Get the maximum valid task_id."""
        cls._ensure_initialized()
        return len(cls._combinations) * cls.TASK_IDS_PER_COMBO

    @classmethod
    def parse_task_id(cls, task_id: int) -> Dict[str, Any]:
        """
        Parse a task_id into its configuration.

        Args:
            task_id: The task ID (1 to max_task_id)

        Returns:
            Dict with:
            - task_id: The original task_id
            - combo_index: Index into combinations list
            - template_ids: Tuple of template IDs in this combination
            - templates: List of (plugin, template_name) tuples
            - variation_seed: Seed for variation within this combination
            - num_tasks: Number of sub-tasks (3-5)

        Raises:
            ValueError: If task_id is out of valid range
        """
        cls._ensure_initialized()

        if task_id < 1:
            raise ValueError("task_id must be >= 1")

        combo_index = (task_id - 1) // cls.TASK_IDS_PER_COMBO
        variation_seed = (task_id - 1) % cls.TASK_IDS_PER_COMBO

        if combo_index >= len(cls._combinations):
            raise ValueError(
                f"task_id {task_id} out of range. "
                f"Valid range: 1 - {cls.max_task_id()}"
            )

        template_ids = cls._combinations[combo_index]
        templates = [cls.TEMPLATES[tid] for tid in template_ids]

        # Derive num_tasks (3-5) from variation_seed
        num_tasks = (variation_seed % 3) + 3

        return {
            "task_id": task_id,
            "combo_index": combo_index,
            "template_ids": template_ids,
            "templates": templates,
            "variation_seed": variation_seed,
            "num_tasks": num_tasks,
        }

    @classmethod
    def get_sub_task_seeds(cls, variation_seed: int, num_tasks: int) -> List[int]:
        """
        Generate deterministic seeds for each sub-task.

        Args:
            variation_seed: Base seed from task_id
            num_tasks: Number of sub-tasks

        Returns:
            List of seeds, one per sub-task
        """
        seeds = []
        for i in range(num_tasks):
            # Use hash for stable seed derivation
            hash_input = f"{variation_seed}:{i}".encode()
            hash_value = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
            seeds.append(hash_value)
        return seeds

    @classmethod
    def select_templates_for_tasks(
        cls,
        template_ids: Tuple[int, ...],
        variation_seed: int,
        num_tasks: int
    ) -> List[Tuple[str, str]]:
        """
        Select which template to use for each sub-task.

        Args:
            template_ids: Available template IDs for this combination
            variation_seed: Seed for random selection
            num_tasks: Number of sub-tasks

        Returns:
            List of (plugin, template_name) for each sub-task
        """
        import random
        rng = random.Random(variation_seed)

        selected = []
        for _ in range(num_tasks):
            tid = rng.choice(template_ids)
            selected.append(cls.TEMPLATES[tid])

        return selected

    @classmethod
    def random_task_id(cls, rng=None) -> int:
        """
        Generate a random valid task_id.

        Args:
            rng: Optional random.Random instance

        Returns:
            A random task_id in valid range
        """
        import random
        cls._ensure_initialized()

        if rng is None:
            rng = random.Random()

        return rng.randint(1, cls.max_task_id())

    @classmethod
    def get_template_info(cls, template_id: int) -> Optional[Tuple[str, str]]:
        """Get (plugin, template_name) for a template ID."""
        return cls.TEMPLATES.get(template_id)

    @classmethod
    def list_templates(cls) -> Dict[int, Tuple[str, str]]:
        """List all registered templates."""
        return cls.TEMPLATES.copy()

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get registry statistics."""
        cls._ensure_initialized()

        combo_by_size = {}
        for combo in cls._combinations:
            size = len(combo)
            combo_by_size[size] = combo_by_size.get(size, 0) + 1

        return {
            "num_templates": len(cls.TEMPLATES),
            "num_combinations": len(cls._combinations),
            "max_task_id": cls.max_task_id(),
            "task_ids_per_combo": cls.TASK_IDS_PER_COMBO,
            "combinations_by_size": combo_by_size,
        }

    @classmethod
    def print_info(cls):
        """Print registry information."""
        stats = cls.get_stats()
        print("=" * 50)
        print("Task Registry Info")
        print("=" * 50)
        print(f"Templates: {stats['num_templates']}")
        print(f"Combinations: {stats['num_combinations']}")
        print(f"Max task_id: {stats['max_task_id']}")
        print(f"Task IDs per combo: {stats['task_ids_per_combo']}")
        print(f"Combinations by size: {stats['combinations_by_size']}")
        print()
        print("Template List:")
        for tid, (plugin, name) in sorted(cls.TEMPLATES.items()):
            print(f"  {tid:3d}: {plugin}/{name}")


# Convenience functions for external use
def parse_task_id(task_id: int) -> Dict[str, Any]:
    """Parse a task_id into configuration. See TaskRegistry.parse_task_id."""
    return TaskRegistry.parse_task_id(task_id)


def max_task_id() -> int:
    """Get maximum valid task_id."""
    return TaskRegistry.max_task_id()


def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics."""
    return TaskRegistry.get_stats()


# Initialize on import
TaskRegistry._ensure_initialized()


if __name__ == "__main__":
    # Demo
    TaskRegistry.print_info()

    print("\nExample task_id parsing:")
    for tid in [1, 10001, 50001, 100000]:
        try:
            config = parse_task_id(tid)
            print(f"\ntask_id={tid}:")
            print(f"  templates: {config['templates']}")
            print(f"  num_tasks: {config['num_tasks']}")
            print(f"  variation_seed: {config['variation_seed']}")
        except ValueError as e:
            print(f"\ntask_id={tid}: {e}")
