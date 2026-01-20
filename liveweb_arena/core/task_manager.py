"""Task manager for generating composite tasks"""

import random
from typing import Dict, List, Optional, Type

from ..plugins.base import BasePlugin, SubTask
from .models import CompositeTask


class TaskManager:
    """
    Manages task generation and composition.
    Uses seed for deterministic task generation.
    """

    def __init__(self, plugins: Dict[str, Type[BasePlugin]]):
        """
        Initialize TaskManager with plugin registry.

        Args:
            plugins: Dictionary mapping plugin name to plugin class
        """
        self._plugin_classes = plugins
        self._plugin_instances: Dict[str, BasePlugin] = {}

    def _get_plugin(self, name: str) -> BasePlugin:
        """Get or create plugin instance"""
        if name not in self._plugin_instances:
            plugin_cls = self._plugin_classes.get(name)
            if not plugin_cls:
                raise ValueError(f"Unknown plugin: {name}. Available: {list(self._plugin_classes.keys())}")
            self._plugin_instances[name] = plugin_cls()
        return self._plugin_instances[name]

    async def generate_composite_task(
        self,
        seed: int,
        num_subtasks: int = 2,
        templates: Optional[List[tuple]] = None,
    ) -> CompositeTask:
        """
        Generate a composite task with multiple sub-tasks.

        Args:
            seed: Random seed for deterministic generation
            num_subtasks: Number of sub-tasks (1-4)
            templates: List of (plugin, template_name) tuples; None = random

        Returns:
            CompositeTask with subtasks and combined_intent
        """
        # Validate num_subtasks
        num_subtasks = max(1, min(4, num_subtasks))

        # Initialize RNG with seed for deterministic generation
        rng = random.Random(seed)

        # Build list of (plugin_name, template_name) for each subtask
        if templates:
            # Use specified templates (cycle if not enough)
            selected_templates = []
            for i in range(num_subtasks):
                selected_templates.append(templates[i % len(templates)])
        else:
            # Random selection from available plugins (no specific template)
            available = list(self._plugin_classes.keys())
            if len(available) == 0:
                raise ValueError("No plugins available")
            selected_templates = [(rng.choice(available), None) for _ in range(num_subtasks)]

        # Generate sub-tasks
        subtasks: List[SubTask] = []

        for i, (plugin_name, template_name) in enumerate(selected_templates):
            plugin = self._get_plugin(plugin_name)
            # Derive seed for this sub-task
            subtask_seed = seed + i * 1000
            subtask = await plugin.generate_task(
                subtask_seed,
                template_name=template_name,
            )
            # Override answer_tag
            subtask.answer_tag = f"answer{i + 1}"
            subtasks.append(subtask)

        # Always include ALL plugin hints (not just selected ones)
        # This ensures agent knows about all available data sources
        plugin_hints: Dict[str, str] = {}
        for plugin_name in self._plugin_classes.keys():
            plugin = self._get_plugin(plugin_name)
            plugin_hints[plugin_name] = plugin.usage_hint

        # Build combined intent (without start_url - Agent decides navigation)
        combined_intent = self._build_combined_intent(subtasks)

        return CompositeTask(
            subtasks=subtasks,
            combined_intent=combined_intent,
            plugin_hints=plugin_hints,
            seed=seed,
        )

    def _build_combined_intent(self, subtasks: List[SubTask]) -> str:
        """Build combined intent (tasks only, no URLs - Agent decides navigation)"""
        # Build task list
        task_lines = []
        for i, subtask in enumerate(subtasks):
            task_lines.append(f"{i + 1}. {subtask.intent}")
            task_lines.append(f"   Answer tag: {subtask.answer_tag}")
            task_lines.append("")

        tasks_text = "\n".join(task_lines)

        # Build answer template
        answer_keys = {f"answer{i + 1}": "..." for i in range(len(subtasks))}
        answer_example = '{"answers": ' + str(answer_keys).replace("'", '"') + '}'

        combined = f"""## Tasks to Complete

{tasks_text}

## Output Requirements

When you have completed all tasks, use the "stop" action with your answers in this JSON format:

```json
{answer_example}
```

Each answer should be a concise, direct response to the corresponding task.
"""
        return combined

    def get_plugin(self, name: str) -> BasePlugin:
        """Get plugin instance by name (for validation)"""
        return self._get_plugin(name)
