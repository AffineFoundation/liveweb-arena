"""Task manager for generating composite tasks"""

import hashlib
import random
from typing import Dict, List, Optional, Type

from ..plugins import DISABLED_PLUGINS
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
        if name in DISABLED_PLUGINS:
            raise ValueError(f"Plugin '{name}' is disabled. Disabled plugins: {DISABLED_PLUGINS}")
        if name not in self._plugin_instances:
            plugin_cls = self._plugin_classes.get(name)
            if not plugin_cls:
                raise ValueError(f"Unknown plugin: {name}. Available: {list(self._plugin_classes.keys())}")
            self._plugin_instances[name] = plugin_cls()
        return self._plugin_instances[name]

    @staticmethod
    def derive_subtask_seed(seed: int, index: int) -> int:
        """Derive a deterministic seed for one subtask inside a composite task."""
        hash_input = f"{seed}:{index}".encode()
        return int(hashlib.sha256(hash_input).hexdigest()[:8], 16)

    def plan_subtasks(
        self,
        seed: int,
        num_subtasks: int = 2,
        templates: Optional[List[tuple]] = None,
    ) -> List[Dict[str, object]]:
        """
        Plan deterministic subtask generation without constructing the full CompositeTask.

        The returned plan is stable and matches the seeds/templates that
        generate_composite_task() will use.
        """
        num_subtasks = max(1, min(4, num_subtasks))
        rng = random.Random(seed)

        if templates:
            selected_templates = []
            for i in range(num_subtasks):
                t = templates[i % len(templates)]
                if len(t) == 2:
                    selected_templates.append((t[0], t[1], None))
                else:
                    selected_templates.append(t)
        else:
            available = list(self._plugin_classes.keys())
            if len(available) == 0:
                raise ValueError("No plugins available")
            selected_templates = [(rng.choice(available), None, None) for _ in range(num_subtasks)]

        plan = []
        for i, (plugin_name, template_name, variant) in enumerate(selected_templates):
            plan.append(
                {
                    "subtask_index": i,
                    "plugin_name": plugin_name,
                    "template_name": template_name,
                    "variant": variant,
                    "subtask_seed": self.derive_subtask_seed(seed, i),
                }
            )
        return plan

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
            templates: List of (plugin, template_name, variant) tuples; None = random.
                       variant can be None for random selection or int for specific variant.

        Returns:
            CompositeTask with subtasks and combined_intent
        """
        plan = self.plan_subtasks(seed=seed, num_subtasks=num_subtasks, templates=templates)

        # Initialize plugins that will be used (some need API data before question generation)
        plugins_to_use = {item["plugin_name"] for item in plan}
        for plugin_name in plugins_to_use:
            plugin = self._get_plugin(plugin_name)
            if hasattr(plugin, 'initialize'):
                plugin.initialize()

        # Generate sub-tasks
        subtasks: List[SubTask] = []

        for i, item in enumerate(plan):
            plugin_name = str(item["plugin_name"])
            template_name = item["template_name"]
            variant = item["variant"]
            plugin = self._get_plugin(plugin_name)
            subtask_seed = int(item["subtask_seed"])
            subtask = await plugin.generate_task(
                subtask_seed,
                template_name=template_name,
                variant=variant,
            )
            # Override answer_tag
            subtask.answer_tag = f"answer{i + 1}"
            subtasks.append(subtask)

        # Build combined intent (without start_url - Agent decides navigation)
        combined_intent = self._build_combined_intent(subtasks)

        return CompositeTask(
            subtasks=subtasks,
            combined_intent=combined_intent,
            plugin_hints={},
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
