"""JSON-only agent policy for browser action parsing and repair"""

import json
import re
from typing import List, Optional, Tuple

from .models import BrowserAction, BrowserObservation, CompositeTask, TrajectoryStep


# Valid action types
VALID_ACTION_TYPES = {
    "goto", "click", "type", "press", "scroll", "wait", "stop",
    "click_role", "type_role"
}

# System prompt template - built dynamically with plugin hints
SYSTEM_PROMPT_BASE = """You are a web automation agent that interacts with real websites to complete tasks.

You have access to a browser and can navigate to any website to gather information.

{available_tools}

{task_description}

## Action Protocol

You must respond with a single JSON object (no markdown, no extra text). The JSON must have this structure:

```
{{
  "thought": "your reasoning about what to do next",
  "action": {{
    "type": "<action_type>",
    "params": {{...}}
  }}
}}
```

## Available Actions

1. **goto** - Navigate to a URL
   ```json
   {{"action": {{"type": "goto", "params": {{"url": "https://example.com"}}}}}}
   ```

2. **click** - Click an element by CSS selector
   ```json
   {{"action": {{"type": "click", "params": {{"selector": "button.submit"}}}}}}
   ```

3. **type** - Type text into an input field
   ```json
   {{"action": {{"type": "type", "params": {{"selector": "input#search", "text": "query", "press_enter": true}}}}}}
   ```

4. **press** - Press a keyboard key
   ```json
   {{"action": {{"type": "press", "params": {{"key": "Enter"}}}}}}
   ```

5. **scroll** - Scroll the page (for loading dynamic content)
   ```json
   {{"action": {{"type": "scroll", "params": {{"direction": "down", "amount": 300}}}}}}
   ```

6. **view_more** - View more truncated content (when page content is cut off)
   ```json
   {{"action": {{"type": "view_more", "params": {{"direction": "down"}}}}}}
   ```
   Use this when you see "... (content below, use view_more direction=down to see)" to view hidden content.

7. **wait** - Wait for a duration
   ```json
   {{"action": {{"type": "wait", "params": {{"seconds": 2}}}}}}
   ```

8. **click_role** - Click by accessibility role (more stable)
   ```json
   {{"action": {{"type": "click_role", "params": {{"role": "button", "name": "Search"}}}}}}
   ```

9. **type_role** - Type into element by accessibility role
   ```json
   {{"action": {{"type": "type_role", "params": {{"role": "textbox", "name": "Search", "text": "query", "press_enter": true}}}}}}
   ```

10. **stop** - Complete the task and submit answers
   ```json
   {{
     "action": {{
       "type": "stop",
       "params": {{
         "format": "json",
         "final": {{
           "answers": {{"answer1": "value1", "answer2": "value2"}}
         }}
       }}
     }}
   }}
   ```

## Tips

- First analyze the task and decide which website to visit
- Use the "goto" action to navigate to the appropriate URL
- Homepage/list data may be inaccurate (no +/- signs, delayed). Always visit detail pages for precise values
- When done with all tasks, use the "stop" action with your answers

## IMPORTANT

- Output ONLY a single JSON object
- Do NOT include markdown code blocks
- Do NOT include any text before or after the JSON
"""

# Step prompt template
STEP_PROMPT_TEMPLATE = """## Current Page State

URL: {url}
Title: {title}

### Accessibility Tree
```
{accessibility_tree}
```

### Recent Actions
{recent_actions}

What is your next action? Remember: output ONLY a JSON object, no markdown or extra text.
"""


class AgentPolicy:
    """
    JSON-only policy for browser action generation and parsing.

    Responsibilities:
    - Build system and step prompts
    - Parse LLM response to BrowserAction
    - Repair malformed JSON (two-stage)
    """

    def __init__(self, max_recent_steps: int = 5):
        """
        Initialize policy.

        Args:
            max_recent_steps: Number of recent steps to include in prompt
        """
        self._max_recent_steps = max_recent_steps
        self._json_repair_count = 0

    @property
    def json_repair_count(self) -> int:
        """Get count of JSON repairs performed"""
        return self._json_repair_count

    def reset_repair_count(self):
        """Reset JSON repair counter"""
        self._json_repair_count = 0

    def build_system_prompt(self, task: CompositeTask) -> str:
        """Build system prompt with task intent and plugin hints"""
        # Build available tools section from plugin hints
        if task.plugin_hints:
            tools_section = "## Available Information Sources\n\n"
            for plugin_name, usage_hint in task.plugin_hints.items():
                tools_section += usage_hint + "\n\n"
        else:
            tools_section = ""

        return SYSTEM_PROMPT_BASE.format(
            available_tools=tools_section,
            task_description=task.combined_intent,
        )

    def build_step_prompt(
        self,
        obs: BrowserObservation,
        trajectory: List[TrajectoryStep],
    ) -> str:
        """Build step prompt with current observation and recent history"""
        # Format recent actions WITH thoughts (so model remembers its reasoning)
        recent = trajectory[-self._max_recent_steps:] if trajectory else []
        if recent:
            action_lines = []
            for step in recent:
                if step.action:
                    # Include thought so model remembers what it found
                    if step.thought:
                        action_lines.append(f"Step {step.step_num} thought: {step.thought}")
                    action_str = f"Step {step.step_num} action: {step.action.action_type}"
                    if step.action.params:
                        action_str += f" {json.dumps(step.action.params)}"
                    action_str += f" -> {step.action_result}"
                    action_lines.append(action_str)
            recent_actions = "\n".join(action_lines) if action_lines else "(no actions yet)"
        else:
            recent_actions = "(no actions yet)"

        return STEP_PROMPT_TEMPLATE.format(
            url=obs.url,
            title=obs.title,
            accessibility_tree=obs.accessibility_tree,
            recent_actions=recent_actions,
        )

    def parse_response(self, raw: str) -> Tuple[Optional[str], Optional[BrowserAction]]:
        """
        Parse LLM response to extract thought and action.

        Returns:
            Tuple of (thought, BrowserAction) or (None, None) on failure
        """
        # Try direct parse first
        parsed = self._try_parse_json(raw)

        # If direct parse fails, try heuristic extraction
        if parsed is None:
            self._json_repair_count += 1
            parsed = self._extract_json_object(raw)

        if parsed is None:
            return None, None

        # Extract thought and action
        thought = parsed.get("thought")
        action_data = parsed.get("action", {})

        if not action_data:
            return thought, None

        action_type = action_data.get("type", "")
        params = action_data.get("params", {})

        # Validate action type
        if action_type not in VALID_ACTION_TYPES:
            # Try to recover - maybe they used a similar name
            action_type_lower = action_type.lower()
            for valid_type in VALID_ACTION_TYPES:
                if valid_type in action_type_lower or action_type_lower in valid_type:
                    action_type = valid_type
                    break
            else:
                # Default to wait if unknown
                action_type = "wait"
                params = {"seconds": 0.5}

        return thought, BrowserAction(action_type=action_type, params=params)

    def _try_parse_json(self, text: str) -> Optional[dict]:
        """Try to parse text as JSON directly"""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    def _extract_json_object(self, text: str) -> Optional[dict]:
        """
        Extract the largest JSON object from text.

        Handles cases where LLM wraps JSON in markdown or adds extra text.
        """
        # Try to find JSON in markdown code block
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        # Find all potential JSON objects by matching braces
        candidates = []
        depth = 0
        start = None

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

        # Try candidates from largest to smallest
        candidates.sort(key=len, reverse=True)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return None
