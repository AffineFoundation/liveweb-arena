"""Data models for WebArena Dynamic"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..plugins.base import SubTask


@dataclass
class BrowserObservation:
    """Observation from browser state"""
    url: str
    title: str
    accessibility_tree: str  # Truncated accessibility tree
    html: Optional[str] = None
    screenshot: Optional[bytes] = None


@dataclass
class BrowserAction:
    """Action to execute in browser"""
    action_type: str  # goto|click|type|press|scroll|wait|stop|click_role|type_role|view_more
    params: dict = field(default_factory=dict)


@dataclass
class CompositeTask:
    """A composite task containing multiple sub-tasks"""
    subtasks: List["SubTask"]  # List of SubTask from plugins.base
    combined_intent: str
    plugin_hints: Dict[str, str]
    seed: int


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory"""
    step_num: int
    observation: BrowserObservation
    thought: Optional[str] = None
    action: Optional[BrowserAction] = None
    action_result: str = ""
    prompt: Optional[str] = None  # Actual prompt sent to LLM
