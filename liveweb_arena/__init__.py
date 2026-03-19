"""LiveWeb Arena - Real-time web interaction evaluation for LLM browser agents"""

__version__ = "0.1.0"

# Core components
from .core.models import BrowserObservation, BrowserAction, CompositeTask, TrajectoryStep
from .plugins.base import BasePlugin, SubTask, ValidationResult

# Avoid importing Playwright-dependent browser code at import time.
# This keeps `liveweb_arena.core.*` usable in minimal/test environments.
try:
    from .core.browser import BrowserEngine, BrowserSession
except ModuleNotFoundError:
    BrowserEngine = None
    BrowserSession = None

__all__ = [
    "__version__",
    # Models
    "BrowserObservation",
    "BrowserAction",
    "CompositeTask",
    "TrajectoryStep",
    # Browser
    "BrowserEngine",
    "BrowserSession",
    # Plugins
    "BasePlugin",
    "SubTask",
    "ValidationResult",
]
