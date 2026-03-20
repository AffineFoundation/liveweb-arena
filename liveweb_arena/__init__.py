"""LiveWeb Arena - Real-time web interaction evaluation for LLM browser agents"""

__version__ = "0.1.0"

# Core components that do not require optional browser runtime deps.
from .core.models import BrowserObservation, BrowserAction, CompositeTask, TrajectoryStep
from .plugins.base import BasePlugin, SubTask, ValidationResult

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


def __getattr__(name: str):
    """Lazy-load browser classes so base imports work without Playwright."""
    if name in {"BrowserEngine", "BrowserSession"}:
        from .core.browser import BrowserEngine, BrowserSession

        return {"BrowserEngine": BrowserEngine, "BrowserSession": BrowserSession}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
