"""ArXiv plugin for browsing and querying academic paper listings."""

from .arxiv import ArxivPlugin

# Import templates to register them
from . import templates as templates  # noqa: F401 — import registers templates

__all__ = ["ArxivPlugin"]
