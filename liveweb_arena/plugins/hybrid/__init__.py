"""Hybrid cross-site query plugin"""

from .hybrid import HybridPlugin

# Import templates to register them
from . import templates

__all__ = ["HybridPlugin"]
