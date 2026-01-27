"""Stooq plugin for financial market data"""

from .stooq import StooqPlugin

# Import templates to register them
from . import templates

__all__ = ["StooqPlugin"]
