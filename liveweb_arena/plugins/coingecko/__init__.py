"""CoinGecko plugin for cryptocurrency data queries"""

from .coingecko import CoinGeckoPlugin

# Import templates to register them
from . import templates

__all__ = ["CoinGeckoPlugin"]
