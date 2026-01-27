"""Weather plugin package"""

from .weather import WeatherPlugin

# Import templates to register them
from . import templates

__all__ = ["WeatherPlugin"]
