"""Taostats question templates"""

from .subnet import SubnetInfoTemplate
from .ranking import SubnetRankingTemplate
from .tokenomics import TokenomicsTemplate
from .validator import ValidatorTemplate
from .comparison import ComparisonTemplate
from .price_change import PriceChangeTemplate
from .variables import SubnetVariable, MetricVariable, SubnetMetric, SubnetSpec, MetricSpec

__all__ = [
    "SubnetInfoTemplate",
    "SubnetRankingTemplate",
    "TokenomicsTemplate",
    "ValidatorTemplate",
    "ComparisonTemplate",
    "PriceChangeTemplate",
    "SubnetVariable",
    "MetricVariable",
    "SubnetMetric",
    "SubnetSpec",
    "MetricSpec",
]
