"""Taostats question templates"""

from .subnet import SubnetInfoTemplate
from .network import NetworkTemplate
from .comparison import ComparisonTemplate
from .analysis import AnalysisTemplate
from .ranking import RankingTemplate
from .price_change import PriceChangeTemplate
from .threshold import ThresholdTemplate
from .multi_condition import MultiConditionTemplate
from .delta import DeltaTemplate
from .range_count import RangeCountTemplate
from .percentage import PercentageTemplate
from .variables import SubnetVariable, MetricVariable, SubnetMetric, SubnetSpec, MetricSpec

__all__ = [
    "SubnetInfoTemplate",
    "NetworkTemplate",
    "ComparisonTemplate",
    "AnalysisTemplate",
    "RankingTemplate",
    "PriceChangeTemplate",
    "ThresholdTemplate",
    "MultiConditionTemplate",
    "DeltaTemplate",
    "RangeCountTemplate",
    "PercentageTemplate",
    "SubnetVariable",
    "MetricVariable",
    "SubnetMetric",
    "SubnetSpec",
    "MetricSpec",
]
