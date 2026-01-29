"""Stooq question templates"""

from .price import StooqPriceTemplate
from .comparison import StooqComparisonTemplate
from .historical import StooqHistoricalTemplate
from .market_summary import StooqMarketSummaryTemplate
from .currency import StooqCurrencyTemplate
from .ranking import StooqRankingTemplate
from .sector_analysis import StooqSectorAnalysisTemplate
from .variables import (
    StockVariable, IndexVariable, CurrencyVariable, CommodityVariable,
    PriceMetricVariable, StockSpec, IndexSpec, CurrencySpec, CommoditySpec,
    MetricSpec, PriceMetric, InstrumentType,
    US_STOCKS, INDICES, CURRENCIES, COMMODITIES,
)

__all__ = [
    "StooqPriceTemplate",
    "StooqComparisonTemplate",
    "StooqHistoricalTemplate",
    "StooqMarketSummaryTemplate",
    "StooqCurrencyTemplate",
    "StooqRankingTemplate",
    "StooqSectorAnalysisTemplate",
    "StockVariable",
    "IndexVariable",
    "CurrencyVariable",
    "CommodityVariable",
    "PriceMetricVariable",
    "StockSpec",
    "IndexSpec",
    "CurrencySpec",
    "CommoditySpec",
    "MetricSpec",
    "PriceMetric",
    "InstrumentType",
    "US_STOCKS",
    "INDICES",
    "CURRENCIES",
    "COMMODITIES",
]
