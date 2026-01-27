"""Variables for Stooq financial data question templates"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from liveweb_arena.core.validators.base import Variable, VariableType


class InstrumentType(Enum):
    """Types of financial instruments on Stooq"""
    STOCK = "stock"
    INDEX = "index"
    CURRENCY = "currency"
    COMMODITY = "commodity"


class PriceMetric(Enum):
    """Price-related metrics available on Stooq"""
    LAST_PRICE = "last_price"
    CHANGE_PERCENT = "change_percent"
    CHANGE_ABSOLUTE = "change_absolute"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


@dataclass
class StockSpec:
    """Specification for a stock symbol"""
    symbol: str  # e.g., "aapl.us"
    display_name: str  # e.g., "Apple (AAPL)"
    exchange: str  # e.g., "US", "UK", "DE"


@dataclass
class IndexSpec:
    """Specification for a market index"""
    symbol: str  # e.g., "^dji"
    display_name: str  # e.g., "Dow Jones Industrial Average"
    region: str  # e.g., "US", "Europe", "Asia"


@dataclass
class CurrencySpec:
    """Specification for a currency pair"""
    symbol: str  # e.g., "eurusd"
    display_name: str  # e.g., "EUR/USD"
    base: str  # e.g., "EUR"
    quote: str  # e.g., "USD"


@dataclass
class CommoditySpec:
    """Specification for a commodity"""
    symbol: str  # e.g., "gc.f"
    display_name: str  # e.g., "Gold"
    category: str  # e.g., "metals", "energy"


@dataclass
class MetricSpec:
    """Specification for a price metric"""
    metric: PriceMetric
    display_name: str
    unit: str = ""
    is_percentage: bool = False


# Popular US stocks with high liquidity
US_STOCKS = [
    StockSpec("aapl.us", "Apple (AAPL)", "US"),
    StockSpec("msft.us", "Microsoft (MSFT)", "US"),
    StockSpec("googl.us", "Alphabet (GOOGL)", "US"),
    StockSpec("amzn.us", "Amazon (AMZN)", "US"),
    StockSpec("nvda.us", "NVIDIA (NVDA)", "US"),
    StockSpec("meta.us", "Meta Platforms (META)", "US"),
    StockSpec("tsla.us", "Tesla (TSLA)", "US"),
    StockSpec("jpm.us", "JPMorgan Chase (JPM)", "US"),
    StockSpec("v.us", "Visa (V)", "US"),
    StockSpec("wmt.us", "Walmart (WMT)", "US"),
    StockSpec("xom.us", "Exxon Mobil (XOM)", "US"),
    StockSpec("ko.us", "Coca-Cola (KO)", "US"),
    StockSpec("dis.us", "Walt Disney (DIS)", "US"),
    StockSpec("nke.us", "Nike (NKE)", "US"),
    StockSpec("intc.us", "Intel (INTC)", "US"),
    StockSpec("amd.us", "AMD (AMD)", "US"),
    StockSpec("coin.us", "Coinbase (COIN)", "US"),
]

# Major global indices
INDICES = [
    IndexSpec("^dji", "Dow Jones Industrial Average", "US"),
    IndexSpec("^spx", "S&P 500", "US"),
    IndexSpec("^ndx", "NASDAQ 100", "US"),
    IndexSpec("^ukx", "FTSE 100", "Europe"),
    IndexSpec("^dax", "DAX", "Europe"),
    IndexSpec("^cac", "CAC 40", "Europe"),
    IndexSpec("^nkx", "Nikkei 225", "Asia"),
    IndexSpec("^hsi", "Hang Seng Index", "Asia"),
    IndexSpec("^kospi", "KOSPI", "Asia"),
]

# Major currency pairs
CURRENCIES = [
    CurrencySpec("eurusd", "EUR/USD", "EUR", "USD"),
    CurrencySpec("gbpusd", "GBP/USD", "GBP", "USD"),
    CurrencySpec("usdjpy", "USD/JPY", "USD", "JPY"),
    CurrencySpec("usdchf", "USD/CHF", "USD", "CHF"),
    CurrencySpec("audusd", "AUD/USD", "AUD", "USD"),
    CurrencySpec("usdcad", "USD/CAD", "USD", "CAD"),
    CurrencySpec("nzdusd", "NZD/USD", "NZD", "USD"),
    CurrencySpec("eurgbp", "EUR/GBP", "EUR", "GBP"),
    CurrencySpec("eurjpy", "EUR/JPY", "EUR", "JPY"),
]

# Major commodities
COMMODITIES = [
    CommoditySpec("gc.f", "Gold Futures", "metals"),
    CommoditySpec("si.f", "Silver Futures", "metals"),
    CommoditySpec("hg.f", "Copper", "metals"),
    CommoditySpec("cl.f", "Crude Oil (WTI)", "energy"),
    CommoditySpec("ng.f", "Natural Gas", "energy"),
    CommoditySpec("zc.f", "Corn", "agriculture"),
    CommoditySpec("zw.f", "Wheat", "agriculture"),
    CommoditySpec("zs.f", "Soybeans", "agriculture"),
    # Spot prices (used by hybrid templates)
    CommoditySpec("xauusd", "Gold", "metals"),
    CommoditySpec("xagusd", "Silver", "metals"),
]


class StockVariable(Variable):
    """Variable for stock symbol selection"""

    def __init__(self, stocks: List[StockSpec] = None):
        super().__init__("stock", VariableType.TEXT)
        self.stocks = stocks or US_STOCKS

    def sample(self, rng: random.Random) -> StockSpec:
        return rng.choice(self.stocks)

    def get_display_value(self, value: StockSpec) -> str:
        return value.display_name

    def get_api_value(self, value: StockSpec) -> str:
        return value.symbol


class IndexVariable(Variable):
    """Variable for market index selection"""

    def __init__(self, indices: List[IndexSpec] = None, regions: List[str] = None):
        super().__init__("index", VariableType.TEXT)
        all_indices = indices or INDICES
        if regions:
            self.indices = [i for i in all_indices if i.region in regions]
        else:
            self.indices = all_indices

    def sample(self, rng: random.Random) -> IndexSpec:
        return rng.choice(self.indices)

    def get_display_value(self, value: IndexSpec) -> str:
        return value.display_name

    def get_api_value(self, value: IndexSpec) -> str:
        return value.symbol


class CurrencyVariable(Variable):
    """Variable for currency pair selection"""

    def __init__(self, currencies: List[CurrencySpec] = None):
        super().__init__("currency", VariableType.TEXT)
        self.currencies = currencies or CURRENCIES

    def sample(self, rng: random.Random) -> CurrencySpec:
        return rng.choice(self.currencies)

    def get_display_value(self, value: CurrencySpec) -> str:
        return value.display_name

    def get_api_value(self, value: CurrencySpec) -> str:
        return value.symbol


class CommodityVariable(Variable):
    """Variable for commodity selection"""

    def __init__(self, commodities: List[CommoditySpec] = None, categories: List[str] = None):
        super().__init__("commodity", VariableType.TEXT)
        all_commodities = commodities or COMMODITIES
        if categories:
            self.commodities = [c for c in all_commodities if c.category in categories]
        else:
            self.commodities = all_commodities

    def sample(self, rng: random.Random) -> CommoditySpec:
        return rng.choice(self.commodities)

    def get_display_value(self, value: CommoditySpec) -> str:
        return value.display_name

    def get_api_value(self, value: CommoditySpec) -> str:
        return value.symbol


class PriceMetricVariable(Variable):
    """Variable for price metric selection"""

    METRICS = {
        PriceMetric.LAST_PRICE: MetricSpec(
            PriceMetric.LAST_PRICE, "current price", "", False
        ),
        PriceMetric.CHANGE_PERCENT: MetricSpec(
            PriceMetric.CHANGE_PERCENT, "percentage change", "%", True
        ),
        PriceMetric.CHANGE_ABSOLUTE: MetricSpec(
            PriceMetric.CHANGE_ABSOLUTE, "price change", "", False
        ),
        PriceMetric.OPEN: MetricSpec(
            PriceMetric.OPEN, "opening price", "", False
        ),
        PriceMetric.HIGH: MetricSpec(
            PriceMetric.HIGH, "daily high", "", False
        ),
        PriceMetric.LOW: MetricSpec(
            PriceMetric.LOW, "daily low", "", False
        ),
    }

    def __init__(self, allowed_metrics: List[PriceMetric] = None):
        super().__init__("metric", VariableType.TEXT)
        self.allowed_metrics = allowed_metrics or [
            PriceMetric.LAST_PRICE,
            PriceMetric.CHANGE_PERCENT,
        ]

    def sample(self, rng: random.Random) -> MetricSpec:
        metric = rng.choice(self.allowed_metrics)
        return self.METRICS[metric]

    def get_display_value(self, value: MetricSpec) -> str:
        return value.display_name

    def get_api_value(self, value: MetricSpec) -> str:
        return value.metric.value
