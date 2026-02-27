"""Hybrid cross-site query templates"""

from liveweb_arena.plugins import DISABLED_PLUGINS

from .top_performer import HybridTopPerformerTemplate
from .ranking import HybridRankingTemplate
from .conditional_branch import HybridConditionalBranchTemplate
from .portfolio import HybridPortfolioRebalanceTemplate
from .anomaly import HybridAnomalyDetectionTemplate
from .chained_decision import HybridChainedDecisionTemplate
from .satisficing_search import HybridSatisficingSearchTemplate

__all__ = [
    "HybridTopPerformerTemplate",
    "HybridRankingTemplate",
    "HybridConditionalBranchTemplate",
    "HybridPortfolioRebalanceTemplate",
    "HybridAnomalyDetectionTemplate",
    "HybridChainedDecisionTemplate",
    "HybridSatisficingSearchTemplate",
]

# cross_domain_calc depends on weather plugin
if "weather" not in DISABLED_PLUGINS:
    from .cross_domain_calc import HybridCrossDomainCalcTemplate
    __all__.append("HybridCrossDomainCalcTemplate")
