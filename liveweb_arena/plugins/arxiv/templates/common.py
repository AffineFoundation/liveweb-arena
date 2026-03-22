"""Shared helpers for ArXiv templates."""

from typing import Any, Dict, List, Optional, Tuple

from liveweb_arena.core.ground_truth_trigger import GroundTruthResult
from liveweb_arena.core.gt_collector import get_current_gt_collector


def get_collected_listing_data(
    category: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[GroundTruthResult]]:
    """Return collected API data for a category listing, or a GT failure."""
    gt_collector = get_current_gt_collector()
    if gt_collector is None:
        return None, GroundTruthResult.system_error("No GT collector")

    collected = gt_collector.get_collected_api_data()
    data = collected.get(f"arxiv:{category}")
    if data is None:
        keys = [k for k in collected if k.startswith("arxiv:")][:5]
        return None, GroundTruthResult.not_collected(
            f"Agent did not visit ArXiv listing for '{category}'. "
            f"Collected keys: {keys}"
        )

    return data, None


def get_papers_from_listing(
    data: Dict[str, Any],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[GroundTruthResult]]:
    """Extract a rank-sorted list of papers from collected listing data."""
    papers = data.get("papers")
    if not papers or not isinstance(papers, dict):
        return None, GroundTruthResult.fail("No papers in collected listing data")

    sorted_papers = sorted(papers.values(), key=lambda p: p["rank"])
    if not sorted_papers:
        return None, GroundTruthResult.fail("Papers dict is empty after sort")

    return sorted_papers, None
