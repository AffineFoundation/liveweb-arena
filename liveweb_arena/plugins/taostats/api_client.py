"""Taostats API client with Bittensor SDK support"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache source name
CACHE_SOURCE = "taostats"


# ============================================================
# Cache Data Fetcher (used by snapshot_integration)
# ============================================================

async def fetch_cache_api_data() -> Optional[Dict[str, Any]]:
    """
    Fetch Bittensor subnet data for all active subnets.

    Returns data structure:
    {
        "_meta": {"source": "taostats", "subnet_count": N},
        "subnets": {
            "1": {"name": "...", "owner": "...", "price": ...},
            ...
        }
    }
    """
    try:
        import bittensor as bt
    except ImportError:
        logger.warning("bittensor package not installed, skipping taostats cache")
        return {"_meta": {"source": CACHE_SOURCE, "subnet_count": 0}, "subnets": {}}

    logger.info("Fetching Bittensor subnet data...")

    result = {
        "_meta": {"source": CACHE_SOURCE, "subnet_count": 0},
        "subnets": {},
    }

    try:
        # Connect to Bittensor network
        subtensor = bt.Subtensor(network="finney")

        # Fetch all subnet info (0-128)
        for subnet_id in range(129):
            try:
                info = subtensor.subnet(subnet_id)
                if info is None:
                    continue

                subnet_data = {}

                # Extract name
                if hasattr(info, 'subnet_name') and info.subnet_name:
                    subnet_data["name"] = info.subnet_name
                elif hasattr(info, 'subnet_identity') and info.subnet_identity:
                    subnet_data["name"] = info.subnet_identity.subnet_name

                # Extract owner
                if hasattr(info, 'owner_coldkey') and info.owner_coldkey:
                    subnet_data["owner"] = info.owner_coldkey

                # Extract price
                if hasattr(info, 'price') and info.price:
                    subnet_data["price"] = float(info.price.tao)

                if subnet_data:
                    result["subnets"][str(subnet_id)] = subnet_data

            except Exception as e:
                logger.debug(f"Failed to fetch subnet {subnet_id}: {e}")
                continue

    except Exception as e:
        logger.error(f"Bittensor connection failed: {e}")
        return result

    result["_meta"]["subnet_count"] = len(result["subnets"])
    logger.info(f"Fetched {len(result['subnets'])} Bittensor subnets")
    return result


async def fetch_single_subnet_data(subnet_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch data for a single subnet.

    Used by page-based cache: each page caches its own subnet's data.

    Args:
        subnet_id: Subnet ID (e.g., "27")

    Returns:
        Dict with subnet data, or empty dict on error
    """
    try:
        import bittensor as bt
    except ImportError:
        logger.warning("bittensor package not installed")
        return {}

    logger.debug(f"Fetching Bittensor data for subnet {subnet_id}...")

    try:
        subnet_id_int = int(subnet_id)
        subtensor = bt.Subtensor(network="finney")
        info = subtensor.subnet(subnet_id_int)

        if info is None:
            return {}

        subnet_data = {}

        # Extract name
        if hasattr(info, 'subnet_name') and info.subnet_name:
            subnet_data["name"] = info.subnet_name
        elif hasattr(info, 'subnet_identity') and info.subnet_identity:
            subnet_data["name"] = info.subnet_identity.subnet_name

        # Extract owner
        if hasattr(info, 'owner_coldkey') and info.owner_coldkey:
            subnet_data["owner"] = info.owner_coldkey

        # Extract price
        if hasattr(info, 'price') and info.price:
            subnet_data["price"] = float(info.price.tao)

        return subnet_data

    except Exception as e:
        logger.debug(f"Failed to fetch subnet {subnet_id}: {e}")
        return {}
