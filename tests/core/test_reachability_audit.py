from liveweb_arena.core.reachability_audit import audit_reachability_failure
from liveweb_arena.plugins.coingecko.coingecko import CoinGeckoPlugin
from liveweb_arena.plugins.stooq.stooq import StooqPlugin
from liveweb_arena.plugins.taostats.taostats import TaostatsPlugin


def test_reachability_audit_classifies_nav_aborted_from_navigation_metadata():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets/73",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("navigation failed"),
        evidence={
            "navigation_metadata": {
                "classification_hint": "env_nav_aborted",
                "navigation_stage": "goto_domcontentloaded",
                "raw_exception_type": "Error",
                "raw_exception_message": "net::ERR_ABORTED; maybe frame was detached",
                "attempt_index": 1,
                "max_attempts": 2,
            }
        },
    )
    assert audit.classification == "env_nav_aborted"
    assert audit.navigation_stage == "goto_domcontentloaded"
    assert audit.raw_exception_type == "Error"


def test_reachability_audit_classifies_target_closed():
    audit = audit_reachability_failure(
        url="https://www.coingecko.com/en/coins/bitcoin",
        plugin_name="coingecko",
        plugin=CoinGeckoPlugin(),
        exception=RuntimeError("Target page, context or browser has been closed"),
        evidence={
            "navigation_metadata": {
                "classification_hint": "env_target_closed",
                "raw_exception_type": "TargetClosedError",
                "raw_exception_message": "Target page, context or browser has been closed",
            }
        },
    )
    assert audit.classification == "env_target_closed"
    assert audit.is_environment_failure is True


def test_reachability_audit_classifies_cdn_blocked():
    audit = audit_reachability_failure(
        url="https://www.coingecko.com/en/coins/bitcoin",
        plugin_name="coingecko",
        plugin=CoinGeckoPlugin(),
        http_status=403,
        evidence={"site_probe": {"cf_ray": "abc123", "server": "cloudflare"}},
    )
    assert audit.classification == "env_cdn_blocked"


def test_reachability_audit_classifies_stooq_invalid_shape():
    audit = audit_reachability_failure(
        url="https://stooq.com/q/?q=WMT",
        plugin_name="stooq",
        plugin=StooqPlugin(),
    )
    assert audit.classification == "model_invalid_url_shape"
    assert audit.is_model_hallucination is True
