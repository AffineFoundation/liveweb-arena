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


def test_reachability_audit_classifies_from_raw_navigation_message_when_outer_error_is_generic():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets/73",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("Too many consecutive action failures (5)"),
        evidence={
            "navigation_metadata": {
                "raw_exception_type": "Error",
                "raw_exception_message": "net::ERR_ABORTED; maybe frame was detached",
                "navigation_stage": "action_click",
            }
        },
    )
    assert audit.classification == "env_nav_aborted"
    assert audit.navigation_stage == "action_click"


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


def test_reachability_audit_classifies_disallowed_domain_as_model_error():
    audit = audit_reachability_failure(
        url="https://finance.yahoo.com/quote/AAPL/",
        plugin_name=None,
        plugin=None,
        reason="Domain not allowed",
        allowed_domains={"stooq.com"},
        evidence={
            "interceptor": {
                "blocked_url": "https://finance.yahoo.com/quote/AAPL/",
                "blocked_domain": "finance.yahoo.com",
                "allowed_domains": ["stooq.com"],
                "blocked_resource_type": "document",
                "blocked_by": "interceptor",
            }
        },
    )
    assert audit.classification == "model_disallowed_domain"
    assert audit.layer == "model"
    assert audit.is_environment_failure is False
    assert audit.is_model_hallucination is True


def test_reachability_audit_classifies_taostats_list_action_timeout():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("Page.click: Timeout 5000ms exceeded"),
        evidence={
            "navigation_metadata": {
                "navigation_stage": "action_click",
                "raw_exception_type": "TimeoutError",
                "raw_exception_message": "Page.click: Timeout 5000ms exceeded",
                "evidence": {"selector": ".rt-th:nth-child(6)"},
            }
        },
    )
    assert audit.classification == "env_taostats_list_action_timeout"
    assert audit.is_environment_failure is True
    assert audit.evidence["page_kind"] == "taostats_list"
    assert audit.evidence["interaction_kind"] == "sort"
    assert audit.evidence["target_locator"] == ".rt-th:nth-child(6)"


def test_reachability_audit_classifies_taostats_invalid_selector_as_model_error():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("Locator failed"),
        evidence={
            "navigation_metadata": {
                "navigation_stage": "action_click",
                "raw_exception_type": "Error",
                "raw_exception_message": "button:contains('30D') is not a valid selector",
                "evidence": {"selector": "button:contains('30D')"},
            }
        },
    )
    assert audit.classification == "model_invalid_selector"
    assert audit.layer == "model"
    assert audit.is_environment_failure is False
    assert audit.is_model_hallucination is True
    assert audit.evidence["page_kind"] == "taostats_list"
    assert audit.evidence["selector_syntax_invalid"] is True


def test_reachability_audit_classifies_taostats_detail_prefetch_invalidated():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets/73",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("Target page, context or browser has been closed"),
        evidence={
            "taostats_prefetch": {
                "page_kind": "taostats_detail",
                "prefetch_phase": "setup_page_for_cache",
                "wait_target": "text=Statistics",
                "background_refresh": False,
            },
            "navigation_metadata": {
                "raw_exception_type": "TargetClosedError",
                "raw_exception_message": "Target page, context or browser has been closed",
            },
        },
    )
    assert audit.classification == "env_taostats_detail_prefetch_invalidated"
    assert audit.is_environment_failure is True
    assert audit.evidence["page_kind"] == "taostats_detail"
    assert audit.evidence["prefetch_phase"] == "setup_page_for_cache"
    assert audit.evidence["wait_target"] == "text=Statistics"


def test_reachability_audit_keeps_taostats_detail_soft_setup_out_of_invalidated():
    audit = audit_reachability_failure(
        url="https://taostats.io/subnets/73",
        plugin_name="taostats",
        plugin=TaostatsPlugin(),
        exception=RuntimeError("Page.click: Timeout 5000ms exceeded"),
        evidence={
            "taostats_prefetch": {
                "page_kind": "taostats_detail",
                "prefetch_phase": "setup_page_for_cache",
                "wait_target": "text=Price Impact",
                "page_body_ready": True,
                "detail_setup_soft_failed": True,
            },
            "navigation_metadata": {
                "raw_exception_type": "TimeoutError",
                "raw_exception_message": "Page.click: Timeout 5000ms exceeded",
            },
        },
    )
    assert audit.classification == "env_nav_timeout"
