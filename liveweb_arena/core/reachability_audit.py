from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from liveweb_arena.core.cache import normalize_url
from liveweb_arena.core.site_probe import probe_site


@dataclass
class ReachabilityAuditResult:
    status: str
    classification: str
    layer: str
    url: str
    normalized_url: str
    domain: str
    plugin_name: Optional[str] = None
    reason: Optional[str] = None
    http_status: Optional[int] = None
    exception_type: Optional[str] = None
    raw_exception_type: Optional[str] = None
    raw_exception_message: Optional[str] = None
    navigation_stage: Optional[str] = None
    resource_type: Optional[str] = None
    attempt_index: Optional[int] = None
    max_attempts: Optional[int] = None
    browser_reused: Optional[bool] = None
    context_reused: Optional[bool] = None
    page_recreated_before_retry: Optional[bool] = None
    is_environment_failure: bool = False
    is_model_hallucination: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def classify_stooq_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "stooq.com" not in host:
        return None
    path = parsed.path.lower()
    query = parsed.query.lower()

    if host.startswith("www."):
        return "env_tls_error"
    if "/q/conv/" in path or "/s/mst/" in path or "quote.php" in path:
        return "model_invalid_url_shape"
    if "q=" in query and "s=" not in query:
        return "model_invalid_url_shape"
    return None


def classify_coingecko_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "coingecko.com" not in host:
        return None
    path = parsed.path.lower()
    if "/coins/" in path:
        slug = path.split("/coins/", 1)[1].split("/", 1)[0]
        # Very loose first-pass hallucination heuristic for stock/company names in crypto namespace.
        if slug in {
            "microsoft",
            "google",
            "exxon-mobil",
            "jpmorgan-chase",
            "tesla",
            "walmart",
            "apple",
            "amazon",
        }:
            return "model_invalid_asset_id"
    return None


def classify_taostats_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "taostats.io" not in host:
        return None
    path = parsed.path.lower()
    if path in {"", "/", "/subnets"}:
        return None
    if path.startswith("/subnets/"):
        return None
    return "model_invalid_url_shape"


def classify_model_hallucination(url: str) -> str | None:
    return classify_stooq_url(url) or classify_coingecko_url(url) or classify_taostats_url(url)


def audit_reachability_failure(
    *,
    url: str,
    plugin_name: str | None,
    plugin: Any | None = None,
    exception: BaseException | None = None,
    reason: str | None = None,
    allowed_domains: set[str] | None = None,
    http_status: int | None = None,
    evidence: dict[str, Any] | None = None,
) -> ReachabilityAuditResult:
    normalized = normalize_url(url) if url else ""
    domain = _domain(url)
    evidence = dict(evidence or {})
    exception_text = f"{type(exception).__name__}: {exception}" if exception is not None else ""
    exception_lower = exception_text.lower()
    plugin_classification = plugin.classify_url(url) if plugin is not None and hasattr(plugin, "classify_url") else None
    hallucination_class = plugin_classification or classify_model_hallucination(url)

    navigation_metadata = evidence.get("navigation_metadata") or {}
    raw_exception_type = navigation_metadata.get("raw_exception_type") or (type(exception).__name__ if exception is not None else None)
    raw_exception_message = navigation_metadata.get("raw_exception_message") or (str(exception) if exception is not None else None)
    navigation_stage = navigation_metadata.get("navigation_stage")
    resource_type = navigation_metadata.get("resource_type")
    attempt_index = navigation_metadata.get("attempt_index")
    max_attempts = navigation_metadata.get("max_attempts")
    browser_reused = navigation_metadata.get("browser_reused")
    context_reused = navigation_metadata.get("context_reused")
    page_recreated_before_retry = navigation_metadata.get("page_recreated_before_retry")

    if hallucination_class is not None:
        is_env = hallucination_class.startswith("env_") or hallucination_class.startswith("ambiguous_")
        return ReachabilityAuditResult(
            status="unreachable",
            classification=hallucination_class,
            layer="model" if not is_env else "tls",
            url=url,
            normalized_url=normalized,
            domain=domain,
            plugin_name=plugin_name,
            reason=reason or exception_text,
            http_status=http_status,
            exception_type=type(exception).__name__ if exception is not None else None,
            raw_exception_type=raw_exception_type,
            raw_exception_message=raw_exception_message,
            navigation_stage=navigation_stage,
            resource_type=resource_type,
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            browser_reused=browser_reused,
            context_reused=context_reused,
            page_recreated_before_retry=page_recreated_before_retry,
            is_environment_failure=is_env,
            is_model_hallucination=not is_env,
            evidence=evidence,
        )

    probe = probe_site(url) if url else None
    if probe is not None:
        evidence.setdefault("site_probe", probe.to_dict())
        if http_status is None:
            http_status = probe.http_status

    if navigation_metadata.get("classification_hint") in {
        "env_nav_aborted",
        "env_target_closed",
        "env_nav_timeout",
        "env_browser_context_invalidated",
    }:
        return ReachabilityAuditResult(
            status="unreachable",
            classification=navigation_metadata["classification_hint"],
            layer="browser",
            url=url,
            normalized_url=normalized,
            domain=domain,
            plugin_name=plugin_name,
            reason=reason or exception_text,
            http_status=http_status,
            exception_type=type(exception).__name__ if exception is not None else None,
            raw_exception_type=raw_exception_type,
            raw_exception_message=raw_exception_message,
            navigation_stage=navigation_stage,
            resource_type=resource_type,
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            browser_reused=browser_reused,
            context_reused=context_reused,
            page_recreated_before_retry=page_recreated_before_retry,
            is_environment_failure=True,
            is_model_hallucination=False,
            evidence=evidence,
        )

    if http_status == 403 and "coingecko" in domain:
        return ReachabilityAuditResult(
            status="unreachable",
            classification="env_cdn_blocked",
            layer="cdn",
            url=url,
            normalized_url=normalized,
            domain=domain,
            plugin_name=plugin_name,
            reason=reason or exception_text,
            http_status=http_status,
            exception_type=type(exception).__name__ if exception is not None else None,
            raw_exception_type=raw_exception_type,
            raw_exception_message=raw_exception_message,
            navigation_stage=navigation_stage,
            resource_type=resource_type,
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            browser_reused=browser_reused,
            context_reused=context_reused,
            page_recreated_before_retry=page_recreated_before_retry,
            is_environment_failure=True,
            is_model_hallucination=False,
            evidence=evidence,
        )

    if probe and probe.exception_type == "SSLError":
        classification = "env_tls_error"
        layer = "tls"
    elif "certificate_verify_failed" in exception_lower or "sslerror" in exception_lower:
        classification = "env_tls_error"
        layer = "tls"
    elif "err_aborted" in exception_lower or "frame was detached" in exception_lower:
        classification = "env_nav_aborted"
        layer = "browser"
    elif "target page, context or browser has been closed" in exception_lower or "targetclosederror" in exception_lower:
        classification = "env_target_closed"
        layer = "browser"
    elif "status=429" in exception_lower:
        classification = "env_api_rate_limited"
        layer = "api"
    elif "empty response for coin_id" in exception_lower:
        classification = "env_api_empty"
        layer = "api"
    elif http_status is not None and 400 <= http_status < 500:
        classification = "env_http_4xx"
        layer = "cdn"
    elif http_status is not None and http_status >= 500:
        classification = "env_http_5xx"
        layer = "cdn"
    else:
        classification = "ambiguous_navigation_failure"
        layer = "browser"

    return ReachabilityAuditResult(
        status="unreachable",
        classification=classification,
        layer=layer,
        url=url,
        normalized_url=normalized,
        domain=domain,
        plugin_name=plugin_name,
        reason=reason or exception_text,
        http_status=http_status,
        exception_type=type(exception).__name__ if exception is not None else None,
        raw_exception_type=raw_exception_type,
        raw_exception_message=raw_exception_message,
        navigation_stage=navigation_stage,
        resource_type=resource_type,
        attempt_index=attempt_index,
        max_attempts=max_attempts,
        browser_reused=browser_reused,
        context_reused=context_reused,
        page_recreated_before_retry=page_recreated_before_retry,
        is_environment_failure=classification.startswith("env_") or classification.startswith("ambiguous_"),
        is_model_hallucination=False,
        evidence=evidence,
    )
