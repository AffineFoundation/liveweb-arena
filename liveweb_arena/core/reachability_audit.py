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


def _matches_allowed_domain(domain: str, allowed_domain: str) -> bool:
    return domain == allowed_domain or domain.endswith("." + allowed_domain)


def _is_taostats_list_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower().rstrip("/")
    return "taostats.io" in host and path in {"", "/subnets"}


def _is_taostats_detail_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    return "taostats.io" in host and (path.startswith("/subnet/") or path.startswith("/subnets/"))


def _infer_taostats_interaction_kind(target_locator: str | None, raw_exception_message: str | None) -> str:
    text = " ".join(part for part in [target_locator or "", raw_exception_message or ""] if part).lower()
    if any(marker in text for marker in ("page-item", "next", "prev", "next page")):
        return "paginate"
    if any(marker in text for marker in ("rt-th", "dt-orderable", "1h", "24h", "1w", "1m", "sort")):
        return "sort"
    if any(marker in text for marker in ("all", "rows:", "dataTables_length", ".dataTables_length")):
        return "show_all"
    return "unknown"


def _is_invalid_selector_message(raw_exception_type: str | None, raw_exception_message: str | None) -> bool:
    text = " ".join(part for part in [raw_exception_type or "", raw_exception_message or ""] if part).lower()
    return any(
        marker in text
        for marker in (
            "not a valid selector",
            "unexpected token",
            "queryselectorall",
            "selector engine",
            "selector is malformed",
            "unknown engine",
        )
    )


def _is_missing_ui_target_message(raw_exception_type: str | None, raw_exception_message: str | None) -> bool:
    text = " ".join(part for part in [raw_exception_type or "", raw_exception_message or ""] if part).lower()
    return "no element found with role" in text or "no element found for selector" in text


def _build_disallowed_domain_audit(
    *,
    url: str,
    normalized: str,
    domain: str,
    plugin_name: str | None,
    reason: str | None,
    http_status: int | None,
    exception: BaseException | None,
    raw_exception_type: str | None,
    raw_exception_message: str | None,
    navigation_stage: str | None,
    resource_type: str | None,
    attempt_index: int | None,
    max_attempts: int | None,
    browser_reused: bool | None,
    context_reused: bool | None,
    page_recreated_before_retry: bool | None,
    evidence: dict[str, Any],
) -> "ReachabilityAuditResult":
    return ReachabilityAuditResult(
        status="unreachable",
        classification="model_disallowed_domain",
        layer="model",
        url=url,
        normalized_url=normalized,
        domain=domain,
        plugin_name=plugin_name,
        reason=reason or "Domain not allowed",
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
        is_environment_failure=False,
        is_model_hallucination=True,
        evidence=evidence,
    )


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
    raw_exception_lower = (raw_exception_message or "").lower()
    navigation_stage = navigation_metadata.get("navigation_stage")
    resource_type = navigation_metadata.get("resource_type")
    attempt_index = navigation_metadata.get("attempt_index")
    max_attempts = navigation_metadata.get("max_attempts")
    browser_reused = navigation_metadata.get("browser_reused")
    context_reused = navigation_metadata.get("context_reused")
    page_recreated_before_retry = navigation_metadata.get("page_recreated_before_retry")
    navigation_evidence = dict(navigation_metadata.get("evidence") or {})
    interceptor_metadata = dict(evidence.get("interceptor") or {})
    if not resource_type:
        resource_type = interceptor_metadata.get("blocked_resource_type") or resource_type

    if allowed_domains and domain:
        normalized_allowed = {(item or "").lower() for item in allowed_domains if item}
        if normalized_allowed and not any(_matches_allowed_domain(domain, allowed) for allowed in normalized_allowed):
            evidence.setdefault("interceptor", interceptor_metadata)
            evidence.setdefault("allowed_domains", sorted(normalized_allowed))
            return _build_disallowed_domain_audit(
                url=url,
                normalized=normalized,
                domain=domain,
                plugin_name=plugin_name,
                reason=reason,
                http_status=http_status,
                exception=exception,
                raw_exception_type=raw_exception_type,
                raw_exception_message=raw_exception_message,
                navigation_stage=navigation_stage,
                resource_type=resource_type,
                attempt_index=attempt_index,
                max_attempts=max_attempts,
                browser_reused=browser_reused,
                context_reused=context_reused,
                page_recreated_before_retry=page_recreated_before_retry,
                evidence=evidence,
            )

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

    combined_lower = " ".join(part for part in [exception_lower, raw_exception_lower] if part)

    if "taostats.io" in domain and _is_taostats_list_url(url):
        target_locator = (
            navigation_evidence.get("selector")
            or navigation_evidence.get("target_locator")
            or (
                f"role={navigation_evidence.get('role')} name={navigation_evidence.get('name')}"
                if navigation_evidence.get("role")
                else None
            )
        )
        interaction_kind = _infer_taostats_interaction_kind(target_locator, raw_exception_message)
        selector_syntax_invalid = _is_invalid_selector_message(raw_exception_type, raw_exception_message)
        missing_ui_target = _is_missing_ui_target_message(raw_exception_type, raw_exception_message)
        if selector_syntax_invalid:
            evidence.update(
                {
                    "page_kind": "taostats_list",
                    "interaction_kind": interaction_kind,
                    "target_locator": target_locator,
                    "selector_syntax_invalid": True,
                }
            )
            return ReachabilityAuditResult(
                status="unreachable",
                classification="model_invalid_selector",
                layer="model",
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
                is_environment_failure=False,
                is_model_hallucination=True,
                evidence=evidence,
            )
        if missing_ui_target:
            evidence.update(
                {
                    "page_kind": "taostats_list",
                    "interaction_kind": interaction_kind,
                    "target_locator": target_locator,
                    "selector_syntax_invalid": False,
                    "ui_target_missing": True,
                }
            )
            return ReachabilityAuditResult(
                status="unreachable",
                classification="model_invalid_ui_target",
                layer="model",
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
                is_environment_failure=False,
                is_model_hallucination=True,
                evidence=evidence,
            )
        if (
            (navigation_stage or "").startswith("action_")
            and (
                "timeout" in combined_lower
                or "too many consecutive action failures" in combined_lower
                or "no element found with role" in combined_lower
            )
        ):
            evidence.update(
                {
                    "page_kind": "taostats_list",
                    "interaction_kind": interaction_kind,
                    "target_locator": target_locator,
                    "selector_syntax_invalid": False,
                }
            )
            return ReachabilityAuditResult(
                status="unreachable",
                classification="env_taostats_list_action_timeout",
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

    taostats_prefetch = dict(evidence.get("taostats_prefetch") or {})
    if not taostats_prefetch:
        taostats_prefetch = {
            key: evidence.get(key)
            for key in ("page_kind", "prefetch_phase", "wait_target", "background_refresh")
            if evidence.get(key) is not None
        }
    if "taostats.io" in domain and _is_taostats_detail_url(url):
        prefetch_phase = taostats_prefetch.get("prefetch_phase")
        wait_target = taostats_prefetch.get("wait_target")
        background_refresh = bool(taostats_prefetch.get("background_refresh", False))
        page_kind = taostats_prefetch.get("page_kind")
        detail_setup_soft_failed = bool(taostats_prefetch.get("detail_setup_soft_failed", False))
        page_body_ready = taostats_prefetch.get("page_body_ready")
        if prefetch_phase or page_kind == "taostats_detail":
            if detail_setup_soft_failed and page_body_ready is True:
                evidence.update(
                    {
                        "page_kind": "taostats_detail",
                        "prefetch_phase": prefetch_phase or "setup_page_for_cache",
                        "wait_target": wait_target,
                        "background_refresh": background_refresh,
                        "page_body_ready": True,
                        "detail_setup_soft_failed": True,
                    }
                )
            else:
                evidence.update(
                    {
                        "page_kind": "taostats_detail",
                        "prefetch_phase": prefetch_phase or "goto",
                        "wait_target": wait_target,
                        "background_refresh": background_refresh,
                    }
                )
                return ReachabilityAuditResult(
                    status="unreachable",
                    classification="env_taostats_detail_prefetch_invalidated",
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

    probe = probe_site(url) if url else None
    if probe is not None:
        evidence.setdefault("site_probe", probe.to_dict())
        if http_status is None:
            http_status = probe.http_status

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
    elif "certificate_verify_failed" in combined_lower or "sslerror" in combined_lower:
        classification = "env_tls_error"
        layer = "tls"
    elif "err_aborted" in combined_lower or "frame was detached" in combined_lower:
        classification = "env_nav_aborted"
        layer = "browser"
    elif "target page, context or browser has been closed" in combined_lower or "targetclosederror" in combined_lower:
        classification = "env_target_closed"
        layer = "browser"
    elif "timeout" in combined_lower:
        classification = "env_nav_timeout"
        layer = "browser"
    elif "handler is closed" in combined_lower or "transport closed" in combined_lower or "connection closed" in combined_lower:
        classification = "env_browser_context_invalidated"
        layer = "browser"
    elif "status=429" in combined_lower:
        classification = "env_api_rate_limited"
        layer = "api"
    elif "empty response for coin_id" in combined_lower:
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
