#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liveweb_arena.core.reachability_audit import ReachabilityAuditResult, classify_model_hallucination
from liveweb_arena.core.site_probe import probe_site
from liveweb_arena.plugins import get_all_plugins


SITE_UNREACHABLE_RE = re.compile(r"Required site unreachable:\s+(https?://\S+)")
DOMAIN_UNREACHABLE_RE = re.compile(r"Required domain unreachable:\s+(https?://\S+)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit unreachable URL causes from LiveWeb logs")
    parser.add_argument("--log-file", type=str, required=True, help="Driver log file to analyze")
    parser.add_argument(
        "--max-unique-urls",
        type=int,
        default=200,
        help="Maximum number of unique URLs to probe",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full JSON report",
    )
    parser.add_argument(
        "--show-top-urls",
        type=int,
        default=20,
        help="How many top URLs to show in the terminal summary",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds for each direct site probe",
    )
    return parser


def _extract_unreachable_urls(log_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = SITE_UNREACHABLE_RE.search(line) or DOMAIN_UNREACHABLE_RE.search(line)
            if match:
                counts[match.group(1)] += 1
    return counts


def _infer_plugin(url: str) -> tuple[str | None, Any | None]:
    host = (urlparse(url).hostname or "").lower()
    candidates: list[tuple[int, int, str, Any]] = []
    for name, plugin_cls in get_all_plugins().items():
        for allowed in getattr(plugin_cls, "allowed_domains", []):
            allowed = allowed.lower()
            if host == allowed or host.endswith("." + allowed) or allowed.endswith("." + host):
                # Prefer exact/specific domains first, and keep hybrid last.
                hybrid_penalty = 1 if name == "hybrid" else 0
                candidates.append((hybrid_penalty, -len(allowed), name, plugin_cls))
    if candidates:
        _penalty, _neg_len, name, plugin_cls = sorted(candidates)[0]
        try:
            return name, plugin_cls()
        except Exception:
            return name, None
    return None, None


def _classify_from_probe(
    url: str,
    plugin_name: str | None,
    plugin: Any | None,
    *,
    probe_timeout: float,
) -> ReachabilityAuditResult:
    model_class = None
    if plugin is not None and hasattr(plugin, "classify_url"):
        model_class = plugin.classify_url(url)
    model_class = model_class or classify_model_hallucination(url)
    if model_class is not None:
        is_env = model_class.startswith("env_") or model_class.startswith("ambiguous_")
        return ReachabilityAuditResult(
            status="unreachable",
            classification=model_class,
            layer="model" if not is_env else "tls",
            url=url,
            normalized_url=url,
            domain=(urlparse(url).hostname or "").lower(),
            plugin_name=plugin_name,
            reason="plugin/url-shape classification",
            is_environment_failure=is_env,
            is_model_hallucination=not is_env,
            evidence={},
        )

    probe = probe_site(url, timeout=probe_timeout)
    evidence = {"site_probe": asdict(probe)}
    domain = (urlparse(url).hostname or "").lower()

    if probe.ok and (probe.http_status or 0) < 400:
        classification = "env_browser_navigation_failure"
        layer = "browser"
        env_failure = True
        model_hallucination = False
    elif probe.exception_type == "SSLError":
        classification = "env_tls_error"
        layer = "tls"
        env_failure = True
        model_hallucination = False
    elif probe.exception_type in {"ConnectTimeout", "ReadTimeout", "Timeout"}:
        classification = "env_prefetch_timeout"
        layer = "browser"
        env_failure = True
        model_hallucination = False
    elif probe.exception_type in {"ConnectionError"}:
        classification = "env_dns_or_connect_error"
        layer = "browser"
        env_failure = True
        model_hallucination = False
    elif probe.http_status == 403 and ("coingecko" in domain or probe.cf_ray or (probe.server or "").lower() == "cloudflare"):
        classification = "env_cdn_blocked"
        layer = "cdn"
        env_failure = True
        model_hallucination = False
    elif probe.http_status is not None and 400 <= probe.http_status < 500:
        classification = "env_http_4xx"
        layer = "cdn"
        env_failure = True
        model_hallucination = False
    elif probe.http_status is not None and probe.http_status >= 500:
        classification = "env_http_5xx"
        layer = "cdn"
        env_failure = True
        model_hallucination = False
    else:
        classification = "ambiguous_navigation_failure"
        layer = "browser"
        env_failure = True
        model_hallucination = False

    return ReachabilityAuditResult(
        status="unreachable",
        classification=classification,
        layer=layer,
        url=url,
        normalized_url=url,
        domain=domain,
        plugin_name=plugin_name,
        reason=probe.reason,
        http_status=probe.http_status,
        exception_type=probe.exception_type,
        is_environment_failure=env_failure,
        is_model_hallucination=model_hallucination,
        evidence=evidence,
    )


def _weighted_ratio(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.most_common()}


def _print_summary(report: dict[str, Any], show_top_urls: int) -> None:
    print("\n**Reachability Audit Summary**")
    print(f"log_file: {report['log_file']}")
    print(f"total_unreachable_events: {report['total_unreachable_events']}")
    print(f"unique_unreachable_urls: {report['unique_unreachable_urls']}")
    print(f"environment_failure_ratio: {report['environment_failure_ratio']:.4f}")
    print(f"model_hallucination_ratio: {report['model_hallucination_ratio']:.4f}")

    print("\nTop classifications:")
    for classification, ratio in report["classification_ratios"].items():
        count = report["classification_counts"][classification]
        print(f"  {classification}: {count} ({ratio:.4f})")

    print("\nEnvironment-only details:")
    for classification, ratio in report["environment_detail_ratios"].items():
        count = report["environment_detail_counts"][classification]
        print(f"  {classification}: {count} ({ratio:.4f})")

    print("\nTop unreachable URLs:")
    for item in report["top_urls"][:show_top_urls]:
        print(
            "  "
            f"{item['count']:>3}x {item['url']} -> {item['classification']} "
            f"(plugin={item['plugin_name']}, layer={item['layer']})"
        )


def main() -> int:
    args = _build_parser().parse_args()
    log_path = Path(args.log_file)
    url_counts = _extract_unreachable_urls(log_path)

    most_common_urls = url_counts.most_common(args.max_unique_urls)
    total_events = sum(url_counts.values())

    classification_counts: Counter[str] = Counter()
    env_detail_counts: Counter[str] = Counter()
    plugin_counts: Counter[str] = Counter()
    env_failures = 0
    model_hallucinations = 0
    rows: list[dict[str, Any]] = []

    for url, count in most_common_urls:
        plugin_name, plugin = _infer_plugin(url)
        audit = _classify_from_probe(url, plugin_name, plugin, probe_timeout=args.probe_timeout)
        classification_counts[audit.classification] += count
        if audit.is_environment_failure:
            env_detail_counts[audit.classification] += count
            env_failures += count
        if audit.is_model_hallucination:
            model_hallucinations += count
        if plugin_name:
            plugin_counts[plugin_name] += count
        row = audit.to_dict()
        row["count"] = count
        rows.append(row)

    report = {
        "log_file": str(log_path),
        "total_unreachable_events": total_events,
        "unique_unreachable_urls": len(url_counts),
        "audited_unique_urls": len(rows),
        "environment_failure_ratio": (env_failures / total_events) if total_events else 0.0,
        "model_hallucination_ratio": (model_hallucinations / total_events) if total_events else 0.0,
        "classification_counts": dict(classification_counts),
        "classification_ratios": _weighted_ratio(classification_counts, total_events),
        "environment_detail_counts": dict(env_detail_counts),
        "environment_detail_ratios": _weighted_ratio(env_detail_counts, total_events),
        "plugin_counts": dict(plugin_counts),
        "top_urls": sorted(rows, key=lambda item: item["count"], reverse=True),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    _print_summary(report, args.show_top_urls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
