from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import requests
import time


@dataclass
class SiteProbeResult:
    ok: bool
    url: str
    final_url: str | None = None
    http_status: int | None = None
    exception_type: str | None = None
    reason: str | None = None
    server: str | None = None
    cf_ray: str | None = None
    location: str | None = None
    body_length: int | None = None
    elapsed_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_site(url: str, timeout: float = 10.0) -> SiteProbeResult:
    start = time.time()
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        return SiteProbeResult(
            ok=response.ok,
            url=url,
            final_url=response.url,
            http_status=response.status_code,
            server=response.headers.get("server"),
            cf_ray=response.headers.get("cf-ray"),
            location=response.headers.get("location"),
            body_length=len(response.text or ""),
            elapsed_ms=int((time.time() - start) * 1000),
            reason=f"http_{response.status_code}" if not response.ok else None,
        )
    except Exception as exc:
        return SiteProbeResult(
            ok=False,
            url=url,
            exception_type=type(exc).__name__,
            reason=str(exc),
            elapsed_ms=int((time.time() - start) * 1000),
        )
