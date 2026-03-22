from __future__ import annotations

STRICT_EVAL_PROFILE = "strict_eval"
FAST_COLLECT_PROFILE = "fast_collect"


def normalize_runtime_profile(profile: str | None) -> str:
    value = (profile or "").strip().lower()
    if value in {"", "eval", STRICT_EVAL_PROFILE}:
        return STRICT_EVAL_PROFILE
    if value in {"collect", FAST_COLLECT_PROFILE}:
        return FAST_COLLECT_PROFILE
    raise ValueError(f"Unknown runtime profile: {profile}")


def runtime_profile_to_behavior_mode(profile: str | None) -> str:
    normalized = normalize_runtime_profile(profile)
    if normalized == FAST_COLLECT_PROFILE:
        return "collect"
    return "eval"


def is_fast_collect_profile(profile: str | None) -> bool:
    return normalize_runtime_profile(profile) == FAST_COLLECT_PROFILE
