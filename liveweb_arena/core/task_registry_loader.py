from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from .runtime_profiles import is_fast_collect_profile, normalize_runtime_profile

STRICT_TASK_REGISTRY_DIR_ENV = "LIVEWEB_STRICT_TASK_REGISTRY_DIR"


def _load_task_registry_module_from_dir(source_dir: Path):
    module_path = source_dir / "liveweb_arena" / "core" / "task_registry.py"
    spec = importlib.util.spec_from_file_location("liveweb_runtime_task_registry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load task registry from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_task_registry_source_dir(runtime_profile: str | None = None) -> Path | None:
    profile = normalize_runtime_profile(runtime_profile)
    if is_fast_collect_profile(profile):
        return None
    raw = os.getenv(STRICT_TASK_REGISTRY_DIR_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def parse_task_id(task_id: int, *, runtime_profile: str | None = None) -> dict[str, Any]:
    source_dir = resolve_task_registry_source_dir(runtime_profile)
    if source_dir is None:
        from .task_registry import parse_task_id as local_parse_task_id

        return local_parse_task_id(task_id)
    module = _load_task_registry_module_from_dir(source_dir)
    return module.parse_task_id(task_id)


def max_task_id(*, runtime_profile: str | None = None) -> int:
    source_dir = resolve_task_registry_source_dir(runtime_profile)
    if source_dir is None:
        from .task_registry import max_task_id as local_max_task_id

        return local_max_task_id()
    module = _load_task_registry_module_from_dir(source_dir)
    return module.max_task_id()
