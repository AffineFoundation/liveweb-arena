from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


SCRIPT_PATH = Path("/home/xmyf/liveweb-arena/scripts/batch_eval.py")
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_stub)
env_stub = types.ModuleType("env")
env_stub.Actor = object
sys.modules.setdefault("env", env_stub)
SPEC = importlib.util.spec_from_file_location("batch_eval_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_eval)


class _FakeBrowser:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeActor:
    def __init__(self):
        self.browser = _FakeBrowser()


def test_maybe_cleanup_actor_falls_back_to_browser_close():
    actor = _FakeActor()
    asyncio.run(batch_eval._maybe_cleanup_actor(actor))
    assert actor.browser.closed is True


def test_write_summary_includes_status_and_error_path(tmp_path: Path):
    result_path = tmp_path / "results.jsonl"
    task_ids_path = tmp_path / "task_ids.txt"
    summary_path = tmp_path / "summary.json"
    error_path = tmp_path / "error.txt"
    task_ids_path.write_text("1\n")
    error_path.write_text("traceback")

    args = argparse.Namespace(
        model="test-model",
        base_url=None,
        server_pool_file="/tmp/server_pool.json",
        max_concurrency=8,
    )

    summary = batch_eval._write_summary(
        summary_path,
        [{"score": 1.0, "success": True, "extra": {"task_id": 1}}],
        task_ids_path=task_ids_path,
        result_path=result_path,
        args=args,
        status="failed",
        error_path=error_path,
    )

    loaded = json.loads(summary_path.read_text())
    assert loaded["status"] == "failed"
    assert loaded["error_path"] == str(error_path)
    assert loaded["mean_score"] == 1.0
    assert loaded["success_rate"] == 1.0
    assert summary["status"] == "failed"


def test_load_task_ids_respects_min_unique_plugins(monkeypatch):
    args = argparse.Namespace(
        task_ids_file=None,
        num_prompts=5,
        seed=123,
        include_plugins="",
        exclude_plugins="openlibrary,weather",
        min_unique_plugins=2,
    )

    task_ids = batch_eval._load_task_ids(args)

    assert len(task_ids) == 5
    for task_id in task_ids:
        cfg = batch_eval.parse_task_id(task_id)
        assert len({plugin for plugin, _ in cfg["templates"]}) >= 2
