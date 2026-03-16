#!/usr/bin/env python3
"""High-throughput batch evaluation for LiveWeb Arena."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import traceback
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import Actor
from liveweb_arena.core.task_registry import TaskRegistry, parse_task_id
from liveweb_arena.utils.llm_client import MultiServerLLMRouter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch evaluation for LiveWeb Arena")
    parser.add_argument("--model", type=str, required=True, help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--server-pool-file",
        type=str,
        default=None,
        help="Optional server_pool.json for multi-server routing; overrides --base-url for request routing",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key (default: API_KEY env var)")
    parser.add_argument("--num-prompts", type=int, default=200, help="Number of prompts/tasks to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to sample deterministic task ids")
    parser.add_argument("--task-ids-file", type=str, default=None, help="Optional newline-delimited task id file")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "eval"), help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="batch_eval", help="Output file prefix")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Max concurrent evaluations")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-task timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-steps", type=int, default=None, help="Per-task max browser steps")
    parser.add_argument("--registry-version", type=str, default=None, help="Override TASK_REGISTRY_VERSION for sampling")
    parser.add_argument(
        "--exclude-plugins",
        type=str,
        default="",
        help="Comma separated plugin names to exclude from sampled tasks",
    )
    parser.add_argument(
        "--include-plugins",
        type=str,
        default="",
        help="Comma separated plugin names to keep; empty means all",
    )
    parser.add_argument(
        "--min-unique-plugins",
        type=int,
        default=1,
        help="Minimum number of unique plugins a sampled task must contain",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-task progress")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Write partial results/summary every N completed tasks",
    )
    return parser


def _load_task_ids(args: argparse.Namespace) -> list[int]:
    if args.task_ids_file:
        task_ids = [
            int(line.strip())
            for line in Path(args.task_ids_file).read_text().splitlines()
            if line.strip()
        ]
        return task_ids[: args.num_prompts]

    include_plugins = {p.strip() for p in args.include_plugins.split(",") if p.strip()}
    exclude_plugins = {p.strip() for p in args.exclude_plugins.split(",") if p.strip()}
    rng = random.Random(args.seed)

    candidate_combo_indices: list[int] = []
    for combo_index, template_ids in enumerate(TaskRegistry._combinations):
        plugins = {TaskRegistry.TEMPLATES[tid][0] for tid in template_ids}
        if len(plugins) < args.min_unique_plugins:
            continue
        if include_plugins and not (plugins & include_plugins):
            continue
        if exclude_plugins and (plugins & exclude_plugins):
            continue
        candidate_combo_indices.append(combo_index)

    rng.shuffle(candidate_combo_indices)
    if not candidate_combo_indices:
        return []

    selected: list[int] = []
    used_task_ids: set[int] = set()
    round_index = 0
    while len(selected) < args.num_prompts:
        added_in_round = 0
        for combo_index in candidate_combo_indices:
            variation_seed = (
                args.seed + combo_index + round_index * 9973
            ) % TaskRegistry.TASK_IDS_PER_COMBO
            task_id = combo_index * TaskRegistry.TASK_IDS_PER_COMBO + variation_seed + 1
            if task_id in used_task_ids:
                continue
            selected.append(task_id)
            used_task_ids.add(task_id)
            added_in_round += 1
            if len(selected) >= args.num_prompts:
                break
        if added_in_round == 0:
            break
        round_index += 1
    return selected


async def _run_one(
    actor: Actor,
    task_id: int,
    args: argparse.Namespace,
    max_concurrency: int,
) -> dict[str, Any]:
    result = await actor.evaluate(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        task_id=task_id,
        timeout=args.timeout,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_concurrency=max_concurrency,
        mode="eval",
        route_key=f"batch-eval:{args.seed}:task:{task_id}",
    )
    result.setdefault("extra", {})
    result["extra"]["task_id"] = task_id
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r.get("score", 0.0)) for r in results]
    successes = [bool(r.get("success", False)) for r in results]
    errored = [r for r in results if r.get("error")]
    by_plugin_scores: dict[str, list[float]] = defaultdict(list)
    by_plugin_success: dict[str, list[int]] = defaultdict(list)
    by_plugin_errors: Counter[str] = Counter()

    for result in results:
        cfg = parse_task_id(int(result.get("extra", {}).get("task_id")))
        plugins = sorted({plugin for plugin, _name in cfg["templates"]})
        plugin_key = "+".join(plugins)
        by_plugin_scores[plugin_key].append(float(result.get("score", 0.0)))
        by_plugin_success[plugin_key].append(1 if result.get("success", False) else 0)
        if result.get("error"):
            by_plugin_errors[plugin_key] += 1

    return {
        "num_tasks": len(results),
        "mean_score": mean(scores) if scores else 0.0,
        "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
        "error_rate": (len(errored) / len(results)) if results else 0.0,
        "num_errors": len(errored),
        "by_plugin": {
            plugin: {
                "count": len(vals),
                "mean_score": mean(vals) if vals else 0.0,
                "success_rate": (sum(by_plugin_success[plugin]) / len(by_plugin_success[plugin]))
                if by_plugin_success[plugin]
                else 0.0,
                "num_errors": by_plugin_errors[plugin],
            }
            for plugin, vals in sorted(by_plugin_scores.items())
        },
    }


def _write_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _build_summary(
    results: list[dict[str, Any]],
    *,
    task_ids_path: Path,
    result_path: Path,
    args: argparse.Namespace,
    status: str,
    error_path: Path | None = None,
) -> dict[str, Any]:
    summary = _summarize(results)
    summary["status"] = status
    summary["task_ids_path"] = str(task_ids_path)
    summary["results_path"] = str(result_path)
    summary["model"] = args.model
    summary["base_url"] = args.base_url
    summary["server_pool_file"] = args.server_pool_file
    summary["max_concurrency"] = args.max_concurrency
    summary["registry_version"] = os.environ.get("TASK_REGISTRY_VERSION")
    if error_path is not None:
        summary["error_path"] = str(error_path)
    return summary


def _write_summary(
    path: Path,
    results: list[dict[str, Any]],
    *,
    task_ids_path: Path,
    result_path: Path,
    args: argparse.Namespace,
    status: str,
    error_path: Path | None = None,
) -> dict[str, Any]:
    summary = _build_summary(
        results,
        task_ids_path=task_ids_path,
        result_path=result_path,
        args=args,
        status=status,
        error_path=error_path,
    )
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


async def _maybe_cleanup_actor(actor: Actor) -> None:
    cleanup = getattr(actor, "cleanup", None)
    if cleanup is not None:
        maybe_awaitable = cleanup()
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable
        return

    browser = getattr(actor, "browser", None)
    if browser is not None and hasattr(browser, "close"):
        maybe_awaitable = browser.close()
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable


async def main() -> int:
    args = _build_parser().parse_args()

    if args.registry_version:
        os.environ["TASK_REGISTRY_VERSION"] = args.registry_version

    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("API key required via --api-key or API_KEY/CHUTES_API_KEY", file=sys.stderr)
        return 1
    args.api_key = api_key
    if not args.base_url and not args.server_pool_file:
        print("Either --base-url or --server-pool-file is required.", file=sys.stderr)
        return 1

    task_ids = _load_task_ids(args)
    if not task_ids:
        print("No task ids selected for evaluation.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    result_path = output_dir / f"{args.output_prefix}_{timestamp}.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_{timestamp}.summary.json"
    task_ids_path = output_dir / f"{args.output_prefix}_{timestamp}.task_ids.txt"
    task_ids_path.write_text("\n".join(str(tid) for tid in task_ids) + "\n")

    print(f"Evaluating {len(task_ids)} tasks with concurrency={args.max_concurrency}")
    print(f"Task IDs saved to: {task_ids_path}")

    llm_router = None
    if args.server_pool_file:
        llm_router = MultiServerLLMRouter.from_server_pool_file(
            args.server_pool_file,
            default_api_key=api_key,
            max_inflight_requests=args.max_concurrency,
        )
    actor = Actor(api_key=api_key, use_cache=True, llm_router=llm_router)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    results: list[dict[str, Any]] = []
    uncaught_error: Exception | None = None
    error_path = output_dir / f"{args.output_prefix}_{timestamp}.error.txt"

    async def guarded(task_id: int) -> dict[str, Any]:
        async with semaphore:
            result = await _run_one(actor, task_id, args, args.max_concurrency)
            if args.verbose:
                print(
                    f"task_id={task_id} score={result.get('score', 0.0):.3f} "
                    f"success={result.get('success', False)} error={bool(result.get('error'))}"
                )
            return result

    try:
        for coro in asyncio.as_completed([guarded(task_id) for task_id in task_ids]):
            results.append(await coro)
            if len(results) % args.flush_every == 0 or len(results) == len(task_ids):
                _write_results_jsonl(result_path, results)
                _write_summary(
                    summary_path,
                    results,
                    task_ids_path=task_ids_path,
                    result_path=result_path,
                    args=args,
                    status="running" if len(results) < len(task_ids) else "completed",
                )
                print(f"Completed {len(results)}/{len(task_ids)} tasks")
    except Exception as exc:
        uncaught_error = exc
        error_path.write_text(traceback.format_exc())
    finally:
        try:
            await _maybe_cleanup_actor(actor)
        except Exception:
            cleanup_trace = traceback.format_exc()
            if error_path.exists():
                error_path.write_text(error_path.read_text() + "\n\nCleanup error:\n" + cleanup_trace)
            else:
                error_path.write_text("Cleanup error:\n" + cleanup_trace)

    _write_results_jsonl(result_path, results)
    final_status = "failed" if uncaught_error is not None else "completed"
    summary = _write_summary(
        summary_path,
        results,
        task_ids_path=task_ids_path,
        result_path=result_path,
        args=args,
        status=final_status,
        error_path=error_path if error_path.exists() else None,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results saved to: {result_path}")
    print(f"Summary saved to: {summary_path}")
    if uncaught_error is not None:
        print(f"Error details saved to: {error_path}", file=sys.stderr)
        raise uncaught_error
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
