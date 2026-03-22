# Downstream Alignment Rules

This document records how downstream `liveweb-arena` is expected to integrate
with downstream benchmark / collection code.

## Runtime Profiles

`liveweb-arena` exposes runtime behavior through two profiles:

- `strict_eval`
- `fast_collect`

The runtime profile controls episode execution only.
It does not define the final benchmark score by itself.

## Integration Contract

Downstream benchmark code is responsible for choosing:

- the runtime profile used to execute an episode
- the judge profile used to score the episode

The intended combinations are:

- strict evaluation:
  - run with `strict_eval`
  - judge with `strict_eval`
- accelerated collection:
  - run with `fast_collect`
  - judge with `strict_eval`

## What Must Stay Out Of Strict Eval

The strict path should remain compatible with upstream semantics.

Do not enable the following by default in `strict_eval`:

- collect-only local recovery
- disallowed-domain correction
- invalid-generated-url correction
- taostats list recovery
- downstream-only prompt profiles
- extra loopguard that changes single-episode semantics

## What Is Allowed In Fast Collect

`fast_collect` may enable runtime-only acceleration such as:

- local recovery
- fail-fast / early-stop
- loop detection
- invalid URL correction
- collection-specific routing and timeout policy

These features are allowed because they help find candidate trajectories faster.
They are not allowed to replace strict judging.

## Source Of Truth

The downstream benchmark repo contains the normative policy for:

- strict evaluation
- collection
- SFT filtering
- RL reward / filtering

See:

- [sampling-eval-rl-policy.md](/home/xmyf/liveweb-capability-bench/docs/sampling-eval-rl-policy.md)

## Current Online-Aligned Reference

The current online-aligned LIVEWEB definition comes from:

- `affine-cortex` `system_config.json`
- `affine-cortex` environment config
- upstream `liveweb-arena` task registry

Downstream runtime behavior should be compatible with that contract when running
in `strict_eval`.
