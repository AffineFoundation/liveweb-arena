# Runtime Profiles In LiveWeb Arena

Downstream `liveweb-arena` now supports two explicit runtime profiles:

- `strict_eval`
- `fast_collect`

These profiles control episode execution behavior. They are intentionally
separate from final benchmark judging.

## `strict_eval`

Use for any evaluation path that needs to remain as close as possible to the
upstream `AffineFoundation/liveweb-arena` semantics.

The strict path should avoid downstream-only acceleration features such as:

- disallowed-domain recovery
- invalid-generated-url recovery
- taostats list recovery
- collect-only prompt profiles
- aggressive loop guards

## `fast_collect`

Use for sampling, SFT data generation, RL feedback collection, and diagnostics.

The collect path may enable:

- local recoveries
- early-stop / fail-fast
- loop detection
- invalid URL correction
- model-specific runtime prompt profiles

These features are allowed to improve runtime efficiency, but they are not the
canonical scoring semantics.

## Integration Rule

`liveweb-arena` exposes runtime behavior. Downstream benchmark code is
responsible for deciding:

- which runtime profile is used to execute an episode
- which judge profile is used to score the final result

For downstream collection, the intended pattern is:

- run with `fast_collect`
- judge with `strict_eval`

For evaluation and upstream parity checks, use:

- run with `strict_eval`
- judge with `strict_eval`

## Code Entry Points

- runtime profile definitions:
  - [liveweb_arena/core/runtime_profiles.py](/home/xmyf/liveweb-arena/liveweb_arena/core/runtime_profiles.py)
- actor runtime selection:
  - [env.py](/home/xmyf/liveweb-arena/env.py)
- agent loop runtime gates:
  - [liveweb_arena/core/agent_loop.py](/home/xmyf/liveweb-arena/liveweb_arena/core/agent_loop.py)
