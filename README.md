# LiveWeb Arena

Real-time web evaluation framework for LLM browser agents.

## Quick Start

```bash
# Install
pip install -e .
playwright install chromium

# Configure
cp .env.example .env
# Edit .env with your API_KEY

# Run evaluation
python eval.py --seed 42 --verbose
```

## Runtime Profiles

Downstream integrations may now choose between two explicit episode runtime
profiles:

- `strict_eval`
- `fast_collect`

`strict_eval` is intended for upstream-compatible capability measurement.
`fast_collect` is intended for accelerated candidate-trajectory generation. The
runtime profile only changes episode execution behavior; final benchmark scoring
should still come from a strict judge.

See:

- [docs/runtime-profiles.md](/home/xmyf/liveweb-arena/docs/runtime-profiles.md)
- [docs/downstream-alignment.md](/home/xmyf/liveweb-arena/docs/downstream-alignment.md)

## Usage

```bash
# Basic evaluation
python eval.py --seed 42

# Specific template
python eval.py --templates weather/current_weather --seed 42

# Multi-task evaluation
python eval.py --num-tasks 3 --seed 42

# Deterministic task by ID
python eval.py --task-id 100001 --seed 42

# View all templates
python eval.py --show-registry
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--seed` | Random seed | random |
| `--task-id` | Deterministic task ID | - |
| `--num-tasks` | Sub-tasks (1-4) | 1 |
| `--templates` | Template(s) to use | random |
| `--model` | LLM model | `zai-org/GLM-4.7-TEE` |
| `--base-url` | API URL | `https://llm.chutes.ai/v1` |
| `--timeout` | Timeout (seconds) | 3600 |
| `--verbose` | Verbose output | false |

## Templates

### Weather (wttr.in) - 6 templates
`location_name`, `time_of_day`, `multi_day`, `current_weather`, `astronomy`, `weather_comparison`

### Stooq (stooq.com) - 7 templates
`stooq_price`, `stooq_comparison`, `stooq_ranking`, `stooq_sector_analysis`, `stooq_currency`, `stooq_volatility`, `stooq_range_position`

### CoinGecko (coingecko.com) - 8 templates
`coingecko_price`, `coingecko_volume`, `coingecko_comparison`, `coingecko_rank`, `coingecko_top_movers`, `coingecko_supply`, `coingecko_ath`, `coingecko_performance`

### Taostats (taostats.io) - 10 templates
`taostats_subnet_info`, `taostats_comparison`, `taostats_analysis`, `taostats_ranking`, `taostats_price_change`, `taostats_threshold`, `taostats_multi_condition`, `taostats_delta`, `taostats_range_count`, `taostats_percentage`

### Hybrid (cross-source) - 3 templates
`hybrid_top_performer`, `hybrid_ranking`, `hybrid_conditional_branch`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_KEY` | LLM API key (required) |
| `COINGECKO_API_KEY` | CoinGecko Pro API key (optional) |
| `TAOSTATS_API_KEY` | Taostats API key (optional) |

## Output

Results saved to `eval/<timestamp>.json`:

```json
{
  "score": 1.0,
  "success": true,
  "extra": {
    "seed": 42,
    "answer_details": [...],
    "conversation": [...]
  }
}
```

## License

MIT
