# Input Schema

The CLI accepts a JSON object through `--input`.

## Required

- `market_data_csv_path`: CSV path with at least `date` and `symbol`; factor execution expects numeric OHLCV fields such as `open`, `high`, `low`, `close`, `vwap`, `volume`, and `amount`.

## Common Optional Fields

- `dry_run`: Boolean. When true, validate paths and output contracts without calling an LLM.
- `output_dir`: Optional output directory used when `--output` is not passed. Relative paths are resolved from the repository root.
- `insample_time_range`: Two-date list used for lightweight Pearson IC, RankIC, and ICIR-style evaluation.
- `outsample_time_range`: Two-date list recorded for later validation context.
- `test_debug_range`: Two-date list used by factor code debug checks; defaults to `insample_time_range`. Legacy `test_range` is still accepted for compatibility.
- `test_symbols`: Symbol list for factor debugging checks; defaults to all symbols in the CSV.
- `label_config`: Object with `price`, `span`, and `preprocess`; default is `{"price": "vwap", "span": 10, "preprocess": "cs_zscore"}`.
- `factor_requirement`: Plain-text requirements inserted into LLM prompts.
- `examples`: Inline seed factors as objects with `code` and `arguments`.
- `seed_factors_json_path`: JSON file containing seed factors.
- `llm_name`: Model name. Leave empty to use `OPENAI_MODEL` from the shell environment, or `gpt-4o-mini` if unset.
- `debate_jobs`: Number of debate jobs. Keep this small for first runs.
- `debate_max_rounds`: Maximum debate turns.
- `debate_max_tokens`: Maximum conversation token budget per debate pair.
- `agent_few_shots`: Number of seed factors sampled per agent.
- `seed_metric_threshold`: Early-stop threshold under the primary metric implied by `evaluation_mode`.
- `factor_metric_threshold`: Legacy fallback threshold used when `icir_threshold` or `rank_icir_threshold` is not set.
- `icir_threshold`: Pearson ICIR acceptance threshold for `pearson_ic` and `hybrid`.
- `rank_icir_threshold`: RankICIR acceptance threshold for `rank_ic` and `hybrid`.
- `factor_correlation_threshold`: Similarity threshold against accepted factors.
- `evaluation_mode`: One of `pearson_ic`, `rank_ic`, or `hybrid`. `pearson_ic` ranks and filters by Pearson `ICIR`; `rank_ic` ranks and filters by `RankICIR`; `hybrid` requires both thresholds to pass and requires the Pearson/RankIC in-sample direction signs to agree.
- `key_metric`: Legacy compatibility field. If `evaluation_mode` is absent, `ICIR` maps to `pearson_ic`, `RankICIR` maps to `rank_ic`, and `HybridICIR` maps to `hybrid`. New configs should prefer `evaluation_mode`.
- `use_factormad_alpha_library`: Whether to read and update a reusable local factor library. The public example keeps this false so dry-run does not mutate the sample library.
- `few_shot_from_library`: Whether evaluated library factors are added to the agent few-shot pool.
- `factormad_alpha_library_path`: Optional library JSON path. Prefer this over the legacy `alpha_library_path` alias.

## Environment

Real LLM runs require local credentials. Recommended:

```bash
cp .env.example .env
# edit .env, fill OPENAI_API_KEY or FACTORMAD_OPENAI_API_KEY
```

Shell environment variables are also supported:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_BASE_URL=https://api.openai.com/v1
```

FactorMAD-specific overrides take precedence:

```bash
export FACTORMAD_OPENAI_API_KEY=...
export FACTORMAD_OPENAI_MODEL=gpt-4o-mini
export FACTORMAD_OPENAI_BASE_URL=https://api.openai.com/v1
```

The runtime loads the repository-root `.env` with `override=False`, so already exported shell variables are not overwritten.

Use `dry_run=true` for installation and output-contract checks.
