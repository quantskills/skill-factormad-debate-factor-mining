# Input Schema

The CLI accepts a JSON object through `--input`.

## Required

- `market_data_path`: CSV file, one Parquet file, or Hive-partitioned Parquet directory. Data must contain `date`, `symbol`, `open`, `high`, `low`, `close`, and `volume`. The legacy `market_data_csv_path` remains supported for CSV-only configurations.
- `market_data_manifest_path`: Required for fundamental or hybrid mining. The manifest supplies field groups, coverage artifacts, and the point-in-time contract. Fundamental runs reject a manifest that does not assert `future_backfill=false`.

## Common Optional Fields

- `dry_run`: Boolean. When true, validate paths and output contracts without calling an LLM.
- `output_dir`: Optional output directory used when `--output` is not passed. Relative paths are resolved from the repository root.
- `insample_time_range`: Two-date list used for lightweight Pearson IC, RankIC, and ICIR-style evaluation.
- `outsample_time_range`: Two-date list recorded for later validation context.
- `test_debug_range`: Two-date list used by factor code debug checks; defaults to `insample_time_range`. Legacy `test_range` is still accepted for compatibility.
- `test_symbols`: Symbol list for factor debugging checks; defaults to all symbols in the CSV.
- `market_data_load_range`: Optional two-date list used to project the loaded panel. Include enough pre-history for the longest rolling window and enough trailing data for the forward-return label.
- `label_config`: Object with `price`, `span`, and `preprocess`; default is `{"price": "vwap", "span": 10, "preprocess": "cs_zscore"}`.
- `factor_requirement`: Plain-text requirements inserted into LLM prompts.
- `data_profile`: Optional lineage name. Defaults to `ohlcv_v1` or `ohlcv_fundamental_pit_v1`.
- `factor_mode`: One of `ohlcv`, `fundamental`, `hybrid`, or `any`. `hybrid` requires each generated candidate to reference at least one market field and one fundamental field.
- `feature_groups`: Manifest field groups exposed to the LLM. If omitted with a manifest, all declared research groups are considered.
- `feature_columns`: Additional explicit fields to expose after schema and coverage validation.
- `min_feature_coverage`: Minimum full-period non-null rate for a fundamental field.
- `min_yearly_coverage`: Minimum yearly non-null rate over `feature_coverage_time_range`.
- `feature_coverage_time_range`: Optional two-date field-availability screening range. It defaults to `insample_time_range`; do not use the sealed out-of-sample period for discovery-time filtering.
- `min_factor_nonnull_rate`: Minimum non-null rate required from a candidate on the debug slice; default `0.01`.
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
- `use_factormad_alpha_library`: Whether to update the writable factor library after the run. The public example keeps this false so dry-run does not mutate the sample library.
- `few_shot_from_library`: Whether evaluated library factors are added to the agent few-shot pool.
- `few_shot_library_paths`: Optional list of read-only factor-library JSON paths. Use this to seed a hybrid run from one or more existing libraries without writing back to them.
- `factormad_alpha_library_path`: Writable library JSON path for newly accepted factors. Prefer this over the legacy `alpha_library_path` alias.

## Fundamental Field Semantics

- Fundamental columns are point-in-time states activated on the manifest-defined trading date and carried forward within each symbol. Repeated daily values do not mean the company reported every day.
- `_ytd` means fiscal-year cumulative, `_ytd_ann` means simple annualization and is not true TTM, `_ttm` means strict trailing twelve months, `_sq` means single quarter, `_yoy` means year-over-year, and `_latest` means the latest disclosed balance.
- Raw availability/activation dates and report metadata are excluded from generated factor inputs. Candidate code may not use negative `shift`, `diff`, or `pct_change`, backward fill, or centered rolling.

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
