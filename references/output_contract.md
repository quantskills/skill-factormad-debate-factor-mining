# Output Contract

Run:

```bash
python scripts/factormad_debate_factor.py --input examples/debate_input.json
```

The public example writes to `output_dir` from `examples/debate_input.json` (`outputs/debate_example`). Pass `--output <dir>` to override it. If neither `--output` nor input JSON `output_dir` is set, the script creates `outputs/debateYYYYMMDDHHMMSS`. The output directory contains:

- `factormad_debate_result.json`: Main result payload.
- `debate_rounds.json`: Per-round debate and validation records.
- `accepted_factors.json`: Accepted factor records.

The CLI is responsible for writing the stable split artifacts. The debate runtime returns in-memory `debate_rounds` and `accepted_factors` to avoid duplicate `factormad_*` split files.

The main result includes:

- `ok`: Boolean success flag.
- `dry_run`: Present and true for dry-run validation.
- `debate_hash`: Debate identifier.
- `job_count`: Number of debate jobs.
- `round_count`: Number of debate rounds.
- `accepted_count`: Number of accepted factors.
- `invalid_count`: Number of rejected or invalid factors.
- `llm_fee`: Estimated LLM fee for the run.
- `best_factor`: Best candidate under the configured lightweight metric.
- `generated_factors`: Candidate factors generated during debate.
- `accepted_factors`: Accepted factors after metric and similarity checks.
- `invalid_factors`: Rejected factors or near-duplicates.
- `debate_rounds`: Per-round records with agent view text, candidate factor, status, and error, also written to `debate_rounds.json`.
- `metric_time_ranges`: Run-level metric labels: `IC` and `ICIR` use `insample_time_range`; `O-IC` and `O-ICIR` use `outsample_time_range`.
- `debate_json_path`, `debate_rounds_path`, `accepted_factors_path`: Artifact paths.

Every factor record in `best_factor`, `generated_factors`, `accepted_factors`, and `invalid_factors` uses the same canonical field names:

- `code`: Python function code for the factor.
- `arguments`: Arguments for the factor function.
- `metric`: Numeric values for `IC`, `ICIR`, `O-IC`, and `O-ICIR`.
- `insample_time_range` and `outsample_time_range`: The two evaluation windows retained on each factor record. Per-metric time-range labels are kept only at the run level.

Legacy inputs using `factor_code`, `factor_arguments`, `best_factor_code`, `best_factor_arguments`, `best_metric`, or `round_records` are still accepted for compatibility, but new outputs use the canonical fields above.

The lightweight metrics are for internal factor selection only. They are not final strategy backtest results.
