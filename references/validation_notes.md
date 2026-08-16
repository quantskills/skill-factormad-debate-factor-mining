# Validation Notes

This skill is a research tool for code-based alpha factor mining.

## Data Assumptions

- Input data is user-provided or generated from a data source the user has rights to use.
- Daily stock data is expected.
- Rows must contain `date` and `symbol`; OHLCV columns are expected for real factor evaluation.
- Fundamental fields must come from a point-in-time panel whose manifest asserts no future backfill. Quarterly observations are activated and then carried forward as daily states.
- `vwap` is used as the default label price when available.

## Internal Checks

Generated factor code is checked for:

- Runnable Python function definition.
- No imports inside generated factor code.
- Output as a pandas Series aligned to the market-data index.
- Non-all-NaN output.
- Minimum candidate non-null coverage.
- Static allowed-field and `factor_mode` checks.
- Static rejection of negative shift/diff/pct_change, backward fill, and centered rolling.
- Basic look-ahead-bias signal via shortened-date rerun.
- Rough scale comparability across symbols.
- Symbol-order stability.

## Limitations

- LLM-generated code can still contain subtle logic errors.
- Static analysis cannot prove the absence of all indirect look-ahead paths; independent point-in-time review remains required.
- Coverage filtering protects availability, not economic validity. Strict TTM fields can be excluded when historical quarterly chains are incomplete; `_ytd_ann` must not be interpreted as a replacement for TTM.
- Lightweight Pearson IC, RankIC, and ICIR-style metrics can overfit and is not a production validation.
- The included toy data and dry-run mode validate mechanics, not factor quality.
- Final evaluation should include out-of-sample testing, transaction costs, turnover, risk exposure, and portfolio-level backtests.

## Risk Boundary

This repository is for research and educational workflows. It does not provide investment advice, trading recommendations, guaranteed returns, or a production-ready strategy.
