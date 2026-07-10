# Validation Notes

This skill is a research tool for code-based alpha factor mining.

## Data Assumptions

- Input data is user-provided or generated from a data source the user has rights to use.
- Daily stock data is expected.
- Rows must contain `date` and `symbol`; OHLCV columns are expected for real factor evaluation.
- `vwap` is used as the default label price when available.

## Internal Checks

Generated factor code is checked for:

- Runnable Python function definition.
- No imports inside generated factor code.
- Output as a pandas Series aligned to the market-data index.
- Non-all-NaN output.
- Basic look-ahead-bias signal via shortened-date rerun.
- Rough scale comparability across symbols.
- Symbol-order stability.

## Limitations

- LLM-generated code can still contain subtle logic errors.
- Lightweight Pearson IC, RankIC, and ICIR-style metrics can overfit and is not a production validation.
- The included toy data and dry-run mode validate mechanics, not factor quality.
- Final evaluation should include out-of-sample testing, transaction costs, turnover, risk exposure, and portfolio-level backtests.

## Risk Boundary

This repository is for research and educational workflows. It does not provide investment advice, trading recommendations, guaranteed returns, or a production-ready strategy.
