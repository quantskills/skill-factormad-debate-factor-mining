---
name: factormad-debate-factor-mining
description: "Run a self-contained FactorMAD-style LLM multi-agent debate workflow for mining interpretable code-based stock alpha factors from OHLCV market data. Use when an agent needs to generate, debate, validate, score, and export candidate alpha factor code for quantitative research."
quantSkills:
  organization: https://github.com/quantskills
  repository: quantskills/skill-factormad-debate-factor-mining
  repository_url: https://github.com/quantskills/skill-factormad-debate-factor-mining
  project_type: skill
  collection: factormad
  license: GPL-3.0-only
  category: tooling
  tags: [factormad, alpha-factor-mining, multi-agent, debate, factor-debug]
  platforms: [codex]
  language: zh-en
  status: active
  validation_level: runnable
  maintainer_type: community
  requires: []
  summary_zh: 使用 FactorMAD 风格的 LLM 多智能体辩论流程从 OHLCV 行情数据中挖掘代码型股票 Alpha 因子。
  summary_en: Mine interpretable code-based stock alpha factors from OHLCV market data with a FactorMAD-style LLM debate workflow.
---

# FactorMAD Debate Factor Mining

Use this skill to run a FactorMAD-style multi-agent debate workflow that proposes, critiques, validates, scores, and exports Python code-based stock alpha factor candidates.

## Core Workflow

1. Read `references/input_schema.md` before preparing the input JSON.
2. Use `dry_run=true` first to validate installation, paths, and output artifacts without an LLM call.
3. For a real run, configure `.env` from `.env.example` or set `OPENAI_API_KEY` / `FACTORMAD_OPENAI_API_KEY`, provide OHLCV market data, and run:

```bash
python scripts/factormad_debate_factor.py --input examples/debate_input.json
```

4. Read `references/output_contract.md` before consuming generated artifacts.
5. Treat lightweight IC/ICIR as internal selection evidence only; run independent out-of-sample and portfolio validation before using any factor downstream.

## Output Contract

Produce:

- `outputs/debate_example/factormad_debate_result.json` or `outputs/debateYYYYMMDDHHMMSS/factormad_debate_result.json`
- `outputs/debate_example/debate_rounds.json` or `outputs/debateYYYYMMDDHHMMSS/debate_rounds.json`
- `outputs/debate_example/accepted_factors.json` or `outputs/debateYYYYMMDDHHMMSS/accepted_factors.json`

## References

- Use `references/source_boundary.md` for allowed data and source boundaries.
- Use `references/input_schema.md` for input fields and environment variables.
- Use `references/output_contract.md` for artifact names and result fields.
- Use `references/validation_notes.md` for assumptions, checks, limitations, and risk boundaries.
