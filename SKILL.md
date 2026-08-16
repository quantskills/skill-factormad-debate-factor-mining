---
name: factormad-debate-factor-mining
description: "Run a self-contained FactorMAD-style LLM multi-agent debate workflow for mining interpretable code-based stock alpha factors from daily OHLCV and point-in-time fundamental data. Use when an agent needs to generate, debate, validate, score, and export OHLCV, fundamental, or hybrid candidate alpha factor code for quantitative research."
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
  summary_zh: 使用 FactorMAD 风格的 LLM 多智能体辩论流程从日频 OHLCV 与时点一致基本面数据中挖掘代码型股票 Alpha 因子。
  summary_en: Mine interpretable code-based stock alpha factors from daily OHLCV and point-in-time fundamental data with a FactorMAD-style LLM debate workflow.
---

```json qsh-form
{
  "version": 1,
  "task": {
    "placeholder": "说明 OHLCV 数据、候选因子约束和实验目标；请上传或指明输入文件",
    "required": true
  },
  "fields": [
    {
      "key": "run_mode",
      "label": "运行模式",
      "type": "select",
      "default": "dry_run",
      "help": "建议先预检安装、路径和产物，再进行真实辩论",
      "options": [
        { "value": "dry_run", "label": "预检（不调用 LLM）" },
        { "value": "debate", "label": "真实多智能体辩论" }
      ]
    },
    {
      "key": "focus",
      "label": "挖掘关注点",
      "type": "textarea",
      "placeholder": "例如：量价背离、短期反转、可解释性或低换手"
    }
  ],
  "prompt_template": "{{#task}}任务与材料：\n{{task}}\n\n{{/task}}{{#attachments}}用户上传的材料（已放入工作区）：\n{{attachments}}\n\n{{/attachments}}按 FactorMAD 工作流以 {{run_mode}} 模式处理用户提供的 OHLCV 数据，{{#focus}}重点关注：{{focus}}；{{/focus}}组织多智能体提出、批判、校验、评分并导出可解释的 Python 股票 Alpha 候选代码，保留辩论轮次与接受因子产物，并明确内部 IC 证据不能替代独立样本外和组合验证，输出中文报告。"
}
```

# FactorMAD Debate Factor Mining

Use this skill to run a FactorMAD-style multi-agent debate workflow that proposes, critiques, validates, scores, and exports Python code-based stock alpha factor candidates.

## Core Workflow

1. Read `references/input_schema.md` before preparing the input JSON.
2. Use `dry_run=true` first to validate installation, paths, and output artifacts without an LLM call.
3. For a real run, configure `.env` from `.env.example` or set `OPENAI_API_KEY` / `FACTORMAD_OPENAI_API_KEY`, provide either OHLCV CSV data or a point-in-time OHLCV/fundamental Parquet panel plus its manifest, and run:

```bash
python scripts/factormad_debate_factor.py --input examples/debate_input.json
```

4. Read `references/output_contract.md` before consuming generated artifacts.
5. Treat lightweight Pearson IC, RankIC, and ICIR-style metrics as internal selection evidence only; run independent out-of-sample and portfolio validation before using any factor downstream.

For fundamental or hybrid mining, require `point_in_time_contract.future_backfill=false` in the manifest. Treat quarterly values as stepwise daily states after their activation date, not as daily publications. Keep `few_shot_library_paths` read-only and write newly accepted factors only to `factormad_alpha_library_path`.

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
