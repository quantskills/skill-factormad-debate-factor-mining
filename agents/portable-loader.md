# Portable Loader: FactorMAD Debate Factor Mining

## 用途

本仓库提供一个可移植的 Skill 入口，用于运行 FactorMAD 风格的 LLM 多智能体辩论流程，从 OHLCV 行情数据中挖掘可解释的代码型股票 Alpha 因子。

## 运行入口

在仓库根目录运行：

```bash
python scripts/factormad_debate_factor.py --input <input-json> [--output <output-dir>]
```

最小示例：

```bash
python scripts/factormad_debate_factor.py --input examples/debate_input.json
```

## 真实运行流程

1. 按照 `references/input_schema.md` 准备输入 JSON。
2. 将 `dry_run` 设置为 `false`。
3. 通过 `market_data_csv_path` 提供 OHLCV 行情数据。
4. 通过 `.env` 或 shell 环境变量配置 LLM 服务。
5. 执行入口命令。

## 环境变量

优先读取：

```text
FACTORMAD_OPENAI_API_KEY
FACTORMAD_OPENAI_MODEL
FACTORMAD_OPENAI_BASE_URL
```

如果未设置，则读取：

```text
OPENAI_API_KEY
OPENAI_MODEL
OPENAI_BASE_URL
```

## 输出产物

稳定输出包括：

```text
factormad_debate_result.json
debate_rounds.json
accepted_factors.json
```

输出目录按以下优先级确定：

1. CLI 参数 `--output`
2. 输入 JSON 中的 `output_dir`
3. 自动创建的 `outputs/debateYYYYMMDDHHMMSS`

## 参考文件

- `SKILL.md`：Agent 使用说明。
- `README.md`：中文说明文档。
- `README.en.md`：英文说明文档。
- `references/input_schema.md`：输入参数说明。
- `references/output_contract.md`：输出文件和字段约定。
- `references/validation_notes.md`：验证边界和限制说明。

## 边界说明

本 Skill 仅用于量化研究自动化。生成的因子和 IC/ICIR 指标不是投资建议、收益承诺或生产交易验证。
