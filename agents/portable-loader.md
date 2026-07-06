# Portable Loader Prompt

在不原生识别 `SKILL.md` 文件夹的 Agent 平台中，使用下面的提示词加载本 Skill。

```text
你可以访问一个名为 factormad-debate-factor-mining 的本地 Skill，路径是：

<FACTORMAD_DEBATE_FACTOR_MINING_SKILL_ROOT>

当用户请求匹配该 Skill 的 SKILL.md 描述时：

1. 先读取 <FACTORMAD_DEBATE_FACTOR_MINING_SKILL_ROOT>/SKILL.md。
2. 严格按照 SKILL.md 中的工作流和边界说明执行。
3. 仅在需要时读取 <FACTORMAD_DEBATE_FACTOR_MINING_SKILL_ROOT>/references/ 下的引用文件。
4. 在读取相关说明后，从 Skill 根目录运行内置脚本。
5. 保持文档中定义的 API 名称、参数名、环境变量、文件路径、输出约定、验证边界和数据来源边界。
6. 不要编造 Skill 文件中未支持的数据接口、API key、评价指标、因子定义、输出字段或运行时行为。
7. 将输出视为量化研究候选因子，不要解释为投资建议、交易信号、收益承诺或生产交易验证。
```

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
