from __future__ import annotations

import json
import logging
import os
import random
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .factormad_compute_runtime import (
    DEFAULT_EXAMPLES,
    DEFAULT_FACTOR_REQUIREMENT,
    LightweightFactorEvaluator,
    SkillChatClient,
    compact_factor_info,
    complete_factor,
    default_key_metric_for_evaluation_mode,
    hash_str,
    load_factormad_market_data,
    normalize_evaluation_mode,
    parse_json,
    resolve_llm_name,
    write_factormad_json,
)
from .factormad_debug_runtime import FactorMADDebugger
from .factormad_alpha_library import (
    append_effective_factors_to_library,
    load_factormad_library,
    resolve_factormad_library_path,
)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[FactorMAD/runtime] {message}", file=sys.stderr, flush=True)


def _finite_metric_value(metric: dict[str, Any], key: str) -> float:
    try:
        value = float(metric.get(key, float("-inf")))
    except Exception:
        return float("-inf")
    return value if np.isfinite(value) else float("-inf")


def _metric_text(factor_info: dict[str, Any] | None, key_metric: str) -> str:
    if not factor_info:
        return "metric=n/a"
    metric = factor_info.get("metric") or {}
    parts: list[str] = []
    for key in (key_metric, "ICIR", "RankICIR", "HybridICIR", "IC", "RankIC", "O-ICIR", "O-RankICIR"):
        if key in metric and key not in {item.split("=")[0] for item in parts}:
            try:
                parts.append(f"{key}={float(metric[key]):.5f}")
            except Exception:
                parts.append(f"{key}={metric[key]}")
    return " ".join(parts) if parts else "metric=n/a"


class FactorMADFactorSet:
    def __init__(
        self,
        factor_infos: list[dict[str, Any]] | None = None,
        metric_threshold: float = 0.2,
        correlation_threshold: float = 0.8,
        key_metric: str = "ICIR",
        evaluation_mode: str = "pearson_ic",
        icir_threshold: float | None = None,
        rank_icir_threshold: float | None = None,
    ):
        self.metric_threshold = metric_threshold
        self.icir_threshold = metric_threshold if icir_threshold is None else float(icir_threshold)
        self.rank_icir_threshold = metric_threshold if rank_icir_threshold is None else float(rank_icir_threshold)
        self.correlation_threshold = correlation_threshold
        self.evaluation_mode = normalize_evaluation_mode(evaluation_mode)
        self.key_metric = key_metric or default_key_metric_for_evaluation_mode(self.evaluation_mode)
        self.factors: dict[str, dict[str, Any]] = {}
        self.invalid_factors: dict[str, dict[str, Any]] = {}
        for info in factor_infos or []:
            if isinstance(info, dict) and "hash" in info:
                self.factors[info["hash"]] = info

    def read_all(self, *, is_sorted: bool = False, contains_invalid: bool = False) -> list[dict[str, Any]]:
        factors = list(self.factors.values())
        if contains_invalid:
            factors += list(self.invalid_factors.values())
        if is_sorted:
            factors = sorted(
                factors,
                key=lambda item: _finite_metric_value(item.get("metric") or {}, self.key_metric),
                reverse=True,
            )
        return factors

    def sample(self) -> dict[str, Any]:
        return deepcopy(random.choice(list(self.factors.values())))

    def correlation_retrieval(
        self,
        query_info: dict[str, Any],
        retrieval_factors: dict[str, dict[str, Any]] | list[dict[str, Any]],
        threshold: float,
    ) -> list[tuple[float, dict[str, Any]]]:
        if isinstance(retrieval_factors, dict):
            retrieval_factors = list(retrieval_factors.values())
        if not retrieval_factors or "value" not in query_info:
            return []
        query_value = pd.to_numeric(pd.Series(query_info["value"]), errors="coerce").sort_index()
        result_list: list[tuple[float, dict[str, Any]]] = []
        for bench_info in retrieval_factors:
            if bench_info.get("hash") == query_info.get("hash") or "value" not in bench_info:
                continue
            bench_value = pd.to_numeric(pd.Series(bench_info["value"]), errors="coerce").sort_index()
            aligned = pd.concat([query_value.rename("query"), bench_value.rename("bench")], axis=1, join="inner").dropna()
            if len(aligned) < 2:
                continue
            corr = aligned["query"].corr(aligned["bench"])
            if corr == corr and float(corr) > threshold:
                result_list.append((float(corr), bench_info))
        return sorted(result_list, key=lambda x: x[0], reverse=True)

    def _passes_metric_threshold(self, factor_info: dict[str, Any]) -> bool:
        metric = factor_info.get("metric") or {}
        if self.evaluation_mode == "pearson_ic":
            return _finite_metric_value(metric, "ICIR") >= self.icir_threshold
        if self.evaluation_mode == "rank_ic":
            return _finite_metric_value(metric, "RankICIR") >= self.rank_icir_threshold
        if self.evaluation_mode == "hybrid":
            return (
                _finite_metric_value(metric, "ICIR") >= self.icir_threshold
                and _finite_metric_value(metric, "RankICIR") >= self.rank_icir_threshold
                and bool(factor_info.get("direction_consistent", False))
            )
        raise ValueError(f"unsupported evaluation_mode: {self.evaluation_mode}")

    def check(self, factor_info: dict[str, Any] | None) -> str:
        if not factor_info:
            return "invalid"
        try:
            if not self._passes_metric_threshold(factor_info):
                return "invalid"
            similar = self.correlation_retrieval(
                factor_info,
                retrieval_factors=self.factors,
                threshold=self.correlation_threshold,
            )
            return "similar" if similar else "effective"
        except Exception:
            return "invalid"

    def append(self, sample: dict[str, Any] | None) -> str:
        status = self.check(sample)
        if not sample or "hash" not in sample:
            return status
        if status == "effective":
            self.factors[sample["hash"]] = sample
        else:
            self.invalid_factors[sample["hash"]] = sample
        return status


class FactorMADDebateAgent:
    def __init__(
        self,
        init_factors: list[dict[str, Any]],
        evaluator: LightweightFactorEvaluator,
        debugger: FactorMADDebugger,
        factorset: FactorMADFactorSet,
        factor_requirement: str,
        llm_name: str,
    ):
        self.init_factors = init_factors
        self.evaluator = evaluator
        self.debugger = debugger
        self.factorset = factorset
        self.factor_requirement = factor_requirement
        self.llm_name = llm_name
        self.system_prompt = self._system_prompt()
        self.gpt = SkillChatClient(self.system_prompt)

    def _stepback(self) -> str:
        sys_prompt = "You are a quantitative factor researcher."
        prompt = "\n".join([
            "Summarize one key hypothesis for mining quantitative factors based on the examples.",
            "The factor requirements are:",
            self.factor_requirement,
            "The example quantitative factors are:",
            json.dumps([
                {
                    "code": factor.get("code"),
                    "arguments": factor.get("arguments"),
                    "metric": factor.get("metric"),
                }
                for factor in self.init_factors
            ], ensure_ascii=False),
            "The hypothesis should focus on a specific aspect, such as volatility, moving averages, normalization, volume-price relation, or fields to use.",
            "Return JSON only: {\"hypothese\": \"<text>\"}",
        ])
        resp = SkillChatClient(sys_prompt).robust_chat(
            prompt,
            llm=self.llm_name,
            temperatures=[0.8, 0.8, 0.8],
            response_type="json_object",
            parse_func=lambda text: parse_json(text, key=["hypothese"]),
        )
        return str(resp.get("hypothese", ""))

    def _system_prompt(self) -> str:
        hypothese = self._stepback()
        return "\n".join([
            "You are a quantitative investment researcher debating with another researcher to improve a quantitative factor.",
            "The factor requirements are:",
            self.factor_requirement,
            "You have already designed these factors:",
            json.dumps([
                {
                    "code": factor.get("code"),
                    "arguments": factor.get("arguments"),
                    "metric": factor.get("metric"),
                }
                for factor in self.init_factors
            ], ensure_ascii=False),
            "Your summarized factor-mining hypothesis is:",
            hypothese,
            "Debate process:",
            "<1> You receive the opposite viewpoint plus factor code/arguments and IC/ICIR metrics.",
            "<2> Summarize the opposing view.",
            "<3> Critique the factor using its code, arguments, and metrics.",
            "<4> Design an improved factor with code and arguments.",
            "If the previous factor failed validation, directly address the failure reason in your critique and revised factor.",
            "Market data can contain NaN, inf, zero volume, zero amount, and zero rolling denominators; protect all divisions and replace final inf values with NaN.",
            "For large A-share data, use fast vectorized pandas/numpy operations and avoid Python loops, nested groupby.apply, or expensive rolling corr unless necessary.",
            "Return JSON only with keys summary, opinion, factor.",
            json.dumps({
                "summary": "<summary>",
                "opinion": "<opinion>",
                "factor": {"code": "<code>", "arguments": {}},
            }, ensure_ascii=False),
        ])

    def _status_prompt(self, status: str) -> str:
        if status == "invalid":
            return "This factor is invalid or too weak under lightweight IC/ICIR evaluation."
        if status == "similar":
            return "This factor is too similar to existing accepted factors."
        if status == "effective":
            return "This factor is effective under lightweight IC/ICIR evaluation."
        return "The status of this factor is unknown."

    def _failure_feedback(self, debug_status: str, error_msg: str | None) -> str:
        details = str(error_msg or "unknown validation error")
        guidance = [
            "The factor you designed failed validation, so it cannot be accepted yet.",
            f"Validation status: {debug_status}.",
            f"Validation error: {details}.",
        ]
        if debug_status == "TIME" or "timed out" in details.lower():
            guidance.append(
                "Please rewrite it with faster vectorized pandas/numpy operations; avoid Python loops, nested groupby.apply, and expensive rolling corr on the full universe."
            )
        if debug_status == "NAN" or any(token in details.lower() for token in ["nan", "inf", "divide", "zero"]):
            guidance.append(
                "Please explicitly handle NaN/inf, zero volume, zero amount, zero prices, and rolling denominators equal to zero; protect divisions with eps and replace final inf values with NaN."
            )
        if debug_status == "DIMENSION":
            guidance.append("Please normalize the factor so values are dimensionless and comparable across symbols.")
        if debug_status == "ORDER":
            guidance.append("Please make the result independent of symbol ordering and aligned to data.index.")
        if debug_status == "FUTURE":
            guidance.append("Please remove look-ahead bias and only use information available at timestamp t or earlier.")
        guidance.append("Please propose a corrected factor in the required JSON format.")
        return "\n".join(guidance)

    def generate_init_factor(self) -> dict[str, Any] | None:
        sys_prompt = "You are a quantitative factor researcher designing a classic quantitative factor."
        prompt = "\n".join([
            "I want to design a quantitative factor. Requirements:",
            self.factor_requirement,
            "Use these examples for format only; do not generate a near-duplicate:",
            "The generated factor must handle NaN/inf, zero volume, zero amount, and zero denominators, and should be vectorized for large A-share data.",
            json.dumps([
                {"code": factor.get("code"), "arguments": factor.get("arguments")}
                for factor in self.init_factors
            ], ensure_ascii=False),
            "Return JSON only with keys code and arguments.",
        ])
        resp = SkillChatClient(sys_prompt).robust_chat(
            prompt,
            llm=self.llm_name,
            temperatures=[0.8, 0.8, 0.8],
            response_type="json_object",
            parse_func=lambda text: parse_json(text, key=["code", "arguments"]),
        )
        factor_code, factor_args, status, _, _ = self.debugger.debug(
            resp["code"],
            resp["arguments"],
            max_debug_num=2,
        )
        if status != "PASS":
            return None
        return complete_factor(self.evaluator, factor_code, factor_args)

    def init_view(self, factor_info: dict[str, Any] | None = None) -> tuple[str | None, dict[str, Any] | None]:
        if factor_info is None:
            factor_info = self.generate_init_factor()
        if factor_info is None:
            return None, None
        init_view = "\n".join([
            "Here is the factor I designed:",
            json.dumps({"code": factor_info["code"], "arguments": factor_info["arguments"]}, ensure_ascii=False),
            "And its lightweight IC/ICIR metrics are:",
            json.dumps(factor_info["metric"], ensure_ascii=False),
            "Please give me your feedback.",
        ])
        self.gpt.reset_conversation([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Please design a factor."},
            {"role": "assistant", "content": init_view},
        ])
        return init_view, factor_info

    def _complete(self, resp: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str, str | None]:
        try:
            raw_factor = resp.get("factor") or {}
            factor_code, factor_args, debug_status, error_msg, _ = self.debugger.debug(
                raw_factor.get("code"),
                raw_factor.get("arguments"),
                max_debug_num=2,
            )
            if debug_status != "PASS":
                return self._failure_feedback(debug_status, error_msg), None, "invalid", error_msg
            factor_info = complete_factor(self.evaluator, factor_code, factor_args)
            status = self.factorset.check(factor_info)
            resp["factor"] = {"code": factor_code, "arguments": factor_args}
            resp["backtest"] = {
                "metric": factor_info.get("metric") if factor_info else {},
                "status": self._status_prompt(status),
            }
            self.gpt.conversation[-1] = {"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)}
            return json.dumps(resp, ensure_ascii=False), factor_info, status, None
        except Exception as exc:
            return self._failure_feedback("OTHER", str(exc)), None, "invalid", str(exc)

    def respond(self, opposite_view: str) -> tuple[str, dict[str, Any] | None, str, str | None]:
        resp = self.gpt.robust_chat(
            opposite_view,
            llm=self.llm_name,
            temperatures=[0.2, 0.2, 0.2],
            response_type="json_object",
            parse_func=lambda text: parse_json(text, key=["summary", "opinion", "factor"]),
        )
        return self._complete(resp)


def _load_seed_examples(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    examples = input_data.get("examples")
    if isinstance(examples, list) and examples:
        return examples

    seed_path = str(input_data.get("seed_factors_json_path") or input_data.get("seed_alphas_json_path") or "").strip()
    if seed_path:
        try:
            with open(seed_path, "r", encoding="utf-8") as file_obj:
                loaded = json.load(file_obj)
            if isinstance(loaded, dict):
                loaded = loaded.get("alphas", [])
            if isinstance(loaded, list):
                code_examples = []
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("factor_code") or item.get("code")
                    args = item.get("factor_arguments") or item.get("arguments") or {}
                    if code and isinstance(args, dict):
                        code_examples.append({"code": code, "arguments": args})
                if code_examples:
                    return code_examples
        except Exception as exc:
            logging.warning("Failed to load seed factors from %s: %s", seed_path, exc)

    return DEFAULT_EXAMPLES


def run_agent_debate(
    *,
    evaluator: LightweightFactorEvaluator,
    debugger: FactorMADDebugger,
    factorset: FactorMADFactorSet,
    factor_requirement: str,
    llm_name: str,
    agent_few_shots: int,
    debate_max_rounds: int,
    debate_max_tokens: int,
    key_metric: str,
    seed_metric_threshold: float,
    show_progress: bool = True,
    job_index: int = 1,
) -> dict[str, Any]:
    total_examples = factorset.read_all()
    if len(total_examples) < agent_few_shots:
        raise ValueError(f"Not enough seed factors. Need {agent_few_shots}, got {len(total_examples)}.")
    _progress(
        show_progress,
        f"job {job_index}: sampling {agent_few_shots} few-shot examples from {len(total_examples)} seed/library factors",
    )
    example_list1 = random.sample(total_examples, agent_few_shots)
    example_list2 = random.sample(total_examples, agent_few_shots)

    _progress(show_progress, f"job {job_index}: initializing debate agents")
    agent1 = FactorMADDebateAgent(example_list1, evaluator, debugger, factorset, factor_requirement, llm_name)
    agent2 = FactorMADDebateAgent(example_list2, evaluator, debugger, factorset, factor_requirement, llm_name)

    def _stop() -> bool:
        return (
            agent1.gpt.conversation_tokens_num() > debate_max_tokens
            or agent2.gpt.conversation_tokens_num() > debate_max_tokens
        )

    def _passes_seed_threshold(factor_info: dict[str, Any]) -> bool:
        metric = factor_info.get("metric") or {}
        if factorset.evaluation_mode == "hybrid":
            return (
                _finite_metric_value(metric, "HybridICIR") >= seed_metric_threshold
                and bool(factor_info.get("direction_consistent", False))
            )
        return _finite_metric_value(metric, key_metric) >= seed_metric_threshold

    debate_round = 0
    debate_hash = hash_str(agent1.system_prompt + agent2.system_prompt, length=8)
    views: list[str] = []
    round_records: list[dict[str, Any]] = []
    generated_factors: list[dict[str, Any]] = []

    _progress(show_progress, f"job {job_index} round 0: generating initial factor")
    view, init_factor_info = agent2.init_view()
    init_source = "llm_generated"
    if init_factor_info is None:
        # Keep the debate moving when the LLM-generated initial factor fails
        # debugger checks. This mirrors the original FactorMAD option of
        # selecting an existing seed factor as the initial view.
        init_source = "seed_fallback"
        _progress(show_progress, f"job {job_index} round 0: initial factor failed validation, using seed fallback")
        init_factor_info = factorset.sample()
        view, init_factor_info = agent2.init_view(init_factor_info)

    views.append(view or "")
    init_status = factorset.append(init_factor_info)
    init_compact = compact_factor_info(init_factor_info)
    init_compact["debate"] = debate_hash
    init_compact["round"] = 0
    init_compact["status"] = init_status
    init_compact["init_source"] = init_source
    generated_factors.append(init_compact)
    _progress(
        show_progress,
        f"job {job_index} round 0: status={init_status} source={init_source} {_metric_text(init_factor_info, key_metric)}",
    )
    round_records.append({
        "round": 0,
        "agent": "agent2",
        "view": view,
        "factor": init_compact,
        "status": init_status,
        "init_source": init_source,
    })
    if _passes_seed_threshold(init_factor_info):
        _progress(
            show_progress,
            f"job {job_index}: early stop because round 0 {key_metric} reaches seed_metric_threshold={seed_metric_threshold}",
        )
        best_factor = init_compact
        return {
            "debate_hash": debate_hash,
            "agent1_factors": [compact_factor_info(x) for x in agent1.init_factors],
            "agent2_factors": [compact_factor_info(x) for x in agent2.init_factors],
            "init_factor": init_compact,
            "views": views,
            "debate_rounds": round_records,
            "generated_factors": generated_factors,
            "best_factor": best_factor,
            "accepted_factors": [compact_factor_info(x) for x in factorset.read_all(is_sorted=True)],
            "invalid_factors": [compact_factor_info(x) for x in factorset.invalid_factors.values()],
            "round_count": 0,
            "llm_fee": agent1.gpt.fee() + agent2.gpt.fee(),
        }

    while debate_round < debate_max_rounds:
        for agent_name, agent in [("agent1", agent1), ("agent2", agent2)]:
            try:
                debate_round += 1
                _progress(
                    show_progress,
                    f"job {job_index} debate round {debate_round}/{debate_max_rounds}: {agent_name} responding",
                )
                view, factor_info, status, error_msg = agent.respond(view or "")
                views.append(view or "")
                compact = compact_factor_info(factor_info)
                if compact:
                    compact["debate"] = debate_hash
                    compact["round"] = debate_round
                    compact["status"] = status
                    generated_factors.append(compact)
                if factor_info:
                    status = factorset.append(factor_info)
                _progress(
                    show_progress,
                    f"job {job_index} debate round {debate_round}: {agent_name} status={status} {_metric_text(factor_info, key_metric)}",
                )
                round_records.append({
                    "round": debate_round,
                    "agent": agent_name,
                    "view": view,
                    "factor": compact,
                    "status": status,
                    "error": error_msg,
                })
                if _stop() or debate_round >= debate_max_rounds:
                    break
            except Exception as exc:
                _progress(show_progress, f"job {job_index} debate round {debate_round}: {agent_name} error={exc}")
                round_records.append({
                    "round": debate_round,
                    "agent": agent_name,
                    "status": "error",
                    "error": str(exc),
                })
                break
        if _stop() or debate_round >= debate_max_rounds:
            break

    valid_generated = [factor for factor in generated_factors if isinstance(factor, dict)]
    best_factor = None
    if valid_generated:
        best_factor = sorted(
            valid_generated,
            key=lambda item: _finite_metric_value(item.get("metric") or {}, key_metric),
            reverse=True,
        )[0]

    return {
        "debate_hash": debate_hash,
        "agent1_factors": [compact_factor_info(x) for x in agent1.init_factors],
        "agent2_factors": [compact_factor_info(x) for x in agent2.init_factors],
        "init_factor": init_compact,
        "views": views,
        "debate_rounds": round_records,
        "generated_factors": generated_factors,
        "best_factor": best_factor,
        "accepted_factors": [compact_factor_info(x) for x in factorset.read_all(is_sorted=True)],
        "invalid_factors": [compact_factor_info(x) for x in factorset.invalid_factors.values()],
        "round_count": debate_round,
        "llm_fee": round(agent1.gpt.fee() + agent2.gpt.fee(), 5),
    }


def run_factormad_debate(input_data: dict[str, Any]) -> dict[str, Any]:
    show_progress = True
    market_data_csv_path = str(input_data.get("market_data_csv_path", "")).strip()
    if not market_data_csv_path:
        raise ValueError("market_data_csv_path is required.")

    _progress(show_progress, f"loading market data: {market_data_csv_path}")
    raw_data = load_factormad_market_data(market_data_csv_path)
    _progress(
        show_progress,
        "market data loaded: "
        f"rows={len(raw_data)} "
        f"symbols={raw_data.index.get_level_values('symbol').nunique()} "
        f"dates={raw_data.index.get_level_values('time').min().date()}..{raw_data.index.get_level_values('time').max().date()}",
    )
    insample = input_data.get("insample_time_range", ["2020-01-01", "2021-12-31"])
    outsample = input_data.get("outsample_time_range", ["2022-01-01", "2022-12-31"])
    test_range = input_data.get("test_debug_range", input_data.get("test_range", insample))
    label_config = input_data.get("label_config", {"price": "vwap", "span": 10, "preprocess": "cs_zscore"})
    demo_symbol = str(input_data.get("demo_symbol", "")).strip()
    test_symbols = input_data.get("test_symbols")
    if not isinstance(test_symbols, list) or not test_symbols:
        test_symbols = raw_data.index.get_level_values("symbol").unique().tolist()

    factor_requirement = str(input_data.get("factor_requirement", DEFAULT_FACTOR_REQUIREMENT)).strip() or DEFAULT_FACTOR_REQUIREMENT
    examples = _load_seed_examples(input_data)
    llm_name = resolve_llm_name(input_data)
    agent_few_shots = int(input_data.get("agent_few_shots", 3) or 3)
    debate_max_rounds = int(input_data.get("debate_max_rounds", 10) or 10)
    debate_max_tokens = int(input_data.get("debate_max_tokens", 32 * 1024) or (32 * 1024))
    seed_metric_threshold = float(input_data.get("seed_metric_threshold", 0.3))
    factor_metric_threshold = float(input_data.get("factor_metric_threshold", 0.2))
    icir_threshold = float(input_data.get("icir_threshold", factor_metric_threshold))
    rank_icir_threshold = float(input_data.get("rank_icir_threshold", factor_metric_threshold))
    factor_correlation_threshold = float(input_data.get("factor_correlation_threshold", 0.8))
    evaluation_mode = normalize_evaluation_mode(
        input_data.get("evaluation_mode"),
        legacy_key_metric=input_data.get("key_metric"),
    )
    key_metric = default_key_metric_for_evaluation_mode(evaluation_mode)
    debate_jobs = max(1, int(input_data.get("debate_jobs", 1) or 1))
    use_alpha_library = _to_bool(input_data.get("use_factormad_alpha_library"), default=False)
    few_shot_from_library = _to_bool(input_data.get("few_shot_from_library"), default=use_alpha_library)
    alpha_library_path = str(
        input_data.get("factormad_alpha_library_path")
        or input_data.get("alpha_library_path")
        or resolve_factormad_library_path()
    )

    _progress(
        show_progress,
        "configuration: "
        f"model={llm_name} jobs={debate_jobs} max_rounds={debate_max_rounds} "
        f"few_shots={agent_few_shots} evaluation_mode={evaluation_mode} key_metric={key_metric} "
        f"icir_threshold={icir_threshold} rank_icir_threshold={rank_icir_threshold}",
    )
    _progress(
        show_progress,
        "label config: "
        f"price={label_config.get('price')} span={label_config.get('span')} preprocess={label_config.get('preprocess')}",
    )

    evaluator = LightweightFactorEvaluator(
        raw_data=raw_data,
        insample_time_range=(insample[0], insample[1]),
        outsample_time_range=(outsample[0], outsample[1]),
        label_config=label_config,
        demo_symbol=demo_symbol,
        evaluation_mode=evaluation_mode,
    )
    debugger = FactorMADDebugger(
        evaluator=evaluator,
        test_range=(test_range[0], test_range[1]),
        test_symbols=test_symbols,
        factor_requirement=factor_requirement,
        examples=examples,
        llm_name=llm_name,
    )

    seed_factor_infos: list[dict[str, Any]] = []
    seed_failures: list[dict[str, Any]] = []
    _progress(show_progress, f"evaluating seed examples: count={len(examples)}")
    for idx, example in enumerate(examples, start=1):
        try:
            info = complete_factor(evaluator, example.get("code"), example.get("arguments"), name=f"seed_{idx:03d}")
            if info is not None:
                seed_factor_infos.append(info)
        except Exception as exc:
            seed_failures.append({"index": idx, "error": str(exc)[:300]})

    _progress(
        show_progress,
        f"seed evaluation finished: usable={len(seed_factor_infos)} failed={len(seed_failures)}",
    )
    if len(seed_factor_infos) < agent_few_shots:
        raise ValueError(f"Not enough seed factors after evaluation. Need {agent_few_shots}, got {len(seed_factor_infos)}.")

    library_factor_infos: list[dict[str, Any]] = []
    library_failures: list[dict[str, Any]] = []
    if use_alpha_library:
        library_items = load_factormad_library(alpha_library_path)
        _progress(show_progress, f"evaluating alpha library: path={alpha_library_path} count={len(library_items)}")
        for item in library_items:
            try:
                info = complete_factor(
                    evaluator,
                    item.get("factor_code") or item.get("code"),
                    item.get("factor_arguments") or item.get("arguments") or {},
                    entry_function=item.get("entry_function") or None,
                    name=str(item.get("name") or ""),
                )
                if info is not None:
                    for key in ("source", "created_at", "workflow_run", "debate", "round"):
                        if key in item and key not in info:
                            info[key] = item[key]
                    library_factor_infos.append(info)
            except Exception as exc:
                library_failures.append({"name": item.get("name"), "hash": item.get("hash"), "error": str(exc)[:300]})
        _progress(
            show_progress,
            f"alpha library evaluation finished: usable={len(library_factor_infos)} failed={len(library_failures)}",
        )

    initial_factor_infos = seed_factor_infos
    if few_shot_from_library and library_factor_infos:
        initial_factor_infos = seed_factor_infos + library_factor_infos

    factorset = FactorMADFactorSet(
        factor_infos=initial_factor_infos,
        metric_threshold=factor_metric_threshold,
        correlation_threshold=factor_correlation_threshold,
        key_metric=key_metric,
        evaluation_mode=evaluation_mode,
        icir_threshold=icir_threshold,
        rank_icir_threshold=rank_icir_threshold,
    )

    job_results: list[dict[str, Any]] = []
    all_round_records: list[dict[str, Any]] = []
    all_generated_factors: list[dict[str, Any]] = []
    total_fee = 0.0

    for job_index in range(1, debate_jobs + 1):
        _progress(show_progress, f"starting debate job {job_index}/{debate_jobs}")
        job_result = run_agent_debate(
            evaluator=evaluator,
            debugger=debugger,
            factorset=factorset,
            factor_requirement=factor_requirement,
            llm_name=llm_name,
            agent_few_shots=agent_few_shots,
            debate_max_rounds=debate_max_rounds,
            debate_max_tokens=debate_max_tokens,
            key_metric=key_metric,
            seed_metric_threshold=seed_metric_threshold,
            show_progress=show_progress,
            job_index=job_index,
        )
        job_result["job_index"] = job_index
        total_fee += float(job_result.get("llm_fee", job_result.get("fee", 0.0)) or 0.0)
        for record in job_result.get("debate_rounds", job_result.get("round_records", [])):
            if isinstance(record, dict):
                record = dict(record)
                record["job_index"] = job_index
                all_round_records.append(record)
        for factor in job_result.get("generated_factors", []):
            if isinstance(factor, dict):
                factor = dict(factor)
                factor["job_index"] = job_index
                all_generated_factors.append(factor)
        job_results.append({
            "job_index": job_index,
            "debate_hash": job_result.get("debate_hash"),
            "round_count": job_result.get("round_count", 0),
            "best_factor": job_result.get("best_factor"),
            "llm_fee": job_result.get("llm_fee", job_result.get("fee", 0.0)),
            "generated_count": len(job_result.get("generated_factors", [])),
        })
        _progress(
            show_progress,
            f"finished debate job {job_index}/{debate_jobs}: "
            f"rounds={job_result.get('round_count', 0)} "
            f"generated={len(job_result.get('generated_factors', []))} "
            f"llm_fee={job_result.get('llm_fee', job_result.get('fee', 0.0))}",
        )

    valid_generated = [factor for factor in all_generated_factors if isinstance(factor, dict)]
    best_factor = None
    if valid_generated:
        best_factor = sorted(
            valid_generated,
            key=lambda item: _finite_metric_value(item.get("metric") or {}, key_metric),
            reverse=True,
        )[0]

    debate_hash = hash_str(json.dumps([job.get("debate_hash") for job in job_results], sort_keys=True), length=8)
    debate_result = {
        "debate_hash": debate_hash,
        "jobs": job_results,
        "job_count": debate_jobs,
        "insample_time_range": [str(insample[0]), str(insample[1])],
        "outsample_time_range": [str(outsample[0]), str(outsample[1])],
        "metric_time_ranges": evaluator.metric_time_ranges(),
        "evaluation_mode": evaluation_mode,
        "key_metric": key_metric,
        "icir_threshold": icir_threshold,
        "rank_icir_threshold": rank_icir_threshold,
        "debate_rounds": all_round_records,
        "generated_factors": all_generated_factors,
        "best_factor": best_factor,
        "accepted_factors": [compact_factor_info(x) for x in factorset.read_all(is_sorted=True)],
        "invalid_factors": [compact_factor_info(x) for x in factorset.invalid_factors.values()],
        "round_count": sum(int(job.get("round_count", 0) or 0) for job in job_results),
        "llm_fee": round(total_fee, 5),
        "library_factor_count_before": len(library_factor_infos),
        "library_failures": library_failures,
        "seed_factor_count": len(seed_factor_infos),
        "few_shot_from_library": few_shot_from_library,
        "use_factormad_alpha_library": use_alpha_library,
        "factormad_alpha_library_path": alpha_library_path,
    }

    if use_alpha_library:
        _progress(show_progress, f"updating alpha library: {alpha_library_path}")
        library_update = append_effective_factors_to_library(
            path=alpha_library_path,
            factors=all_generated_factors,
            key_metric=key_metric,
            metadata={
                "market_data_csv_path": market_data_csv_path,
                "insample_time_range": insample,
                "outsample_time_range": outsample,
            },
        )
        debate_result["library_update"] = library_update
        debate_result["library_factor_count_after"] = library_update.get("after_count", 0)
    debate_result["seed_failures"] = seed_failures
    debate_result["library_failures"] = debate_result.get("library_failures", [])
    debate_result["accepted_count"] = len(debate_result.get("accepted_factors", []))
    debate_result["invalid_count"] = len(debate_result.get("invalid_factors", []))
    debate_result["ok"] = bool(debate_result.get("best_factor"))

    _progress(
        show_progress,
        "debate summary: "
        f"ok={debate_result['ok']} "
        f"rounds={debate_result['round_count']} "
        f"accepted={debate_result['accepted_count']} "
        f"invalid={debate_result['invalid_count']} "
        f"llm_fee={debate_result['llm_fee']}",
    )

    result_path = write_factormad_json(debate_result, prefix="factormad_debate_result_")
    debate_result["debate_json_path"] = result_path
    # Overwrite the same file so the payload contains its own path.
    with open(result_path, "w", encoding="utf-8") as file_obj:
        json.dump(debate_result, file_obj, ensure_ascii=False, indent=2, default=str)
        file_obj.write("\n")
    return debate_result
