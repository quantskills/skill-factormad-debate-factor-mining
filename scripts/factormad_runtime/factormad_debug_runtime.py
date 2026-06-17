from __future__ import annotations

import json
import logging
import traceback
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
    load_factor_payload,
    load_factormad_market_data,
    parse_error,
    parse_func_name,
    parse_json,
    run_with_timeout,
    select_factors_from_payload,
    shift_date,
    resolve_llm_name,
    write_factormad_json,
)


class FactorMADDebugger:
    def __init__(
        self,
        evaluator: LightweightFactorEvaluator,
        test_range: tuple[str, str],
        test_symbols: list[str],
        factor_requirement: str,
        examples: list[dict[str, Any]],
        llm_name: str,
        timeout_seconds: int = 30,
    ):
        self.evaluator = evaluator
        self.test_range = test_range
        self.test_symbols = test_symbols
        self.factor_requirement = factor_requirement
        self.examples = examples
        self.llm_name = llm_name
        self.timeout_seconds = timeout_seconds

    def _system_prompt(self) -> str:
        sample_examples = self.examples[:3] if self.examples else []
        return "\n".join([
            "Assuming you are a helpful expert in quantitative investment research, revise a factor calculation function.",
            "Please revise the factor while meeting the following requirements:",
            self.factor_requirement,
            "Data quality requirements:",
            "- Market data may contain suspended rows, missing values, NaN, inf, zero volume, zero amount, and zero rolling denominators.",
            "- Every division must protect the denominator with eps or replace zero denominators with NaN before division.",
            "- Explicitly handle np.inf and -np.inf in the returned factor by replacing them with np.nan.",
            "- Rolling mean/std/corr may produce NaN or zero denominators; guard these cases and avoid all-NaN outputs.",
            "- Do not fill invalid factor values with arbitrary constants such as 0 unless it is economically justified.",
            "- Keep the final output aligned to data.index and independent of symbol order.",
            "Performance requirements:",
            "- Prefer vectorized pandas/numpy operations.",
            "- Avoid Python loops over symbols and avoid expensive groupby.apply when groupby.transform or direct grouped rolling can work.",
            "Keep the factor function name consistent with the original function.",
            "Here are examples of the required factor format:",
            json.dumps(sample_examples, ensure_ascii=False),
            "Return JSON only:",
            json.dumps({"code": "<code>", "arguments": {"<param>": "<value>"}}, ensure_ascii=False),
        ])

    def _suggestion_prompt(self, status: str) -> str:
        suggestions = {
            "PASS": "The factor function is correct and can be used for lightweight IC/ICIR evaluation.",
            "OTHER": "Please carefully review the code for bugs and correct them.",
            "EXEC": "Please fix syntax/runtime-definition errors in the factor function.",
            "IMPORT": "Do not import any modules; pandas is available as pd and numpy as np.",
            "LINE": "Please simplify the factor function to reduce code length.",
            "TIME": "Please optimize the factor function to run faster. Avoid Python loops, nested groupby.apply, rolling corr on very large frames, and repeated full-data recalculation.",
            "KEY": "Please use only allowed data fields: open, high, low, close, vwap, volume, amount.",
            "OUTPUT": "Please return a pandas Series aligned to the input dataframe index.",
            "NAN": "Please fix all-NaN output, often caused by division by zero, zero volume/amount, rolling std equal to zero, or rolling windows with too few valid values.",
            "FUTURE": "Please remove look-ahead bias. Use rolling/shift only with past values.",
            "DIMENSION": "Please make factor values dimensionless or cross-sectionally comparable.",
            "ORDER": "Please make the result independent of symbol ordering.",
        }
        return suggestions.get(status, suggestions["OTHER"])

    def _refine_prompt(self, code: str, args: dict[str, Any], status: str, error_msg: str | None) -> str:
        return "\n".join([
            "I am working on this quantitative factor function:",
            code,
            "Arguments:",
            json.dumps(args, ensure_ascii=False),
            "The factor failed validation with this status and error:",
            str(status),
            str(error_msg),
            self._suggestion_prompt(status),
            "When fixing the factor, explicitly consider NaN/inf/empty values, zero volume, zero amount, zero price denominators, and rolling denominators equal to zero.",
            "Use a small eps such as 1e-8 for numeric denominators where appropriate, and replace final np.inf/-np.inf with np.nan.",
            "The revised function must remain simple, fast, vectorized, and must not use future data.",
            "Please output only JSON with keys code and arguments.",
        ])

    def _check(self, code: str, args: dict[str, Any], entry_function: str | None = None) -> tuple[str, str | None]:
        if code is None or args is None:
            return "OTHER", "Error: missing factor code or arguments"
        if not isinstance(args, dict):
            return "OTHER", "Error: factor arguments must be a JSON object"

        try:
            _ = parse_func_name(code, entry_function=entry_function)
            _ = self.evaluator._factor_namespace(code)
        except Exception as exc:
            return "EXEC", str(exc)

        if "import " in code:
            return "IMPORT", "Error: factor code imports modules"
        if code.count("\n") > 80:
            return "LINE", "Error: factor code has too many lines"

        def test_func() -> pd.Series:
            return self.evaluator.calc(
                code,
                args,
                entry_function=entry_function,
                test_range=self.test_range,
                test_symbols=self.test_symbols,
            )

        try:
            factor_series = run_with_timeout(test_func, self.timeout_seconds)
        except TimeoutError:
            return "TIME", "Error: factor execution timed out"
        except Exception:
            return "OTHER", parse_error(traceback.format_exc(), code)

        if not isinstance(factor_series, pd.Series):
            return "OUTPUT", "Error: factor output is not a pandas Series"
        if factor_series.isna().all():
            return "NAN", "Error: all factor values are NaN"

        try:
            test_end = pd.Timestamp(self.test_range[1])
            test_start = pd.Timestamp(self.test_range[0])
            if test_end - test_start > pd.Timedelta(days=120):
                debug_range_wo = (self.test_range[0], shift_date(self.test_range[1], -100))
                factor_series_wo = self.evaluator.calc(
                    code,
                    args,
                    entry_function=entry_function,
                    test_range=debug_range_wo,
                    test_symbols=self.test_symbols,
                )
                factor_slice = factor_series.loc[pd.Timestamp(debug_range_wo[0]):pd.Timestamp(debug_range_wo[1])]
                diff = (factor_series_wo.fillna(0) - factor_slice.fillna(0)).abs()
                denom = factor_series_wo.fillna(0).abs() + 1e-8
                if float((diff / denom).max()) >= 1e-3:
                    return "FUTURE", "Error: possible look-ahead bias"
        except Exception:
            return "FUTURE", "Error: possible look-ahead bias"

        try:
            factor_std = factor_series.unstack().std(axis=0)
            q01 = float(factor_std.quantile(0.01))
            q99 = float(factor_std.quantile(0.99))
            if np.isfinite(q01) and np.isfinite(q99) and q99 / (q01 + 1e-8) >= 1e2:
                return "DIMENSION", "Error: factor scale differs greatly across symbols"
        except Exception:
            return "DIMENSION", "Error: factor scale differs greatly across symbols"

        try:
            reversed_symbols = list(reversed(self.test_symbols))
            if reversed_symbols:
                factor_reversed = self.evaluator.calc(
                    code,
                    args,
                    entry_function=entry_function,
                    test_range=self.test_range,
                    test_symbols=reversed_symbols,
                )
                aligned = factor_series.fillna(0).sort_index()
                reversed_aligned = factor_reversed.fillna(0).sort_index()
                diff = (aligned - reversed_aligned).abs() / (aligned.abs() + 1e-8)
                if float(diff.max()) >= 1e-3:
                    return "ORDER", "Error: factor result changes when symbol order changes"
        except Exception:
            return "ORDER", "Error: factor result changes when symbol order changes"

        return "PASS", None

    def debug(
        self,
        code: str,
        args: dict[str, Any],
        *,
        entry_function: str | None = None,
        max_debug_num: int = 2,
    ) -> tuple[str | None, dict[str, Any] | None, str, str | None, list[dict[str, Any]]]:
        gpt = None
        debug_num = 0
        current_code = code
        current_args = args
        records: list[dict[str, Any]] = []
        last_status = "OTHER"
        last_error_msg: str | None = None

        try:
            while True:
                status, error_msg = self._check(current_code, current_args, entry_function=entry_function)
                last_status, last_error_msg = status, error_msg
                records.append({
                    "round": debug_num,
                    "status": status,
                    "error": error_msg,
                    "code": current_code,
                    "arguments": current_args,
                })
                if status == "PASS":
                    logging.info("FactorMAD factor code check passed")
                    return current_code, current_args, status, None, records
                if debug_num >= max_debug_num:
                    break
                debug_num += 1
                if gpt is None:
                    gpt = SkillChatClient(self._system_prompt())
                resp = gpt.robust_chat(
                    self._refine_prompt(current_code, current_args, status, error_msg),
                    llm=self.llm_name,
                    temperatures=[0.2, 0.3, 0.4, 0.5],
                    response_type="json_object",
                    parse_func=lambda text: parse_json(text, key=["code", "arguments"]),
                )
                current_code = resp["code"]
                current_args = resp["arguments"]
        except Exception as exc:
            logging.error(traceback.format_exc())
            last_status = "OTHER"
            last_error_msg = str(exc)
        return None, None, last_status, last_error_msg, records


def _load_input_factors(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    result_path = str(input_data.get("debate_json_path") or input_data.get("factormad_result_json_path") or "").strip()
    if result_path:
        payload = load_factor_payload(result_path)
        selection = str(input_data.get("selection", "all")).strip() or "all"
        top_n = input_data.get("top_n")
        return select_factors_from_payload(payload, selection=selection, top_n=top_n)

    raw_factors = input_data.get("factors")
    if isinstance(raw_factors, list):
        return [item for item in raw_factors if isinstance(item, dict)]

    code = str(input_data.get("factor_code", "")).strip()
    if code:
        args = input_data.get("factor_arguments", input_data.get("arguments", {}))
        return [{
            "name": str(input_data.get("name", input_data.get("alpha_name", "factormad_factor"))),
            "code": code,
            "entry_function": str(input_data.get("entry_function", "")).strip() or None,
            "arguments": args if isinstance(args, dict) else {},
            "metric": input_data.get("metric", {}),
        }]
    return []


def run_factormad_debug(input_data: dict[str, Any]) -> dict[str, Any]:
    market_data_csv_path = str(input_data.get("market_data_csv_path", "")).strip()
    if not market_data_csv_path:
        raise ValueError("market_data_csv_path is required.")

    raw_data = load_factormad_market_data(market_data_csv_path)
    insample = input_data.get("insample_time_range", ["2020-01-01", "2021-12-31"])
    outsample = input_data.get("outsample_time_range", ["2022-01-01", "2022-12-31"])
    test_range = input_data.get("test_debug_range", input_data.get("test_range", insample))
    label_config = input_data.get("label_config", {"price": "vwap", "span": 10, "preprocess": "cs_zscore"})
    test_symbols = input_data.get("test_symbols")
    if not isinstance(test_symbols, list) or not test_symbols:
        test_symbols = raw_data.index.get_level_values("symbol").unique().tolist()
    factor_requirement = str(input_data.get("factor_requirement", DEFAULT_FACTOR_REQUIREMENT)).strip() or DEFAULT_FACTOR_REQUIREMENT
    examples = input_data.get("examples")
    if not isinstance(examples, list) or not examples:
        examples = DEFAULT_EXAMPLES
    llm_name = resolve_llm_name(input_data)
    max_debug_num = int(input_data.get("max_debug_num", 2) or 2)

    evaluator = LightweightFactorEvaluator(
        raw_data=raw_data,
        insample_time_range=(insample[0], insample[1]),
        outsample_time_range=(outsample[0], outsample[1]),
        label_config=label_config,
        demo_symbol=str(input_data.get("demo_symbol", "")),
    )
    debugger = FactorMADDebugger(
        evaluator=evaluator,
        test_range=(test_range[0], test_range[1]),
        test_symbols=test_symbols,
        factor_requirement=factor_requirement,
        examples=examples,
        llm_name=llm_name,
    )

    input_factors = _load_input_factors(input_data)
    if not input_factors:
        raise ValueError("No factors to debug. Pass factor_code, factors, or debate_json_path.")

    debugged_factors: list[dict[str, Any]] = []
    failed_factors: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    key_metric = str(input_data.get("key_metric", "ICIR")).strip() or "ICIR"

    for idx, factor in enumerate(input_factors, start=1):
        name = str(factor.get("name") or factor.get("entry_function") or f"factor_{idx:03d}")
        code = str(factor.get("factor_code") or factor.get("code") or "").strip()
        args = factor.get("factor_arguments", factor.get("arguments", {}))
        if not isinstance(args, dict):
            args = {}
        entry_function = str(factor.get("entry_function") or "").strip() or None
        fixed_code, fixed_args, status, error_msg, records = debugger.debug(
            code,
            args,
            entry_function=entry_function,
            max_debug_num=max_debug_num,
        )
        for record in records:
            record["factor_name"] = name
            debug_records.append(record)
        if status != "PASS" or fixed_code is None or fixed_args is None:
            failed_factors.append({"name": name, "status": status, "error": error_msg})
            continue
        info = complete_factor(evaluator, fixed_code, fixed_args, entry_function=entry_function, name=f"{name}_debugged")
        if info is None:
            failed_factors.append({"name": name, "status": "EVAL", "error": "metric calculation failed"})
            continue
        compact = compact_factor_info(info)
        compact["original_name"] = name
        compact["before_metric"] = factor.get("metric", {})
        compact["after_metric"] = compact.get("metric", {})
        debugged_factors.append(compact)

    best_factor = None
    if debugged_factors:
        best_factor = sorted(
            debugged_factors,
            key=lambda item: item.get("metric", {}).get(key_metric, float("-inf")),
            reverse=True,
        )[0]

    result_payload = {
        "input_factor_count": len(input_factors),
        "debugged_count": len(debugged_factors),
        "failed_count": len(failed_factors),
        "insample_time_range": [str(insample[0]), str(insample[1])],
        "outsample_time_range": [str(outsample[0]), str(outsample[1])],
        "metric_time_ranges": evaluator.metric_time_ranges(),
        "debugged_factors": debugged_factors,
        "failed_factors": failed_factors,
        "best_factor": best_factor,
        "ok": bool(best_factor),
    }
    debug_rounds_path = write_factormad_json(debug_records, prefix="factormad_debug_rounds_")
    debugged_factors_path = write_factormad_json(debugged_factors, prefix="factormad_debugged_factors_")
    result_payload.update({
        "debug_rounds_path": debug_rounds_path,
        "debugged_factors_path": debugged_factors_path,
    })
    debug_json_path = write_factormad_json(result_payload, prefix="factormad_debug_result_")
    result_payload["debug_json_path"] = debug_json_path
    # Overwrite the same file so the payload contains its own path.
    with open(debug_json_path, "w", encoding="utf-8") as file_obj:
        json.dump(result_payload, file_obj, ensure_ascii=False, indent=2, default=str)
        file_obj.write("\n")
    return result_payload
