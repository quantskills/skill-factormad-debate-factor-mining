from __future__ import annotations

import json
import logging
import os
import re
import signal
import time
import traceback
from copy import deepcopy
from hashlib import md5
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .alpha_compute import build_alpha_output_frame
from .alpha_ops import (
    load_market_data_frame,
    write_dataframe_csv,
    write_json_output,
)
from .factormad_alpha_library import (
    load_factormad_library,
    resolve_factormad_library_path,
)


DEFAULT_FACTOR_REQUIREMENT = (
    '<1> This factor will be used in the following scenario: '
    '{"Objective": "Design a quantitative factor to predict the future returns of stocks.", '
    '"Market": "China A stock market", "Data Frequency": "Daily", '
    '"Data Introduction": {"open": "The opening price.", "high": "The highest price.", '
    '"low": "The lowest price.", "close": "The closing price.", '
    '"vwap": "The average execution price fallback field.", '
    '"volume": "The trading volume.", "amount": "The trading amount."}, '
    '"Main Metric": "The cross-sectional ICIR between factor value and future returns."}.\n'
    '<2> The factor function should be simple and efficient, using pandas/numpy operations only.\n'
    '<3> The factor value should be dimensionless or cross-sectionally comparable.\n'
    '<4> Avoid look-ahead bias: do not use future data at timestamp t.\n'
    '<5> The function should take a multi-index dataframe data with levels time and symbol. '
    'Available columns are ["open", "high", "low", "close", "vwap", "volume", "amount"].\n'
    '<6> The function may take int/float/string parameters as needed.\n'
    '<7> The output must be a pandas Series aligned to the input index.'
)


DEFAULT_EXAMPLES = [
    {
        "code": """def macd(data, short_window, long_window, mid):
    close = data["close"]
    ema_short = close.groupby("symbol", group_keys=False).apply(lambda x: x.ewm(span=short_window).mean())
    ema_long = close.groupby("symbol", group_keys=False).apply(lambda x: x.ewm(span=long_window).mean())
    diff = ema_short - ema_long
    dea = diff.groupby("symbol", group_keys=False).apply(lambda x: x.ewm(span=mid).mean())
    macd = 2 * (diff - dea)
    return macd
""",
        "arguments": {"short_window": 12, "long_window": 26, "mid": 9},
    },
    {
        "code": """def price_volume_correlation(data, window):
    return data.groupby("symbol", group_keys=False).apply(lambda x: x["close"].rolling(window).corr(x["volume"]))
""",
        "arguments": {"window": 100},
    },
    {
        "code": """def volatility(data, window):
    close = data["close"]
    ret = close.groupby("symbol", group_keys=False).apply(lambda x: x.pct_change().fillna(0))
    return ret.groupby("symbol", group_keys=False).apply(lambda x: x.rolling(window).std())
""",
        "arguments": {"window": 30},
    },
    {
        "code": """def volume_spike_factor(data, window):
    volume = data["volume"]
    avg_volume = volume.groupby("symbol", group_keys=False).apply(lambda x: x.rolling(window).mean())
    return volume / avg_volume
""",
        "arguments": {"window": 10},
    },
    {
        "code": """def price_drawdown_factor(data, window):
    close = data["close"]
    rolling_max = close.groupby("symbol", group_keys=False).apply(lambda x: x.rolling(window).max())
    return (close - rolling_max) / rolling_max
""",
        "arguments": {"window": 60},
    },
    {
        "code": """def amplitude(data, window):
    amplitude_value = (data["high"] - data["low"]) / data["low"]
    return amplitude_value.groupby("symbol", group_keys=False).apply(lambda x: x.rolling(window).mean())
""",
        "arguments": {"window": 10},
    },
]


def hash_str(plain_text: str, length: int | None = None) -> str:
    hash_code = md5(plain_text.encode("utf-8")).hexdigest()
    return hash_code[:length] if length is not None else hash_code


def calculate_tokens_num(text: str) -> int:
    if not isinstance(text, str):
        text = str(text)
    try:
        import tiktoken

        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def parse_json(text: str, key: str | list[str] | None = None) -> Any:
    def _key_filter(json_dict: dict[str, Any], key_value: str | list[str] | None) -> Any:
        if key_value is None:
            return json_dict
        if isinstance(key_value, list):
            return {k: json_dict.get(k, None) for k in key_value}
        if isinstance(key_value, str):
            return json_dict.get(key_value, None)
        raise AssertionError(f"Unsupported key type {type(key_value)}")

    try:
        return _key_filter(json.loads(text), key)
    except Exception:
        pattern = re.compile(r"```json\n([\s\S]*?)\n```")
        match = pattern.search(text)
        if not match:
            raise AssertionError(f"JSON parse failed\n{text}")
        return _key_filter(json.loads(match.group(1)), key)


def parse_error(traceback_text: str, code_text: str) -> str:
    line_msg = ""
    for line in traceback_text.splitlines():
        if 'File "<string>", line' in line:
            line_msg = line
            break

    error_text_list = ["Error:"]
    if traceback_text.splitlines():
        error_text_list.append(traceback_text.splitlines()[0])
    if line_msg:
        error_text_list.append(line_msg)
        line_match = re.search(r'File "<string>", line (\d+)', line_msg)
        if line_match:
            line_number = int(line_match.group(1))
            lines = code_text.splitlines()
            if 0 < line_number <= len(lines):
                error_text_list.append(lines[line_number - 1])
    if traceback_text.splitlines():
        error_text_list.append(traceback_text.splitlines()[-1])
    return "\n".join(error_text_list)


def shift_date(date0: str | pd.Timestamp, days: int) -> str | pd.Timestamp:
    if isinstance(date0, str):
        return (pd.Timestamp(date0) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return date0 + pd.Timedelta(days=days)


def compact_factor_info(info: dict[str, Any] | None, *, include_value: bool = False) -> dict[str, Any] | None:
    if info is None:
        return None
    out: dict[str, Any] = {}
    for key, value in info.items():
        if key == "value" and not include_value:
            continue
        if isinstance(value, pd.Series):
            out[key] = {
                "count": int(value.count()),
                "nan_count": int(value.isna().sum()),
                "preview": [float(x) if pd.notna(x) else None for x in value.dropna().head(5).tolist()],
            }
        else:
            out[key] = value
    out.pop("factor_code", None)
    out.pop("factor_arguments", None)
    return out


def parse_factor_function_names(code_text: str) -> list[str]:
    return re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code_text, flags=re.M)


def parse_func_name(code_text: str, entry_function: str | None = None) -> str:
    names = parse_factor_function_names(code_text)
    if entry_function:
        if entry_function not in names:
            raise ValueError(f"entry_function '{entry_function}' not found in factor code. Available: {names}")
        return entry_function
    if not names:
        raise ValueError("Unable to parse factor function name.")
    return names[-1]


def _raise_timeout(signum, frame):
    raise TimeoutError("factor execution timed out")


def run_with_timeout(func: Callable[[], Any], timeout_seconds: int) -> Any:
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return func()
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _calc_csic(factor_series: pd.Series, label_series: pd.Series) -> pd.Series:
    data = pd.concat([factor_series.rename("factor"), label_series.rename("label")], axis=1).dropna()
    if data.empty:
        return pd.Series(dtype="float64")
    return data.groupby("time").apply(lambda x: x["factor"].corr(x["label"], method="spearman"))


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def metric_time_ranges(
    insample_time_range: tuple[str, str] | list[str],
    outsample_time_range: tuple[str, str] | list[str],
) -> dict[str, dict[str, Any]]:
    insample = [str(insample_time_range[0]), str(insample_time_range[1])]
    outsample = [str(outsample_time_range[0]), str(outsample_time_range[1])]
    return {
        "IC": {"sample": "insample", "time_range": insample},
        "ICIR": {"sample": "insample", "time_range": insample},
        "O-IC": {"sample": "outsample", "time_range": outsample},
        "O-ICIR": {"sample": "outsample", "time_range": outsample},
    }


def load_factormad_market_data(market_data_csv_path: str) -> pd.DataFrame:
    raw = load_market_data_frame({"market_data_csv_path": market_data_csv_path})
    df = raw.copy()
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"market data is missing required columns for FactorMAD: {sorted(missing)}")

    for col in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    has_true_amount = "amount" in df.columns
    if not has_true_amount:
        df["amount"] = df["close"] * df["volume"]
    if "vwap" not in df.columns or df["vwap"].isna().all():
        volume = df["volume"].where(df["volume"] != 0)
        raw_vwap = df["amount"] / volume
        if has_true_amount and "adjfactor" in df.columns:
            adjfactor = pd.to_numeric(df["adjfactor"], errors="coerce").fillna(1.0)
            df["vwap"] = raw_vwap * adjfactor
        else:
            df["vwap"] = raw_vwap

    time_index = pd.to_datetime(df["date"].str.replace("-", "", regex=False).str[:8], format="%Y%m%d", errors="coerce")
    df = df.assign(time=time_index).dropna(subset=["time", "symbol"])
    df = df.sort_values(["time", "symbol"]).drop_duplicates(["time", "symbol"], keep="last")
    df = df.set_index(["time", "symbol"]).sort_index()
    df.index = df.index.set_names(["time", "symbol"])
    return df


def load_skill_dotenv() -> None:
    """Load the skill-local .env without overriding exported variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


def get_env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip()
    return default


class SkillChatClient:
    def __init__(self, system_prompt: str | None = None):
        load_skill_dotenv()
        from openai import OpenAI

        self.api_base = get_env_first("FACTORMAD_OPENAI_BASE_URL", "OPENAI_BASE_URL") or None
        self.api_key = get_env_first("FACTORMAD_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing OpenAI-compatible API key for FactorMAD skills. "
                "Set FACTORMAD_OPENAI_API_KEY or OPENAI_API_KEY in the shell, "
                "or copy .env.example to .env and fill it locally."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.llm_system_prompt = system_prompt or "You are a helpful assistant."
        self.token_fee_dict = {
            "gpt-5.4": (2.5, 10.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
        }
        self.total_fee = 0.0
        self.reset_conversation()

    def reset_conversation(self, conversation: list[dict[str, str]] | None = None) -> None:
        self.conversation = conversation or [{"role": "system", "content": self.llm_system_prompt}]

    def conversation_tokens_num(self) -> int:
        return calculate_tokens_num("\n".join(item["content"] for item in self.conversation))

    def fee(self) -> float:
        return round(self.total_fee, 5)

    def _update_fee(self, prompt_tokens: int, completion_tokens: int, llm: str) -> None:
        request_fee, reply_fee = self.token_fee_dict.get(llm, (0.0, 0.0))
        self.total_fee += prompt_tokens * request_fee / 1_000_000
        self.total_fee += completion_tokens * reply_fee / 1_000_000

    def chat(
        self,
        request: str,
        llm: str,
        temperature: float,
        response_type: str = "text",
        parse_func: Callable[[str], Any] | None = None,
    ) -> Any:
        messages = self.conversation + [{"role": "user", "content": str(request)}]
        response = self.client.chat.completions.create(
            model=llm,
            temperature=temperature,
            messages=messages,
            response_format={"type": response_type},
        )
        reply = response.choices[0].message.content
        if reply is None:
            raise AssertionError("chat request failed: empty reply")
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._update_fee(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0), llm)
        parsed = parse_func(reply) if parse_func is not None else reply
        self.conversation.append({"role": "user", "content": str(request)})
        self.conversation.append({"role": "assistant", "content": reply})
        return parsed

    def robust_chat(
        self,
        request: str,
        llm: str,
        temperatures: list[float],
        response_type: str = "text",
        parse_func: Callable[[str], Any] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for temperature in temperatures:
            try:
                return self.chat(
                    request,
                    llm=llm,
                    temperature=temperature,
                    response_type=response_type,
                    parse_func=parse_func,
                )
            except Exception as exc:
                last_error = exc
                logging.error("LLM chat error: %s", exc)
                time.sleep(3)
        raise AssertionError(f"LLM robust chat failed: {last_error}")


class LightweightFactorEvaluator:
    def __init__(
        self,
        raw_data: pd.DataFrame,
        insample_time_range: tuple[str, str],
        outsample_time_range: tuple[str, str],
        label_config: dict[str, Any],
        demo_symbol: str = "",
    ):
        self.raw_data = raw_data.sort_index()
        self.insample_time_range = insample_time_range
        self.outsample_time_range = outsample_time_range
        self.label_config = label_config
        self.demo_symbol = demo_symbol
        self._prepare_labels()

    def _prepare_labels(self) -> None:
        label_price = str(self.label_config.get("price", "vwap"))
        label_span = int(self.label_config.get("span", 10))
        label_preprocess = self.label_config.get("preprocess")
        if label_price not in self.raw_data.columns:
            raise ValueError(f"label price column '{label_price}' not found in market data")

        label_price_data = self.raw_data[label_price].unstack()
        label_data = (label_price_data.shift(-label_span - 1) / label_price_data.shift(-1) - 1)
        if label_preprocess == "cs_zscore":
            row_std = label_data.std(axis=1).replace(0, np.nan)
            label_data = label_data.sub(label_data.mean(axis=1), axis=0).div(row_std, axis=0)
        elif label_preprocess == "cs_rank":
            label_data = label_data.rank(axis=1, pct=True)
        elif label_preprocess in (None, "", "none"):
            pass
        else:
            raise ValueError("Only support cs_zscore and cs_rank label preprocess for FactorMAD.")

        try:
            label_series = label_data.stack(future_stack=True)
        except TypeError:
            label_series = label_data.stack(dropna=False)
        insample = (pd.Timestamp(self.insample_time_range[0]), pd.Timestamp(self.insample_time_range[1]))
        outsample = (pd.Timestamp(self.outsample_time_range[0]), pd.Timestamp(self.outsample_time_range[1]))
        self.insample_label_data = label_series.loc[insample[0]:insample[1], :].sort_index()
        self.outsample_label_data = label_series.loc[outsample[0]:outsample[1], :].sort_index()

    def calc(
        self,
        factor_code: str,
        factor_args: dict[str, Any],
        *,
        entry_function: str | None = None,
        raw_data: pd.DataFrame | None = None,
        test_range: tuple[str, str] | None = None,
        test_symbols: list[str] | None = None,
        timeout_seconds: int = 0,
    ) -> pd.Series:
        data = self.raw_data if raw_data is None else raw_data
        if test_range is not None:
            slices = (pd.Timestamp(test_range[0]), pd.Timestamp(test_range[1]))
            data = data.loc[slices[0]:slices[1], :]
        if test_symbols:
            current_symbols = data.index.get_level_values("symbol").unique().tolist()
            selected = [symbol for symbol in test_symbols if symbol in current_symbols]
            if selected:
                data = data.loc[(slice(None), selected), :]

        namespace = self._factor_namespace(factor_code)
        func_name = parse_func_name(factor_code, entry_function=entry_function)
        func = namespace[func_name]

        def _run() -> pd.Series:
            value = func(data.copy(), **factor_args)
            if isinstance(value, pd.DataFrame):
                if value.shape[1] == 1:
                    value = value.iloc[:, 0]
                else:
                    value = value.stack()
            if not isinstance(value, pd.Series):
                raise TypeError("Factor function must return a pandas Series.")
            value.index = value.index.set_names(["time", "symbol"])
            return value.sort_index().astype("float32")

        return run_with_timeout(_run, timeout_seconds)

    def _factor_namespace(self, factor_code: str) -> dict[str, Any]:
        if re.search(r"^\s*(import|from)\s+", factor_code, flags=re.M):
            raise ValueError("Factor code must not import modules; use provided pd and np.")
        namespace: dict[str, Any] = {
            "pd": pd,
            "np": np,
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "pow": pow,
                "range": range,
                "round": round,
                "set": set,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            },
        }
        forbidden = ("open(", "eval(", "exec(", "__import__", "subprocess", "os.", "sys.")
        if any(token in factor_code for token in forbidden):
            raise ValueError("Factor code contains forbidden operations.")
        exec(compile(factor_code, "<factormad_factor>", "exec"), namespace)
        return namespace

    def metric_time_ranges(self) -> dict[str, dict[str, Any]]:
        return metric_time_ranges(self.insample_time_range, self.outsample_time_range)

    def calc_metric(self, factor_series: pd.Series) -> dict[str, float]:
        if factor_series.isna().all():
            return {"IC": float("nan"), "ICIR": float("nan"), "O-IC": float("nan"), "O-ICIR": float("nan")}

        insample = (pd.Timestamp(self.insample_time_range[0]), pd.Timestamp(self.insample_time_range[1]))
        outsample = (pd.Timestamp(self.outsample_time_range[0]), pd.Timestamp(self.outsample_time_range[1]))
        insample_factor = factor_series.loc[insample[0]:insample[1], :]
        outsample_factor = factor_series.loc[outsample[0]:outsample[1], :]

        insample_csic = _calc_csic(insample_factor.copy(), self.insample_label_data.copy())
        outsample_csic = _calc_csic(outsample_factor.copy(), self.outsample_label_data.copy())
        if len(insample_csic) and _safe_float(insample_csic.mean()) < 0:
            insample_csic = -insample_csic
            outsample_csic = -outsample_csic

        insample_mean = _safe_float(insample_csic.mean()) if len(insample_csic) else float("nan")
        insample_std = _safe_float(insample_csic.std()) if len(insample_csic) else float("nan")
        outsample_mean = _safe_float(outsample_csic.mean()) if len(outsample_csic) else float("nan")
        outsample_std = _safe_float(outsample_csic.std()) if len(outsample_csic) else float("nan")
        return {
            "IC": round(insample_mean, 5) if pd.notna(insample_mean) else float("nan"),
            "ICIR": round(insample_mean / insample_std, 5) if insample_std not in (0.0, float("nan")) and pd.notna(insample_std) else float("nan"),
            "O-IC": round(outsample_mean, 5) if pd.notna(outsample_mean) else float("nan"),
            "O-ICIR": round(outsample_mean / outsample_std, 5) if outsample_std not in (0.0, float("nan")) and pd.notna(outsample_std) else float("nan"),
        }

    def backtest(self, factor_code: str, factor_args: dict[str, Any], *, entry_function: str | None = None) -> dict[str, Any]:
        factor_series = self.calc(factor_code, factor_args, entry_function=entry_function).rename("factor")
        metric_dict = self.calc_metric(factor_series)
        value = pd.Series(dtype="float32")
        available_symbols = factor_series.unstack().columns.tolist() if len(factor_series) else []
        demo_symbol = self.demo_symbol if self.demo_symbol in available_symbols else (available_symbols[0] if available_symbols else "")
        if demo_symbol:
            value = factor_series.unstack()[demo_symbol]
        return {
            "factor_series": factor_series,
            "metric": metric_dict,
            "insample_time_range": [str(self.insample_time_range[0]), str(self.insample_time_range[1])],
            "outsample_time_range": [str(self.outsample_time_range[0]), str(self.outsample_time_range[1])],
            "demo_symbol": demo_symbol,
            "value": value,
        }


def complete_factor(
    evaluator: LightweightFactorEvaluator,
    code: str | None,
    args: dict[str, Any] | None,
    *,
    entry_function: str | None = None,
    name: str = "",
) -> dict[str, Any] | None:
    if code is None or args is None:
        return None
    res_dict = evaluator.backtest(code, args, entry_function=entry_function)
    parsed_entry = parse_func_name(code, entry_function=entry_function)
    factor_info: dict[str, Any] = {
        "name": name or parsed_entry,
        "code": code,
        "entry_function": parsed_entry,
        "hash": hash_str(code + json.dumps(args, sort_keys=True, default=str), length=8),
        "arguments": args,
        "value": res_dict["value"],
        "metric": res_dict["metric"],
        "insample_time_range": res_dict["insample_time_range"],
        "outsample_time_range": res_dict["outsample_time_range"],
    }
    return factor_info


def _series_to_matrix(series: pd.Series) -> pd.DataFrame:
    frame = series.copy()
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("Factor series must use a two-level MultiIndex: time, symbol.")
    frame.index = frame.index.set_names(["time", "symbol"])
    matrix = frame.unstack()
    matrix.index = pd.to_datetime(matrix.index).strftime("%Y%m%d")
    matrix.columns = matrix.columns.astype(str)
    return matrix


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip()).strip("_").lower()
    return slug or "factormad"


def _unique_column_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used_names:
            return candidate
        suffix += 1


def load_factor_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"FactorMAD result JSON not found: {path}")
    with payload_path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError("FactorMAD result JSON must contain an object.")
    return loaded


def select_factors_from_payload(payload: dict[str, Any], selection: str = "best", top_n: int | None = None) -> list[dict[str, Any]]:
    selection = str(selection or "best").strip().lower()
    if selection == "best":
        best = payload.get("best_factor") or {}
        if best:
            return [best]
        if payload.get("best_factor_code"):
            return [{
                "name": "best_factor",
                "code": payload.get("best_factor_code"),
                "arguments": payload.get("best_factor_arguments", {}),
                "metric": payload.get("best_metric", {}),
            }]
    candidates: list[dict[str, Any]] = []
    # Use the most refined available collection. Do not merge generated factors
    # with accepted/debugged factors when selection="all"; that would reintroduce
    # rejected or duplicate candidates into downstream debug/compute.
    for key in ("debugged_factors", "accepted_factors", "generated_factors", "candidates"):
        raw = payload.get(key)
        if isinstance(raw, list):
            candidates.extend([item for item in raw if isinstance(item, dict)])
            break
    if selection == "effective":
        candidates = [
            item for item in candidates
            if str(item.get("status") or "").strip().lower() == "effective"
        ]
    if top_n is not None:
        candidates = candidates[: int(top_n)]
    return candidates


def load_factors_from_library(path: str | None = None, top_n: int | None = None) -> list[dict[str, Any]]:
    factors = load_factormad_library(resolve_factormad_library_path(path))
    if top_n is not None:
        factors = factors[: int(top_n)]
    return factors


def compute_factormad_alpha(
    *,
    market_data_csv_path: str,
    factors: list[dict[str, Any]],
) -> dict[str, Any]:
    if not factors:
        raise ValueError("No FactorMAD factors were provided for computation.")
    raw_csv_data = load_market_data_frame({"market_data_csv_path": market_data_csv_path})
    raw_factor_data = load_factormad_market_data(market_data_csv_path)
    evaluator = LightweightFactorEvaluator(
        raw_data=raw_factor_data,
        insample_time_range=(
            raw_factor_data.index.get_level_values("time").min().strftime("%Y-%m-%d"),
            raw_factor_data.index.get_level_values("time").max().strftime("%Y-%m-%d"),
        ),
        outsample_time_range=(
            raw_factor_data.index.get_level_values("time").min().strftime("%Y-%m-%d"),
            raw_factor_data.index.get_level_values("time").max().strftime("%Y-%m-%d"),
        ),
        label_config={"price": "vwap", "span": 1},
    )

    alpha_results: dict[str, pd.DataFrame] = {}
    alpha_columns: list[str] = []
    alpha_names: dict[str, str] = {}
    failed_factors: list[dict[str, Any]] = []
    used_columns: set[str] = set()

    for idx, factor in enumerate(factors, start=1):
        code = str(factor.get("factor_code") or factor.get("code") or "").strip()
        args = factor.get("factor_arguments", factor.get("arguments", {}))
        if not isinstance(args, dict):
            args = {}
        name = str(factor.get("name") or factor.get("entry_function") or f"factor_{idx:03d}")
        entry_function = str(factor.get("entry_function") or "").strip() or None
        try:
            series = evaluator.calc(code, args, entry_function=entry_function)
            matrix = _series_to_matrix(series)
            base_name = _slugify_name(name)
            col = _unique_column_name(f"alpha_factormad_{idx:03d}_{base_name}", used_columns)
            used_columns.add(col)
            alpha_results[col] = matrix
            alpha_columns.append(col)
            alpha_names[col] = name
        except Exception as exc:
            failed_factors.append({"name": name, "error": str(exc)[:300]})

    out, nan_counts = build_alpha_output_frame(raw_csv_data, alpha_results, alpha_order=alpha_columns)
    alpha_csv_path = write_dataframe_csv(out, prefix="factormad_alpha_signal_")
    return {
        "alpha_csv_path": alpha_csv_path,
        "row_count": len(out),
        "alpha_columns": alpha_columns,
        "alpha_names": alpha_names,
        "nan_counts": nan_counts,
        "computed_count": len(alpha_columns),
        "failed_count": len(failed_factors),
        "failed_factors": failed_factors,
        "ok": len(alpha_columns) > 0,
    }


def resolve_llm_name(input_data: dict[str, Any]) -> str:
    load_skill_dotenv()
    llm_name = str(input_data.get("llm_name", "")).strip()
    return llm_name or get_env_first("FACTORMAD_OPENAI_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")


def write_factormad_json(payload: Any, *, prefix: str) -> str:
    return write_json_output(payload, prefix=prefix)
