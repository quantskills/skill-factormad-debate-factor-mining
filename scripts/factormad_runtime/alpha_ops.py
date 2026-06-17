"""Alpha operations helper library for jiaocha_skills.

Provides:
  - ``build_matrices``  — pivot a raw market-data DataFrame into per-field
                          date × symbol matrices.
  - ``AlphaEngine``     — bound alpha-computation methods and an eval namespace.
  - ``translate_alpha`` — LLM-powered expression → Python-code translator.
  - ``HELPER_DOCS``     — prompt documentation string used by the LLM translator.
"""

from __future__ import annotations

import os
import re
from typing import Any
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_skill_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip()
    return default


# ─── Fields that are not numeric / should not be pivoted ─────────────────────
_NON_NUMERIC = frozenset({
    "date", "symbol", "name", "trade_status", "dominant_id",
    "exchange", "trading_code", "underlying_symbol", "trading_date",
    "minute", "datetime",
})

# ─── Prompt documentation exposed to the LLM ─────────────────────────────────
HELPER_DOCS: str = """\
AVAILABLE FIELD VARIABLES (each is a date × symbol DataFrame):
  open, high, low, close, volume, amount
  (use _ref("field_name") to access other fields by name)
  IMPORTANT: Only these 6 fields exist. Do NOT use vwap or any other field.

AVAILABLE FUNCTIONS:
  rank(x)               — cross-sectional percentile rank at each date (0–1)
    zscore(x)             — cross-sectional z-score at each date
  ts_rank(x, n)         — time-series percentile rank over n periods
    ts_zscore(x, n)       — rolling z-score over n periods
  delay(x, n)           — lag x by n periods
  delta(x, n)           — x - delay(x, n)
  returns(x, n=1)       — pct_change over n periods (default x=close)
    adv(n)                — rolling mean of volume over n periods
  ts_mean(x, n)         — rolling mean
  ts_std(x, n)          — rolling standard deviation
  ts_max(x, n)          — rolling maximum
  ts_min(x, n)          — rolling minimum
  ts_sum(x, n)          — rolling sum
    ts_argmax(x, n)       — periods since rolling maximum within the last n bars
    ts_argmin(x, n)       — periods since rolling minimum within the last n bars
  correlation(x, y, n)  — rolling Pearson correlation per symbol
  covariance(x, y, n)   — rolling covariance per symbol
  scale(x, a=1)         — cross-sectional rescaling so |sum| == a
  decay_linear(x, n)    — linearly-decayed weighted moving average
    sign(x), log(x), abs(x), power(x, n), signed_power(x, n)
    min(x, y), max(x, y), clip(x, lower, upper)
  Standard arithmetic:  +, -, *, /  between DataFrames and scalars
"""


# ─── Matrix builder ───────────────────────────────────────────────────────────

def build_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot a raw market-data DataFrame into per-field date × symbol matrices.

    Args:
        df: DataFrame with at least ``date`` and ``symbol`` columns.

    Returns:
        Mapping of field name → date-indexed, symbol-columned DataFrame.
    """
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["date", "symbol"])

    numeric_cols = [c for c in df.columns if c not in _NON_NUMERIC]
    matrices: dict[str, pd.DataFrame] = {}
    for col in numeric_cols:
        try:
            matrices[col] = df.pivot(index="date", columns="symbol", values=col).astype(float)
        except (TypeError, ValueError):
            continue

    has_true_amount = "amount" in matrices
    adjfactor = matrices.get("adjfactor", 1.0)

    if not has_true_amount and all(k in matrices for k in ("close", "volume")):
        if "adjfactor" in matrices:
            matrices["amount"] = matrices["close"] / adjfactor.replace(0, np.nan) * matrices["volume"]
        else:
            matrices["amount"] = matrices["close"] * matrices["volume"]

    if "vwap" not in matrices and all(k in matrices for k in ("amount", "volume")):
        matrices["vwap"] = matrices["amount"] / matrices["volume"].replace(0, np.nan) * adjfactor

    return matrices


def load_market_data_frame(input_data: dict[str, Any]) -> pd.DataFrame:
    """Load market data from a CSV path.

    Args:
        input_data: Parsed skill input JSON.

    Returns:
        A normalized DataFrame with at least ``date`` and ``symbol`` columns.

    Raises:
        ValueError: If the input is missing or malformed.
        FileNotFoundError: If a referenced CSV path does not exist.
    """
    market_data_csv_path = str(input_data.get("market_data_csv_path", "")).strip()

    if not market_data_csv_path:
        raise ValueError("market_data_csv_path is required. Run get_market_data first and pass its CSV path.")
    if not os.path.isfile(market_data_csv_path):
        raise FileNotFoundError(f"market_data_csv_path not found: {market_data_csv_path}")
    df_raw = pd.read_csv(market_data_csv_path)

    if df_raw.empty:
        raise ValueError("market_data_csv_path points to an empty CSV.")
    if "date" not in df_raw.columns or "symbol" not in df_raw.columns:
        raise ValueError("market data CSV must include 'date' and 'symbol' columns.")

    df_raw = df_raw.copy()
    df_raw["date"] = df_raw["date"].astype(str)
    return df_raw


def _resolve_output_dir() -> Path:
    """Return the project-local directory where skill outputs should be written."""
    configured_dir = os.environ.get("FACTORMAD_OUTPUT_DIR", "").strip() or os.environ.get("PANDA_SKILL_OUTPUT_DIR", "").strip()
    if configured_dir:
        output_dir = Path(configured_dir)
    else:
        output_dir = Path.cwd() / ".skill_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _build_output_path(*, prefix: str, suffix: str) -> Path:
    """Create a unique output path inside the configured project-local output dir."""
    output_dir = _resolve_output_dir()
    stem = prefix.rstrip("_") or "output"
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = output_dir / f"{stem}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_dataframe_csv(df: pd.DataFrame, *, prefix: str) -> str:
    """Write a DataFrame to a project-local CSV and return its absolute path."""
    output_path = _build_output_path(prefix=prefix, suffix=".csv")

    df_to_write = df.copy()
    if "date" in df_to_write.columns:
        df_to_write["date"] = df_to_write["date"].astype(str)
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv

        table = pa.Table.from_pandas(df_to_write, preserve_index=False)
        pacsv.write_csv(table, output_path)
    except Exception:
        df_to_write.to_csv(output_path, index=False)
    return str(output_path)


def write_text_output(text: str, *, prefix: str, suffix: str = ".txt") -> str:
    """Write text to a project-local file and return its absolute path."""
    output_path = _build_output_path(prefix=prefix, suffix=suffix)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def write_json_output(payload: Any, *, prefix: str) -> str:
    """Write a JSON payload to a project-local file and return its absolute path."""
    output_path = _build_output_path(prefix=prefix, suffix=".json")
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return str(output_path)


def evaluate_alpha_expression(
    alpha_expr: str,
    df_raw: pd.DataFrame,
    *,
    api_key: str = "",
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[pd.DataFrame, str]:
    """Translate and evaluate an alpha expression on normalized market data."""
    alpha_expr = alpha_expr.strip()
    if not alpha_expr:
        raise ValueError("alpha_expr is required.")

    matrices = build_matrices(df_raw)
    engine = AlphaEngine(matrices)
    python_expr = translate_alpha(
        alpha_expr,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    try:
        result = eval(python_expr, engine.eval_ns())  # noqa: S307
        return engine._to_df(result), python_expr
    except NameError as exc:
        missing_name = str(exc).split("'")[1] if "'" in str(exc) else None
        if missing_name:
            available = ", ".join(sorted(engine.matrices.keys()))
            raise ValueError(
                f"Alpha evaluation failed: field '{missing_name}' is not available in market data. Available: {available}"
            ) from exc
        raise ValueError(f"Alpha evaluation failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - error shape covered via skills
        raise ValueError(f"Alpha evaluation failed: {exc}") from exc


# ─── AlphaEngine ─────────────────────────────────────────────────────────────

class AlphaEngine:
    """Holds per-invocation data matrices and exposes alpha-computation methods.

    All methods accept DataFrames *or* scalars/Series and always return a
    date × symbol DataFrame aligned to the input matrices.

    Usage::

        engine = AlphaEngine(build_matrices(df_raw))
        result_df = eval(python_expr, engine.eval_ns())
    """

    def __init__(self, matrices: dict[str, pd.DataFrame]) -> None:
        if not matrices:
            raise ValueError(
                "Market data does not contain any numeric fields that can be used in alpha expressions."
            )
        self.matrices = matrices
        self._template: pd.DataFrame = next(iter(matrices.values()))

    # ── internal utilities ────────────────────────────────────────────────────

    def _ref(self, name: str) -> pd.DataFrame:
        """Return a field matrix by name with a clear error if unavailable."""
        if name not in self.matrices:
            available = ", ".join(self.matrices.keys())
            raise KeyError(f"Field '{name}' not in market data. Available: {available}")
        return self.matrices[name]

    def _to_df(self, x: Any) -> pd.DataFrame:
        """Coerce scalar / Series / ndarray to a date × symbol DataFrame."""
        if isinstance(x, pd.DataFrame):
            return x
        ref = self._template
        if isinstance(x, (int, float, np.integer, np.floating)):
            return pd.DataFrame(float(x), index=ref.index, columns=ref.columns)
        if isinstance(x, pd.Series):
            return pd.DataFrame(
                [x.values] * len(ref), index=ref.index, columns=ref.columns
            )
        raise TypeError(f"Cannot coerce {type(x).__name__} to DataFrame")

    # ── cross-sectional operations ────────────────────────────────────────────

    def rank(self, x: Any) -> pd.DataFrame:
        """Cross-sectional percentile rank at each date (0–1)."""
        return self._to_df(x).rank(axis=1, pct=True)

    def zscore(self, x: Any) -> pd.DataFrame:
        """Cross-sectional z-score at each date."""
        x = self._to_df(x)
        row_std = x.std(axis=1).replace(0, np.nan)
        return x.sub(x.mean(axis=1), axis=0).div(row_std, axis=0)

    def scale(self, x: Any, a: float = 1.0) -> pd.DataFrame:
        """Cross-sectionally rescale so the sum of absolute values equals ``a``."""
        x = self._to_df(x)
        row_sum = x.abs().sum(axis=1).replace(0, np.nan)
        return x.div(row_sum, axis=0) * a

    # ── time-series operations ────────────────────────────────────────────────

    def delay(self, x: Any, n: int) -> pd.DataFrame:
        """Value of ``x`` lagged by ``n`` periods."""
        return self._to_df(x).shift(n)

    def delta(self, x: Any, n: int) -> pd.DataFrame:
        """``x`` minus its value ``n`` periods ago."""
        x = self._to_df(x)
        return x - x.shift(n)

    def returns(self, x: Any = None, n: int = 1) -> pd.DataFrame:
        """Percentage change over ``n`` periods. Defaults to ``close``."""
        base = self._to_df(x) if x is not None else self._ref("close")
        return base.pct_change(n)

    def adv(self, n: int) -> pd.DataFrame:
        """Average daily volume over the past ``n`` periods."""
        return self.ts_mean(self._ref("volume"), n)

    def ts_mean(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling mean over the past ``n`` periods."""
        return self._to_df(x).rolling(n, min_periods=1).mean()

    def ts_std(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling standard deviation over the past ``n`` periods."""
        return self._to_df(x).rolling(n, min_periods=2).std()

    def ts_max(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling maximum over the past ``n`` periods."""
        return self._to_df(x).rolling(n, min_periods=1).max()

    def ts_min(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling minimum over the past ``n`` periods."""
        return self._to_df(x).rolling(n, min_periods=1).min()

    def ts_sum(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling sum over the past ``n`` periods."""
        return self._to_df(x).rolling(n, min_periods=1).sum()

    def ts_rank(self, x: Any, n: int) -> pd.DataFrame:
        """Time-series percentile rank of the current value within the past ``n`` periods."""
        def _rank_last(w: np.ndarray) -> float:
            s = pd.Series(w)
            return float(s.rank(pct=True).iloc[-1])

        return self._to_df(x).apply(
            lambda col: col.rolling(n, min_periods=max(1, n // 2)).apply(
                _rank_last, raw=False
            )
        )

    def ts_zscore(self, x: Any, n: int) -> pd.DataFrame:
        """Rolling z-score of ``x`` over ``n`` periods."""
        x = self._to_df(x)
        rolling_mean = x.rolling(n, min_periods=1).mean()
        rolling_std = x.rolling(n, min_periods=2).std().replace(0, np.nan)
        return (x - rolling_mean) / rolling_std

    def ts_argmax(self, x: Any, n: int) -> pd.DataFrame:
        """Periods since the rolling maximum within the last ``n`` bars."""
        return self._to_df(x).apply(
            lambda col: col.rolling(n, min_periods=1).apply(
                lambda w: float(len(w) - 1 - int(np.argmax(w))), raw=True
            )
        )

    def ts_argmin(self, x: Any, n: int) -> pd.DataFrame:
        """Periods since the rolling minimum within the last ``n`` bars."""
        return self._to_df(x).apply(
            lambda col: col.rolling(n, min_periods=1).apply(
                lambda w: float(len(w) - 1 - int(np.argmin(w))), raw=True
            )
        )

    def decay_linear(self, x: Any, n: int) -> pd.DataFrame:
        """Linearly-decayed weighted moving average over ``n`` periods."""
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()
        return self._to_df(x).apply(
            lambda col: col.rolling(n, min_periods=n).apply(
                lambda w: float(np.dot(w, weights)), raw=True
            )
        )

    # ── pairwise operations ───────────────────────────────────────────────────

    def correlation(self, x: Any, y: Any, n: int) -> pd.DataFrame:
        """Rolling Pearson correlation between ``x`` and ``y`` over ``n`` periods."""
        x, y = self._to_df(x), self._to_df(y)
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for sym in x.columns:
            if sym in y.columns:
                result[sym] = x[sym].rolling(n, min_periods=2).corr(y[sym])
        return result

    def covariance(self, x: Any, y: Any, n: int) -> pd.DataFrame:
        """Rolling covariance between ``x`` and ``y`` over ``n`` periods."""
        x, y = self._to_df(x), self._to_df(y)
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for sym in x.columns:
            if sym in y.columns:
                result[sym] = x[sym].rolling(n, min_periods=2).cov(y[sym])
        return result

    # ── element-wise operations ───────────────────────────────────────────────

    def sign(self, x: Any) -> pd.DataFrame:
        return pd.DataFrame(np.sign(self._to_df(x)), index=self._template.index, columns=self._template.columns)

    def log(self, x: Any) -> pd.DataFrame:
        return np.log(self._to_df(x))

    def abs(self, x: Any) -> pd.DataFrame:
        return self._to_df(x).abs()

    def power(self, x: Any, n: Any) -> pd.DataFrame:
        return self._to_df(x) ** n

    def signed_power(self, x: Any, n: Any) -> pd.DataFrame:
        x = self._to_df(x)
        return np.sign(x) * (x.abs() ** n)

    def min(self, x: Any, y: Any) -> pd.DataFrame:
        return self._to_df(x).combine(self._to_df(y), np.minimum)

    def max(self, x: Any, y: Any) -> pd.DataFrame:
        return self._to_df(x).combine(self._to_df(y), np.maximum)

    def clip(self, x: Any, lower: Any, upper: Any) -> pd.DataFrame:
        x = self._to_df(x)
        lower_value = self._to_df(lower) if isinstance(lower, (pd.DataFrame, pd.Series)) else lower
        upper_value = self._to_df(upper) if isinstance(upper, (pd.DataFrame, pd.Series)) else upper
        return self.min(self.max(x, lower_value), upper_value)

    # ── eval namespace ────────────────────────────────────────────────────────

    def eval_ns(self) -> dict[str, Any]:
        """Return a namespace dict suitable for ``eval(python_expr, ns)``."""
        ns: dict[str, Any] = {
            # alpha functions (unbound from self for clean eval usage)
            "rank":         self.rank,
            "zscore":       self.zscore,
            "ts_rank":      self.ts_rank,
            "ts_zscore":    self.ts_zscore,
            "delay":        self.delay,
            "delta":        self.delta,
            "returns":      self.returns,
            "adv":          self.adv,
            "ts_mean":      self.ts_mean,
            "ts_std":       self.ts_std,
            "ts_max":       self.ts_max,
            "ts_min":       self.ts_min,
            "ts_sum":       self.ts_sum,
            "ts_argmax":    self.ts_argmax,
            "ts_argmin":    self.ts_argmin,
            "correlation":  self.correlation,
            "covariance":   self.covariance,
            "scale":        self.scale,
            "decay_linear": self.decay_linear,
            "sign":         self.sign,
            "log":          self.log,
            "abs":          self.abs,
            "power":        self.power,
            "signed_power": self.signed_power,
            "min":          self.min,
            "max":          self.max,
            "clip":         self.clip,
            # field access helper
            "_ref":         self._ref,
            # standard libs
            "np":           np,
            "pd":           pd,
        }
        # expose each field matrix by name; 'open' is aliased to 'open_'
        for name, mat in self.matrices.items():
            ns[name] = mat
        ns["open_"] = self.matrices.get("open")
        return ns


# ─── LLM translator ──────────────────────────────────────────────────────────

def translate_alpha(
    alpha_expr: str,
    *,
    api_key: str = "",
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Use an LLM to translate an alpha formula string to a Python expression.

    Falls back to returning ``alpha_expr`` unchanged if the LLM call fails.

    Args:
        alpha_expr: The raw alpha formula string.
        api_key:    OpenAI-compatible API key.
        base_url:   Optional custom base URL.
        model:      Model name.

    Returns:
        A Python expression string ready to be ``eval``'d against
        the namespace produced by :meth:`AlphaEngine.eval_ns`.
    """
    system_prompt = (
        "You are an expert quantitative researcher. Convert alpha factor expressions "
        "to a single Python expression using ONLY the helpers listed below.\n\n"
        + HELPER_DOCS
        + "\nRULES:\n"
        "1. Output ONLY the Python expression — no assignments, no imports, no markdown.\n"
        "2. The expression must evaluate to a DataFrame (date rows × symbol columns).\n"
        "3. Use the exact variable/function names listed above.\n"
        "4. Do NOT invent functions not in the list.\n"
        "5. When the alpha uses 'open', write it as 'open_' (reserved word workaround).\n"
    )
    try:
        from openai import OpenAI  # type: ignore

        _load_skill_dotenv()
        resolved_api_key = api_key or _env_first("FACTORMAD_OPENAI_API_KEY", "OPENAI_API_KEY")
        resolved_base_url = base_url or _env_first("FACTORMAD_OPENAI_BASE_URL", "OPENAI_BASE_URL") or None
        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this alpha expression:\n{alpha_expr}"},
            ],
            temperature=0,
            max_tokens=512,
        )
        code = (response.choices[0].message.content or "").strip()
        code = re.sub(r"```[a-z]*\n?", "", code).replace("```", "").strip()
        return code
    except Exception:
        return alpha_expr  # graceful fallback
