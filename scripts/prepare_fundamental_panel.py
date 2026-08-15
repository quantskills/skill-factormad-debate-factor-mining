#!/usr/bin/env python3
"""Build a point-in-time daily OHLCV and fundamental research panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY_COLUMNS = ["symbol", "date", "quarter", "if_adjusted"]
MARKET_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
]
DAILY_COLUMNS = [
    "date",
    "symbol",
    "market_value",
    "float_market_value",
    "turnover_rate",
    "roe_ttm",
    "is_open",
    "is_st",
]
SHARE_COLUMNS = [
    "date",
    "symbol",
    "circulation_a",
    "free_circulation",
    "total",
    "total_a",
]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_date(series: pd.Series, name: str) -> pd.Series:
    compact = (
        series.astype("string")
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        examples = series.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"{name} contains invalid dates: {examples}")
    return parsed


def _available_parquet_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    return sorted(path for path in root.rglob("data.parquet") if path.is_file())


def _read_partitioned(
    root: Path,
    *,
    columns: list[str] | None = None,
    report_year_min: int | None = None,
    report_year_max: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _available_parquet_files(root):
        report_year = None
        for part in path.parts:
            if part.startswith("report_year="):
                report_year = int(part.split("=", 1)[1])
                break
        if report_year_min is not None and report_year is not None and report_year < report_year_min:
            continue
        if report_year_max is not None and report_year is not None and report_year > report_year_max:
            continue
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows == 0:
            continue
        available = set(pq.ParquetFile(path).schema_arrow.names)
        selected = columns
        if columns is not None:
            selected = [column for column in columns if column in available]
        frames.append(pd.read_parquet(path, columns=selected))
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_year_month_dataset(
    root: Path,
    *,
    columns: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _available_parquet_files(root):
        year = month = None
        for part in path.parts:
            if part.startswith("year="):
                year = int(part.split("=", 1)[1])
            elif part.startswith("month="):
                month = int(part.split("=", 1)[1])
        if year is not None and (year < start_date.year or year > end_date.year):
            continue
        if year == start_date.year and month is not None and month < start_date.month:
            continue
        if year == end_date.year and month is not None and month > end_date.month:
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise ValueError(f"no rows found in daily dataset: {root}")
    result = pd.concat(frames, ignore_index=True)
    result["date"] = _normalize_date(result["date"], f"{root.name}.date")
    result["symbol"] = result["symbol"].astype("string").str.upper().str.strip()
    return result.loc[result["date"].between(start_date, end_date)].copy()


def load_trading_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_parquet(path)
    required = {"nature_date", "next_trade_date", "is_trade"}
    missing = sorted(required - set(calendar.columns))
    if missing:
        raise ValueError(f"trading calendar is missing columns: {missing}")
    calendar = calendar.loc[:, ["nature_date", "next_trade_date", "is_trade"]].copy()
    calendar["nature_date"] = _normalize_date(calendar["nature_date"], "nature_date")
    calendar["next_trade_date"] = _normalize_date(calendar["next_trade_date"], "next_trade_date")
    if calendar["nature_date"].duplicated().any():
        raise ValueError("trading calendar contains duplicate nature_date rows")
    return calendar.sort_values("nature_date").reset_index(drop=True)


def attach_activation_dates(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result["available_date"] = pd.Series(dtype="datetime64[ns]")
        result["activation_date"] = pd.Series(dtype="datetime64[ns]")
        return result
    result = frame.copy()
    result["available_date"] = _normalize_date(result["available_date"], "available_date")
    mapping = calendar.set_index("nature_date")["next_trade_date"]
    result["activation_date"] = result["available_date"].map(mapping)
    if result["activation_date"].isna().any():
        examples = result.loc[result["activation_date"].isna(), "available_date"].head(5).tolist()
        raise ValueError(f"announcement dates are outside the trading calendar: {examples}")
    return result


def load_statement_events(
    store_root: Path,
    calendar: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    base_root = store_root / "canonical" / "fundamentals_pit" / "schema=v1"
    extended_root = store_root / "canonical" / "fundamentals_pit_extended" / "schema=v1"
    report_year_max = end_date.year
    base = _read_partitioned(base_root, report_year_max=report_year_max)
    extended = _read_partitioned(extended_root, report_year_max=report_year_max)
    if base.empty or extended.empty:
        raise ValueError("base or extended fundamentals are empty")
    for name, frame in (("base", base), ("extended", extended)):
        missing = sorted(set(KEY_COLUMNS + ["available_date"]) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} fundamentals are missing columns: {missing}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} fundamentals contain duplicate statement keys")
    base_keys = pd.MultiIndex.from_frame(base[KEY_COLUMNS])
    extended_keys = pd.MultiIndex.from_frame(extended[KEY_COLUMNS])
    if not base_keys.equals(extended_keys):
        base_key_set = set(base_keys.tolist())
        extended_key_set = set(extended_keys.tolist())
        if base_key_set != extended_key_set:
            raise ValueError("base and extended statement keys do not match")
        extended = extended.set_index(KEY_COLUMNS).loc[base_key_set].reset_index()
    metadata = set(KEY_COLUMNS + ["available_date", "report_year", "report_quarter", "is_warmup"])
    base_value_columns = [column for column in base.columns if column not in metadata]
    extended_value_columns = [column for column in extended.columns if column not in metadata]
    right = extended.loc[:, KEY_COLUMNS + extended_value_columns].copy()
    statements = base.merge(right, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    statements["symbol"] = statements["symbol"].astype("string").str.upper().str.strip()
    statements["quarter"] = statements["quarter"].astype("string").str.lower().str.strip()
    valid_quarter = statements["quarter"].str.fullmatch(r"\d{4}q[1-4]", na=False)
    if not valid_quarter.all():
        raise ValueError("fundamentals contain invalid report quarters")
    statements["report_ordinal"] = (
        statements["quarter"].str[:4].astype(int) * 4
        + statements["quarter"].str[-1].astype(int)
        - 1
    )
    statements = attach_activation_dates(statements, calendar)
    statements = statements.loc[statements["available_date"].le(end_date)].copy()
    value_columns = base_value_columns + extended_value_columns
    for column in value_columns:
        statements[column] = pd.to_numeric(statements[column], errors="coerce")
    flow_fields = [column for column in value_columns if column.startswith(("is_", "cfs_"))]
    balance_fields = [column for column in value_columns if column.startswith("bs_")]
    if len(flow_fields) + len(balance_fields) != len(value_columns):
        unknown = sorted(set(value_columns) - set(flow_fields) - set(balance_fields))
        raise ValueError(f"unclassified fundamental fields: {unknown}")
    source = {
        "base_root": str(base_root),
        "extended_root": str(extended_root),
        "base_rows": int(len(base)),
        "extended_rows": int(len(extended)),
        "usable_rows": int(len(statements)),
        "flow_fields": flow_fields,
        "balance_fields": balance_fields,
        "if_adjusted_counts": {
            str(key): int(value)
            for key, value in statements["if_adjusted"].value_counts(dropna=False).items()
        },
    }
    return statements, flow_fields, balance_fields, source


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return numerator / denominator


def _single_quarter(
    known: dict[int, dict[str, float]],
    ordinal: int,
    field: str,
) -> float:
    current = known.get(ordinal, {}).get(field, float("nan"))
    if ordinal % 4 == 0:
        return current
    previous = known.get(ordinal - 1, {}).get(field, float("nan"))
    if not math.isfinite(current) or not math.isfinite(previous):
        return float("nan")
    return current - previous


def _ttm_value(
    known: dict[int, dict[str, float]],
    ordinal: int,
    field: str,
) -> float:
    values = [_single_quarter(known, current, field) for current in range(ordinal - 3, ordinal + 1)]
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return float(sum(values))


def build_statement_snapshots(
    statements: pd.DataFrame,
    flow_fields: list[str],
    balance_fields: list[str],
) -> pd.DataFrame:
    """Build event snapshots without allowing late old reports to roll state back."""
    if statements.empty:
        raise ValueError("statement event frame is empty")
    value_fields = flow_fields + balance_fields
    ordered = statements.sort_values(["symbol", "activation_date", "report_ordinal", "available_date"])
    output: list[dict[str, Any]] = []
    for symbol, symbol_frame in ordered.groupby("symbol", sort=False):
        known: dict[int, dict[str, float]] = {}
        available_by_ordinal: dict[int, pd.Timestamp] = {}
        latest_ordinal: int | None = None
        for activation_date, event_frame in symbol_frame.groupby("activation_date", sort=True):
            event_available_date = event_frame["available_date"].max()
            for row in event_frame.itertuples(index=False):
                ordinal = int(row.report_ordinal)
                known[ordinal] = {
                    field: float(getattr(row, field)) if pd.notna(getattr(row, field)) else float("nan")
                    for field in value_fields
                }
                available_by_ordinal[ordinal] = pd.Timestamp(row.available_date)
                latest_ordinal = ordinal if latest_ordinal is None else max(latest_ordinal, ordinal)
            if latest_ordinal is None:
                continue
            latest = known[latest_ordinal]
            record: dict[str, Any] = {
                "symbol": symbol,
                "fund_statement_activation_date": pd.Timestamp(activation_date),
                "fund_statement_available_date": pd.Timestamp(event_available_date),
                "fund_latest_report_available_date": available_by_ordinal[latest_ordinal],
                "fund_report_ordinal": latest_ordinal,
                "fund_report_year": latest_ordinal // 4,
                "fund_report_quarter": latest_ordinal % 4 + 1,
            }
            for field in flow_fields:
                prefix = f"fund_{field}"
                single = _single_quarter(known, latest_ordinal, field)
                ttm = _ttm_value(known, latest_ordinal, field)
                previous_ytd = known.get(latest_ordinal - 4, {}).get(field, float("nan"))
                current_ytd = latest.get(field, float("nan"))
                record[f"{prefix}_ytd"] = current_ytd
                record[f"{prefix}_sq"] = single
                record[f"{prefix}_ttm"] = ttm
                record[f"{prefix}_yoy"] = _safe_ratio(current_ytd, previous_ytd) - 1.0
            for field in balance_fields:
                prefix = f"fund_{field}"
                current = latest.get(field, float("nan"))
                previous = known.get(latest_ordinal - 4, {}).get(field, float("nan"))
                record[f"{prefix}_latest"] = current
                record[f"{prefix}_yoy"] = _safe_ratio(current, previous) - 1.0
            output.append(record)
    result = pd.DataFrame(output)
    if result.empty:
        raise ValueError("no statement snapshots were generated")
    return result.sort_values(["fund_statement_activation_date", "symbol"]).reset_index(drop=True)


def _midpoint(frame: pd.DataFrame, low: str, high: str) -> pd.Series:
    low_value = pd.to_numeric(frame.get(low), errors="coerce")
    high_value = pd.to_numeric(frame.get(high), errors="coerce")
    return pd.concat([low_value, high_value], axis=1).mean(axis=1, skipna=True)


def load_forecast_events(
    store_root: Path,
    calendar: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    root = store_root / "canonical" / "financial_forecast" / "schema=v1"
    frame = _read_partitioned(root, report_year_max=end_date.year)
    if frame.empty:
        return frame
    frame = attach_activation_dates(frame, calendar)
    frame = frame.loc[frame["available_date"].le(end_date)].copy()
    result = frame.loc[:, ["symbol", "available_date", "activation_date"]].copy()
    result["fund_forecast_np_mid"] = _midpoint(frame, "forecast_np_floor", "forecast_np_ceiling")
    result["fund_forecast_eps_mid"] = _midpoint(frame, "forecast_eps_floor", "forecast_eps_ceiling")
    result["fund_forecast_growth_mid"] = _midpoint(
        frame, "forecast_growth_rate_floor", "forecast_growth_rate_ceiling"
    )
    result["fund_forecast_yoy"] = pd.to_numeric(
        frame.get("net_profit_yoy_const_forecast"), errors="coerce"
    )
    result = result.rename(
        columns={
            "available_date": "fund_forecast_available_date",
            "activation_date": "fund_forecast_activation_date",
        }
    )
    return (
        result.sort_values(["fund_forecast_activation_date", "symbol"])
        .drop_duplicates(["symbol", "fund_forecast_activation_date"], keep="last")
        .reset_index(drop=True)
    )


def load_performance_events(
    store_root: Path,
    calendar: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    root = store_root / "canonical" / "financial_performance" / "schema=v1"
    frame = _read_partitioned(root, report_year_max=end_date.year)
    if frame.empty:
        return frame
    frame = attach_activation_dates(frame, calendar)
    frame = frame.loc[frame["available_date"].le(end_date)].copy()
    selected = {
        "operating_revenue_yoy": "fund_performance_revenue_yoy",
        "net_profit_parent_yoy": "fund_performance_profit_yoy",
        "net_cash_flow_operating_yoy": "fund_performance_cfo_yoy",
        "roe_weighted": "fund_performance_roe",
        "total_assets_growth_rate": "fund_performance_asset_growth",
    }
    result = frame.loc[:, ["symbol", "available_date", "activation_date"]].copy()
    for source, target in selected.items():
        result[target] = pd.to_numeric(frame.get(source), errors="coerce")
    result = result.rename(
        columns={
            "available_date": "fund_performance_available_date",
            "activation_date": "fund_performance_activation_date",
        }
    )
    return (
        result.sort_values(["fund_performance_activation_date", "symbol"])
        .drop_duplicates(["symbol", "fund_performance_activation_date"], keep="last")
        .reset_index(drop=True)
    )


def _merge_asof_by_symbol(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    daily_date: str,
    event_date: str,
) -> pd.DataFrame:
    if events.empty:
        return daily
    left = daily.sort_values([daily_date, "symbol"]).reset_index(drop=True)
    right = events.sort_values([event_date, "symbol"]).reset_index(drop=True)
    # pandas 3 distinguishes Python- and Arrow-backed StringDtype in merge keys.
    left["symbol"] = left["symbol"].astype(str).to_numpy(dtype=object)
    right["symbol"] = right["symbol"].astype(str).to_numpy(dtype=object)
    return pd.merge_asof(
        left,
        right,
        by="symbol",
        left_on=daily_date,
        right_on=event_date,
        direction="backward",
        allow_exact_matches=True,
    )


def _add_ratio_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel
    eps = 1e-12

    def ratio(name: str, numerator: str, denominator: str) -> None:
        if numerator not in result or denominator not in result:
            return
        den = pd.to_numeric(result[denominator], errors="coerce")
        num = pd.to_numeric(result[numerator], errors="coerce")
        result[name] = num.div(den.where(den.abs().gt(eps))).replace([np.inf, -np.inf], np.nan)

    def annualized_ytd_ratio(name: str, numerator: str, denominator: str) -> None:
        if numerator not in result or denominator not in result or "fund_report_quarter" not in result:
            return
        quarter = pd.to_numeric(result["fund_report_quarter"], errors="coerce")
        annualization = 4.0 / quarter.where(quarter.between(1, 4))
        den = pd.to_numeric(result[denominator], errors="coerce")
        num = pd.to_numeric(result[numerator], errors="coerce") * annualization
        result[name] = num.div(den.where(den.abs().gt(eps))).replace([np.inf, -np.inf], np.nan)

    ratio("fund_earnings_yield_ttm", "fund_is_n_income_attr_p_ttm", "market_value")
    ratio("fund_book_to_market", "fund_bs_total_hldr_eqy_exc_min_int_latest", "market_value")
    ratio("fund_sales_to_market_ttm", "fund_is_total_revenue_ttm", "market_value")
    ratio("fund_cfo_yield_ttm", "fund_cfs_net_cash_operating_ttm", "market_value")
    annualized_ytd_ratio(
        "fund_earnings_yield_ytd_ann", "fund_is_n_income_attr_p_ytd", "market_value"
    )
    annualized_ytd_ratio(
        "fund_sales_to_market_ytd_ann", "fund_is_total_revenue_ytd", "market_value"
    )
    annualized_ytd_ratio(
        "fund_cfo_yield_ytd_ann", "fund_cfs_net_cash_operating_ytd", "market_value"
    )
    ratio("fund_gross_margin_ttm", "fund_is_gross_profit_ttm", "fund_is_total_revenue_ttm")
    ratio("fund_operating_margin_ttm", "fund_is_operate_profit_ttm", "fund_is_total_revenue_ttm")
    ratio("fund_net_margin_ttm", "fund_is_n_income_attr_p_ttm", "fund_is_total_revenue_ttm")
    ratio("fund_roa_ttm", "fund_is_n_income_attr_p_ttm", "fund_bs_total_assets_latest")
    ratio(
        "fund_roe_calc_ttm",
        "fund_is_n_income_attr_p_ttm",
        "fund_bs_total_hldr_eqy_exc_min_int_latest",
    )
    ratio("fund_gross_margin_ytd", "fund_is_gross_profit_ytd", "fund_is_total_revenue_ytd")
    ratio("fund_operating_margin_ytd", "fund_is_operate_profit_ytd", "fund_is_total_revenue_ytd")
    ratio("fund_net_margin_ytd", "fund_is_n_income_attr_p_ytd", "fund_is_total_revenue_ytd")
    annualized_ytd_ratio(
        "fund_roa_ytd_ann", "fund_is_n_income_attr_p_ytd", "fund_bs_total_assets_latest"
    )
    annualized_ytd_ratio(
        "fund_roe_ytd_ann",
        "fund_is_n_income_attr_p_ytd",
        "fund_bs_total_hldr_eqy_exc_min_int_latest",
    )
    ratio("fund_cfo_to_assets_ttm", "fund_cfs_net_cash_operating_ttm", "fund_bs_total_assets_latest")
    ratio("fund_cfo_to_profit_ttm", "fund_cfs_net_cash_operating_ttm", "fund_is_n_income_attr_p_ttm")
    annualized_ytd_ratio(
        "fund_cfo_to_assets_ytd_ann",
        "fund_cfs_net_cash_operating_ytd",
        "fund_bs_total_assets_latest",
    )
    ratio("fund_cfo_to_profit_ytd", "fund_cfs_net_cash_operating_ytd", "fund_is_n_income_attr_p_ytd")
    ratio("fund_debt_to_assets", "fund_bs_total_liab_latest", "fund_bs_total_assets_latest")
    ratio("fund_current_ratio", "fund_bs_total_cur_assets_latest", "fund_bs_total_cur_liab_latest")
    ratio("fund_cash_to_debt", "fund_bs_money_cap_latest", "fund_bs_total_liab_latest")
    ratio("fund_rd_intensity_ttm", "fund_is_rd_exp_ttm", "fund_is_total_revenue_ttm")
    ratio("fund_rd_intensity_ytd", "fund_is_rd_exp_ytd", "fund_is_total_revenue_ytd")
    ratio("fund_capex_to_assets_ttm", "fund_cfs_cash_paid_asset_ttm", "fund_bs_total_assets_latest")
    annualized_ytd_ratio(
        "fund_capex_to_assets_ytd_ann",
        "fund_cfs_cash_paid_asset_ytd",
        "fund_bs_total_assets_latest",
    )
    ratio("fund_inventory_to_assets", "fund_bs_inventory_latest", "fund_bs_total_assets_latest")
    ratio(
        "fund_receivable_to_assets",
        "fund_bs_notes_accts_receiv_latest",
        "fund_bs_total_assets_latest",
    )
    ratio("fund_goodwill_to_assets", "fund_bs_goodwill_latest", "fund_bs_total_assets_latest")
    if {
        "fund_is_n_income_attr_p_ttm",
        "fund_cfs_net_cash_operating_ttm",
        "fund_bs_total_assets_latest",
    }.issubset(result.columns):
        numerator = result["fund_is_n_income_attr_p_ttm"] - result["fund_cfs_net_cash_operating_ttm"]
        denominator = result["fund_bs_total_assets_latest"].where(
            result["fund_bs_total_assets_latest"].abs().gt(eps)
        )
        result["fund_accruals_ttm"] = numerator.div(denominator).replace([np.inf, -np.inf], np.nan)
    if {
        "fund_is_n_income_attr_p_ytd",
        "fund_cfs_net_cash_operating_ytd",
        "fund_bs_total_assets_latest",
        "fund_report_quarter",
    }.issubset(result.columns):
        quarter = pd.to_numeric(result["fund_report_quarter"], errors="coerce")
        annualization = 4.0 / quarter.where(quarter.between(1, 4))
        numerator = (
            result["fund_is_n_income_attr_p_ytd"] - result["fund_cfs_net_cash_operating_ytd"]
        ) * annualization
        denominator = result["fund_bs_total_assets_latest"].where(
            result["fund_bs_total_assets_latest"].abs().gt(eps)
        )
        result["fund_accruals_ytd_ann"] = numerator.div(denominator).replace(
            [np.inf, -np.inf], np.nan
        )
    return result


def _field_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    all_columns = list(columns)
    groups: dict[str, list[str]] = {
        "market": [column for column in MARKET_COLUMNS if column in all_columns],
        "daily_fundamental": [
            column
            for column in [
                "market_value",
                "float_market_value",
                "turnover_rate",
                "roe_ttm",
                "circulation_a",
                "free_circulation",
                "total",
                "total_a",
            ]
            if column in all_columns
        ],
        "value": [
            column
            for column in all_columns
            if column
            in {
                "fund_earnings_yield_ttm",
                "fund_earnings_yield_ytd_ann",
                "fund_book_to_market",
                "fund_sales_to_market_ttm",
                "fund_sales_to_market_ytd_ann",
                "fund_cfo_yield_ttm",
                "fund_cfo_yield_ytd_ann",
            }
        ],
        "quality": [
            column
            for column in all_columns
            if column
            in {
                "fund_gross_margin_ttm",
                "fund_operating_margin_ttm",
                "fund_net_margin_ttm",
                "fund_roa_ttm",
                "fund_roe_calc_ttm",
                "fund_gross_margin_ytd",
                "fund_operating_margin_ytd",
                "fund_net_margin_ytd",
                "fund_roa_ytd_ann",
                "fund_roe_ytd_ann",
                "fund_cfo_to_assets_ttm",
                "fund_cfo_to_profit_ttm",
                "fund_cfo_to_assets_ytd_ann",
                "fund_cfo_to_profit_ytd",
                "fund_accruals_ttm",
                "fund_accruals_ytd_ann",
            }
        ],
        "solvency": [
            column
            for column in all_columns
            if column
            in {"fund_debt_to_assets", "fund_current_ratio", "fund_cash_to_debt"}
        ],
        "investment": [
            column
            for column in all_columns
            if column
            in {
                "fund_rd_intensity_ttm",
                "fund_rd_intensity_ytd",
                "fund_capex_to_assets_ttm",
                "fund_capex_to_assets_ytd_ann",
                "fund_inventory_to_assets",
                "fund_receivable_to_assets",
                "fund_goodwill_to_assets",
            }
        ],
        "events": [
            column
            for column in all_columns
            if column.startswith("fund_forecast_") or column.startswith("fund_performance_")
        ],
        "statement_flows": [
            column
            for column in all_columns
            if column.startswith("fund_is_") or column.startswith("fund_cfs_")
        ],
        "statement_balances": [
            column for column in all_columns if column.startswith("fund_bs_")
        ],
    }
    groups["growth"] = [
        column
        for column in all_columns
        if column.startswith("fund_") and column.endswith("_yoy")
    ]
    return {key: sorted(set(values)) for key, values in groups.items()}


def _coverage_report(panel: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sample = panel.loc[panel["date"].between(start, end)]
    fundamental = [
        column
        for column in sample.columns
        if column.startswith("fund_")
        and not column.endswith(("_date", "_ordinal", "_year", "_quarter"))
    ]
    rows: list[dict[str, Any]] = []
    for column in fundamental:
        values = sample[column]
        rows.append(
            {
                "field": column,
                "rows": int(len(values)),
                "nonnull_rows": int(values.notna().sum()),
                "nonnull_rate": float(values.notna().mean()),
                "finite_rows": int(np.isfinite(pd.to_numeric(values, errors="coerce")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["nonnull_rate", "field"], ascending=[False, True])


def _yearly_coverage_report(
    panel: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        current = _coverage_report(panel, start=year_start, end=year_end)
        current.insert(0, "year", year)
        rows.append(current)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_partitioned_panel(panel: pd.DataFrame, root: Path, compression: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    panel = panel.assign(_year=panel["date"].dt.year, _month=panel["date"].dt.month)
    for (year, month), frame in panel.groupby(["_year", "_month"], sort=True):
        destination = root / f"year={int(year)}" / f"month={int(month):02d}" / "data.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = frame.drop(columns=["_year", "_month"]).sort_values(["date", "symbol"])
        output.to_parquet(destination, index=False, compression=compression)
        outputs.append(
            {
                "year": int(year),
                "month": int(month),
                "rows": int(len(output)),
                "path": str(destination),
                "size_bytes": int(destination.stat().st_size),
                "sha256": _sha256_file(destination),
            }
        )
    return outputs


def build_panel(args: argparse.Namespace) -> dict[str, Any]:
    market_path = Path(args.market_csv).expanduser().resolve()
    store_root = Path(args.data_store_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    factor_start = pd.Timestamp(args.factor_start_date)
    factor_end = pd.Timestamp(args.factor_end_date)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started_at = datetime.now().isoformat()
    try:
        calendar_path = store_root / "canonical" / "trading_calendar" / "schema=v1" / "data.parquet"
        calendar = load_trading_calendar(calendar_path)
        statements, flow_fields, balance_fields, statement_source = load_statement_events(
            store_root, calendar, end_date=end_date
        )
        snapshots = build_statement_snapshots(statements, flow_fields, balance_fields)

        market = pd.read_csv(market_path, usecols=MARKET_COLUMNS)
        market["date"] = _normalize_date(market["date"], "market.date")
        market["symbol"] = market["symbol"].astype("string").str.upper().str.strip()
        market = market.loc[market["date"].between(start_date, end_date)].copy()
        if market.empty:
            raise ValueError("market CSV has no rows in the requested range")
        if market.duplicated(["date", "symbol"]).any():
            raise ValueError("market CSV contains duplicate (date, symbol) rows")

        daily = _read_year_month_dataset(
            store_root / "canonical" / "equity_daily" / "schema=v1",
            columns=DAILY_COLUMNS,
            start_date=start_date,
            end_date=end_date,
        )
        shares = _read_year_month_dataset(
            store_root / "canonical" / "stock_shares" / "schema=v1",
            columns=SHARE_COLUMNS,
            start_date=start_date,
            end_date=end_date,
        )
        for name, frame in (("equity_daily", daily), ("stock_shares", shares)):
            if frame.duplicated(["date", "symbol"]).any():
                raise ValueError(f"{name} contains duplicate (date, symbol) rows")
        panel = market.merge(daily, on=["date", "symbol"], how="left", validate="one_to_one")
        panel = panel.merge(shares, on=["date", "symbol"], how="left", validate="one_to_one")
        panel = _merge_asof_by_symbol(
            panel,
            snapshots,
            daily_date="date",
            event_date="fund_statement_activation_date",
        )

        forecast = load_forecast_events(store_root, calendar, end_date=end_date)
        performance = load_performance_events(store_root, calendar, end_date=end_date)
        panel = _merge_asof_by_symbol(
            panel,
            forecast,
            daily_date="date",
            event_date="fund_forecast_activation_date",
        )
        panel = _merge_asof_by_symbol(
            panel,
            performance,
            daily_date="date",
            event_date="fund_performance_activation_date",
        )

        if "fund_statement_available_date" in panel:
            panel["fund_statement_age_days"] = (
                panel["date"] - panel["fund_statement_available_date"]
            ).dt.days.astype("float32")
            panel["fund_statement_event_today"] = panel["date"].eq(
                panel["fund_statement_activation_date"]
            ).astype("int8")
        if "fund_forecast_available_date" in panel:
            panel["fund_forecast_age_days"] = (
                panel["date"] - panel["fund_forecast_available_date"]
            ).dt.days.astype("float32")
            panel["fund_forecast_event_today"] = panel["date"].eq(
                panel["fund_forecast_activation_date"]
            ).astype("int8")
        if "fund_performance_available_date" in panel:
            panel["fund_performance_age_days"] = (
                panel["date"] - panel["fund_performance_available_date"]
            ).dt.days.astype("float32")
            panel["fund_performance_event_today"] = panel["date"].eq(
                panel["fund_performance_activation_date"]
            ).astype("int8")
        panel = _add_ratio_features(panel)
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        for activation_column in (
            "fund_statement_activation_date",
            "fund_forecast_activation_date",
            "fund_performance_activation_date",
        ):
            if activation_column in panel:
                invalid = panel[activation_column].notna() & panel[activation_column].gt(panel["date"])
                if invalid.any():
                    raise ValueError(f"future activation dates detected in {activation_column}")

        coverage = _coverage_report(panel, start=factor_start, end=factor_end)
        yearly_coverage = _yearly_coverage_report(panel, start=factor_start, end=factor_end)
        quality_dir = staging / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        coverage_path = quality_dir / "field_coverage.parquet"
        coverage.to_parquet(coverage_path, index=False, compression=args.compression)
        yearly_coverage_path = quality_dir / "field_coverage_by_year.parquet"
        yearly_coverage.to_parquet(yearly_coverage_path, index=False, compression=args.compression)

        snapshot_path = staging / "data" / "statement_events" / "data.parquet"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_parquet(snapshot_path, index=False, compression=args.compression)
        panel_root = staging / "data" / "panel" / "schema=v1"
        partitions = _write_partitioned_panel(panel, panel_root, args.compression)
        final_columns = list(panel.columns)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "point_in_time_contract": {
                "availability_column": "available_date",
                "activation_policy": "next_trading_day",
                "fill_policy": "backward_asof_within_symbol_only",
                "future_backfill": False,
                "statement_flow_semantics": "year_to_date retained and converted to single-quarter, strict TTM, and YoY at each event snapshot; annualized YTD ratios are explicitly suffixed ytd_ann and never substituted for TTM",
                "revision_policy": "late old-quarter events update known history but never roll back the latest known report quarter",
            },
            "limitations": [
                "The source currently contains at most one stored record per symbol-quarter, so historical revision completeness is unverified.",
                "Many source statement rows are adjusted records with availability substantially later than the report period; the panel uses the conservative stored availability date.",
                "Historical point-in-time industry membership is not included.",
            ],
            "date_range": {
                "panel_start": start_date.strftime("%Y-%m-%d"),
                "panel_end": end_date.strftime("%Y-%m-%d"),
                "factor_coverage_start": factor_start.strftime("%Y-%m-%d"),
                "factor_coverage_end": factor_end.strftime("%Y-%m-%d"),
            },
            "rows": int(len(panel)),
            "symbols": int(panel["symbol"].nunique()),
            "dates": int(panel["date"].nunique()),
            "columns": final_columns,
            "column_count": len(final_columns),
            "field_groups": _field_groups(final_columns),
            "sources": {
                "market_csv": {
                    "path": str(market_path),
                    "size_bytes": int(market_path.stat().st_size),
                    "sha256": _sha256_file(market_path),
                },
                "data_store_root": str(store_root),
                "data_store_partition_manifest": {
                    "path": str(store_root / "state" / "partition_manifest.json"),
                    "sha256": _sha256_file(store_root / "state" / "partition_manifest.json"),
                },
                "trading_calendar": str(calendar_path),
                "statements": statement_source,
                "forecast_rows": int(len(forecast)),
                "performance_rows": int(len(performance)),
            },
            "artifacts": {
                "panel_root": str(output_root / "data" / "panel" / "schema=v1"),
                "statement_events": str(output_root / "data" / "statement_events" / "data.parquet"),
                "field_coverage": str(output_root / "quality" / "field_coverage.parquet"),
                "field_coverage_by_year": str(
                    output_root / "quality" / "field_coverage_by_year.parquet"
                ),
            },
            "partitions": [
                {
                    **item,
                    "path": str(output_root / Path(item["path"]).relative_to(staging)),
                }
                for item in partitions
            ],
        }
        quality = {
            "status": "passed",
            "checks": {
                "row_count_matches_market": int(len(panel)) == int(len(market)),
                "unique_date_symbol": not panel.duplicated(["date", "symbol"]).any(),
                "no_future_statement_activation": not (
                    panel["fund_statement_activation_date"].notna()
                    & panel["fund_statement_activation_date"].gt(panel["date"])
                ).any(),
                "statement_features_present": any(
                    column.startswith("fund_is_") or column.startswith("fund_bs_")
                    for column in panel.columns
                ),
                "ratio_features_present": "fund_earnings_yield_ytd_ann" in panel.columns,
            },
            "rows": int(len(panel)),
            "symbols": int(panel["symbol"].nunique()),
            "dates": int(panel["date"].nunique()),
            "fundamental_feature_count": int(sum(column.startswith("fund_") for column in panel.columns)),
            "coverage_min": float(coverage["nonnull_rate"].min()) if not coverage.empty else None,
            "coverage_median": float(coverage["nonnull_rate"].median()) if not coverage.empty else None,
            "coverage_max": float(coverage["nonnull_rate"].max()) if not coverage.empty else None,
        }
        if not all(quality["checks"].values()):
            raise ValueError(f"quality checks failed: {quality['checks']}")
        _write_json(staging / "quality" / "summary.json", quality)
        _write_json(staging / "manifests" / "fundamental_panel_manifest.json", manifest)
        staging.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-csv", required=True, help="Existing FactorMAD OHLCV CSV")
    parser.add_argument("--data-store-root", required=True, help="Root containing canonical datasets")
    parser.add_argument("--output-root", required=True, help="New experiment output directory")
    parser.add_argument("--start-date", default="2007-01-04")
    parser.add_argument("--end-date", default="2022-01-11")
    parser.add_argument("--factor-start-date", default="2010-01-04")
    parser.add_argument("--factor-end-date", default="2021-12-31")
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_panel(args)
    print(
        json.dumps(
            {
                "ok": True,
                "output_root": str(Path(args.output_root).expanduser().resolve()),
                "rows": manifest["rows"],
                "symbols": manifest["symbols"],
                "columns": manifest["column_count"],
                "partitions": len(manifest["partitions"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
