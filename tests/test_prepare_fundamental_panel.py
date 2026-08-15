from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_fundamental_panel import (  # noqa: E402
    _add_ratio_features,
    attach_activation_dates,
    build_statement_snapshots,
    _merge_asof_by_symbol,
)


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nature_date": pd.to_datetime(["2020-04-30", "2020-05-01", "2020-05-04"]),
            "next_trade_date": pd.to_datetime(["2020-05-04", "2020-05-04", "2020-05-05"]),
            "is_trade": [1, 0, 1],
        }
    )


def test_activation_uses_next_trading_day() -> None:
    events = pd.DataFrame({"available_date": [20200430, 20200501]})
    result = attach_activation_dates(events, _calendar())
    assert result["activation_date"].tolist() == [pd.Timestamp("2020-05-04")] * 2


def test_statement_snapshots_convert_ytd_and_do_not_roll_back() -> None:
    rows = [
        ("000001.SZ", "2020-04-30", "2020-05-04", 2020 * 4, 10.0, 100.0),
        ("000001.SZ", "2020-08-30", "2020-08-31", 2020 * 4 + 1, 30.0, 110.0),
        ("000001.SZ", "2020-10-30", "2020-11-02", 2020 * 4 + 2, 60.0, 120.0),
        ("000001.SZ", "2021-03-30", "2021-03-31", 2020 * 4 + 3, 100.0, 130.0),
        # A late old-quarter record must not replace the 2020Q4 latest-report state.
        ("000001.SZ", "2021-04-15", "2021-04-16", 2019 * 4 + 3, 80.0, 90.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "available_date",
            "activation_date",
            "report_ordinal",
            "is_revenue",
            "bs_total_assets",
        ],
    )
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame["activation_date"] = pd.to_datetime(frame["activation_date"])
    result = build_statement_snapshots(frame, ["is_revenue"], ["bs_total_assets"])

    q4 = result.loc[result["fund_statement_activation_date"].eq(pd.Timestamp("2021-03-31"))].iloc[0]
    assert q4["fund_is_revenue_ytd"] == 100.0
    assert q4["fund_is_revenue_sq"] == 40.0
    assert q4["fund_is_revenue_ttm"] == 100.0
    assert q4["fund_bs_total_assets_latest"] == 130.0

    late = result.iloc[-1]
    assert late["fund_report_year"] == 2020
    assert late["fund_report_quarter"] == 4
    assert late["fund_bs_total_assets_latest"] == 130.0


def test_ytd_annualized_ratios_are_explicit_and_do_not_replace_ttm() -> None:
    panel = pd.DataFrame(
        {
            "fund_report_quarter": [2],
            "market_value": [100.0],
            "fund_is_n_income_attr_p_ytd": [10.0],
            "fund_is_n_income_attr_p_ttm": [16.0],
            "fund_is_total_revenue_ytd": [50.0],
            "fund_is_total_revenue_ttm": [90.0],
            "fund_is_gross_profit_ytd": [20.0],
            "fund_bs_total_assets_latest": [200.0],
        }
    )

    result = _add_ratio_features(panel)

    assert result.loc[0, "fund_earnings_yield_ttm"] == 0.16
    assert result.loc[0, "fund_earnings_yield_ytd_ann"] == 0.20
    assert result.loc[0, "fund_gross_margin_ytd"] == 0.40


def test_daily_asof_join_never_backfills_before_activation() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-05-01", "2020-05-04", "2020-05-05"]),
            "symbol": ["000001.SZ"] * 3,
        }
    )
    events = pd.DataFrame(
        {
            "symbol": pd.Series(["000001.SZ"], dtype="string"),
            "activation_date": pd.to_datetime(["2020-05-04"]),
            "fund_value": [1.0],
        }
    )
    result = _merge_asof_by_symbol(
        daily,
        events,
        daily_date="date",
        event_date="activation_date",
    )
    assert np.isnan(result.loc[0, "fund_value"])
    assert result.loc[1:, "fund_value"].tolist() == [1.0, 1.0]
