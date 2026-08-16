from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from factormad_debate_factor import _dry_run  # noqa: E402
from factormad_runtime.alpha_ops import (  # noqa: E402
    inspect_market_data_source,
    load_market_data_frame,
)
from factormad_runtime.factormad_compute_runtime import (  # noqa: E402
    LightweightFactorEvaluator,
    prepare_factormad_market_data,
)
from factormad_runtime.factormad_alpha_library import load_factormad_library  # noqa: E402
from factormad_runtime.factormad_feature_catalog import (  # noqa: E402
    analyze_factor_code,
    build_factor_requirement,
    build_feature_catalog,
)


def _panel(tmp_path: Path) -> tuple[Path, Path]:
    panel_root = tmp_path / "panel" / "schema=v1"
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(pd.date_range("2021-01-04", periods=12, freq="B")):
        for symbol_index, symbol in enumerate(("000001.SZ", "000002.SZ", "000003.SZ")):
            close = 10.0 + date_index * 0.1 + symbol_index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0 + symbol_index,
                    "amount": close * (1000.0 + symbol_index),
                    "vwap": close,
                    "fund_book_to_market": 0.4 + symbol_index * 0.1,
                    "fund_sparse_ttm": np.nan if date_index else 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    partition = panel_root / "year=2021" / "month=1"
    partition.mkdir(parents=True)
    frame.to_parquet(partition / "data.parquet", index=False)

    overall_path = tmp_path / "field_coverage.parquet"
    yearly_path = tmp_path / "field_coverage_by_year.parquet"
    pd.DataFrame(
        {
            "field": ["fund_book_to_market", "fund_sparse_ttm"],
            "nonnull_rate": [1.0, 1.0 / 12.0],
        }
    ).to_parquet(overall_path, index=False)
    pd.DataFrame(
        {
            "year": [2021, 2021],
            "field": ["fund_book_to_market", "fund_sparse_ttm"],
            "nonnull_rate": [1.0, 1.0 / 12.0],
        }
    ).to_parquet(yearly_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rows": len(frame),
                "symbols": 3,
                "date_range": {
                    "panel_start": "2021-01-04",
                    "panel_end": "2021-01-19",
                },
                "point_in_time_contract": {
                    "activation_policy": "next_trading_day",
                    "fill_policy": "backward_asof_within_symbol_only",
                    "future_backfill": False,
                },
                "field_groups": {
                    "market": [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                        "vwap",
                    ],
                    "value": ["fund_book_to_market", "fund_sparse_ttm"],
                },
                "artifacts": {
                    "field_coverage": overall_path.name,
                    "field_coverage_by_year": yearly_path.name,
                },
            }
        ),
        encoding="utf-8",
    )
    return panel_root, manifest_path


def _hybrid_input(tmp_path: Path) -> dict[str, object]:
    panel_root, manifest_path = _panel(tmp_path)
    return {
        "market_data_path": str(panel_root),
        "market_data_manifest_path": str(manifest_path),
        "market_data_load_range": ["2021-01-04", "2021-01-19"],
        "insample_time_range": ["2021-01-04", "2021-01-13"],
        "outsample_time_range": ["2021-01-14", "2021-01-19"],
        "feature_groups": ["market", "value"],
        "factor_mode": "hybrid",
        "min_feature_coverage": 0.2,
        "min_yearly_coverage": 0.2,
        "label_config": {"price": "vwap", "span": 1},
    }


def test_csv_loader_remains_backward_compatible(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "date": ["2021-01-04", "2021-01-05"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.1],
        }
    ).to_csv(path, index=False)
    result = load_market_data_frame(
        {"market_data_csv_path": str(path)},
        columns=["close"],
        start_date="2021-01-05",
    )
    assert list(result.columns) == ["date", "symbol", "close"]
    assert result["date"].tolist() == ["2021-01-05"]


def test_csv_loader_parses_integer_yyyymmdd(tmp_path: Path) -> None:
    path = tmp_path / "integer_dates.csv"
    pd.DataFrame(
        {
            "date": [20210104, 20210105],
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.1],
        }
    ).to_csv(path, index=False)
    result = load_market_data_frame({"market_data_csv_path": str(path)}, columns=["close"])
    assert result["date"].tolist() == ["2021-01-04", "2021-01-05"]


def test_partitioned_parquet_projects_columns_and_dates(tmp_path: Path) -> None:
    input_data = _hybrid_input(tmp_path)
    summary = inspect_market_data_source(input_data)
    assert summary["partitioned"] is True
    assert summary["row_count"] == 36
    result = load_market_data_frame(
        input_data,
        columns=["close", "fund_book_to_market"],
        start_date="2021-01-08",
        end_date="2021-01-11",
    )
    assert set(result.columns) == {
        "date",
        "symbol",
        "close",
        "fund_book_to_market",
    }
    assert result["date"].min() == "2021-01-08"
    assert result["date"].max() == "2021-01-11"


def test_catalog_filters_sparse_fields_and_documents_pit_semantics(tmp_path: Path) -> None:
    input_data = _hybrid_input(tmp_path)
    summary = inspect_market_data_source(input_data)
    catalog = build_feature_catalog(input_data, summary["columns"])
    assert "fund_book_to_market" in catalog["fundamental_fields"]
    assert "fund_sparse_ttm" not in catalog["fundamental_fields"]
    assert catalog["point_in_time_contract"]["future_backfill"] is False
    requirement = build_factor_requirement(input_data, catalog)
    assert "_ytd_ann is simple annualization and is not true TTM" in requirement
    assert "Use at least one market/OHLCV field and at least one fundamental field." in requirement


def test_fundamental_fields_require_pit_manifest() -> None:
    input_data = {
        "factor_mode": "hybrid",
        "feature_columns": ["fund_book_to_market"],
    }
    columns = ["date", "symbol", "open", "high", "low", "close", "volume", "fund_book_to_market"]
    try:
        build_feature_catalog(input_data, columns)
    except ValueError as exc:
        assert "market_data_manifest_path" in str(exc)
    else:
        raise AssertionError("fundamental catalog accepted data without a PIT manifest")


def test_static_analysis_enforces_hybrid_mode_and_future_safety(tmp_path: Path) -> None:
    input_data = _hybrid_input(tmp_path)
    summary = inspect_market_data_source(input_data)
    catalog = build_feature_catalog(input_data, summary["columns"])
    hybrid_code = (
        "def hybrid(data):\n"
        "    return data['close'] * data['fund_book_to_market']\n"
    )
    referenced, violations = analyze_factor_code(
        hybrid_code,
        {},
        catalog,
        enforce_mode=True,
    )
    assert referenced == {"close", "fund_book_to_market"}
    assert violations == []

    _, missing_domain = analyze_factor_code(
        "def market_only(data):\n    return data['close']\n",
        {},
        catalog,
        enforce_mode=True,
    )
    assert any(item["status"] == "KEY" for item in missing_domain)

    _, future = analyze_factor_code(
        "def future(data):\n    return data['close'].shift(-1) * data['fund_book_to_market']\n",
        {},
        catalog,
        enforce_mode=True,
    )
    assert any(item["status"] == "FUTURE" for item in future)

    _, unknown = analyze_factor_code(
        "def unknown(data):\n    return data.get('not_a_field') * data['close']\n",
        {},
        catalog,
        enforce_mode=True,
    )
    assert any("unknown data field" in item["message"] for item in unknown)


def test_evaluator_lazy_loads_only_referenced_columns(tmp_path: Path) -> None:
    input_data = _hybrid_input(tmp_path)
    raw_data, catalog, _ = prepare_factormad_market_data(input_data)
    assert "fund_book_to_market" not in raw_data.columns
    evaluator = LightweightFactorEvaluator(
        raw_data,
        ("2021-01-04", "2021-01-13"),
        ("2021-01-14", "2021-01-19"),
        {"price": "vwap", "span": 1},
        feature_catalog=catalog,
        market_data_input=input_data,
    )
    code = (
        "def hybrid(data):\n"
        "    assert set(data.columns) == {'close', 'fund_book_to_market'}\n"
        "    return data['close'] * data['fund_book_to_market']\n"
    )
    value = evaluator.calc(code, {})
    assert value.notna().all()
    assert len(value) == 36


def test_hybrid_dry_run_records_data_contract(tmp_path: Path) -> None:
    input_data = _hybrid_input(tmp_path)
    output_dir = tmp_path / "output"
    result = _dry_run(input_data, output_dir)
    assert result["ok"] is True
    assert result["feature_catalog"]["factor_mode"] == "hybrid"
    assert "fund_book_to_market" in result["best_factor"]["referenced_fields"]
    assert result["best_factor"]["market_data_manifest_sha256"]
    assert (output_dir / "factormad_debate_result.json").is_file()


def test_read_only_library_does_not_create_missing_file(tmp_path: Path) -> None:
    library_path = tmp_path / "missing_library.json"
    try:
        load_factormad_library(library_path, create_if_missing=False)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing read-only library did not raise")
    assert not library_path.exists()
