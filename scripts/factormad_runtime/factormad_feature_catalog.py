from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


INDEX_COLUMNS = ("date", "symbol")
REQUIRED_MARKET_FIELDS = ("open", "high", "low", "close", "volume")
OPTIONAL_MARKET_FIELDS = ("vwap", "amount", "adjfactor")
MARKET_FIELDS = frozenset(REQUIRED_MARKET_FIELDS + OPTIONAL_MARKET_FIELDS)
DEFAULT_FUNDAMENTAL_GROUPS = (
    "market",
    "daily_fundamental",
    "value",
    "quality",
    "solvency",
    "investment",
    "events",
    "statement_flows",
    "statement_balances",
    "growth",
)
EXCLUDED_FIELD_SUFFIXES = ("_available_date", "_activation_date")
EXCLUDED_FIELDS = frozenset(
    {
        "fund_latest_report_available_date",
        "fund_report_ordinal",
        "fund_report_year",
        "fund_report_quarter",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def resolve_market_data_path(input_data: dict[str, Any]) -> Path:
    raw_path = str(
        input_data.get("market_data_path")
        or input_data.get("market_data_csv_path")
        or ""
    ).strip()
    if not raw_path:
        raise ValueError("market_data_path or market_data_csv_path is required.")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"market data path not found: {path}")
    return path


def _load_manifest(input_data: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    raw_path = str(input_data.get("market_data_manifest_path") or "").strip()
    if not raw_path:
        return {}, None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"market data manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("market data manifest must be a JSON object")
    return payload, path


def _coverage_maps(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    start_year: int,
    end_year: int,
) -> tuple[dict[str, float], dict[str, float]]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}

    def _artifact_path(value: Any) -> Path:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute() and manifest_path is not None:
            path = manifest_path.parent / path
        return path

    overall_path = _artifact_path(artifacts.get("field_coverage"))
    yearly_path = _artifact_path(artifacts.get("field_coverage_by_year"))
    overall: dict[str, float] = {}
    yearly_min: dict[str, float] = {}
    if overall_path.is_file():
        frame = pd.read_parquet(overall_path, columns=["field", "nonnull_rate"])
        overall = {
            str(row.field): float(row.nonnull_rate)
            for row in frame.itertuples(index=False)
            if pd.notna(row.nonnull_rate)
        }
    if yearly_path.is_file():
        frame = pd.read_parquet(yearly_path, columns=["year", "field", "nonnull_rate"])
        frame = frame.loc[frame["year"].between(start_year, end_year)]
        if not frame.empty:
            expected_years = set(range(start_year, end_year + 1))
            yearly_min = {
                str(field): min(
                    {
                        int(row.year): float(row.nonnull_rate)
                        for row in group.itertuples(index=False)
                        if pd.notna(row.nonnull_rate)
                    }.get(year, 0.0)
                    for year in expected_years
                )
                for field, group in frame.groupby("field")
            }
    return overall, yearly_min


def _field_is_metadata(field: str) -> bool:
    return field in EXCLUDED_FIELDS or field.endswith(EXCLUDED_FIELD_SUFFIXES)


def build_feature_catalog(
    input_data: dict[str, Any],
    available_columns: Iterable[str],
) -> dict[str, Any]:
    available = {str(column) for column in available_columns}
    if set(REQUIRED_MARKET_FIELDS).issubset(available):
        available.update({"amount", "vwap"})
    manifest, manifest_path = _load_manifest(input_data)
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    manifest_panel_path = str(artifacts.get("panel_root") or "").strip()
    if manifest_panel_path:
        declared_panel = Path(manifest_panel_path).expanduser()
        if not declared_panel.is_absolute() and manifest_path is not None:
            declared_panel = manifest_path.parent / declared_panel
        actual_panel = resolve_market_data_path(input_data)
        if declared_panel.resolve() != actual_panel:
            raise ValueError(
                "market_data_manifest_path describes a different panel_root: "
                f"{declared_panel.resolve()} != {actual_panel}"
            )
    manifest_groups = manifest.get("field_groups") if isinstance(manifest.get("field_groups"), dict) else {}
    requested_groups = _string_list(input_data.get("feature_groups"))
    if not requested_groups:
        requested_groups = list(DEFAULT_FUNDAMENTAL_GROUPS if manifest_groups else ("market",))
    unknown_groups = sorted(set(requested_groups) - set(manifest_groups) - {"market"})
    if unknown_groups and manifest_groups:
        raise ValueError(f"unknown feature_groups: {unknown_groups}")

    insample = input_data.get("insample_time_range", ["2020-01-01", "2021-12-31"])
    coverage_range = input_data.get("feature_coverage_time_range", insample)
    if not isinstance(coverage_range, list) or len(coverage_range) != 2:
        raise ValueError("feature_coverage_time_range must be a two-date list")
    start_year = pd.Timestamp(coverage_range[0]).year
    end_year = pd.Timestamp(coverage_range[1]).year
    overall_coverage, yearly_min_coverage = _coverage_maps(
        manifest,
        manifest_path=manifest_path,
        start_year=start_year,
        end_year=end_year,
    )
    min_coverage = float(input_data.get("min_feature_coverage", 0.0) or 0.0)
    min_yearly_coverage = float(input_data.get("min_yearly_coverage", 0.0) or 0.0)

    requested_fields: set[str] = set()
    resolved_groups: dict[str, list[str]] = {}
    if manifest_groups:
        for group in requested_groups:
            group_fields = _string_list(manifest_groups.get(group, []))
            resolved_groups[group] = group_fields
            requested_fields.update(group_fields)
    else:
        resolved_groups["market"] = sorted(MARKET_FIELDS & available)
        requested_fields.update(resolved_groups["market"])
    explicit_fields = set(_string_list(input_data.get("feature_columns")))
    requested_fields.update(explicit_fields)
    requested_fields.update(MARKET_FIELDS & available)

    included: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for field in sorted(requested_fields):
        reason = ""
        if field in INDEX_COLUMNS:
            reason = "index_column"
        elif _field_is_metadata(field):
            reason = "point_in_time_metadata"
        elif field not in available:
            reason = "missing_from_data"
        elif field.startswith("fund_") and overall_coverage.get(field, 1.0) < min_coverage:
            reason = "overall_coverage_below_threshold"
        elif (
            field.startswith("fund_")
            and min_yearly_coverage > 0.0
            and yearly_min_coverage.get(field, 0.0) < min_yearly_coverage
        ):
            reason = "yearly_coverage_below_threshold"
        if reason:
            excluded.append({"field": field, "reason": reason})
        else:
            included.add(field)

    market_fields = sorted(included & MARKET_FIELDS)
    fundamental_fields = sorted(included - MARKET_FIELDS)
    factor_mode = str(input_data.get("factor_mode") or "").strip().lower()
    if not factor_mode:
        factor_mode = "hybrid" if fundamental_fields else "ohlcv"
    if factor_mode not in {"ohlcv", "fundamental", "hybrid", "any"}:
        raise ValueError("factor_mode must be one of: ohlcv, fundamental, hybrid, any")
    if factor_mode in {"fundamental", "hybrid"} and not fundamental_fields:
        raise ValueError(f"factor_mode={factor_mode} requires at least one fundamental field")

    group_fields: dict[str, list[str]] = {}
    for group, fields in resolved_groups.items():
        selected = sorted(set(fields) & included)
        if selected:
            group_fields[group] = selected
    if market_fields:
        group_fields["market"] = market_fields
    explicit_selected = sorted(explicit_fields & included)
    if explicit_selected:
        group_fields["explicit"] = explicit_selected
    grouped = set().union(*group_fields.values()) if group_fields else set()
    ungrouped = sorted(included - grouped)
    if ungrouped:
        group_fields["other"] = ungrouped
    data_profile = str(input_data.get("data_profile") or "").strip()
    if not data_profile:
        data_profile = "ohlcv_fundamental_pit_v1" if fundamental_fields else "ohlcv_v1"

    point_in_time = manifest.get("point_in_time_contract", {})
    if fundamental_fields and manifest_path is None:
        raise ValueError("fundamental fields require market_data_manifest_path")
    if fundamental_fields and point_in_time.get("future_backfill") is not False:
        raise ValueError("fundamental manifest must assert future_backfill=false")
    return {
        "data_profile": data_profile,
        "factor_mode": factor_mode,
        "requested_groups": requested_groups,
        "source_columns": sorted(available),
        "field_groups": group_fields,
        "allowed_fields": sorted(included),
        "market_fields": market_fields,
        "fundamental_fields": fundamental_fields,
        "excluded_fields": excluded,
        "coverage": {field: overall_coverage.get(field) for field in sorted(included)},
        "yearly_min_coverage": {
            field: yearly_min_coverage.get(field) for field in sorted(included)
        },
        "coverage_years": [start_year, end_year],
        "min_feature_coverage": min_coverage,
        "min_yearly_coverage": min_yearly_coverage,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": _sha256(manifest_path) if manifest_path else None,
        "manifest_panel_path": str(Path(manifest_panel_path).expanduser()) if manifest_panel_path else None,
        "manifest_rows": manifest.get("rows"),
        "point_in_time_contract": point_in_time,
    }


def build_factor_requirement(input_data: dict[str, Any], catalog: dict[str, Any]) -> str:
    group_lines = [
        f"- {group}: {', '.join(fields)}"
        for group, fields in catalog.get("field_groups", {}).items()
    ]
    mode = catalog["factor_mode"]
    mode_rule = {
        "ohlcv": "Use only market/OHLCV fields.",
        "fundamental": "Use at least one fundamental field.",
        "hybrid": "Use at least one market/OHLCV field and at least one fundamental field.",
        "any": "Use any allowed fields that support a coherent economic hypothesis.",
    }[mode]
    lines = [
        "Design an interpretable daily China A-share factor to predict future cross-sectional returns.",
        "Use pandas/numpy only, return a pandas Series aligned to the input MultiIndex (time, symbol), and use direct data[\"field\"] access.",
        "Use only information available at timestamp t or earlier. Do not use negative shift/diff/pct_change, backward fill, or centered rolling windows.",
        "Protect every denominator, replace final +/-inf with NaN, and make values dimensionless or cross-sectionally comparable.",
        mode_rule,
        "Allowed fields by group:",
        *group_lines,
    ]
    if catalog.get("fundamental_fields"):
        lines.extend(
            [
                "Fundamental fields are point-in-time, forward-filled states that change only after their activation trading date; they are not newly published every day.",
                "Suffix semantics: _ytd is fiscal-year cumulative; _ytd_ann is simple annualization and is not true TTM; _ttm is strict trailing twelve months; _sq is single-quarter; _yoy is year-over-year; _latest is the latest disclosed balance.",
                "Do not infer information from excluded raw available/activation date columns.",
            ]
        )
    user_requirement = str(input_data.get("factor_requirement") or "").strip()
    if user_requirement:
        lines.extend(["Additional user requirements:", user_requirement])
    return "\n".join(lines)


def extract_referenced_fields(code: str, allowed_fields: Iterable[str]) -> set[str]:
    tree = ast.parse(code)
    allowed = set(allowed_fields)
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            slices: list[ast.AST]
            if isinstance(node.slice, (ast.List, ast.Tuple)):
                slices = list(node.slice.elts)
            else:
                slices = [node.slice]
            for item in slices:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    if item.value in allowed:
                        referenced.add(item.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "data"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in allowed
        ):
            referenced.add(node.args[0].value)
    return referenced


def _numeric_value(node: ast.AST, arguments: dict[str, Any]) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        value = arguments.get(node.id)
        return float(value) if isinstance(value, (int, float)) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _numeric_value(node.operand, arguments)
        return -value if value is not None else None
    return None


def analyze_factor_code(
    code: str,
    arguments: dict[str, Any],
    catalog: dict[str, Any],
    *,
    enforce_mode: bool,
) -> tuple[set[str], list[dict[str, str]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return set(), [{"status": "EXEC", "message": str(exc)}]
    allowed = set(catalog.get("allowed_fields", []))
    referenced = extract_referenced_fields(code, allowed)
    violations: list[dict[str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "data":
            slices = list(node.slice.elts) if isinstance(node.slice, (ast.List, ast.Tuple)) else [node.slice]
            for item in slices:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    violations.append(
                        {"status": "KEY", "message": "data fields must use literal string access"}
                    )
                elif item.value not in allowed:
                    violations.append(
                        {"status": "KEY", "message": f"unknown data field: {item.value}"}
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "data"
            and node.args
        ):
            field_node = node.args[0]
            if not isinstance(field_node, ast.Constant) or not isinstance(field_node.value, str):
                violations.append(
                    {"status": "KEY", "message": "data.get fields must be literal strings"}
                )
            elif field_node.value not in allowed:
                violations.append(
                    {"status": "KEY", "message": f"unknown data field: {field_node.value}"}
                )
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method in {"bfill", "backfill"}:
            violations.append({"status": "FUTURE", "message": "bfill uses future observations"})
        if method in {"shift", "diff", "pct_change"}:
            period_node = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg in {"periods", "n"}:
                    period_node = keyword.value
            value = _numeric_value(period_node, arguments) if period_node is not None else 1.0
            if value is not None and value < 0:
                violations.append(
                    {"status": "FUTURE", "message": f"negative {method} uses future observations"}
                )
        if method == "rolling":
            for keyword in node.keywords:
                if keyword.arg == "center" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    violations.append(
                        {"status": "FUTURE", "message": "centered rolling uses future observations"}
                    )
        if method == "fillna":
            for keyword in node.keywords:
                if keyword.arg == "method" and isinstance(keyword.value, ast.Constant) and keyword.value.value in {"bfill", "backfill"}:
                    violations.append(
                        {"status": "FUTURE", "message": "backward fill uses future observations"}
                    )
        if method == "interpolate":
            for keyword in node.keywords:
                if (
                    keyword.arg == "limit_direction"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value in {"backward", "both"}
                ):
                    violations.append(
                        {"status": "FUTURE", "message": "backward interpolation uses future observations"}
                    )
        if (
            method == "roll"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ):
            shift_node = node.args[1] if len(node.args) > 1 else None
            for keyword in node.keywords:
                if keyword.arg == "shift":
                    shift_node = keyword.value
            value = _numeric_value(shift_node, arguments) if shift_node is not None else None
            if value is not None and value < 0:
                violations.append(
                    {"status": "FUTURE", "message": "negative np.roll uses future observations"}
                )

    if not referenced:
        violations.append(
            {"status": "KEY", "message": "factor code does not directly reference an allowed data field"}
        )
    if enforce_mode:
        market = referenced & set(catalog.get("market_fields", []))
        fundamental = referenced & set(catalog.get("fundamental_fields", []))
        mode = catalog.get("factor_mode", "any")
        if mode == "ohlcv" and fundamental:
            violations.append({"status": "KEY", "message": "ohlcv mode forbids fundamental fields"})
        elif mode == "fundamental" and not fundamental:
            violations.append({"status": "KEY", "message": "fundamental mode requires a fundamental field"})
        elif mode == "hybrid" and (not market or not fundamental):
            violations.append(
                {"status": "KEY", "message": "hybrid mode requires market and fundamental fields"}
            )
    return referenced, violations
