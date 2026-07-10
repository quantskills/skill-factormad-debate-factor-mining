#!/usr/bin/env python3
"""CLI entrypoint for FactorMAD debate factor mining.

The script accepts a JSON input file and writes auditable artifacts into an
output directory. Use ``dry_run=true`` in the input JSON to validate the skill
without calling an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from factormad_runtime.factormad_compute_runtime import (
    default_key_metric_for_evaluation_mode,
    normalize_evaluation_mode,
)
from factormad_runtime.factormad_debate_runtime import run_factormad_debate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_output_dir() -> str:
    return f"outputs/debate{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    return payload


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, default=str)
        file_obj.write("\n")
    return str(path)


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[FactorMAD] {message}", file=sys.stderr, flush=True)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _resolve_path(raw_path: str, *, input_path: Path) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [
        input_path.parent / path,
        REPO_ROOT / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((input_path.parent / path).resolve())


def _resolve_output_dir(raw_path: str, *, input_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _output_dir_from_args(input_path: Path, explicit_output: str | None) -> Path:
    input_path = input_path.expanduser().resolve()
    if explicit_output:
        return Path(explicit_output).expanduser()
    input_data = _load_json(input_path)
    configured_output = str(input_data.get("output_dir") or "").strip()
    if configured_output:
        return _resolve_output_dir(configured_output, input_path=input_path)
    return Path(_default_output_dir())


def _normalize_input_paths(input_data: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    normalized = dict(input_data)
    for key in (
        "market_data_csv_path",
        "seed_factors_json_path",
        "seed_alphas_json_path",
        "factormad_alpha_library_path",
        "alpha_library_path",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = _resolve_path(value.strip(), input_path=input_path)
    return normalized


def _metric_time_ranges(insample_time_range: list[str], outsample_time_range: list[str]) -> dict[str, dict[str, Any]]:
    insample = [str(insample_time_range[0]), str(insample_time_range[1])]
    outsample = [str(outsample_time_range[0]), str(outsample_time_range[1])]
    return {
        "IC": {"sample": "insample", "time_range": insample},
        "ICIR": {"sample": "insample", "time_range": insample},
        "RankIC": {"sample": "insample", "time_range": insample},
        "RankICIR": {"sample": "insample", "time_range": insample},
        "HybridICIR": {"sample": "insample", "time_range": insample},
        "O-IC": {"sample": "outsample", "time_range": outsample},
        "O-ICIR": {"sample": "outsample", "time_range": outsample},
        "O-RankIC": {"sample": "outsample", "time_range": outsample},
        "O-RankICIR": {"sample": "outsample", "time_range": outsample},
        "O-HybridICIR": {"sample": "outsample", "time_range": outsample},
    }


def _market_data_summary(market_data_csv_path: str) -> dict[str, Any]:
    frame = pd.read_csv(market_data_csv_path)
    required = {"date", "symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market data CSV missing required columns: {missing}")
    return {
        "market_data_csv_path": market_data_csv_path,
        "row_count": int(len(frame)),
        "symbol_count": int(frame["symbol"].astype(str).nunique()),
        "start_date": str(frame["date"].min()),
        "end_date": str(frame["date"].max()),
        "columns": list(frame.columns),
    }


def _dry_run(input_data: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    market_data_csv_path = str(input_data.get("market_data_csv_path", "")).strip()
    if not market_data_csv_path:
        raise ValueError("market_data_csv_path is required.")
    summary = _market_data_summary(market_data_csv_path)
    insample = input_data.get("insample_time_range", ["2020-01-01", "2021-12-31"])
    outsample = input_data.get("outsample_time_range", ["2022-01-01", "2022-12-31"])
    metric_ranges = _metric_time_ranges(insample, outsample)
    evaluation_mode = normalize_evaluation_mode(
        input_data.get("evaluation_mode"),
        legacy_key_metric=input_data.get("key_metric"),
    )
    key_metric = default_key_metric_for_evaluation_mode(evaluation_mode)
    legacy_threshold = float(input_data.get("factor_metric_threshold", 0.2))
    icir_threshold = float(input_data.get("icir_threshold", legacy_threshold))
    rank_icir_threshold = float(input_data.get("rank_icir_threshold", legacy_threshold))
    factor_code = (
        "def dry_run_close_to_vwap_gap(data, window=5):\n"
        "    eps = 1e-8\n"
        "    close = pd.to_numeric(data['close'], errors='coerce').astype(float)\n"
        "    vwap = pd.to_numeric(data['vwap'], errors='coerce').astype(float)\n"
        "    raw = close.div(vwap.abs() + eps).sub(1.0).replace([np.inf, -np.inf], np.nan)\n"
        "    return raw.groupby(level='symbol', sort=False).transform(lambda s: s.rolling(window, min_periods=2).mean())\n"
    )
    best_factor = {
        "name": "dry_run_close_to_vwap_gap",
        "code": factor_code,
        "entry_function": "dry_run_close_to_vwap_gap",
        "arguments": {"window": 5},
        "metric": {
            "IC": 0.0,
            "ICIR": 0.0,
            "RankIC": 0.0,
            "RankICIR": 0.0,
            "HybridICIR": 0.0,
            "O-IC": 0.0,
            "O-ICIR": 0.0,
            "O-RankIC": 0.0,
            "O-RankICIR": 0.0,
            "O-HybridICIR": 0.0,
        },
        "pearson_direction": 1,
        "rank_ic_direction": 1,
        "direction_consistent": True,
        "evaluation_mode": evaluation_mode,
        "insample_time_range": metric_ranges["IC"]["time_range"],
        "outsample_time_range": metric_ranges["O-IC"]["time_range"],
        "status": "dry_run",
        "source": "dry_run",
    }
    debate_rounds = [{
        "round": 0,
        "agent": "dry_run",
        "status": "dry_run",
        "view": "Dry run validates input, output files, and artifact contract without an LLM call.",
        "factor": best_factor,
    }]
    accepted_factors = [best_factor]
    rounds_path = _write_json(output_dir / "debate_rounds.json", debate_rounds)
    accepted_path = _write_json(output_dir / "accepted_factors.json", accepted_factors)
    result = {
        "ok": True,
        "dry_run": True,
        "debate_hash": "dry-run",
        "job_count": 0,
        "round_count": 0,
        "llm_fee": 0.0,
        "market_data_summary": summary,
        "debate_rounds": debate_rounds,
        "generated_factors": accepted_factors,
        "accepted_factors": accepted_factors,
        "invalid_factors": [],
        "accepted_count": 1,
        "invalid_count": 0,
        "best_factor": best_factor,
        "insample_time_range": best_factor["insample_time_range"],
        "outsample_time_range": best_factor["outsample_time_range"],
        "metric_time_ranges": metric_ranges,
        "evaluation_mode": evaluation_mode,
        "key_metric": key_metric,
        "icir_threshold": icir_threshold,
        "rank_icir_threshold": rank_icir_threshold,
        "debate_rounds_path": rounds_path,
        "accepted_factors_path": accepted_path,
    }
    result["debate_json_path"] = str(output_dir / "factormad_debate_result.json")
    _write_json(output_dir / "factormad_debate_result.json", result)
    return result


def _stable_result_paths(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    debate_rounds = result.get("debate_rounds", result.get("round_records", []))
    accepted_factors = result.get("accepted_factors", [])
    result = dict(result)
    result.pop("round_records", None)
    _progress(True, "writing stable output artifacts")
    result["debate_rounds_path"] = _write_json(output_dir / "debate_rounds.json", debate_rounds)
    result["accepted_factors_path"] = _write_json(output_dir / "accepted_factors.json", accepted_factors)
    result["debate_json_path"] = str(output_dir / "factormad_debate_result.json")
    _write_json(output_dir / "factormad_debate_result.json", result)
    return result


def run_from_paths(input_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_input_data = _load_json(input_path)
    _progress(True, f"input file: {input_path}")
    _progress(True, f"output directory: {output_dir}")

    input_data = _normalize_input_paths(raw_input_data, input_path=input_path)

    os.environ["FACTORMAD_OUTPUT_DIR"] = str(output_dir)
    os.environ["PANDA_SKILL_OUTPUT_DIR"] = str(output_dir)
    os.environ["FACTORMAD_STABLE_OUTPUTS"] = "1"

    if _to_bool(input_data.get("dry_run"), default=False):
        _progress(True, "dry run started")
        result = _dry_run(input_data, output_dir)
        _progress(True, f"dry run finished: accepted={result.get('accepted_count', 0)}")
        return result

    try:
        _progress(True, "real debate run started")
        result = run_factormad_debate(input_data)
        result = _stable_result_paths(result, output_dir)
        _progress(
            True,
            "real debate run finished: "
            f"ok={result.get('ok', False)} "
            f"rounds={result.get('round_count', 0)} "
            f"accepted={result.get('accepted_count', 0)} "
            f"invalid={result.get('invalid_count', 0)}",
        )
    except Exception as exc:
        _progress(True, f"run failed: {exc}")
        result = {
            "ok": False,
            "error": str(exc),
            "input_path": str(input_path),
            "output_dir": str(output_dir),
        }
        result["debate_json_path"] = str(output_dir / "factormad_debate_result.json")
        _write_json(output_dir / "factormad_debate_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FactorMAD debate factor mining.")
    parser.add_argument("--input", required=True, help="Path to a JSON input file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for result JSON artifacts. Overrides input JSON output_dir. Defaults to outputs/debateYYYYMMDDHHMMSS.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = _output_dir_from_args(input_path, args.output)
    result = run_from_paths(input_path, output_dir)
    print(json.dumps({
        "ok": result.get("ok", False),
        "dry_run": result.get("dry_run", False),
        "debate_json_path": result.get("debate_json_path"),
        "debate_rounds_path": result.get("debate_rounds_path"),
        "accepted_factors_path": result.get("accepted_factors_path"),
        "round_count": result.get("round_count", 0),
        "accepted_count": result.get("accepted_count", 0),
        "invalid_count": result.get("invalid_count", 0),
        "llm_fee": result.get("llm_fee", 0.0),
        "error": result.get("error"),
    }, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

