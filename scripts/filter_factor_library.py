#!/usr/bin/env python3
"""Create auditable curated subsets of a FactorMAD factor library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "ICIR",
    "RankICIR",
    "O-ICIR",
    "O-RankICIR",
    "O-HybridICIR",
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
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


def _load_library(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"factor library does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("factors")
    if not isinstance(payload, list):
        raise ValueError("factor library must be a JSON list or an object containing 'factors'")
    if not all(isinstance(factor, dict) for factor in payload):
        raise ValueError("every factor-library entry must be a JSON object")
    return payload


def _metric_values(factor: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    metric = factor.get("metric")
    if not isinstance(metric, dict):
        return {}, ["missing_metric_object"]

    values: dict[str, float] = {}
    reasons: list[str] = []
    for name in REQUIRED_METRICS:
        try:
            value = float(metric[name])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"missing_or_invalid_metric:{name}")
            continue
        if not math.isfinite(value):
            reasons.append(f"nonfinite_metric:{name}")
            continue
        values[name] = value
    return values, reasons


def evaluate_factor(
    factor: dict[str, Any],
    *,
    min_o_hybrid_icir: float,
    max_decay: float | None,
    require_direction_consistent: bool,
) -> dict[str, Any]:
    """Evaluate one factor without mutating its library record."""
    values, reasons = _metric_values(factor)
    pearson_retention = rank_retention = worst_retention = decay = None

    if str(factor.get("status") or "").strip().lower() != "effective":
        reasons.append("status_not_effective")

    if require_direction_consistent and factor.get("direction_consistent") is not True:
        reasons.append("direction_inconsistent")

    o_hybrid = values.get("O-HybridICIR")
    if o_hybrid is not None and o_hybrid < min_o_hybrid_icir:
        reasons.append("o_hybrid_icir_below_threshold")

    icir = values.get("ICIR")
    rank_icir = values.get("RankICIR")
    o_icir = values.get("O-ICIR")
    o_rank_icir = values.get("O-RankICIR")
    if None not in (icir, rank_icir, o_icir, o_rank_icir):
        assert icir is not None and rank_icir is not None
        assert o_icir is not None and o_rank_icir is not None
        if icir <= 0.0:
            reasons.append("nonpositive_insample_metric:ICIR")
        if rank_icir <= 0.0:
            reasons.append("nonpositive_insample_metric:RankICIR")
        if icir > 0.0 and rank_icir > 0.0:
            pearson_retention = o_icir / icir
            rank_retention = o_rank_icir / rank_icir
            worst_retention = min(pearson_retention, rank_retention)
            decay = 1.0 - worst_retention
            minimum_retention = None if max_decay is None else 1.0 - max_decay
            if (
                minimum_retention is not None
                and worst_retention < minimum_retention - 1e-12
            ):
                reasons.append("decay_above_threshold")

    reasons = list(dict.fromkeys(reasons))
    return {
        "name": factor.get("name"),
        "hash": factor.get("hash"),
        "kept": not reasons,
        "reasons": reasons,
        "metrics": {name: values.get(name) for name in REQUIRED_METRICS},
        "pearson_retention": pearson_retention,
        "rank_retention": rank_retention,
        "worst_retention": worst_retention,
        "decay": decay,
    }


def filter_factor_library(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    min_o_hybrid_icir: float = 0.2,
    max_decay: float | None = None,
    require_direction_consistent: bool = False,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    )
    if input_path == output_path:
        raise ValueError("output path must differ from the source library path")
    if manifest_path in {input_path, output_path}:
        raise ValueError("manifest path must differ from input and output paths")
    if not math.isfinite(min_o_hybrid_icir):
        raise ValueError("min_o_hybrid_icir must be finite")
    if max_decay is not None and (not math.isfinite(max_decay) or not 0.0 <= max_decay <= 1.0):
        raise ValueError("max_decay must be between 0 and 1")

    source_sha256 = _sha256_file(input_path)
    factors = _load_library(input_path)
    decisions = [
        evaluate_factor(
            factor,
            min_o_hybrid_icir=min_o_hybrid_icir,
            max_decay=max_decay,
            require_direction_consistent=require_direction_consistent,
        )
        for factor in factors
    ]
    kept = [factor for factor, decision in zip(factors, decisions) if decision["kept"]]
    _write_json(output_path, kept)

    reason_counts = Counter(
        reason
        for decision in decisions
        for reason in decision["reasons"]
    )
    minimum_retention = None if max_decay is None else 1.0 - max_decay
    manifest = {
        "schema_version": 1,
        "kind": "factormad_factor_library_filter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "sha256": source_sha256,
            "factor_count": len(factors),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "factor_count": len(kept),
        },
        "rules": {
            "min_o_hybrid_icir": min_o_hybrid_icir,
            "max_decay": max_decay,
            "minimum_retention": minimum_retention,
            "require_direction_consistent": require_direction_consistent,
            "inclusive_comparison": True,
            "pearson_retention_formula": "O-ICIR / ICIR",
            "rank_retention_formula": "O-RankICIR / RankICIR",
            "worst_retention_formula": "min(pearson_retention, rank_retention)",
            "decay_formula": "1 - worst_retention",
        },
        "counts": {
            "input": len(factors),
            "kept": len(kept),
            "rejected": len(factors) - len(kept),
            "rejection_reasons": dict(sorted(reason_counts.items())),
        },
        "decisions": decisions,
    }
    _write_json(manifest_path, manifest)
    if _sha256_file(input_path) != source_sha256:
        raise RuntimeError("source factor library changed while filtering")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered FactorMAD library and an auditable manifest."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source factor-library JSON")
    parser.add_argument("--output", required=True, type=Path, help="Filtered factor-library JSON")
    parser.add_argument("--manifest", type=Path, help="Audit manifest path")
    parser.add_argument("--min-o-hybrid-icir", type=float, default=0.2)
    parser.add_argument(
        "--max-decay",
        type=float,
        help="Maximum decay in [0, 1]; omit to apply no decay filter",
    )
    parser.add_argument("--require-direction-consistent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = filter_factor_library(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        min_o_hybrid_icir=args.min_o_hybrid_icir,
        max_decay=args.max_decay,
        require_direction_consistent=args.require_direction_consistent,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "source": manifest["source"],
                "output": manifest["output"],
                "rules": manifest["rules"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
