from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_factor_library import filter_factor_library  # noqa: E402


def _factor(
    name: str,
    *,
    icir: float = 0.4,
    rank_icir: float = 0.5,
    o_icir: float = 0.32,
    o_rank_icir: float = 0.4,
    o_hybrid_icir: float | None = None,
    direction_consistent: bool = True,
    status: str = "effective",
) -> dict[str, object]:
    return {
        "name": name,
        "hash": hashlib.md5(name.encode()).hexdigest()[:8],
        "code": f"def {name}(data):\n    return data['close']\n",
        "arguments": {},
        "metric": {
            "ICIR": icir,
            "RankICIR": rank_icir,
            "O-ICIR": o_icir,
            "O-RankICIR": o_rank_icir,
            "O-HybridICIR": min(o_icir, o_rank_icir)
            if o_hybrid_icir is None
            else o_hybrid_icir,
        },
        "direction_consistent": direction_consistent,
        "status": status,
    }


def test_filter_applies_o_hybrid_decay_and_direction_without_mutating_source(tmp_path: Path) -> None:
    factors = [
        _factor("keep_at_inclusive_boundary"),
        _factor("reject_o_hybrid", o_icir=0.19, o_rank_icir=0.4),
        _factor("reject_decay", o_icir=0.3, o_rank_icir=0.39),
        _factor("reject_direction", direction_consistent=False),
        _factor("reject_status", status="similar"),
    ]
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "curated.json"
    manifest_path = tmp_path / "curated.manifest.json"
    input_path.write_text(json.dumps(factors, indent=2) + "\n", encoding="utf-8")
    source_bytes = input_path.read_bytes()

    manifest = filter_factor_library(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        min_o_hybrid_icir=0.2,
        max_decay=0.2,
        require_direction_consistent=True,
    )

    assert input_path.read_bytes() == source_bytes
    assert json.loads(output_path.read_text(encoding="utf-8")) == [factors[0]]
    assert manifest["counts"]["input"] == 5
    assert manifest["counts"]["kept"] == 1
    assert manifest["counts"]["rejected"] == 4
    assert manifest["counts"]["rejection_reasons"] == {
        "decay_above_threshold": 2,
        "direction_inconsistent": 1,
        "o_hybrid_icir_below_threshold": 1,
        "status_not_effective": 1,
    }
    assert manifest["decisions"][0]["worst_retention"] == pytest.approx(0.8)
    assert manifest["decisions"][0]["decay"] == pytest.approx(0.2)
    assert manifest["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["output"]["factor_count"] == 1


def test_filter_can_create_o_hybrid_only_quality_tier(tmp_path: Path) -> None:
    factors = [
        _factor("stable"),
        _factor("large_decay_but_high_oos", icir=1.0, rank_icir=1.0, o_icir=0.25, o_rank_icir=0.3),
        _factor("weak_oos", o_icir=0.1, o_rank_icir=0.1),
    ]
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "quality.json"
    input_path.write_text(json.dumps(factors), encoding="utf-8")

    manifest = filter_factor_library(
        input_path=input_path,
        output_path=output_path,
        min_o_hybrid_icir=0.2,
        require_direction_consistent=True,
    )

    assert [factor["name"] for factor in json.loads(output_path.read_text())] == [
        "stable",
        "large_decay_but_high_oos",
    ]
    assert manifest["rules"]["max_decay"] is None
    assert manifest["rules"]["minimum_retention"] is None
    assert manifest["counts"]["kept"] == 2


def test_filter_rejects_nonpositive_insample_denominator(tmp_path: Path) -> None:
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "quality.json"
    input_path.write_text(json.dumps([_factor("bad", rank_icir=0.0)]), encoding="utf-8")

    manifest = filter_factor_library(
        input_path=input_path,
        output_path=output_path,
        min_o_hybrid_icir=0.2,
        max_decay=0.2,
    )

    assert json.loads(output_path.read_text()) == []
    assert manifest["counts"]["rejection_reasons"] == {
        "nonpositive_insample_metric:RankICIR": 1
    }
