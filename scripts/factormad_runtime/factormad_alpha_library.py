from __future__ import annotations

import json
import math
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any


FACTORMAD_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "examples" / "factormad_alpha_library"
DEFAULT_FACTORMAD_LIBRARY_PATH = FACTORMAD_LIBRARY_DIR / "library.json"


def resolve_factormad_library_path(path: str | None = None) -> Path:
    raw_path = str(path or "").strip()
    if not raw_path:
        return DEFAULT_FACTORMAD_LIBRARY_PATH
    return Path(raw_path).expanduser().resolve()


def ensure_factormad_library(path: str | Path | None = None) -> Path:
    library_path = resolve_factormad_library_path(str(path) if path is not None else None)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    if not library_path.exists():
        library_path.write_text("[]\n", encoding="utf-8")
    return library_path


def _code_hash(code: str, arguments: dict[str, Any]) -> str:
    payload = code + json.dumps(arguments or {}, sort_keys=True, default=str)
    return md5(payload.encode("utf-8")).hexdigest()[:8]


def _factor_key(factor: dict[str, Any]) -> str:
    hash_value = str(factor.get("hash") or "").strip()
    if hash_value:
        return hash_value
    code = str(factor.get("factor_code") or factor.get("code") or "")
    args = factor.get("factor_arguments", factor.get("arguments", {}))
    if not isinstance(args, dict):
        args = {}
    return _code_hash(code, args)


def normalize_library_factor(factor: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(factor, dict):
        return None
    code = str(factor.get("factor_code") or factor.get("code") or "").strip()
    if not code:
        return None
    args = factor.get("factor_arguments", factor.get("arguments", {}))
    if not isinstance(args, dict):
        args = {}
    name = str(factor.get("name") or factor.get("entry_function") or "factormad_factor").strip()
    entry_function = factor.get("entry_function")
    normalized = {
        "name": name or "factormad_factor",
        "code": code,
        "entry_function": entry_function,
        "hash": str(factor.get("hash") or _code_hash(code, args)),
        "arguments": args,
        "metric": factor.get("metric", {}) if isinstance(factor.get("metric"), dict) else {},
        "status": str(factor.get("status") or "effective"),
        "source": str(factor.get("source") or "factormad"),
    }
    for key in (
        "debate",
        "round",
        "created_at",
        "updated_at",
        "workflow_run",
        "insample_time_range",
        "outsample_time_range",
        "market_data_csv_path",
        "market_data_path",
        "data_profile",
        "factor_mode",
        "referenced_fields",
        "market_data_manifest_path",
        "market_data_manifest_sha256",
        "label_config",
        "evaluation_mode",
        "pearson_direction",
        "rank_ic_direction",
        "direction_consistent",
    ):
        if key in factor:
            normalized[key] = factor[key]
    return normalized


def _is_finite_metric_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def has_finite_library_metrics(factor: dict[str, Any], *, required_metrics: tuple[str, ...] = ("IC", "ICIR")) -> bool:
    if not isinstance(factor, dict):
        return False
    metric = factor.get("metric")
    if not isinstance(metric, dict):
        return False
    return all(_is_finite_metric_value(metric.get(key)) for key in required_metrics)


def is_library_effective_factor(factor: dict[str, Any], *, exclude_seed: bool = True) -> bool:
    if not isinstance(factor, dict):
        return False
    name = str(factor.get("name") or "")
    if exclude_seed and name.startswith("seed_"):
        return False
    if str(factor.get("status") or "").strip().lower() != "effective":
        return False
    return has_finite_library_metrics(factor)


def load_factormad_library(
    path: str | Path | None = None,
    *,
    create_if_missing: bool = True,
) -> list[dict[str, Any]]:
    library_path = resolve_factormad_library_path(str(path) if path is not None else None)
    if create_if_missing:
        library_path = ensure_factormad_library(library_path)
    elif not library_path.is_file():
        raise FileNotFoundError(f"FactorMAD library not found: {library_path}")
    try:
        loaded = json.loads(library_path.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError:
        loaded = []
    if isinstance(loaded, dict):
        loaded = loaded.get("factors", [])
    if not isinstance(loaded, list):
        return []
    factors: list[dict[str, Any]] = []
    for item in loaded:
        normalized = normalize_library_factor(item)
        if normalized is not None and is_library_effective_factor(normalized):
            factors.append(normalized)
    return _dedupe_factors(factors)


def save_factormad_library(path: str | Path | None, factors: list[dict[str, Any]], *, key_metric: str = "ICIR") -> Path:
    library_path = ensure_factormad_library(path)
    normalized = [item for item in (normalize_library_factor(factor) for factor in factors) if item is not None]
    normalized = _dedupe_factors(normalized, key_metric=key_metric)
    normalized = sorted(
        normalized,
        key=lambda item: _metric_value(item, key_metric),
        reverse=True,
    )
    library_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return library_path


def _metric_value(factor: dict[str, Any], key_metric: str) -> float:
    try:
        value = float(factor.get("metric", {}).get(key_metric, float("-inf")))
    except Exception:
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _dedupe_factors(factors: list[dict[str, Any]], *, key_metric: str = "ICIR") -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for factor in factors:
        normalized = normalize_library_factor(factor)
        if normalized is None:
            continue
        key = _factor_key(normalized)
        current = by_key.get(key)
        if current is None or _metric_value(normalized, key_metric) > _metric_value(current, key_metric):
            by_key[key] = normalized
    return list(by_key.values())


def append_effective_factors_to_library(
    *,
    path: str | Path | None,
    factors: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    key_metric: str = "ICIR",
) -> dict[str, Any]:
    library_path = ensure_factormad_library(path)
    existing = load_factormad_library(library_path)
    before_count = len(existing)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = dict(metadata or {})
    incoming: list[dict[str, Any]] = []
    for factor in factors or []:
        normalized = normalize_library_factor(factor)
        if normalized is None or not is_library_effective_factor(normalized):
            continue
        normalized.update({k: v for k, v in metadata.items() if v not in (None, "")})
        normalized.setdefault("created_at", now)
        normalized["updated_at"] = now
        incoming.append(normalized)
    merged = _dedupe_factors(existing + incoming, key_metric=key_metric)
    save_factormad_library(library_path, merged, key_metric=key_metric)
    after = load_factormad_library(library_path)
    existing_keys = {_factor_key(factor) for factor in existing}
    added_count = sum(1 for factor in after if _factor_key(factor) not in existing_keys)
    return {
        "library_path": str(library_path),
        "before_count": before_count,
        "after_count": len(after),
        "candidate_count": len(incoming),
        "added_count": added_count,
    }
