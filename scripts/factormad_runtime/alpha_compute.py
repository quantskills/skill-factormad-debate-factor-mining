from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Callable

import pandas as pd

_PROCESS_ENGINE = None


def default_alpha_n_jobs() -> int:
    """Return the hardware maximum logical CPU count for alpha computation."""
    return max(1, os.cpu_count() or 1)


def build_alpha_output_frame(
    source_df: pd.DataFrame,
    alpha_results: dict[str, pd.DataFrame],
    alpha_order: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a wide alpha output frame using the original date/symbol grid."""
    base = source_df[["date", "symbol"]].copy()
    base["date"] = base["date"].astype(str)
    base["symbol"] = base["symbol"].astype(str)
    base = base.drop_duplicates().sort_values(["date", "symbol"]).reset_index(drop=True)
    base_index = pd.MultiIndex.from_frame(base[["date", "symbol"]])

    columns: dict[str, object] = {
        "date": base["date"].to_numpy(),
        "symbol": base["symbol"].to_numpy(),
    }
    nan_counts: dict[str, int] = {}
    ordered_names = alpha_order or sorted(alpha_results)
    for alpha_name in ordered_names:
        if alpha_name not in alpha_results:
            continue
        alpha_df = alpha_results[alpha_name]
        frame = alpha_df.copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        try:
            stacked = frame.stack(future_stack=True)
        except TypeError:
            stacked = frame.stack(dropna=False)
        stacked.index.names = ["date", "symbol"]
        values = stacked.reindex(base_index).round(8)
        columns[alpha_name] = values.to_numpy()
        nan_counts[alpha_name] = int(values.isna().sum())

    return pd.DataFrame(columns), nan_counts


def _compute_process_alpha(name: str) -> tuple[str, pd.DataFrame | None]:
    if _PROCESS_ENGINE is None:
        return name, None
    try:
        alpha_number = int(name.rsplit("_", 1)[1])
        method = getattr(_PROCESS_ENGINE, f"alpha{alpha_number:03d}", None)
        if method is None:
            return name, None
        value = method()
    except Exception:
        return name, None
    if value is None or isinstance(value, int):
        return name, None
    return name, value


def compute_alpha_methods(
    *,
    names: list[str],
    method_getter: Callable[[str], Callable[[], pd.DataFrame] | None],
    n_jobs: int | None = None,
    show_progress: bool = True,
    label: str = "alphas",
) -> dict[str, pd.DataFrame]:
    """Compute independent alpha methods with optional thread parallelism."""
    total = len(names)
    if total == 0:
        return {}

    workers = default_alpha_n_jobs() if n_jobs in (None, "") else max(1, int(n_jobs))
    results: dict[str, pd.DataFrame] = {}

    def _compute(name: str) -> tuple[str, pd.DataFrame | None]:
        method = method_getter(name)
        if method is None:
            return name, None
        try:
            value = method()
        except Exception:
            return name, None
        if value is None or isinstance(value, int):
            return name, None
        return name, value

    def _print_progress(done: int) -> None:
        if not show_progress or total <= 1:
            return
        width = 24
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{label}] |{bar}| {done}/{total}", end="", flush=True)
        if done == total:
            print(flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(_compute, name): name for name in names}
        for done, future in enumerate(as_completed(future_to_name), start=1):
            result_name, value = future.result()
            if value is not None:
                results[result_name] = value
            _print_progress(done)

    return results


def compute_engine_alpha_methods(
    *,
    engine,
    names: list[str],
    n_jobs: int | None = None,
    show_progress: bool = True,
    label: str = "alphas",
    backend: str = "process",
) -> dict[str, pd.DataFrame]:
    """Compute independent engine alpha methods, using forked workers by default."""
    global _PROCESS_ENGINE

    total = len(names)
    if total == 0:
        return {}

    workers = default_alpha_n_jobs() if n_jobs in (None, "") else max(1, int(n_jobs))
    workers = min(workers, total)
    results: dict[str, pd.DataFrame] = {}

    def _print_progress(done: int) -> None:
        if not show_progress or total <= 1:
            return
        width = 24
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{label}] |{bar}| {done}/{total}", end="", flush=True)
        if done == total:
            print(flush=True)

    def _collect(executor) -> dict[str, pd.DataFrame]:
        collected: dict[str, pd.DataFrame] = {}
        future_to_name = {executor.submit(_compute_process_alpha, name): name for name in names}
        for done, future in enumerate(as_completed(future_to_name), start=1):
            result_name, value = future.result()
            if value is not None:
                collected[result_name] = value
            _print_progress(done)
        return collected

    if backend == "process" and "fork" in mp.get_all_start_methods():
        _PROCESS_ENGINE = engine
        context = mp.get_context("fork")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                results = _collect(executor)
        finally:
            _PROCESS_ENGINE = None
        return results

    _PROCESS_ENGINE = engine
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return _collect(executor)
    finally:
        _PROCESS_ENGINE = None
