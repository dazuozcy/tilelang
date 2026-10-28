"""do_bench(msprof) wrapper — the TileOPs msprof-mode latency entry point.

New contract (Ascend A5):

- End-to-end kernel latency comes exclusively from
  ``tilelang.profiler.do_bench(fn, backend="msprof")`` (milliseconds), the
  repository's authoritative latency source.  This module is a thin
  wrapper around it: no profiling subprocess script generation and no
  ``OpBasicInfo*.csv`` parsing.  Single-kernel latency scope (the pre-A5
  ``--kernel-name`` convention) is restored via ``TILEOPS_MSPROF_KERNEL_NAME``,
  forwarded to ``do_bench(..., kernel_name=...)``.
- The wrapper packages the measured call as ``functools.partial(functor,
  *args)`` (the bare functor when *args* is empty) and delegates timing
  to :func:`tilelang.profiler.do_bench`.

Environment mapping (semantics preserved from the previous implementation):

- ``TILEOPS_MSPROF_LAUNCH_COUNT`` → ``do_bench(..., _n_repeat=...)``
  (default ``0`` = auto).
- ``TILEOPS_MSPROF_WARM_UP`` → ``do_bench(..., _n_warmup=...)``
  (default ``0`` = auto).
- ``TILEOPS_PROF_MODE`` (``msprof`` | ``events``) is handled at the
  :mod:`tileops._perf.benchmark_base` dispatch layer.

Cycle-domain attribution (report auxiliary metrics):

- :func:`profile_kernel_msprof_detail` wraps
  ``do_bench(fn, backend="msprof_detail")``, which returns a per-kernel
  ``KernelProfile`` (``dur_ns``, AIC/AIV total cycles, per-pipe busy
  ratios in [0, 1]).  It provides the **bottleneck pipe utilization**
  (busiest pipe's busy ratio × 100) — a purely measured cycle-domain
  metric with no clock-frequency or theoretical-peak assumptions.
- The report's primary ``Ratio(%)`` (roofline performance ratio, computed
  by the CANN msprof tool's internal SoC model) comes from the separate
  ``msprof op --aic-metrics=Roofline`` attribution pass in
  :mod:`tileops._perf.msprof_roofline`; the pipe utilization here is
  the fallback source when that pass is disabled or unsupported.
- ``TILEOPS_MSPROF_DETAIL_REPEAT`` → ``_n_repeat`` for the detail pass
  (default ``5``).  ``TILEOPS_MSPROF_DETAIL=0`` disables the detail pass
  (handled in :mod:`tileops._perf.benchmark_base`).

Deprecated environment variables (read but ignored, each logged once per
call): ``TILEOPS_MSPROF_OUTPUT_DIR``, ``TILEOPS_MSPROF_KEEP_OUTPUT`` —
latency is uniformly measured via do_bench(msprof), so persistent
profiling output directories no longer apply.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable

from tilelang.profiler import do_bench

__all__ = ["MSPROF_RESULT_METADATA", "bench_kernel_msprof", "profile_kernel_msprof_detail"]

_logger = logging.getLogger("tileops._perf.msprof")


#: Minimal per-case msprof metadata attached to benchmark results measured
#: in msprof mode.  This is the schema consumed by the reporting layer.
MSPROF_RESULT_METADATA: dict[str, str] = {
    "backend": "msprof",
    "latency_source": "do_bench(msprof)",
}

_DEPRECATED_ENV_VARS = (
    "TILEOPS_MSPROF_OUTPUT_DIR",
    "TILEOPS_MSPROF_KEEP_OUTPUT",
)


def _env_kernel_name() -> str | None:
    """Return the ``TILEOPS_MSPROF_KERNEL_NAME`` override, or None when unset.

    Restores the pre-A5 ``--kernel-name`` single-kernel latency scope: when
    set, ``do_bench(msprof)`` reports only the matching kernel instead of the
    whole callable (op-layer permute/contiguous excluded for multi-dim reduce).
    """
    return os.environ.get("TILEOPS_MSPROF_KERNEL_NAME") or None


def _warn_deprecated_env() -> None:
    """Log one line per deprecated env var that is still set, then ignore it."""
    for var in _DEPRECATED_ENV_VARS:
        if os.environ.get(var):
            _logger.warning("%s 已废弃，时延统一走 do_bench(msprof)；该配置将被忽略。", var)


def _env_int(name: str, default: int = 0) -> int:
    """Read an integer env var, falling back to *default* on garbage values."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _logger.warning("%s=%r 不是合法整数，使用默认值 %d。", name, raw, default)
        return default


def bench_kernel_msprof(
    functor: Callable,
    args: tuple = (),
    *,
    warmup: float = 25.0,
    rep: float = 100.0,
    return_mode: str = "mean",
) -> tuple[float, str | None]:
    """Benchmark *functor* via ``do_bench(fn, backend="msprof")``.

    Args:
        functor: The callable (typically an Op instance or kernel entry)
            to measure.
        args: Tuple of positional arguments forwarded to *functor*.  When
            non-empty, the measured callable is
            ``functools.partial(functor, *args)``; when empty, *functor*
            itself is passed through untouched.
        warmup: Target warmup budget in milliseconds (do_bench default).
        rep: Target total measurement budget in milliseconds (do_bench
            default).
        return_mode: Aggregation over the measured repeats
            (``"mean"``/``"median"``/``"min"``/``"max"``).

    Returns:
        ``(latency_ms, prof_output_dir)`` where *latency_ms* is the
        do_bench(msprof) latency in milliseconds and *prof_output_dir*
        is always ``None`` (the do_bench wrapper does not expose a
        persistent profiling output directory).

    Raises:
        TypeError: If *args* is not a tuple, or propagated from
            :func:`tilelang.profiler.do_bench` on failure.
    """
    if not isinstance(args, tuple):
        raise TypeError(f"bench_kernel_msprof expects a tuple of args, got {type(args).__name__}.")

    _warn_deprecated_env()

    n_warmup = _env_int("TILEOPS_MSPROF_WARM_UP", 0)
    n_repeat = _env_int("TILEOPS_MSPROF_LAUNCH_COUNT", 0)

    fn = functools.partial(functor, *args) if args else functor
    latency_ms = do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        _n_warmup=n_warmup,
        _n_repeat=n_repeat,
        backend="msprof",
        return_mode=return_mode,
        kernel_name=_env_kernel_name(),
    )
    return float(latency_ms), None


#: Pipe fields of the msprof_detail ``KernelProfile`` (ratios in [0, 1]).
_DETAIL_PIPE_FIELDS = (
    "aic_mad",
    "aic_scalar",
    "aic_mte1",
    "aic_mte2",
    "aic_mte3",
    "aic_fixpipe",
    "aiv_vec",
    "aiv_scalar",
    "aiv_mte2",
    "aiv_mte3",
)


def profile_kernel_msprof_detail(
    functor: Callable,
    args: tuple = (),
    *,
    n_repeat: int | None = None,
) -> dict[str, Any]:
    """Collect cycle-domain pipe utilization via ``do_bench(msprof_detail)``.

    Wraps ``do_bench(fn, backend="msprof_detail")``, whose return value is
    the profiler's per-kernel ``KernelProfile`` (``dur_ns``, AIC/AIV total
    cycles, per-pipe busy ratios relative to each side's total cycles).
    Returns a JSON-serializable dict; the report's ``Ratio(%)`` is the
    ``bottleneck_ratio`` × 100 (busiest pipe's busy ratio).

    Args:
        functor: The callable to profile (Op instance or kernel entry).
        args: Positional arguments forwarded to *functor* (packaged as a
            ``functools.partial`` when non-empty).
        n_repeat: Measured iterations inside the profiling window.  When
            ``None``, ``TILEOPS_MSPROF_DETAIL_REPEAT`` applies (default 5).

    Returns:
        Dict with ``dur_us``, ``aiv_total_cycles``, ``aic_total_cycles``,
        ``pipes`` (per-pipe busy ratios), ``bottleneck_pipe`` and
        ``bottleneck_ratio``.

    Raises:
        TypeError: If *args* is not a tuple, or propagated from do_bench.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"profile_kernel_msprof_detail expects a tuple of args, got {type(args).__name__}."
        )

    if n_repeat is None:
        n_repeat = _env_int("TILEOPS_MSPROF_DETAIL_REPEAT", 5)

    fn = functools.partial(functor, *args) if args else functor
    profile = do_bench(
        fn, backend="msprof_detail", _n_repeat=max(1, int(n_repeat)), kernel_name=_env_kernel_name()
    )

    pipes = {field: float(getattr(profile, field, 0.0) or 0.0) for field in _DETAIL_PIPE_FIELDS}
    bottleneck_pipe = max(pipes, key=pipes.get)
    return {
        "dur_us": float(profile.dur_ns) / 1000.0,
        "aiv_total_cycles": int(getattr(profile, "aiv_total_cycles", 0) or 0),
        "aic_total_cycles": int(getattr(profile, "aic_total_cycles", 0) or 0),
        "pipes": pipes,
        "bottleneck_pipe": bottleneck_pipe,
        "bottleneck_ratio": pipes[bottleneck_pipe],
    }
