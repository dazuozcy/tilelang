"""Benchmark base class — device-agnostic timing and reporting.

Timing backends:

**msprof mode** (default): :func:`tileops._perf.msprof.bench_kernel_msprof`
wraps ``tilelang.profiler.do_bench(fn, backend="msprof")`` — the single
authoritative end-to-end latency source (milliseconds).  No profiling
subprocess or CSV parsing is involved; optional single-kernel scope is
available via ``TILEOPS_MSPROF_KERNEL_NAME``.

**events mode**: :func:`bench_kernel` uses device Event timing
(``torch.npu.Event`` / ``torch.cuda.Event``) — includes launch overhead
(~50-60us per call) but is the standard fallback when msprof is
unavailable or fails.

Theoretical utilization metrics (TFLOPs, TB/s) are derived by dividing
the manifest-declared FLOP/byte formulas by the measured latency;
cycle-level attribution is out of scope here (it belongs to
``do_bench(..., backend="msprof_detail")`` collection during tuning).

Other preserved behaviors:

- Input tensor cloning per iteration (address diversity) in events mode.
- ``workloads_to_params`` and ``ManifestBenchmark`` read from the
  standalone manifest.
- ``_get_env_metadata`` queries the device backend (``npu-smi`` for NPU,
  ``nvidia-smi`` for CUDA).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

import pytest
import torch

from tileops.backend.device import get_device_backend
from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)

try:
    from tileops._perf.msprof import (
        MSPROF_RESULT_METADATA,
        bench_kernel_msprof,
        profile_kernel_msprof_detail,
    )
    from tileops._perf.msprof_roofline import collect_roofline_msprof
except ImportError:
    bench_kernel_msprof = None  # type: ignore[assignment]
    profile_kernel_msprof_detail = None  # type: ignore[assignment]
    collect_roofline_msprof = None  # type: ignore[assignment]
    MSPROF_RESULT_METADATA = {"backend": "msprof", "latency_source": "do_bench(msprof)"}

W = TypeVar("W")

_logger = logging.getLogger("tileops._perf")

_bench_results = threading.local()

# Timestamp fixed at first use within a process (reset by
# ``BenchmarkReport.clear()``) so that per-op appends during a run and the
# session-finish report dump land in the same log file.
_profile_log_stamp: Optional[str] = None
_profile_log_stamp_lock = threading.Lock()


def profile_run_log_path(prof_mode: Optional[str] = None) -> str:
    """Return the session-stable profile log path.

    Format: ``profile_run_{prof_mode}_{YYYYmmdd_HHMMSS}.log`` — the
    profiling mode (``msprof`` / ``events``) and a timestamp are embedded
    in the filename so successive runs never overwrite each other's
    logs.  The timestamp is resolved once per session (see
    ``_profile_log_stamp``); the mode defaults to ``TILEOPS_PROF_MODE``
    (``msprof``).

    Args:
        prof_mode: Profiling mode for the filename; when ``None``, the
            ``TILEOPS_PROF_MODE`` env var (default ``msprof``) is used.
    """
    global _profile_log_stamp
    with _profile_log_stamp_lock:
        if _profile_log_stamp is None:
            _profile_log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stamp = _profile_log_stamp
    mode = (prof_mode or os.environ.get("TILEOPS_PROF_MODE", "msprof")).lower().strip()
    return f"profile_run_{mode}_{stamp}.log"


def profile_run_json_path(prof_mode: Optional[str] = None) -> str:
    """Return the JSON companion path for :func:`profile_run_log_path`."""
    return str(Path(profile_run_log_path(prof_mode)).with_suffix(".json"))


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 5,
    n_repeat: int = 10,
    n_trials: int = 3,
) -> float:
    """Benchmark a kernel with device-event timing.

    Protocol:
      1. Run *n_warmup* un-timed iterations with cache flush.
      2. For each of *n_trials* trials, time *n_repeat* iterations using
         device Events.  Cache is flushed before every iteration.
         Input tensors are cloned each iteration so the kernel sees fresh
         addresses.
      3. Report the median trial mean.

    Uses ``backend.Event(enable_timing=True)``.  Latency includes ~50-60us
    launch overhead per call — acceptable for benchmarks where kernel time
    typically dominates.

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(f"bench_kernel expects a tuple of args, got {type(args).__name__}.")

    backend = get_device_backend()
    has_args = len(args) > 0

    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30

    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size() for a, m in zip(args, tensor_mask, strict=True) if m
        )
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]

            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "bench_kernel: inputs total %.2f GiB; skipping per-iteration cloning.",
                total_bytes / (1 << 30),
            )
            arg_pool = None

            def _run(i):
                return fn(*args)
    else:
        arg_pool = None

        def _run(i):
            return fn()

    for i in range(n_warmup):
        backend.cache_flush()
        _run(i)
    backend.synchronize()

    trial_means: list[float] = []
    for _ in range(n_trials):
        start_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]

        for i in range(n_repeat):
            backend.cache_flush()
            backend.synchronize()
            start_events[i].record()
            _run(i)
            end_events[i].record()
        backend.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
        trial_means.append(sum(times) / len(times))

    if arg_pool is not None:
        del arg_pool
    backend.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _bench_with_mode(
    functor: Callable,
    args: tuple = (),
    mode: Optional[str] = None,
) -> tuple[float, str]:
    """Dispatch to events or msprof timing based on *mode* / env var.

    Returns ``(latency_ms, prof_mode)`` where *prof_mode* is either
    ``"events"`` or ``"msprof"``.

    The default mode is ``msprof`` (end-to-end latency via
    ``do_bench(fn, backend="msprof")``).  Set ``TILEOPS_PROF_MODE=events``
    (or *mode*="events") to use device-event timing instead.  If msprof
    is unavailable or fails, falls back to events with a warning — the
    fallback is surfaced in the report by the analyzer's
    requested/actual mismatch warning.
    """
    mode = mode or os.environ.get("TILEOPS_PROF_MODE", "msprof")
    mode = mode.lower().strip()

    if mode == "msprof":
        if bench_kernel_msprof is None:
            _logger.warning(
                "TILEOPS_PROF_MODE=msprof but the msprof wrapper is unavailable; "
                "falling back to events."
            )
            return bench_kernel(functor, args=args), "events"
        try:
            latency_ms, _prof_output_dir = bench_kernel_msprof(functor, args=args)
            return latency_ms, "msprof"
        except (TypeError, RuntimeError, FileNotFoundError) as e:
            _logger.warning(
                "do_bench(msprof) profiling failed (%s: %s); falling back to events.",
                type(e).__name__,
                e,
            )
            return bench_kernel(functor, args=args), "events"

    return bench_kernel(functor, args=args), "events"


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking."""

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self, functor: Any, *inputs: Any, mode: Optional[str] = None) -> dict:
        with torch.no_grad():
            latency, prof_mode = _bench_with_mode(functor, args=inputs, mode=mode)
        result = self._build_result(latency)
        if prof_mode == "msprof":
            # msprof metadata: latency comes from do_bench(msprof); optional
            # single-kernel scope via TILEOPS_MSPROF_KERNEL_NAME (kernel_name=).
            result["msprof"] = dict(MSPROF_RESULT_METADATA)
            self._attach_roofline_metrics(result, functor, inputs)
        result["prof_mode"] = prof_mode
        return result

    def _attach_roofline_metrics(self, result: dict[str, Any], functor: Any, inputs: tuple) -> None:
        """Attach the msprof-tool-computed roofline ``Ratio(%)`` (primary) with
        a pipe-utilization fallback.

        Primary path: one ``msprof op --aic-metrics=Roofline`` subprocess pass
        (:func:`tileops._perf.msprof_roofline.collect_roofline_msprof`)
        reads the roofline performance ratio that the CANN msprof tool
        computes with its internal SoC model (``GM Read + Write`` point,
        op selected by duration match to the do_bench latency).  This is the
        report's historical ``Ratio(%)`` semantics; no tileops-side peak
        model is involved.  Latency authority stays with do_bench(msprof).

        Fallback path (roofline disabled/failed/unsupported functor, e.g. a
        closure): the bottleneck pipe utilization from
        ``do_bench(backend="msprof_detail")`` — a measured cycle-domain
        metric with a different denominator (the busiest pipe's own total
        cycles).  The active source is recorded in ``msprof.ratio_source``.

        Both passes are non-fatal: a case simply reports no Ratio when both
        fail.  ``TILEOPS_MSPROF_ROOFLINE=0`` skips the roofline pass;
        ``TILEOPS_MSPROF_DETAIL=0`` skips the pipe pass.
        """
        msprof_meta = dict(result.get("msprof") or {})
        attached = False

        if collect_roofline_msprof is not None and os.environ.get(
            "TILEOPS_MSPROF_ROOFLINE", "1"
        ).strip() not in ("0", "false", "off"):
            try:
                with torch.no_grad():
                    roofline = collect_roofline_msprof(
                        functor, args=inputs, latency_us=result.get("latency_us")
                    )
            except TypeError as e:
                _logger.info("roofline pass unsupported for this functor (%s); pipe fallback.", e)
                roofline = None
            except Exception as e:  # noqa: BLE001 - profiling failure must not kill the case
                _logger.warning(
                    "msprof roofline pass failed (%s: %s); trying pipe fallback.",
                    type(e).__name__,
                    e,
                )
                roofline = None
            if roofline:
                for key in (
                    "Ratio(%)",
                    "AI(Ops/Byte)",
                    "Perf(TOps/s)",
                    "Computility(TOps/s)",
                    "GM_Roofline_BW(TB/s)",
                ):
                    if key in roofline:
                        result[key] = roofline[key]
                msprof_meta.update(
                    {
                        "ratio_source": "msprof_roofline_bin",
                        "roofline_op": roofline.get("roofline_op"),
                        "roofline_freq_mhz": roofline.get("roofline_freq_mhz"),
                    }
                )
                attached = True

        if (
            not attached
            and profile_kernel_msprof_detail is not None
            and os.environ.get("TILEOPS_MSPROF_DETAIL", "1").strip() not in ("0", "false", "off")
        ):
            try:
                with torch.no_grad():
                    detail = profile_kernel_msprof_detail(functor, args=inputs)
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "msprof_detail profiling failed (%s: %s); Ratio(%%) omitted for this case.",
                    type(e).__name__,
                    e,
                )
                detail = None
            if detail:
                result["Ratio(%)"] = round(float(detail["bottleneck_ratio"]) * 100.0, 2)
                result["bottleneck_pipe"] = detail["bottleneck_pipe"]
                result["pipe_utilization"] = {
                    pipe: round(ratio, 4) for pipe, ratio in detail["pipes"].items() if ratio > 0
                }
                if "AI(Ops/Byte)" not in result:
                    flops = self.calculate_flops()
                    memory = self.calculate_memory()
                    if flops is not None and memory:
                        result["AI(Ops/Byte)"] = round(flops / memory, 2)
                msprof_meta.update(
                    {
                        "ratio_source": "msprof_detail_pipe",
                        "bottleneck_pipe": detail["bottleneck_pipe"],
                        "aiv_total_cycles": detail["aiv_total_cycles"],
                        "aic_total_cycles": detail["aic_total_cycles"],
                    }
                )
                attached = True

        if attached:
            result["msprof"] = msprof_meta
        else:
            result["msprof"] = msprof_meta or dict(MSPROF_RESULT_METADATA)

    def _build_result(self, latency: float) -> dict:
        """Attach theoretical utilization metrics from the manifest formulas.

        ``tflops``/``bandwidth_tbs`` are the manifest-declared FLOP/byte
        counts divided by the measured latency (milliseconds).
        """
        latency_us = latency * 1000.0
        result: dict[str, Any] = {"latency_us": latency_us}

        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9

        return result


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params."""
    workloads = load_workloads(op_name)
    shape_key, allowed = _workload_contract(op_name)
    params = []
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing {shape_key!r}."
            )
        unknown = sorted(
            repr(k)
            for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown keys {unknown}."
            )
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w, shape_key) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            param_args = (shape, dtype, dict(extra)) if include_extra else (shape, dtype)
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


class ManifestBenchmark(BenchmarkBase[Any]):
    """Generic benchmark that reads FLOP/memory from an Op instance.

    Usage::

        op = LogSumExpFwdOp(dtype=dtype, dim=0)
        bm = ManifestBenchmark("LogSumExpFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(self, op_name: str, op: Any, workload: Any):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _extract_op_config(op: object) -> Optional[dict]:
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config
    kernel = getattr(op, "kernel", None)
    op_config = getattr(kernel, "config", None) if kernel is not None else None
    if op_config:
        return op_config
    cache = getattr(op, "_kernel_cache", None)
    if cache:
        try:
            first_kernel = next(iter(cache.values()))
        except StopIteration:
            first_kernel = None
        if first_kernel is not None:
            op_config = getattr(first_kernel, "config", None)
            if op_config:
                return op_config
    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report."""

    _records: dict = {}
    _prof_mode: str = "msprof"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert benchmark values to stable JSON-compatible objects."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(k): BenchmarkReport._json_safe(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [BenchmarkReport._json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _case_id(name: str, params: dict) -> str:
        """Build a deterministic identifier from the serializable case parameters."""
        explicit = params.get("case_id")
        if explicit:
            return str(explicit)
        payload = json.dumps(
            BenchmarkReport._json_safe(params),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{name}-{digest}"

    @staticmethod
    def _case_label(params: dict[str, Any]) -> str:
        """Return the human-readable pytest/workload label for a benchmark case."""
        explicit = params.get("label")
        if explicit not in (None, ""):
            return str(explicit)

        current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
        nodeid = current_test.rsplit(" (", 1)[0]
        left = nodeid.rfind("[")
        if left >= 0 and nodeid.endswith("]"):
            param_id = nodeid[left + 1 : -1]
            dtype = params.get("dtype")
            if dtype not in (None, ""):
                suffix = f"-{str(dtype).removeprefix('torch.')}"
                if param_id.endswith(suffix):
                    param_id = param_id[: -len(suffix)]
            if param_id:
                return param_id

        for key, value in params.items():
            if key == "shape" or key.endswith("_shape"):
                if isinstance(value, (list, tuple)):
                    return "x".join(str(dim) for dim in value)
                return str(value)
        return "N/A"

    @staticmethod
    def set_prof_mode(mode: str) -> None:
        BenchmarkReport._prof_mode = mode

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            op_config = _extract_op_config(op_or_name)

        def _is_serializable(v: Any) -> bool:
            if v is None or isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, (list, tuple)):
                return all(_is_serializable(x) for x in v)
            if isinstance(v, dict):
                return all(isinstance(k, str) and _is_serializable(x) for k, x in v.items())
            return False

        filtered_params = {
            k: v
            for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        filtered_params = BenchmarkReport._json_safe(filtered_params)
        label = BenchmarkReport._case_label(filtered_params)
        filtered_params.pop("label", None)
        record_entry = {
            "case_id": BenchmarkReport._case_id(name, filtered_params),
            "label": label,
            "params": filtered_params,
            "result": BenchmarkReport._json_safe(result),
            "tag": tag,
        }
        if op_config:
            record_entry["config"] = BenchmarkReport._json_safe(op_config)
        BenchmarkReport._records.setdefault(name, []).append(record_entry)

        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        entry = {"tag": tag, "op": name, **result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        _logger.info(
            "op=%s module=%s tag=%s latency_us=%.4f tflops=%.2f",
            name,
            op_module or "N/A",
            tag,
            result.get("latency_us", 0),
            result.get("tflops", 0),
        )

    @staticmethod
    def dump(path: Optional[str] = None) -> None:
        if not BenchmarkReport._records:
            return

        if path is None:
            path = os.environ.get("TILEOPS_BENCHMARK_REPORT_PATH")
        if path is None:
            path = profile_run_log_path(BenchmarkReport._prof_mode)

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        backend = get_device_backend()
        lines = [
            "# NPU Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(backend.env_metadata())
        lines.append(f"- **Profiling mode**: {BenchmarkReport._prof_mode}")
        lines.append("")

        default_result_keys = [
            "latency_us",
            "tflops",
            "bandwidth_tbs",
        ]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            tag_entries: dict[str, list] = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)
            for key in reversed(("latency_us", "Ratio(%)")):
                if key in result_keys:
                    result_keys.remove(key)
                    result_keys.insert(0, key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                trailing_keys = []
                for k in ("dtype", "shape"):
                    if k in param_keys:
                        param_keys.remove(k)
                        trailing_keys.append(k)
                has_config = any("config" in e for e in tag_group)
                result_headers = {
                    "latency_us": "Latency (us)",
                    "Ratio(%)": "Ratio (%)",
                }
                header_parts = (
                    ["Label"] + [result_headers.get(key, key) for key in result_keys] + param_keys
                )
                if has_config:
                    header_parts.append("config")
                header_parts.extend(trailing_keys)
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry.get("label") or "N/A")]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if val is None:
                            row.append("N/A")
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    row.extend(str(entry["params"].get(k, "")) for k in param_keys)
                    if has_config:
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    for tk in trailing_keys:
                        row.append(str(entry["params"].get(tk, "")))
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with path_obj.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        json_path = path_obj.with_suffix(".json")
        BenchmarkReport.dump_json(str(json_path))
        print(f"\nBenchmark report saved to {path_obj}")
        print(f"Structured benchmark report saved to {json_path}")

    @staticmethod
    def to_dict() -> dict[str, Any]:
        """Return all collected benchmark records in a stable interchange format."""
        backend = get_device_backend()
        records = []
        for operator, entries in BenchmarkReport._records.items():
            for entry in entries:
                records.append({"operator": operator, **BenchmarkReport._json_safe(entry)})
        return {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "profiling_mode_requested": BenchmarkReport._prof_mode,
            "environment": backend.env_metadata(),
            "records": records,
        }

    @staticmethod
    def dump_json(path: Optional[str] = None) -> Optional[Path]:
        """Write the structured benchmark records and return the output path."""
        if not BenchmarkReport._records:
            return None
        output = Path(path or profile_run_json_path(BenchmarkReport._prof_mode))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(BenchmarkReport.to_dict(), f, indent=2, ensure_ascii=False)
        return output

    @staticmethod
    def clear() -> None:
        BenchmarkReport._records.clear()
        BenchmarkReport._prof_mode = "msprof"
        # Reset the session timestamp so the next run gets a fresh
        # ``profile_run_{mode}_{timestamp}.log``.
        global _profile_log_stamp
        with _profile_log_stamp_lock:
            _profile_log_stamp = None
