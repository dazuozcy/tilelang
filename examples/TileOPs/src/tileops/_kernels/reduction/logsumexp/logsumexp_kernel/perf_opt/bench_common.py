"""bench_common.py -- shared Stage 4 measurement helpers (do_bench(msprof) only).

Callable boundary for every measurement (baseline / candidate / final):
the JIT kernel object returned by the Stage 3 factory surface, called once
per invocation on a preallocated (M, N) contiguous input::

    kernel = _logsumexp_kernel_single(M, N, dtype)(block_m)          # single
    kernel = _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)   # tiled
    y = kernel(x)

This is exactly the callable the TileOPs wrapper invokes (wrapper adds
view-only reshapes except for 3d-multidim-reduce, where the op layer adds a
permute+contiguous copy kernel; kernel-level msprof latency therefore
isolates the tunable TileLang kernel for that workload too).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch

PERF_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next(parent for parent in Path(__file__).resolve().parents if (parent / "tilelang").is_dir() and (parent / ".agents").is_dir()))
OP_DIR = os.path.dirname(PERF_OPT_DIR)  # logsumexp_kernel/
WORKLOAD_INVENTORY = os.path.join(PERF_OPT_DIR, "workload_inventory.json")
PERF_RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")
PROFILES_DIR = os.path.join(PERF_OPT_DIR, "profiles")
LOGS_DIR = os.path.join(PERF_OPT_DIR, "logs")

BASELINE_IMPLS = {
    "_logsumexp_kernel_single": os.path.join(OP_DIR, "_logsumexp_kernel_single.py"),
    "_logsumexp_kernel_tiled": os.path.join(OP_DIR, "_logsumexp_kernel_tiled.py"),
}

WARMUP_MS = 25.0
REP_MS = 100.0

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_inventory() -> dict:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as f:
        return json.load(f)


def tune_workloads(kernel_id: str | None = None) -> list[dict]:
    items = [w for w in load_inventory()["workloads"] if w.get("kind") == "tune"]
    if kernel_id is not None:
        items = [w for w in items if w["kernel_id"] == kernel_id]
    return items


def load_impl(impl_path: str):
    """Import a kernel implementation module by file path."""
    module_name = os.path.splitext(os.path.basename(impl_path))[0]
    spec = importlib.util.spec_from_file_location(f"_stage4_{module_name}_{id(impl_path)}", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import implementation file: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_kernel(module, workload: dict):
    """Build the JIT kernel object for a workload from an impl module."""
    p = workload["params"]
    dtype = workload["dtype"]
    if workload["kernel_id"] == "_logsumexp_kernel_single":
        return getattr(module, "_logsumexp_kernel_single")(p["M"], p["N"], dtype)(p["block_m"])
    return getattr(module, "_logsumexp_kernel_tiled")(p["M"], p["N"], dtype, p["tile_n"])(p["block_m"])


def make_runner(impl_path: str, workload: dict, seed: int = 0):
    """Return (run_fn, x): run_fn launches the target kernel exactly once.

    The input tensor is generated once (deterministic seed) and reused across
    all timed calls; the boundary is identical for every implementation.
    """
    module = load_impl(impl_path)
    p = workload["params"]
    M, N = p["M"], p["N"]
    torch_dtype = _DTYPES[workload["dtype"]]
    gen = torch.Generator(device="npu").manual_seed(seed)
    x = torch.randn(M, N, dtype=torch_dtype, device="npu", generator=gen)
    kernel = build_kernel(module, workload)
    torch.npu.synchronize()

    def run_fn():
        kernel(x)

    run_fn.impl_path = impl_path  # type: ignore[attr-defined]
    run_fn.workload = workload  # type: ignore[attr-defined]
    return run_fn, x


def golden_check(impl_path: str, workload: dict, seed: int = 0) -> float:
    """Quick precision sanity of the impl on this workload vs torch golden.

    Returns max abs diff (finite-masked). Full L0 suites run via the impl
    file's own ``--level L0`` entry (see run_experiments.py).
    """
    module = load_impl(impl_path)
    p = workload["params"]
    M, N = p["M"], p["N"]
    torch_dtype = _DTYPES[workload["dtype"]]
    gen = torch.Generator(device="npu").manual_seed(seed)
    x = torch.randn(M, N, dtype=torch_dtype, device="npu", generator=gen)
    kernel = build_kernel(module, workload)
    y = kernel(x)
    torch.npu.synchronize()
    ref = torch.logsumexp(x.float(), dim=-1).to(torch_dtype)
    tol = 1e-2 if torch_dtype in (torch.float16, torch.bfloat16) else 1e-3
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol)
    return float((y.float() - ref.float()).abs().max().item())


def msprof_samples_us(run_fn, samples: int = 5) -> list[float]:
    """One measurement = several do_bench(msprof) samples (returns microseconds).

    early_stop_baseline disabled: controlled comparisons must not mix the
    event-based early-stop estimate with msprof samples.
    """
    from tilelang.profiler import do_bench

    out = []
    for _ in range(samples):
        ms = do_bench(
            run_fn,
            backend="msprof",
            warmup=WARMUP_MS,
            rep=REP_MS,
            early_stop_baseline=None,
        )
        out.append(float(ms) * 1000.0)
    return out


def msprof_detail(run_fn, n_repeat: int = 5) -> dict:
    """Deep-dive pipe ratios via do_bench(backend='msprof_detail')."""
    from tilelang.profiler import do_bench

    prof = do_bench(run_fn, backend="msprof_detail", _n_repeat=max(1, n_repeat))
    pipes = {}
    for field in (
        "aic_mad", "aic_scalar", "aic_mte1", "aic_mte2", "aic_mte3", "aic_fixpipe",
        "aiv_vec", "aiv_scalar", "aiv_mte2", "aiv_mte3",
    ):
        pipes[field] = float(getattr(prof, field, 0.0) or 0.0)
    aiv_pipes = {k: v for k, v in pipes.items() if k.startswith("aiv_")}
    bottleneck = max(aiv_pipes, key=aiv_pipes.get) if any(aiv_pipes.values()) else "none"
    return {
        "dur_us": float(prof.dur_ns) / 1000.0,
        "aiv_total_cycles": int(getattr(prof, "aiv_total_cycles", 0) or 0),
        "aic_total_cycles": int(getattr(prof, "aic_total_cycles", 0) or 0),
        "pipes": pipes,
        "bottleneck_pipe": bottleneck,
        "bottleneck_ratio": aiv_pipes.get(bottleneck, 0.0),
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def timestamp_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def median_us(samples: list[float]) -> float:
    return statistics.median(samples)


def variance_pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return abs(a - b) / a * 100.0


def archive_detail(profile_stage: str, kernel_id: str, workload_id: str, candidate_id: str, detail: dict) -> str:
    """Archive a msprof_detail dict under profiles/; returns the archive dir."""
    d = os.path.join(PROFILES_DIR, profile_stage, kernel_id, f"{workload_id}_{candidate_id}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "msprof_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    return d
