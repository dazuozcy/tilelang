"""Benchmarks for the LogSumExp SIMD-variant op (reduction family, A5).

Ported from the GPU TileOPs ``benchmarks/ops/bench_softmax.py`` (LogSumExp
case only).  This file targets the independent SIMD-mode variant
``LogSumExpFwdOpSimd`` — the SIMT variant's benchmark lives in
``benchmarks/ops/bench_softmax.py`` and is untouched.

Measures latency, TFLOPS, and DRAM bandwidth against the PyTorch baseline
(``torch.logsumexp`` — a native operator, kept as the performance
baseline). Workload shapes and roofline formulas are loaded from the ops
manifest (tileops/manifest/).

Measurement authority: end-to-end latency is measured solely by
``tilelang.profiler.do_bench(fn, backend="msprof")`` (milliseconds),
wrapped by ``ManifestBenchmark.profile``. Cycle-level attribution is done
in npusim.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.logsumexp_simd import LogSumExpFwdOpSimd
from tileops.workloads.reduction import LogSumExpSimdWorkload

# Op name constant

_LOGSUMEXP_SIMD_OP = "LogSumExpFwdOpSimd"


# LogSumExp (SIMD) benchmark


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LOGSUMEXP_SIMD_OP, include_extra=True),
)
def test_logsumexp_simd_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = LogSumExpSimdWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = LogSumExpFwdOpSimd(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_LOGSUMEXP_SIMD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
