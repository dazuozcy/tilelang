"""Phase 0 probe: verify benchmark workload -> kernel mapping empirically.

Expands the benchmark parametrization exactly like
benchmarks/ops/bench_logsumexp_simd.py does
(workloads_to_params('LogSumExpFwdOpSimd', include_extra=True)), then runs
op.forward per case and records which kernel the wrapper dispatches to and
with which (block_m, tile_n).
"""

import json
import sys

import torch

sys.path.insert(0, "/home/z00875089/tilelang-fork/examples/TileOPs")

from tileops.benchmark.benchmark_base import workloads_to_params  # noqa: E402
from tileops.ops.reduction.logsumexp_simd import LogSumExpFwdOpSimd  # noqa: E402
from tileops.workloads.reduction import LogSumExpSimdWorkload  # noqa: E402

rows = []
for params in workloads_to_params("LogSumExpFwdOpSimd", include_extra=True):
    shape, dtype, op_params = params.values  # pytest.param
    op_params.setdefault("dim", -1)
    test = LogSumExpSimdWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = LogSumExpFwdOpSimd(dtype=dtype, **op_params)
    y = op.forward(*inputs)
    torch.npu.synchronize()
    kern = op.kernel
    rows.append(
        {
            "pytest_id": params.id,
            "shape": list(shape),
            "dtype": str(dtype).split(".")[-1],
            "dim": op_params.get("dim", -1),
            "M": kern.M,
            "N": kern.N,
            "block_m": kern.config["block_m"],
            "tile_n": kern.config["tile_n"],
            "kernel_class": type(kern).__name__,
            "variant": "single" if kern.config["tile_n"] == 0 else "tiled",
            "y_shape": list(y.shape),
        }
    )
    print(json.dumps(rows[-1], ensure_ascii=False))

with open("probe_mapping.json", "w") as f:
    json.dump(rows, f, indent=2)
print("OK")
