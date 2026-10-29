"""Benchmarks for the IndexedExpertMLP op (moe family, Ascend A5).

Ported from the GPU TileOPs ``benchmarks/ops/bench_fused_moe_experts.py``
(``test_indexed_expert_mlp_bench``) and ``benchmarks/_cases/moe.py`` (the
``IndexedExpertMLPFwdOp`` entry). Workload shapes and dtypes are loaded from
the ops manifest (``src/tileops/manifest/spec/moe.yaml``).

Port notes:

- Multi-input op: ``workloads_to_params`` requires exactly one tensor input
  and raises ``KeyError`` here, so a manifest-driven ``_shape_dtype_params``
  helper generates the (T, E, K, H, F, dtype, op_params) params instead.
- GPU comparators are not portable: the staged pipeline
  (``MoEPrePermuteFwdOp`` / ``MoEExpertMLPFwdOp`` / ``MoEPostPermuteFwdOp``)
  is not migrated to A5, and the vLLM Triton / FlashInfer CUTLASS baselines
  are CUDA-only. The A5 baseline is a native-torch routed-experts
  implementation (gathered per-route ``torch.bmm`` + silu + weighted
  ``index_add_``) — a genuine performance baseline, distinct from the fp32
  accuracy reference kept in the correctness tests.
- ``routed_scaling_factor`` is applied inside the baseline (neither the op
  nor the baseline takes it as a launch argument).
- Measurement authority: end-to-end latency is measured solely by
  ``tilelang.profiler.do_bench(fn, backend="msprof")`` (milliseconds),
  wrapped by ``ManifestBenchmark.profile``. Cycle-level attribution is done
  in npusim.
"""

import pytest
import torch

from tileops._ops.moe.indexed_expert_mlp import IndexedExpertMLPFwdOp
from tileops._perf.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from workloads.moe import IndexedExpertMLPWorkload

# Op name constant

_INDEXED_EXPERT_MLP_OP = "IndexedExpertMLPFwdOp"

# Manifest-driven multi-input parametrization

_ALLOWED_KEYS = frozenset({"T", "E", "K", "H", "F", "routed_scaling_factor"})


def _shape_dtype_params(op_name: str) -> list:
    """Convert the manifest workload rows of *op_name* to pytest params.

    Each row expands over its ``dtypes``: ``(T, E, K, H, F, dtype, op_params)``,
    with ``op_params`` carrying the op's constructor parameters
    (``routed_scaling_factor``).
    """
    params = []
    for w in load_workloads(op_name):
        unknown = sorted(k for k in w if k not in _ALLOWED_KEYS and k not in ("dtypes", "label"))
        if unknown:
            raise KeyError(f"workload {w.get('label', w)!r} of {op_name!r} has unknown keys.")
        op_params = {k: v for k, v in w.items() if k in ("routed_scaling_factor",)}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    w["T"],
                    w["E"],
                    w["K"],
                    w["H"],
                    w["F"],
                    dtype,
                    dict(op_params),
                    id=f"{w['label']}-{dtype_str}",
                )
            )
    return params


# Native-torch performance baseline


def _torch_routed_experts(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Routed expert MLP in native torch ops (bmm per route, silu, weighted reduce).

    A performance baseline, not the accuracy golden: the gate/up and down
    projections run in the input dtype via gathered batched matmuls, the
    gated activation and the weighted reduce accumulate in float32.
    """
    tokens, top_k = topk_ids.shape
    experts, ffn2, hidden_size = w_gate_up.shape
    ffn = ffn2 // 2
    idx = topk_ids.reshape(-1).to(torch.int64)
    routes = tokens * top_k
    a = hidden.unsqueeze(1).expand(tokens, top_k, hidden_size).reshape(routes, 1, hidden_size)
    gate_up = torch.bmm(a, w_gate_up[idx].transpose(1, 2)).reshape(routes, ffn2)
    gate, up = gate_up[:, :ffn].float(), gate_up[:, ffn:].float()
    act = torch.nn.functional.silu(gate) * up
    route_out = torch.bmm(act.to(w_down.dtype).unsqueeze(1), w_down[idx].transpose(1, 2))
    out = torch.zeros(tokens, hidden_size, dtype=torch.float32, device=hidden.device)
    token_of_route = torch.arange(tokens, device=hidden.device).repeat_interleave(top_k)
    out.index_add_(0, token_of_route, route_out.reshape(routes, hidden_size).float())
    return (out * scale).to(hidden.dtype)


# IndexedExpertMLP benchmark


@pytest.mark.parametrize(
    "T, E, K, H, F, dtype, op_params",
    _shape_dtype_params(_INDEXED_EXPERT_MLP_OP),
)
def test_indexed_expert_mlp_bench(
    T: int, E: int, K: int, H: int, F: int, dtype: torch.dtype, op_params: dict
) -> None:
    workload = IndexedExpertMLPWorkload(T=T, E=E, K=K, H=H, F=F, dtype=dtype, **op_params)
    inputs = workload.gen_inputs()

    op = IndexedExpertMLPFwdOp(**op_params)
    bm = ManifestBenchmark(_INDEXED_EXPERT_MLP_OP, op, workload)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    scale = op.routed_scaling_factor

    def baseline_fn(hidden, w1, w2, topk_weights, topk_ids):
        return _torch_routed_experts(hidden, w1, w2, topk_weights, topk_ids, scale)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
