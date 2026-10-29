"""Correctness tests for the IndexedExpertMLP op (moe family, Ascend A5).

Ported from the GPU TileOPs ``tests/ops/test_fused_moe_experts.py`` — only
the ``IndexedExpertMLPFwdOp`` cases are ported; the other moe ops
(``FusedMoEExpertsFwdOp`` and friends) are not yet migrated to A5 and their
cases are deliberately excluded (importing them would fail).

Port notes:

- ``test_small_route_branch_matches_reference`` exercised the staged op's
  dispatch into the indexed branch; here the same route-pattern matrix
  (dispersed / reused / duplicate-fallback / second-route-group) runs
  against ``IndexedExpertMLPFwdOp`` directly, preserving the kernel-side
  coverage. The staged dispatch predicate belongs to the (unmigrated)
  staged op.
- The roofline pricing assertion is adapted from
  ``test_a_call_prices_the_experts_its_routing_reads`` /
  ``tests/test_roofline_oracle.py`` (active experts priced, routing
  metadata priced).
- The torch.compile / CUDA-graph cases of the GPU file are CUDA-specific
  infrastructure (compile boundary, graph capture) and are not portable.
- Numerical policy preserved: elementwise ``atol = rtol = 1e-2`` plus a
  normalized squared error bound of ``1e-4`` (GPU ``moe_verification(2)`` /
  ``_moe_close``), computed in fp64.
- Tolerances follow the A5 dtype tiers (fp16/bf16 → 1e-2); the golden is
  computed with torch on the A5 device.

Spec-conformant interface:
  IndexedExpertMLPFwdOp(routed_scaling_factor=scale)
"""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._ops.moe.indexed_expert_mlp import IndexedExpertMLPFwdOp
from tileops.backend.device import get_device_backend
from workloads.moe import IndexedExpertMLPWorkload, ref_routed_experts


def _device() -> str:
    """Device name for manual tensor creation (device backend surface)."""
    return get_device_backend().name


# Numerical policy (ported from GPU workloads/moe.py _moe_close +
# workloads/numerics.assert_normalized_error)


def assert_normalized_error(got: torch.Tensor, expected: torch.Tensor, bound: float = 1e-3) -> None:
    """Bound squared error / combined energy, with identical nonfinite values."""
    assert got.shape == expected.shape and got.dtype == expected.dtype
    error = torch.zeros((), device=got.device, dtype=torch.float64)
    energy = torch.zeros_like(error)
    # Reduced a megabyte of elements at a time: a long output reduced whole
    # holds an FP64 temporary the size of the input, and the sum is the same
    # either way.
    chunk = 1 << 20
    for a, b in zip(got.reshape(-1).split(chunk), expected.reshape(-1).split(chunk), strict=True):
        a, b = a.double(), b.double()
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b)), "nonfinite mask mismatch"
        torch.testing.assert_close(
            a.masked_fill(finite, 0), b.masked_fill(finite, 0), rtol=0, atol=0, equal_nan=True
        )
        a, b = a.masked_fill(~finite, 0), b.masked_fill(~finite, 0)
        error += ((a - b) ** 2).sum()
        energy += (a * a + b * b).sum()
    assert error <= bound * energy, f"normalized squared error: {error / energy}"


def _moe_compare(got: torch.Tensor, expected: torch.Tensor) -> None:
    """Elementwise ``atol = rtol = 1e-2`` and normalized squared error at most 1e-4."""
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
    assert_normalized_error(got, expected, bound=1e-4)


# Shared case helpers (ported from GPU tests/ops/test_fused_moe_experts.py)


def _small_route_case(ids, dtype=torch.bfloat16):
    """An indexed-path call routed by *ids*: few routes per expert, H = 128, F = 256."""
    T, K = len(ids), len(ids[0])
    E = max(8, max(max(row) for row in ids) + 1)
    hidden = torch.randn(T, 128, dtype=dtype, device=_device()) * 0.1
    w1 = torch.randn(E, 512, 128, dtype=dtype, device=_device()) * 0.02
    w2 = torch.randn(E, 128, 256, dtype=dtype, device=_device()) * 0.02
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device=_device()), -1)
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=_device())
    return IndexedExpertMLPFwdOp(), (hidden, w1, w2, weights, topk_ids)


# Manifest-shaped workload case (exercises IndexedExpertMLPWorkload)


class IndexedExpertMLPTest(IndexedExpertMLPWorkload, TestBase):
    """TestBase view of the indexed expert MLP workload (ref_program inherited)."""


# IndexedExpertMLPFwdOp — small-route cases


@pytest.mark.smoke
def test_the_indexed_op_scales_its_output():
    """The op multiplies the reduced output by routed_scaling_factor."""
    _, args = _small_route_case([[0, 1], [2, 3]])
    out = IndexedExpertMLPFwdOp(routed_scaling_factor=2.5)(*args)
    expected = ref_routed_experts(*args, scale=2.5)
    _moe_compare(out, expected)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "ids",
    [
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        [[0, 1], [0, 1], [0, 2], [0, 2]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, expert] for expert in range(1, 18)],
    ],
    ids=["dispersed", "reused", "duplicate-fallback", "second-route-group"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_small_route_matches_reference(ids, dtype):
    """Every route pattern the grouped dispatcher serves matches the reference."""
    op, args = _small_route_case(ids, dtype)
    out = op(*args)
    _moe_compare(out, ref_routed_experts(*args))


# IndexedExpertMLPFwdOp — dispatch variants


@pytest.mark.full
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_single_token_direct_dispatch(dtype):
    """One token skips the route-statistics pass (direct dispatch, no metadata)."""
    op, args = _small_route_case([[0, 3]], dtype)
    out = op(*args)
    _moe_compare(out, ref_routed_experts(*args))


@pytest.mark.full
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dims_off_the_tile_grid(dtype):
    """H and F off the tile grid still match the reference (masked epilogue path)."""
    T, E, K, H, F = 4, 8, 2, 100, 96
    hidden = torch.randn(T, H, dtype=dtype, device=_device()) * 0.1
    w1 = torch.randn(E, 2 * F, H, dtype=dtype, device=_device()) * 0.02
    w2 = torch.randn(E, H, F, dtype=dtype, device=_device()) * 0.02
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device=_device()), -1)
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=_device())
    inputs = (hidden, w1, w2, weights, topk_ids)
    out = IndexedExpertMLPFwdOp()(*inputs)
    _moe_compare(out, ref_routed_experts(*inputs))


@pytest.mark.full
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_manifest_shaped_call_matches_reference(dtype):
    """A workload-generated grouped call (T > 1, manifest dims) matches the reference."""
    test = IndexedExpertMLPTest(T=8, E=16, K=4, H=512, F=256, dtype=dtype)
    inputs = test.gen_inputs()
    op = IndexedExpertMLPFwdOp()
    test.check(op, *inputs, compare=_moe_compare)


# IndexedExpertMLPFwdOp — interface contract


@pytest.mark.smoke
def test_rejects_unsupported_dtype():
    """The op serves float16 | bfloat16 only (manifest dtype contract)."""
    T, E, K, H, F = 2, 8, 2, 64, 32
    hidden = torch.randn(T, H, dtype=torch.float32, device=_device())
    w1 = torch.randn(E, 2 * F, H, dtype=torch.float32, device=_device())
    w2 = torch.randn(E, H, F, dtype=torch.float32, device=_device())
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device=_device()), -1)
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=_device())
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        IndexedExpertMLPFwdOp()(hidden, w1, w2, weights, topk_ids)


@pytest.mark.smoke
def test_writes_into_out_buffer():
    """With ``out=`` the op writes the result into the given buffer and returns it."""
    op, args = _small_route_case([[0, 1], [2, 3]])
    out = torch.empty_like(args[0])
    returned = op(*args, out=out)
    assert returned is out
    _moe_compare(out, ref_routed_experts(*args))


@pytest.mark.smoke
def test_roofline_prices_the_experts_its_routing_reads():
    """The roofline prices only the experts ``topk_ids`` selects, plus the routing."""
    op, args = _small_route_case([[0, 3], [3, 7], [0, 7], [3, 0]])
    op(*args)
    T, K, H, F = 4, 2, 128, 256
    elem = args[0].element_size()
    active = 3  # experts 0, 3 and 7
    expected = active * 3 * F * H * elem + 2 * T * H * elem + T * K * (4 + 4)
    assert op.eval_roofline()[1] == expected
    assert op.roofline_data_terms() == {"active_experts": active}


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
