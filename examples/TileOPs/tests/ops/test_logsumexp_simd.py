"""Correctness tests for the LogSumExp SIMD-variant op (reduction family, A5).

Ported from the GPU TileOPs ``tests/ops/test_softmax.py`` (LogSumExp cases
only).  This file targets the independent SIMD-mode variant
``LogSumExpFwdOpSimd`` — the SIMT variant's tests live in
``tests/ops/test_softmax.py`` and are untouched.

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, keepdim, and
multi-dim reduction, validated against the PyTorch reference (computed on
the A5 device via the golden path).

Smoke tests (first params) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.

Spec-conformant interface:
  LogSumExpFwdOpSimd(dtype=dtype, dim=dim, keepdim=keepdim)
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.reduction.logsumexp_simd import LogSumExpFwdOpSimd
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import LogSumExpSimdWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    """Device name for manual tensor creation (device backend surface)."""
    return get_device_backend().name


# Tolerances (A5 tiers from docs/gpu_to_ascend_adaptation.md §1.8:
# fp8→1e-1, bf16/fp16→1e-2, fp32→1e-3)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-3
    # float16 / bfloat16
    return 1e-2, 1e-2


# LogSumExp (SIMD) — spec-conformant interface (shape, dim, keepdim, dtype)


class LogSumExpSimdFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param(
                    (32, 256), -1, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param((32, 256), -1, torch.float16, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, marks=pytest.mark.smoke),
                # dim=-1: dtypes x pow2/non-pow2
                pytest.param((32, 300), -1, torch.float32, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.bfloat16, marks=pytest.mark.full),
                # dim=-1, tail-M
                pytest.param((33, 256), -1, torch.float32, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.bfloat16, marks=pytest.mark.full),
                # dim=-1, 3D input
                pytest.param((2, 16, 256), -1, torch.float32, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.bfloat16, marks=pytest.mark.full),
                # dim=-1, 4D input
                pytest.param((2, 4, 8, 256), -1, torch.float32, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.bfloat16, marks=pytest.mark.full),
                # dim=-1, large-N (triggers N-tiling path)
                pytest.param((4, 32768), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32768), -1, torch.bfloat16, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (single-tile path)
                pytest.param((33, 300), -1, torch.float32, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (multi-tile, masked loads)
                pytest.param((33, 33000), -1, torch.float16, marks=pytest.mark.full),
                # dim=-1, non-aligned M + large-N tiled path
                pytest.param((33, 32768), -1, torch.float16, marks=pytest.mark.full),
                # dim=0
                pytest.param((256, 32), 0, torch.float32, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.bfloat16, marks=pytest.mark.full),
                # dim=1 (middle dim for 3D)
                pytest.param((2, 256, 16), 1, torch.float32, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogSumExpSimdTest(LogSumExpSimdWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@LogSumExpSimdFixture
def test_logsumexp_simd_op(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    test = LogSumExpSimdTest(shape, dtype, dim=dim)
    op = LogSumExpFwdOpSimd(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# LogSumExp (SIMD) — keepdim=True (exercises _reshape_output keepdim path)


class LogSumExpSimdKeepdimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=-1 (last dim, no transpose)
                pytest.param((32, 256), -1, torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.float16, marks=pytest.mark.smoke),
                pytest.param((2, 16, 256), -1, torch.float32, marks=pytest.mark.full),
                # dim=0 (non-last dim, exercises transpose + keepdim)
                pytest.param((256, 32), 0, torch.float32, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, marks=pytest.mark.full),
                # dim=1 (middle dim, 3D)
                pytest.param((2, 256, 16), 1, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpSimdKeepdimFixture
def test_logsumexp_simd_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    """Test logsumexp (SIMD) with keepdim=True — output retains reduced dim as size 1."""
    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = LogSumExpFwdOpSimd(dtype=dtype, dim=dim, keepdim=True)

    y_ref = torch.logsumexp(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"keepdim logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# Non-contiguous input test (spec interface)


class LogSumExpSimdNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((32, 300), torch.float32, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpSimdNonContigFixture
def test_logsumexp_simd_non_contiguous(shape: tuple, dtype: torch.dtype) -> None:
    """Test logsumexp (SIMD) with non-contiguous input."""
    m, n = shape
    x_full = torch.randn(m, n * 2, dtype=dtype, device=_device())
    x = x_full[:, :n]

    op = LogSumExpFwdOpSimd(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# 1D input test (spec interface)


class LogSumExpSimd1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpSimd1DFixture
def test_logsumexp_simd_1d(n: int, dtype: torch.dtype) -> None:
    """Test logsumexp (SIMD) with 1D input -- output should be a scalar."""
    x = torch.randn(n, dtype=dtype, device=_device())
    op = LogSumExpFwdOpSimd(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# Multi-dim acceptance: LogSumExpFwdOpSimd accepts list dims (multi-dim
# reduction is supported).


@pytest.mark.smoke
def test_logsumexp_simd_accepts_multidim() -> None:
    """LogSumExpFwdOpSimd must accept list dim without error (multi-dim is supported)."""
    x = torch.randn(4, 8, device=_device(), dtype=torch.float32)
    op = LogSumExpFwdOpSimd(dtype=torch.float32, dim=[0, 1])
    y = op(x)
    y_ref = torch.logsumexp(x.float(), dim=[0, 1])
    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
