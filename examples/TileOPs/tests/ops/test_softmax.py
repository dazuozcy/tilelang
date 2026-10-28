"""Correctness tests for the LogSumExp op (reduction family, Ascend A5).

Ported from the GPU TileOPs ``tests/ops/test_softmax.py`` (shared
softmax-family test file). Only the LogSumExp tests are ported — the
softmax/log_softmax ops are not yet migrated to A5 and their cases are
deliberately excluded; this file keeps the shared family file name declared
in the manifest ``source.test``.

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, keepdim, and
multi-dim reduction, validated against the PyTorch reference (computed on
the A5 device via the golden path).

Smoke tests (first param) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.

Spec-conformant interface:
  LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.reduction.logsumexp import LogSumExpFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import LogSumExpWorkload
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


# LogSumExp — spec-conformant interface (shape, dim, keepdim, dtype)


class LogSumExpFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param((32, 256), -1, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]),
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


class LogSumExpTest(LogSumExpWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@LogSumExpFixture
def test_logsumexp_op(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    test = LogSumExpTest(shape, dtype, dim=dim)
    op = LogSumExpFwdOp(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# LogSumExp — keepdim=True (exercises _reshape_output keepdim path)


class LogSumExpKeepdimFixture(FixtureBase):
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


@LogSumExpKeepdimFixture
def test_logsumexp_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    """Test logsumexp with keepdim=True — output retains reduced dim as size 1."""
    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=True)

    y_ref = torch.logsumexp(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"keepdim logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# Non-contiguous input test (spec interface)


class LogSumExpNonContigFixture(FixtureBase):
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


@LogSumExpNonContigFixture
def test_logsumexp_non_contiguous(shape: tuple, dtype: torch.dtype) -> None:
    """Test logsumexp with non-contiguous input."""
    m, n = shape
    x_full = torch.randn(m, n * 2, dtype=dtype, device=_device())
    x = x_full[:, :n]

    op = LogSumExpFwdOp(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# 1D input test (spec interface)


class LogSumExp1DFixture(FixtureBase):
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


@LogSumExp1DFixture
def test_logsumexp_1d(n: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 1D input -- output should be a scalar."""
    x = torch.randn(n, dtype=dtype, device=_device())
    op = LogSumExpFwdOp(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# Multi-dim acceptance: LogSumExpFwdOp accepts list dims (multi-dim
# reduction is supported).


@pytest.mark.smoke
def test_logsumexp_accepts_multidim() -> None:
    """LogSumExpFwdOp must accept list dim without error (multi-dim is supported)."""
    x = torch.randn(4, 8, device=_device(), dtype=torch.float32)
    op = LogSumExpFwdOp(dtype=torch.float32, dim=[0, 1])
    y = op(x)
    y_ref = torch.logsumexp(x.float(), dim=[0, 1])
    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
