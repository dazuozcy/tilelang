"""LogSumExp forward kernel wrapper and Kernel class (Ascend A5).

Implements a 2-pass online algorithm for:
  - logsumexp: y[i] = max_i + log(sum_i(exp(x[i,j] - max_i)))

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in on-chip memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

The TileLang kernel functions (``_logsumexp_kernel_single`` /
``_logsumexp_kernel_tiled``) are the GPU implementations extracted verbatim
from the GPU TileOPs repository into
``src/tileops/_kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py``; they
serve as the algorithm reference for the conductor's ``target="ascend"``
re-implementation (Stage 1-3).  This module ports the dispatcher, the
``ascend::`` custom_op wrapper, and the ``LogSumExpKernel`` class with the
standard A5 adaptations (K5-K9 of ``docs/gpu_to_ascend_adaptation.md``):

- ``supported_archs = ["Ascend950"]`` (dav-3510) instead of CUDA SM ints.
- The on-chip budget (``device_smem_budget``) keeps its role as the
  per-core workspace check with UB semantics (248 KiB physical / 216 KiB
  under SimtVF) via the ``tileops.backend.device`` backend surface.
- Autotune surface (``autotune_configs`` / ``autotune`` /
  ``_tile_n_candidates`` / ``tune`` param) removed — heuristic config
  selection only; ``init_config(config)`` without ``tune``.
- ``threads`` removed from the wrapper, ``default_config``, and the
  ``forward`` call (reduction ops: the re-implemented kernel callable takes
  ``block_m`` only).

Boundary handling for non-aligned N is performed inside the kernel via
masked loads and -inf fills, eliminating host-side ``F.pad`` from the
forward path.  In the multi-tile path, only the last tile uses element-wise
masked loads; all preceding tiles use the fast vectorized T.copy path since
their columns are fully in-bounds.
"""

import functools
from typing import Optional

import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .logsumexp_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .logsumexp_kernel import _logsumexp_kernel_single, _logsumexp_kernel_tiled
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .logsumexp_kernel.perf_opt._logsumexp_kernel_single import _logsumexp_kernel_single
from .logsumexp_kernel.perf_opt._logsumexp_kernel_tiled import _logsumexp_kernel_tiled

__all__ = ["LogSumExpKernel"]


# Dispatch (ported verbatim from the GPU kernel module; selects between the
# extracted single-tile and tiled kernel builders).


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel(M: int, N: int, dtype: str, tile_n: int = 0):
    """Build the appropriate logsumexp kernel."""
    if tile_n == 0:
        return _logsumexp_kernel_single(M, N, dtype)
    return _logsumexp_kernel_tiled(M, N, dtype, tile_n)


# custom_op wrapper for torch.compile compatibility


@torch.library.custom_op("ascend::logsumexp_fwd", mutates_args=())
def _logsumexp_fwd_wrapped(
    M: int,
    N: int,
    dtype_str: str,
    block_m: int,
    tile_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _logsumexp_kernel(M, N, dtype_str, tile_n)(block_m)(x)


@_logsumexp_fwd_wrapped.register_fake
def _(M, N, dtype_str, block_m, tile_n, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# Kernel class


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class LogSumExpKernel(Kernel):
    """LogSumExp forward kernel.

    Supports Ascend950 (dav-3510) architectures. Uses 256-element alignment
    for on-chip copies. Implements a 2-pass online algorithm.

    For large N that does not fit in on-chip memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: Must be "logsumexp" (kept for API consistency with the
            softmax-family kernels).
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        device_index: Device index for the on-chip memory budget query.
            When ``None``, the current device is used (via the device
            backend surface).
    """

    supported_archs: list[str] = ["Ascend950"]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if op_kind != "logsumexp":
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected 'logsumexp'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = _elem_bytes(dtype)
        self._smem_budget = device_smem_budget(device_index)

        # tile_n is baked into the kernel at build time, so we pre-compute
        # it from the heuristic block_m in default_config.
        self._tile_n = self.default_config["tile_n"]
        self.kernel = _logsumexp_kernel(
            self.M,
            self.N,
            self.dtype_str,
            self._tile_n,
        )

        self.init_config(config)

        # If the caller supplied an explicit tile_n (e.g. from a previous
        # tuning result), honour it.  Only fall back to the heuristic when
        # tile_n was not provided.
        caller_tile_n = config.get("tile_n") if config is not None else None
        if caller_tile_n is not None:
            target_tile_n = caller_tile_n
        else:
            target_tile_n = self._tile_n_for_block_m(self.config["block_m"])
        if target_tile_n != self._tile_n:
            self._tile_n = target_tile_n
            self.kernel = _logsumexp_kernel(
                self.M,
                self.N,
                self.dtype_str,
                self._tile_n,
            )
        self.config["tile_n"] = self._tile_n

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Uses the device's actual on-chip memory budget (the Ascend UB
        budget via ``device_smem_budget``) so that large-N workloads can
        use fewer, larger tiles or even the single-tile fast path.

        Both paths are subject to the MAX_SINGLE_TILE_COLS column
        cap (TileLang's vectorizer fails at the 32768 column boundary).
        """
        budget = self._smem_budget
        # Single-tile path: cap by column count and budget.
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            tile_n = compute_tile_n(block_m, self._elem_bytes, self.N_padded, budget=budget)
            if tile_n == self.N_padded:
                return 0
        # Tiled path (logsumexp uses 1 shared buffer).
        # Cap the budget so tile_n stays within the column limit.
        col_budget = MAX_SINGLE_TILE_COLS * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded, budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on the on-chip memory budget.

        For the single-tile path (tile_n == 0), prefer the largest
        block_m that fits in on-chip memory.

        For the tiled path, prefer the block_m that minimises the
        number of N-tiles (maximises tile_n) to reduce global memory
        passes.  Among configs with equal tile count, prefer smaller
        block_m for better occupancy.
        """
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8, 16]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                # Single-tile is always better: prefer larger block_m
                best_bm = bm
                best_tile_n = tn
            elif best_tile_n == 0:
                pass
            else:
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "tile_n": best_tile_n}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logsumexp kernel.

        Accepts an ``(M, N)`` tensor.  Boundary handling for non-aligned
        ``N`` is performed inside the kernel (masked loads + ``-inf``
        fill), so no host-side ``F.pad`` is needed.
        """
        tile_n = self._tile_n

        return _logsumexp_fwd_wrapped(
            self.M,
            self.N,
            self.dtype_str,
            self.config["block_m"],
            tile_n,
            x,
        )
