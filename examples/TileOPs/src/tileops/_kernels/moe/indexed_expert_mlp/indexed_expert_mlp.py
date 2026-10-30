"""Indexed small-route expert MLP kernels (Ascend A5).

Ported from the GPU TileOPs ``src/tileops/_kernels/moe/indexed_expert_gemm.py``
(route-indexed small-M expert MLP: route statistics, the indexed expert GEMM
template with its gate/up and down specializations, and the weighted reduce).

Pipeline structure preserved from the GPU source (route-major expert MLP):

1. ``IndexedRouteStatsKernel`` — count each expert's routes, rank every route
   inside its expert, and list up to ``num_tokens`` routes per expert, so a
   GEMM block computes a whole same-expert route group.
2. ``IndexedExpertGateUpKernel`` — the gate/up projection of every route with
   the SiLU gated activation fused into the epilogue.
3. ``IndexedExpertDownKernel`` — the down projection of every route's
   activated hidden row (one row per route, ``route_input=True``).
4. ``IndexedWeightedReduceKernel`` — apply the routing weights and reduce the
   route axis into one row per token, scaled by ``routed_scaling_factor``.

The TileLang kernel functions (``_route_stats`` / ``_indexed_expert_gemm`` /
``_weighted_reduce``) are the GPU implementations extracted verbatim from the
GPU TileOPs repository into
``src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py``
(plus the ``_route_metadata_size`` helper appended per the add-ascend-op S3
"Helper function extraction (mandatory)" rule); they serve as the algorithm
reference for the conductor's ``target="ascend"`` re-implementation
(Stage 1-3).  Until then, launching through the extracted CUDA-target
factories fails (the extracted callables still take GPU launch parameters
the A5 configs no longer carry) — this is expected and is why runtime
verification happens at Stage 5 after integration.

A5 adaptations (K5-K9 of ``docs/gpu_to_ascend_adaptation.md``):

- ``supported_archs = ["Ascend950"]`` (dav-3510) instead of CUDA SM ints.
- The shared-memory opt-in probe (``torch.cuda.get_device_properties`` via
  ``get_shared_memory_optin``) is replaced by the UB budget semantics: the
  per-block workspace budget is queried through the
  ``tileops.backend.device`` backend surface (248 KiB physical / 216 KiB
  usable under SimtVF).
- The GPU ``Entry`` / ``entry_for`` build-identity machinery maps to plain
  constructor caching in the Op (the A5 Op base has no ``kernel_for``
  dispatch); the build identities are preserved verbatim as cache keys.
- ``threads`` is removed from every ``default_config`` and ``forward`` call
  (K3/K9): the re-implemented Ascend callables declare their thread domain
  internally (``T.SimtVF(threads=...)``) instead of taking it as a launch
  argument. The GEMM template keeps ``block_n`` / ``block_k`` /
  ``num_stages``; the single-block route-stats and weighted-reduce factories
  carry no tunable config.
- No autotune surface exists on A5 (K8): heuristic config selection only.
"""

import dataclasses
from typing import Optional

import torch

from tileops._kernels.kernel_base import Kernel
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .indexed_expert_mlp_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .indexed_expert_mlp_kernel import (
#     _indexed_expert_gemm,
#     _route_metadata_size,
#     _route_stats,
#     _weighted_reduce,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .indexed_expert_mlp_kernel.perf_opt._indexed_expert_gemm import (
    _indexed_expert_gemm,
    TUNED_DOWN_CONFIG,
    TUNED_GATEUP_CONFIG,
)
from .indexed_expert_mlp_kernel.perf_opt._route_stats import _route_metadata_size
from .indexed_expert_mlp_kernel.perf_opt._route_stats import _route_stats
from .indexed_expert_mlp_kernel.perf_opt._weighted_reduce import _weighted_reduce
from tileops.backend.device import get_device_backend

__all__ = [
    "IndexedExpertCall",
    "IndexedExpertDownKernel",
    "IndexedExpertGEMMTemplate",
    "IndexedExpertGateUpKernel",
    "IndexedRouteStatsKernel",
    "IndexedWeightedReduceKernel",
]


@dataclasses.dataclass(frozen=True)
class IndexedExpertCall:
    """One route-major expert MLP call, shared by its four kernel interfaces.

    Ported from GPU ``tileops._kernels.moe.call_spec.IndexedExpertCall``
    (the CallSpec base is A5-absent; the call-record fields and the branch
    predicate the kernels build on are preserved).
    """

    num_tokens: int = 0
    top_k: int = 0
    num_experts: int = 0
    hidden_size: int = 0
    ffn_size: int = 0
    dtype: Optional[torch.dtype] = None
    routed_scaling_factor: float = 1.0

    @property
    def grouped_dispatch(self) -> bool:
        """Whether routes that share an expert are computed as one group.

        Several tokens make groups worth describing: the route-statistics pass runs and
        both GEMMs are built to read its metadata. One token has one route per expert at
        most, so each GEMM computes its own route and no metadata is produced.
        """
        return self.num_tokens > 1


class IndexedRouteStatsKernel(Kernel):
    """Describe same-expert route groups on device.

    One build per route extents (and device). Launches a single block over
    the whole metadata layout.
    """

    supported_archs = ["Ascend950"]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        *,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device_index = device_index
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.kernel = _route_stats(num_tokens, top_k, num_experts)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # GPU default was {"threads": 128}; threads is removed on A5 (K3/K9).
        return {}

    def forward(self, expert_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        expected = (self.num_tokens, self.top_k)
        if tuple(expert_ids.shape) != expected or expert_ids.dtype is not torch.int32:
            raise ValueError(f"expert_ids must be {expected} int32")
        size = _route_metadata_size(self.num_experts, self.num_tokens * self.top_k, self.num_tokens)
        if tuple(out.shape) != (size,) or out.dtype is not torch.int32:
            raise ValueError(f"out must be ({size},) int32")
        self.kernel(**self.config)(expert_ids, out)
        return out


class IndexedExpertGEMMTemplate(Kernel):
    """Compute expert-selected rows directly from ``expert_ids``.

    ``a`` is ``[T,K]`` or ``[T,top_k,K]``; weights are ``[E,N,K]`` or
    ``[E,2N,K]`` for a gated activation; output is ``[T,top_k,N]``.
    """

    supported_archs = ["Ascend950"]

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        *,
        activation: str = "none",
        route_input: bool = False,
        dispatch_mode: str = "direct",
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device_index = device_index
        if activation not in ("none", "silu_and_mul"):
            raise ValueError("activation must be 'none' or 'silu_and_mul'")
        if dispatch_mode not in ("direct", "grouped"):
            raise ValueError("dispatch_mode must be 'direct' or 'grouped'")
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.n = n
        self.k = k
        self.dtype = dtype
        self.activation = activation
        self.route_input = route_input
        self.dispatch_mode = dispatch_mode
        self.kernel = _indexed_expert_gemm(
            num_tokens,
            top_k,
            num_experts,
            n,
            k,
            self.dtype_str,
            activation,
            route_input,
            dispatch_mode,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # --- perf_opt (Stage 4 tuned) ---
        # The tuned kernels were selected for block_k=512. That config only fits
        # on-grid shapes (k % 512 == 0 and n % 128 == 0): the kernel then skips
        # the off-grid W-staging buffer. All 8 benchmark workloads are on-grid at
        # this config. Off-grid shapes (correctness-test small H/F) fall through
        # to the baseline heuristic below — block_k=512 there would allocate the
        # off-grid W-staging buffer (n_gemm * block_k) and overflow UB.
        if self.n % 128 == 0 and self.k % 512 == 0:
            return dict(TUNED_GATEUP_CONFIG if self.activation != "none" else TUNED_DOWN_CONFIG)
        # --- baseline (Stage 3) heuristic (off-grid fallback) ---
        block_n = (
            128 if self.route_input and (self.num_tokens == 1 or self.num_tokens >= 16) else 64
        )
        block_k = 128
        num_stages = 3 if 1 < self.num_tokens < 16 else 2
        if self.route_input and (
            self.num_tokens == 1 or (self.dispatch_mode == "grouped" and 4 <= self.num_tokens <= 8)
        ):
            block_k = 256
            num_stages = 2
        # A stage holds the 16-row A tile and the weight tile. Where the stages do not fit a
        # block's workspace, take fewer, then a narrower K step. A5: the budget is the
        # on-chip UB (216 KiB usable under SimtVF) via the device backend surface (K6).
        rows = 16 + (2 if self.activation != "none" else 1) * block_n
        limit = get_device_backend().shared_memory_budget(self.device_index)
        while num_stages > 1 and num_stages * rows * block_k * self.dtype.itemsize > limit:
            num_stages -= 1
        while block_k > 64 and num_stages * rows * block_k * self.dtype.itemsize > limit:
            block_k //= 2
        return {
            "block_n": block_n,
            "block_k": block_k,
            "num_stages": num_stages,
        }

    def forward(
        self,
        a: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        route_metadata: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_a = (
            (self.num_tokens, self.top_k, self.k) if self.route_input else (self.num_tokens, self.k)
        )
        expected_weight_n = 2 * self.n if self.activation != "none" else self.n
        if tuple(a.shape) != expected_a:
            raise ValueError(f"a must have shape {expected_a}")
        if tuple(weights.shape) != (self.num_experts, expected_weight_n, self.k):
            raise ValueError(
                f"weights must have shape {(self.num_experts, expected_weight_n, self.k)}"
            )
        if expert_ids.shape != (self.num_tokens, self.top_k) or expert_ids.dtype is not torch.int32:
            raise ValueError("expert_ids must be [num_tokens, top_k] int32")
        if a.dtype is not self.dtype or weights.dtype is not self.dtype:
            raise TypeError("a and weights must match the template dtype")
        if route_metadata is None:
            if self.dispatch_mode == "grouped":
                raise ValueError("route_metadata is required for grouped dispatch")
            route_metadata = expert_ids.reshape(-1)[:1]
        expected_metadata = (
            _route_metadata_size(self.num_experts, self.num_tokens * self.top_k, self.num_tokens)
            if self.dispatch_mode == "grouped"
            else 1
        )
        if (
            tuple(route_metadata.shape) != (expected_metadata,)
            or route_metadata.dtype is not torch.int32
        ):
            raise ValueError(f"route_metadata must be ({expected_metadata},) int32")
        expected_out = (self.num_tokens, self.top_k, self.n)
        if out is None:
            out = a.new_empty(expected_out)
        if tuple(out.shape) != expected_out or out.dtype is not self.dtype:
            raise ValueError(f"out must be {expected_out} with dtype {self.dtype}")
        self.kernel(**self.config)(a, weights, expert_ids, route_metadata, out)
        return out


class IndexedExpertGateUpKernel(IndexedExpertGEMMTemplate):
    """The gate/up projection on the indexed template, with SiLU fused into the epilogue.

    ``grouped_dispatch`` decides whether the program reads route metadata; it is a
    parameter of this one program, so it belongs in the build identity and not in a
    second implementation.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        *,
        dispatch_mode: str = "direct",
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_tokens,
            num_experts,
            top_k,
            n,
            k,
            dtype,
            activation="silu_and_mul",
            route_input=False,
            dispatch_mode=dispatch_mode,
            config=config,
            device_index=device_index,
        )


class IndexedExpertDownKernel(IndexedExpertGEMMTemplate):
    """The down projection on the indexed template, reading one row per route."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        *,
        dispatch_mode: str = "direct",
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_tokens,
            num_experts,
            top_k,
            n,
            k,
            dtype,
            activation="none",
            route_input=True,
            dispatch_mode=dispatch_mode,
            config=config,
            device_index=device_index,
        )


class IndexedWeightedReduceKernel(Kernel):
    """Apply routing weights and reduce route-major expert output to tokens.

    One build per token count, route width, hidden width, scale, dtype and device.
    """

    supported_archs = ["Ascend950"]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype,
        scaling: float,
        *,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device_index = device_index
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.kernel = _weighted_reduce(
            num_tokens, top_k, hidden_size, self.dtype_str, float(scaling)
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # GPU default was {"threads": 256}; threads is removed on A5 (K3/K9).
        return {}

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(expert_output.shape) != (self.num_tokens, self.top_k, self.hidden_size):
            raise ValueError("expert_output has the wrong shape")
        if tuple(topk_weights.shape) != (self.num_tokens, self.top_k):
            raise ValueError("topk_weights has the wrong shape")
        if tuple(output.shape) != (self.num_tokens, self.hidden_size):
            raise ValueError("output has the wrong shape")
        if expert_output.dtype is not self.dtype or output.dtype is not self.dtype:
            raise TypeError("expert_output and output must match the kernel dtype")
        if topk_weights.dtype is not torch.float32:
            raise TypeError("topk_weights must be float32")
        self.kernel(**self.config)(expert_output, topk_weights, output)
        return output
