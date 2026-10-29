"""Indexed small-route expert MLP operator (Ascend A5).

Ported from the GPU TileOPs
``src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py``.

Route-major expert MLP with device-side reuse dispatch. It computes what a
staged permute/expert-MLP/post-permute pipeline computes for ``silu_and_mul``
without biases, route by route: each of the ``T * K`` routes is a row of its
expert's GEMM, dispatched per route rather than per expert segment. Routes
that share an expert are grouped, and only the leader copies that expert's
weights, so the minimum DRAM traffic the roofline prices is one read per
distinct expert.

A5 adaptations (O1-O6 of ``docs/gpu_to_ascend_adaptation.md``):

- Device checks use ``backend.is_device_tensor(x)`` via
  ``get_device_backend()`` instead of ``x.is_cuda``.
- The GPU ``Target`` selection parameter maps to the A5 ``kernel_map``
  override; there is no target registry on A5.
- The GPU ``call_spec`` / ``kernel_for`` / ``Entry`` dispatch machinery maps
  to a plain per-key kernel cache whose keys preserve the GPU build
  identities; the branch predicate (``grouped_dispatch = num_tokens > 1``)
  is preserved verbatim.
- The op-specific flow (input reading → route metadata → gate/up → down →
  weighted reduce → optional out buffer) and the roofline formula
  (``routed_expert_mlp_roofline``: only the experts ``topk_ids`` selects are
  read) are preserved unchanged.
- The manifest contract's ``topk_ids`` value requirement (in range
  ``[0, E)``) is a caller guarantee, as on GPU: kernels trust it and no
  runtime data check is performed (checking would synchronise the device in
  the forward path). The roofline's active-expert term reads ``topk_ids``
  lazily, only when ``eval_roofline()``/``roofline_data_terms()`` is called.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops._kernels.moe.indexed_expert_mlp import (
    IndexedExpertCall,
    IndexedExpertDownKernel,
    IndexedExpertGateUpKernel,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)
from tileops._kernels.moe.indexed_expert_mlp.indexed_expert_mlp import _route_metadata_size
from tileops._ops.op_base import Op
from tileops.backend.device import get_device_backend

__all__ = ["IndexedExpertMLPFwdOp"]

_DTYPE_CONTRACT = "hidden_states, w_gate_up and w_down must be float16 or bfloat16"


class IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch.

    Args:
        routed_scaling_factor: Scalar applied to the final reduced output.
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        routed_scaling_factor: float = 1.0,
        *,
        kernel_map: Optional[Dict[str, type]] = None,
    ) -> None:
        if routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        self.routed_scaling_factor = routed_scaling_factor
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self._last_call: IndexedExpertCall | None = None
        self._last_topk_ids: torch.Tensor | None = None

    @property
    def default_kernel_map(self) -> Dict[str, type]:
        return {
            "route_stats": IndexedRouteStatsKernel,
            "expert_gate_up": IndexedExpertGateUpKernel,
            "expert_down": IndexedExpertDownKernel,
            "weighted_reduce": IndexedWeightedReduceKernel,
        }

    # Validation

    def _validate(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        backend = get_device_backend()
        for name, tensor in (
            ("hidden_states", hidden_states),
            ("w_gate_up", w_gate_up),
            ("w_down", w_down),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
        ):
            if not backend.is_device_tensor(tensor):
                raise ValueError(f"{name} must be a {backend.name} tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(_DTYPE_CONTRACT)
        if w_gate_up.dtype is not hidden_states.dtype or w_down.dtype is not hidden_states.dtype:
            raise ValueError(_DTYPE_CONTRACT)
        if topk_weights.dtype is not torch.float32:
            raise ValueError("topk_weights must be float32")
        if topk_ids.dtype is not torch.int32:
            raise ValueError("topk_ids must be int32")
        if hidden_states.ndim != 2 or w_gate_up.ndim != 3 or w_down.ndim != 3:
            raise ValueError(
                "hidden_states must be [T, H], w_gate_up [E, 2 * F, H] and w_down [E, H, F]"
            )
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError("topk_weights and topk_ids must be [T, K]")
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        experts2, hidden2, ffn = w_down.shape
        if tokens <= 0 or top_k <= 0:
            raise ValueError("T and K must be positive")
        if top_k > experts:
            raise ValueError(f"K ({top_k}) must not exceed E ({experts})")
        if hidden <= 0 or ffn <= 0:
            raise ValueError("H and F must be positive")
        if ffn2 != 2 * ffn:
            raise ValueError(f"w_gate_up second dim ({ffn2}) must be 2 * F ({2 * ffn})")
        if experts2 != experts or hidden2 != hidden:
            raise ValueError(
                f"w_down must be [E, H, F] with E={experts}, H={hidden}, got "
                f"[{experts2}, {hidden2}, {ffn}]"
            )

    # Kernel dispatch (GPU build identities preserved as cache keys)

    def _kernel_for(self, key: str, call: IndexedExpertCall, device_index: Optional[int]):
        """Return the cached kernel *key* builds for *call*, creating it on a miss.

        The cache keys mirror the GPU ``entry_for`` build identities, so two
        calls that would compile the same program share one kernel.
        """
        dtype = call.dtype
        if key == "route_stats":
            identity = (key, call.num_tokens, call.top_k, call.num_experts, device_index)
            if identity not in self._kernel_cache:
                self._kernel_cache[identity] = IndexedRouteStatsKernel(
                    call.num_tokens,
                    call.top_k,
                    call.num_experts,
                    device_index=device_index,
                )
        elif key == "expert_gate_up":
            identity = (
                key,
                call.num_tokens,
                call.num_experts,
                call.top_k,
                call.ffn_size,
                call.hidden_size,
                dtype,
                call.grouped_dispatch,
                device_index,
            )
            if identity not in self._kernel_cache:
                self._kernel_cache[identity] = IndexedExpertGateUpKernel(
                    call.num_tokens,
                    call.num_experts,
                    call.top_k,
                    call.ffn_size,
                    call.hidden_size,
                    dtype,
                    dispatch_mode="grouped" if call.grouped_dispatch else "direct",
                    device_index=device_index,
                )
        elif key == "expert_down":
            identity = (
                key,
                call.num_tokens,
                call.num_experts,
                call.top_k,
                call.hidden_size,
                call.ffn_size,
                dtype,
                call.grouped_dispatch,
                device_index,
            )
            if identity not in self._kernel_cache:
                self._kernel_cache[identity] = IndexedExpertDownKernel(
                    call.num_tokens,
                    call.num_experts,
                    call.top_k,
                    call.hidden_size,
                    call.ffn_size,
                    dtype,
                    dispatch_mode="grouped" if call.grouped_dispatch else "direct",
                    device_index=device_index,
                )
        elif key == "weighted_reduce":
            identity = (
                key,
                call.num_tokens,
                call.top_k,
                call.hidden_size,
                dtype,
                call.routed_scaling_factor,
                device_index,
            )
            if identity not in self._kernel_cache:
                self._kernel_cache[identity] = IndexedWeightedReduceKernel(
                    call.num_tokens,
                    call.top_k,
                    call.hidden_size,
                    dtype,
                    call.routed_scaling_factor,
                    device_index=device_index,
                )
        else:
            raise KeyError(f"unknown kernel key {key!r}")
        return self._kernel_cache[identity]

    # Forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the routed expert MLP and reduce each token's routes.

        Args:
            hidden_states: $[T \\times H]$ tokens.
            w_gate_up: $[E \\times 2F \\times H]$ stacked gate and up weights.
            w_down: $[E \\times H \\times F]$ down weights.
            topk_weights: $[T \\times K]$ float32 routing weights.
            topk_ids: $[T \\times K]$ int32 expert of each route, each in $[0, E)$.
            out: Optional $[T \\times H]$ buffer the result is written into.

        Returns:
            The $[T \\times H]$ weighted and reduced result: ``out`` when given, else a
            new tensor.
        """
        self._validate(hidden_states, w_gate_up, w_down, topk_weights, topk_ids)
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        device_index = hidden_states.device.index
        # The spec also sizes the route metadata and the intermediate rows.
        call = IndexedExpertCall(
            num_tokens=tokens,
            top_k=top_k,
            num_experts=experts,
            hidden_size=hidden,
            ffn_size=ffn2 // 2,
            dtype=hidden_states.dtype,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        self._last_call = call
        self._last_topk_ids = topk_ids
        metadata = None
        if call.grouped_dispatch:
            size = _route_metadata_size(call.num_experts, tokens * top_k, tokens)
            metadata = torch.empty(size, dtype=torch.int32, device=topk_ids.device)
            self._kernel_for("route_stats", call, device_index)(topk_ids, metadata)
        hidden_rows = hidden_states.new_empty(tokens, top_k, call.ffn_size)
        route_output = hidden_states.new_empty(tokens, top_k, hidden)
        gate_up = self._kernel_for("expert_gate_up", call, device_index)
        gate_up(hidden_states, w_gate_up, topk_ids, metadata, out=hidden_rows)
        down = self._kernel_for("expert_down", call, device_index)
        down(hidden_rows, w_down, topk_ids, metadata, out=route_output)
        output = hidden_states.new_empty(tokens, hidden) if out is None else out
        self._kernel_for("weighted_reduce", call, device_index)(route_output, topk_weights, output)
        return output

    # Roofline

    def roofline_data_terms(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow.

        Reads ``topk_ids`` of the last forward lazily (a device-to-host sync —
        kept out of the forward path, as on GPU).
        """
        if self._last_topk_ids is None:
            raise RuntimeError(
                f"{type(self).__name__}.roofline_data_terms() requires a prior forward() call"
            )
        ids = self._last_topk_ids.detach().to("cpu", torch.int64)
        return {"active_experts": int(ids.unique().numel())}

    def eval_roofline(self) -> tuple[int, int]:
        """``(flops, bytes)`` of the last forward — ``routed_expert_mlp_roofline``.

        FLOPs are the expert MLP of every route plus the weighted combine,
        and the scale per output element when the scaling factor is not one.
        Bytes price one read of the selected experts' weights (3 * F * H
        elements per expert: 2F*H gate/up + H*F down), the tokens read and
        the output written, and the routing metadata.
        """
        if self._last_call is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        call = self._last_call
        tokens, top_k = call.num_tokens, call.top_k
        ffn, hidden = call.ffn_size, call.hidden_size
        elem_bytes = call.dtype.itemsize
        routes = tokens * top_k
        # One route: the gate/up GEMM (2 * 2F * H) + silu_and_mul epilogue
        # (6 * F) + the down GEMM (2 * H * F).
        expert_mlp_flops = 6 * ffn * hidden + 6 * ffn
        flops = routes * (expert_mlp_flops + 2 * hidden)
        if call.routed_scaling_factor != 1.0:
            flops += tokens * hidden
        active_experts = self.roofline_data_terms()["active_experts"]
        weight_bytes = active_experts * (3 * ffn * hidden) * elem_bytes
        nbytes = weight_bytes + 2 * tokens * hidden * elem_bytes + routes * (4 + 4)
        return flops, nbytes
