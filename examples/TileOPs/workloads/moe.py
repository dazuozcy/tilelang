"""Workload definitions for the moe op family (Ascend A5).

Ported from the GPU TileOPs ``workloads/moe.py`` — only the
``IndexedExpertMLPWorkload`` chain is ported (this op is the first moe
migration); the other moe ops are not yet migrated to A5 and their
workload classes are deliberately excluded.

Adaptation of the GPU manifest-driven ``CallWorkload`` machinery: the A5
workload takes the call's dimensions and dtype directly (the A5 manifest
holds the same workload rows) and reproduces the GPU input-generation
semantics:

- hidden scaled by 0.1, both expert weights by 0.02 (keeps fp16
  accumulation over H = 7168 finite; GPU ``MoEExpertsWorkload.gen_inputs``),
- routing weights normalized with ``softmax(dim=-1)`` over each token's K
  routes,
- ``topk_ids`` rows of K distinct experts in ``[0, E)`` (the GPU manifest
  ``topk_ids(T, K, E)`` value generator),
- the GPU ``_scale_routed`` step is a no-op for this op's contract (the
  activation is fixed to ``silu_and_mul`` and the signature has no
  ``b_gate_up``/``b_down`` biases), so it is not carried over.

Device placement goes through the device backend surface (W1 of
``docs/gpu_to_ascend_adaptation.md``) instead of the GPU ``run_device()``.
"""

from __future__ import annotations

import torch

from tileops.backend.device import get_device_backend
from workloads.workload_base import WorkloadBase

__all__ = ["IndexedExpertMLPWorkload", "gated_activation", "ref_routed_experts"]


def gated_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """The gated activation over the last axis, half as wide, in the input's dtype.

    Ported from GPU ``workloads/moe.py`` (the reference definition of each
    gated activation and of its gate/up column order). ``swigluoai`` is
    computed in fp32.
    """
    if activation == "swigluoai":
        # gpt-oss's fixed alpha and limit.
        alpha, limit = 1.702, 7.0
        gate = gate_up[..., ::2].float().clamp(max=limit)
        up = gate_up[..., 1::2].float().clamp(min=-limit, max=limit)
        return ((up + 1) * gate * torch.sigmoid(alpha * gate)).to(gate_up.dtype)
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu_and_mul":
        return torch.nn.functional.gelu(gate) * up
    raise ValueError(f"unknown gated activation: {activation}")


def ref_routed_experts(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu_and_mul",
    scale: float = 1.0,
    *,
    b_gate_up: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """``scale * sum_k w[t, k] * (down_k(act(gate_up_k(h[t]) + b_gate_up[k])) + b_down[k])``

    Over each token's routed experts, in fp32 per expert, cast to the hidden dtype; each
    bias joins its GEMM's fp32 result, ``b_down`` before the routing weight scales it.

    Ported verbatim from GPU ``workloads/moe.py`` — the ground-truth reference the
    A5 golden path computes with torch on the NPU.
    """
    output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    ids = topk_ids.to(torch.int64)
    for e in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gate_up = hidden[t_idx].float() @ w_gate_up[e].float().T
        if b_gate_up is not None:
            gate_up = gate_up + b_gate_up[e].float()
        down = gated_activation(gate_up, activation) @ w_down[e].float().T
        if b_down is not None:
            down = down + b_down[e].float()
        output.index_add_(0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1))
    return (output * scale).to(hidden.dtype)


def _sample_topk_ids(T: int, K: int, E: int, device: torch.device | str) -> torch.Tensor:
    """``[T, K]`` int32 rows of K distinct experts in ``[0, E)``.

    Mirrors the GPU manifest ``topk_ids(T, K, E)`` value generator
    (``rows of k distinct random values in [0, experts)``). Sampled on the
    CPU generator so no device RNG support is required, then moved to the
    run device.
    """
    return torch.stack([torch.randperm(E)[:K] for _ in range(T)]).to(torch.int32).to(device)


class IndexedExpertMLPWorkload(WorkloadBase):
    """One ``IndexedExpertMLPFwdOp`` call: tokens, expert weights and their routing.

    The inputs and reference are the expert MLP's (GPU ``MoEExpertsWorkload``
    semantics). ``gen_inputs()`` returns the five call-time inputs in
    signature order:
    ``(hidden_states, w_gate_up, w_down, topk_weights, topk_ids)``.
    """

    def __init__(
        self,
        T: int,
        E: int,
        K: int,
        H: int,
        F: int,
        dtype: torch.dtype,
        routed_scaling_factor: float = 1.0,
    ):
        self.T = T
        self.E = E
        self.K = K
        self.H = H
        self.F = F
        self.dtype = dtype
        self.routed_scaling_factor = routed_scaling_factor

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        backend = get_device_backend()
        device = backend.name
        # Small scales keep fp16 accumulation over H = 7168 finite (GPU
        # MoEExpertsWorkload); routing weights sum to one.
        hidden = torch.randn(self.T, self.H, dtype=self.dtype, device=device) * 0.1
        w_gate_up = torch.randn(self.E, 2 * self.F, self.H, dtype=self.dtype, device=device) * 0.02
        w_down = torch.randn(self.E, self.H, self.F, dtype=self.dtype, device=device) * 0.02
        topk_ids = _sample_topk_ids(self.T, self.K, self.E, device)
        topk_weights = torch.randn(self.T, self.K, dtype=torch.float32, device=device)
        return (
            hidden,
            w_gate_up,
            w_down,
            topk_weights.softmax(dim=-1),
            topk_ids,
        )

    def ref_program(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return ref_routed_experts(
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            "silu_and_mul",
            self.routed_scaling_factor,
        )
