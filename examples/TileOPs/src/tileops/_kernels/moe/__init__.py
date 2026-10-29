"""MoE kernel family (Ascend A5).

Kernels for A5 are added one by one through the add-ascend-op flow
(tilelang-op-conductor Stage 0 scaffolding + Stages 1-3 kernel generation +
Stage 5 integration). The 910B kernel implementations from the source
repository are deliberately not ported.

Currently hosts the indexed small-route expert MLP kernels.
"""

from tileops._kernels.moe.indexed_expert_mlp import (
    IndexedExpertDownKernel,
    IndexedExpertGateUpKernel,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)

__all__ = [
    "IndexedExpertDownKernel",
    "IndexedExpertGateUpKernel",
    "IndexedRouteStatsKernel",
    "IndexedWeightedReduceKernel",
]
