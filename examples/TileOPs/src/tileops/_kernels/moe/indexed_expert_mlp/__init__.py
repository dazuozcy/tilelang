"""Indexed small-route expert MLP kernel package (Ascend A5).

Hosts the kernel wrapper (:mod:`tileops._kernels.moe.indexed_expert_mlp.indexed_expert_mlp`)
and the extracted GPU TileLang kernel reference file. Re-exports the kernel
classes so the family-level ``__init__.py`` can import them by package name.
"""

from tileops._kernels.moe.indexed_expert_mlp.indexed_expert_mlp import (
    IndexedExpertCall,
    IndexedExpertDownKernel,
    IndexedExpertGateUpKernel,
    IndexedExpertGEMMTemplate,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)

__all__ = [
    "IndexedExpertCall",
    "IndexedExpertDownKernel",
    "IndexedExpertGEMMTemplate",
    "IndexedExpertGateUpKernel",
    "IndexedRouteStatsKernel",
    "IndexedWeightedReduceKernel",
]
