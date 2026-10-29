"""Reduction kernel family (Ascend A5).

Reduction kernels for A5 are added one by one through the add-ascend-op flow
(tilelang-op-conductor Stage 0 scaffolding + Stages 1-3 kernel generation +
Stage 5 integration). The 910B kernel implementations from the source
repository are deliberately not ported.

This package hosts the backend-agnostic shared primitives in
:mod:`tileops._kernels.reduction._primitives` plus the per-op kernel packages
(each op gets its own subfolder holding the kernel class file and the
extracted GPU TileLang kernel reference file).
"""

from tileops._kernels.reduction.logsumexp import LogSumExpKernel
from tileops._kernels.reduction.logsumexp_simd import LogSumExpSimdKernel

__all__ = ["LogSumExpKernel", "LogSumExpSimdKernel"]
