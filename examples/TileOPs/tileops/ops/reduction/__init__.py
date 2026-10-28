"""Reduction Op family (Ascend A5).

Hosts the shared ``_ReduceOpBase`` framework class plus per-op Op classes
added one by one through the add-ascend-op flow (tilelang-op-conductor
Stage 0 scaffolding + Stages 1-3 kernel generation + Stage 5 integration).
The 910B Op implementations from the source repository are deliberately not
ported.
"""

from tileops.ops.reduction.logsumexp import LogSumExpFwdOp
from tileops.ops.reduction.logsumexp_simd import LogSumExpFwdOpSimd
from tileops.ops.reduction.reduce import _ReduceOpBase

__all__ = ["LogSumExpFwdOp", "LogSumExpFwdOpSimd", "_ReduceOpBase"]
