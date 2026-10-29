"""MoE Op family (Ascend A5).

Ops are added one by one through the add-ascend-op flow (tilelang-op-conductor
Stage 0 scaffolding + Stages 1-3 kernel generation + Stage 5 integration).
The 910B Op implementations from the source repository are deliberately not
ported.

Currently hosts the indexed small-route expert MLP op.
"""

from tileops._ops.moe.indexed_expert_mlp import IndexedExpertMLPFwdOp

__all__ = ["IndexedExpertMLPFwdOp"]
