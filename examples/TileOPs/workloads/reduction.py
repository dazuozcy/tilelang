"""Workload definitions for the reduction op family (Ascend A5).

Skeleton + per-op workload classes: concrete Workload classes for reduction
ops are appended per-op by the add-ascend-op flow (tilelang-op-conductor
Stage 0 scaffolding); the 910B workload definitions from the source
repository are deliberately not ported.
"""

from workloads.workload_base import RandnWorkload


class LogSumExpWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""


class LogSumExpSimdWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOpSimd (spec interface: shape + dtype).

    Independent SIMD-mode variant of LogSumExpWorkload — same input
    generation (randn via the device backend), separate class so the two
    operator variants stay isolated.
    """
