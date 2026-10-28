"""Ascend A5 benchmark framework.

Extracted from TileOPs (GPU/TileLang-based) and adapted for the Ascend A5
(dav-3510, ``target="ascend"``) backend.  No dependency on the original
TileOPs repository.

Layer structure (mirrors the original end-to-end flow):

    manifest  ── spec source-of-truth (YAML)
        │
    workloads ── input generation (shape + dtype)
        │
    ops       ── Op ABC: validation, reshape, kernel dispatch, roofline
        │
    kernels   ── Kernel ABC: device-specific implementation
        │
    testing   ── TestBase: correctness vs PyTorch reference
        │
    benchmark ── latency / TFLOPS / bandwidth (do_bench(msprof))

All device-specific code is funneled through :mod:`tileops.device`,
the single device adaptation surface.

The operator registries (manifest YAML files, kernel/op/workload families)
start empty: 910B kernel implementations are deliberately not ported, and
A5 kernels are generated end-to-end by tilelang-op-conductor (Stage 0
scaffolding via the add-ascend-op flow, Stages 1-3 kernel generation,
Stage 5 integration).
"""

from tileops.device import get_device_backend

__all__ = ["get_device_backend"]
