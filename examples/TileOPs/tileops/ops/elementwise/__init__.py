"""Elementwise Op family (Ascend A5).

Empty scaffold: elementwise ops for A5 are added one by one through the
add-ascend-op flow (tilelang-op-conductor Stage 0 scaffolding + Stages 1-3
kernel generation + Stage 5 integration). The 910B Op implementations from
the source repository are deliberately not ported.
"""
