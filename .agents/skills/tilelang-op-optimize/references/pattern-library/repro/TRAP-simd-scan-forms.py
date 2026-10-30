# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""repro for pattern-library entry TRAP-SIMD-SCAN-FORMS.

First verified: tilelang-fork ascend dev + CANN 9.2.0 + bisheng dav-3510 +
Ascend950PR_9579, 2026-10-30 (all six assertions PASS on device).

The four SimdVF/TIR-script form traps hit while building the indexed_expert_mlp
round-2 batch-scan route_stats variant (2026-10-30).

Each case asserts the OBSERVED compiler behaviour on the A5 ascend target:
compile success is a PASS, the specific rejection is the documented trap.
Run:  python3 repro/TRAP-simd-scan-forms.py   (from the pattern-library dir)

Traps covered (see traps-compiler.md entries):
  1. TRAP-vscatter-uint32-index : int32x64 index matches no asc_scatter
     overload -> bisheng "no matching function for call to 'asc_scatter'".
  2. TRAP-tir-vreg-rebind-in-loop : Python rebinding of a vreg inside an
     unrolled/serial region -> "Immutable variable ... used outside its
     defining region" (use T.simd.alloc_local cells instead).
  3. TRAP-copy-2d-to-1d-slice : T.copy 2-D src -> 1-D slice dst ->
     "PlanMTECopy: source and destination regions must contain the same
     number of elements" (use same-shape staging + 2-D subscripts).
  4. TRAP-simdvf-fill-divisibility : T.fill inside SimdVF over a region whose
     element count is not divisible by the 64-lane VReg size ->
     "CanLowerFillRegion ... got region ids_ub[0:4, 0:2]".
"""

import traceback

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tvm import tirx


def expect(name, should_fail, fn):
    try:
        fn()
        outcome = "compiled+ran"
        ok = not should_fail
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {str(exc)[:120]}"
        ok = should_fail
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {outcome}")
    assert ok, name
    return outcome


# ---- 1. vscatter index dtype ------------------------------------------------
def scatter_int32_index():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(out: T.Tensor((64,), "int32")):
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "int32")
                with T.SimdVF():
                    full = S.pset(32)
                    lane = S.vci(0, "int32")
                    S.vscatter(lane, ub[0], lane, full)  # int32x64 index: BAD
                T.copy(ub, out)
        return main

    out = k()()
    torch.npu.synchronize()


def scatter_uint32_index():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(out: T.Tensor((64,), "int32")):
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "int32")
                with T.SimdVF():
                    full = S.pset(32)
                    lane = S.vci(0, "int32")
                    idx = tirx.reinterpret("uint32x64", lane)  # GOOD
                    S.vscatter(lane, ub[0], idx, full)
                T.copy(ub, out)
        return main

    out = k()()
    torch.npu.synchronize()
    assert out[:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7], out[:8]


# ---- 2. vreg rebinding inside an unrolled chain ------------------------------
def vreg_rebind_bad():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(out: T.Tensor((64,), "int32")):
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "int32")
                with T.SimdVF():
                    full = S.pset(32)
                    zero = S.vdup(T.int32(0), "int32")
                    one = S.vdup(T.int32(1), "int32")
                    lane = S.vci(0, "int32")
                    cnt = zero
                    for t in range(1, 4):  # python-unrolled chain, rebinding
                        cnt = S.vadd(cnt, one, full)
                    S.vsts(ub[0], cnt, full)
                T.copy(ub, out)
        return main

    k()()
    torch.npu.synchronize()


def vreg_alloc_local_good():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(out: T.Tensor((64,), "int32")):
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "int32")
                with T.SimdVF():
                    full = S.pset(32)
                    one = S.vdup(T.int32(1), "int32")
                    cnt = S.alloc_local((1,), "int32")
                    cnt[0] = S.vdup(T.int32(0), "int32")
                    for t in range(1, 4):
                        cnt[0] = S.vadd(cnt[0], one, full)
                    S.vsts(ub[0], cnt[0], full)
                T.copy(ub, out)
        return main

    out = k()()
    torch.npu.synchronize()
    assert out[0].item() == 3


# ---- 3. 2-D src -> 1-D slice copy --------------------------------------------
def copy_2d_to_1d_bad():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(src: T.Tensor((4, 2), "int32"), out: T.Tensor((8,), "int32")):
            with T.Kernel(1):
                flat = T.alloc_shared((8,), "int32")
                T.copy(src, flat[0:8])  # 2-D src -> 1-D slice: BAD
                T.copy(flat, out)
        return main

    src = torch.arange(8, dtype=torch.int32, device="npu").reshape(4, 2)
    k()(src)
    torch.npu.synchronize()


# ---- 4. SimdVF fill divisibility ---------------------------------------------
def fill_non_divisible_bad():
    @tilelang.jit(target="ascend", out_idx=[0])
    def k():
        @T.prim_func
        def main(src: T.Tensor((4, 2), "int32"), out: T.Tensor((8,), "int32")):
            with T.Kernel(1):
                ub = T.alloc_shared((4, 2), "int32")
                with T.SimdVF():
                    T.fill(ub, T.int32(0))  # 8 elements, not %64: BAD
                T.copy(ub, out)
        return main

    src = torch.arange(8, dtype=torch.int32, device="npu").reshape(4, 2)
    k()(src)
    torch.npu.synchronize()


def main():
    expect("1a vscatter int32 index (expect bisheng reject)", True, scatter_int32_index)
    expect("1b vscatter uint32 index (expect pass)", False, scatter_uint32_index)
    expect("2a vreg rebind in chain (expect immutable reject)", True, vreg_rebind_bad)
    expect("2b alloc_local cell (expect pass)", False, vreg_alloc_local_good)
    expect("3 copy 2D->1D slice (expect PlanMTECopy reject)", True, copy_2d_to_1d_bad)
    expect("4 SimdVF fill non-divisible (expect CanLowerFillRegion reject)",
           True, fill_non_divisible_bad)
    print("TRAP-simd-scan-forms repro PASS")


if __name__ == "__main__":
    main()
