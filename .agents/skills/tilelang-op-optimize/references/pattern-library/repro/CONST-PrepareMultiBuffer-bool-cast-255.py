# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: PrepareMultiBuffer crashes on a runtime guard wrapping a pipelined GEMM.

CONST id: CONST-PrepareMultiBuffer-bool-cast-255
toolchain: tilelang-fork ascend (dav-3510), 2026-10-29
首次验证: tilelang-fork ascend dev build (dav-3510) + CANN 9.2.0 + Ascend950PR_9579, 2026-10-29
（task indexed_expert_mlp Stage 3 任务内实测 REPRO_OK；2026-10-30 evolver 机械转正——除以上两行登记戳外与任务侧原件逐字节一致）

Root cause
----------
`T.gemm` writing an L0C accumulator inside a runtime ``if`` guard (here
``if is_leader != 0:``) puts the accumulator's *write* (gemm) and *read*
(``T.copy(res, acc_ub)``) under differing conditional guards.  During
``PrepareMultiBuffer`` -> ``MultiBufferPlanBuilder::Build`` ->
``AnalyzeStorageEpoch`` -> ``arith::Analyzer::CanProveEqual``, the guard
expressions are simplified and the arith simplifier folds
``cast(bool, sub(const, const))`` via ``make_const(bool, v)`` with ``v > 1``,
which trips ``IntImm::IntImm``'s ``value == 0 || value == 1`` check:

    ValueError: Check failed: (value == 0 || value == 1) is false:
    255 exceeds range of bool

The same crash fires for *any* runtime conditional guard wrapping the
pipelined gemm, regardless of comparison form (``!= 0`` / ``== 1`` / ``> 0`` /
``>= 1``) and regardless of whether the guard reads a ``T.alloc_var`` mutable
or a plain let-bound scalar.  It is *not* triggered by ``clear_accum=(ko==0)``.

Workaround
----------
Gate only the AIV1 epilogue (write-back) with the runtime guard and leave the
Cube pipeline unconditional.  Non-leader blocks compute a redundant-but-identical
tile and simply never write it back.  See
``_indexed_expert_gemm.py`` epilogue ``if sid == 1: if is_leader != 0:``.

Run: python CONST-PrepareMultiBuffer-bool-cast-255.py
"""

import sys

sys.path.insert(0, "/home/z00875089/tilelang-fork")

import tilelang
import tilelang.ascend.language as T
from tvm.target import Target

BM, BN, BK, NS = 16, 64, 64, 2
KBLOCKS = 4
GRID = 8


def _build():
    @T.prim_func
    def main(
        A: T.Tensor((GRID, 4, BK), "float16"),
        W: T.Tensor((8, BN, BK), "float16"),
        C: T.Tensor((GRID, BN), "float32"),
    ):
        with T.MixedKernel(GRID, sids=2) as (pid, sid):
            a_ub = T.alloc_shared((BM, BK), "float16")
            acc_ub = T.alloc_shared((BM, BN), "float32")
            a_l1 = T.alloc_l1((BM, BK), "float16")
            w_l1 = T.alloc_l1((BN, BK), "float16")
            res = T.alloc_l0c((BM, BN), "float32")
            route = pid
            expert = pid % 8
            is_leader = T.alloc_var("int32", init=1)
            if is_leader != 0:  # runtime guard wrapping the pipelined gemm
                for ko in T.Pipelined(KBLOCKS, num_stages=NS):
                    if sid == 0:
                        with T.SimtVF(threads=128):
                            for mi in T.serial(BM):
                                for ki in T.Parallel(BK):
                                    a_ub[mi, ki] = A[route, 0, ki]
                        T.copy(a_ub, a_l1)
                    T.copy(W[expert, :, :], w_l1)
                    T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=True)
                    if sid == 1:
                        T.copy(res, acc_ub)
                        with T.SimtVF(threads=128):
                            for mi in T.serial(BM):
                                for j in T.Parallel(BN):
                                    C[route, j] = acc_ub[mi, j]
    return main


def main():
    with Target("ascend"):
        func = _build()
        try:
            tilelang.lower(func, target="ascend")
        except ValueError as e:
            msg = str(e)
            assert "exceeds range of bool" in msg, f"unexpected failure: {msg}"
            print("REPRO_OK: reproduces the PrepareMultiBuffer bool-cast crash")
            print(f"  first line: {msg.strip().splitlines()[0][:100]}")
            return
        raise AssertionError(
            "REPRO_MISS: lowering no longer crashes; the arith simplifier bug "
            "appears fixed — re-verify the capability-gap entry after toolchain "
            "upgrade and consider removing this repro."
        )


if __name__ == "__main__":
    main()
