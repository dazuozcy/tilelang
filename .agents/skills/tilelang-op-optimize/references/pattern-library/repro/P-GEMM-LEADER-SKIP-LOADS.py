# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""repro skeleton for pattern-library entry P-GEMM-LEADER-SKIP-LOADS
(indexed_expert_mlp Stage 4 round 3).

First verified: tilelang-fork ascend dev + CANN 9.2.0 + bisheng dav-3510 +
Ascend950PR_9579, 2026-10-30 (the delta pair was compiled + L0-passed and the
effect measured on device in the origin task; this file is the extracted
before/after skeleton, see __doc__).

Optimization point: in a grouped MoE GEMM where non-leader blocks must still
run the Cube pipeline (the CONST-PrepareMultiBuffer-bool-cast-255 trap forbids
a runtime guard around T.gemm), the redundant W GM->L1 copies of non-leader
blocks are the traffic amplifier (measured 1.58x t32 / 2.32x t64 / 2.27x qwen
on the indexed expert MLP). Guard ONLY the W copies with the leader predicate
-- T.gemm stays unguarded and non-leader blocks consume stale w_l1 content,
which is unobservable because the epilogue guard already skips their
write-back.

Measured effect (do_bench(msprof), interleaved in-process protocol):
  gate/up ds-v3-t32-fp16: baseline@128:256:2 151259us -> 128:512:2 119308us
  down    ds-v3-t32-fp16: baseline@128:128:2 166493us -> 128:512:2  79600us
The guard is the single structural delta; the config move is orthogonal.
See origin task perf_opt/opt_log.md rounds 3 + final for the full tables.

This file is a BEFORE/AFTER delta skeleton extracted from the branch pair
(_indexed_expert_gemm.py vs perf_opt/_indexed_expert_gemm_round3_leaderload.py
of the origin task); the full kernel context lives there.  It is executable
only in that task tree -- here it documents the minimal change.

BEFORE (baseline, on-grid fast path inside the K soft-pipeline):

    if on_grid:  # static direct copy fast path (zero guard cost)
        T.copy(
            weights[
                expert,
                nb * block_n : (nb + 1) * block_n,
                ko * block_k : (ko + 1) * block_k,
            ],
            w_l1[0:block_n, :],
        )
        if gated:
            T.copy(
                weights[
                    expert,
                    n + nb * block_n : n + (nb + 1) * block_n,
                    ko * block_k : (ko + 1) * block_k,
                ],
                w_l1[block_n : 2 * block_n, :],
            )
    T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))

AFTER (leader-only W loads; T.gemm still unguarded):

    # guard ONLY the W copies -- NOT T.gemm (a runtime guard around T.gemm
    # puts the res L0C accumulator's write and read under differing guards
    # and crashes PrepareMultiBuffer's storage-epoch analysis).  Non-leader
    # blocks keep whatever stale w_l1 content is present; their tile is
    # never written back, so the stale data is unobservable.
    if on_grid and is_leader != 0:
        T.copy(
            weights[
                expert,
                nb * block_n : (nb + 1) * block_n,
                ko * block_k : (ko + 1) * block_k,
            ],
            w_l1[0:block_n, :],
        )
        if gated:
            T.copy(
                weights[
                    expert,
                    n + nb * block_n : n + (nb + 1) * block_n,
                    ko * block_k : (ko + 1) * block_k,
                ],
                w_l1[block_n : 2 * block_n, :],
            )
    T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))

Insight: with the ND2NZ short-strided-segment MTE2 throughput pinned at
~2 B/cyc/AIC regardless of channel (AIC direct vs AIV plain staging) and
segment length (bk 256/512/1024 all plateau), the only lever that moves the
wall is TRAFFIC VOLUME: skipping the non-leader amplification converts the
measured redundancy factor directly into latency.  The guard boundary is the
trap's own edge: everything except the L0C accumulator's write/read pair is
guardable, so the W copies (UB/L1 buffers, not L0C) take the predicate while
T.gemm keeps the unconditional path.
"""

if __name__ == "__main__":
    print(__doc__)
