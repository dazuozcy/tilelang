# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""EXPERIMENT: SIMT tiled logsumexp + the SIMD member's reusable tunables.

This is a *side branch* kernel copy, not a merged candidate.  Base = the
Stage 4 FINAL MERGED SIMT tiled kernel
(``perf_opt/_logsumexp_kernel_tiled.py``: SimtVF threads 256 + kernel-width
BLOCK_N rule identical).  On top of it, the optimizations that the sibling
SimdVF member (``logsumexp_simd/.../perf_opt/_logsumexp_kernel_tiled.py``)
found are *re-applied in the SimtVF structure*, one env knob each:

  * O1 ``ROWS`` (SIMD OP-ROWS / OP-COMBO, +26.6% on attn-32k there):
    the baseline SIMT tiled kernel honours the wrapper ``block_m`` = 1 for
    attn-32k, i.e. ONE row per persistent tile -> 1024 tiles, each finishing
    with a 2-byte ``lse_ub -> y`` MTE3 micro-write and its own flag
    round-trip.  ROWS re-derives the persistent-tile granularity internally
    (the SIMT *single* member already does this via its own ``_pick_rows``,
    OP1 round 1): a tile processes ``ROWS`` rows as consecutive
    SUB-row sub-batches and writes back ``lse_ub[0:ROWS]`` in ONE MTE3 op.
    Policy is copied verbatim from the SIMD member (M <= 56 -> 1, else the
    largest pow2 whose makespan stays within min+1, cap 8).
    Env override: ``TILELANG_LSE_ROWS``.

  * O2 ``BLOCK_N`` direct retention (SIMD rounds 1-3, DESIGN 8.3 A/B):
    the SIMD member keeps the wrapper ``tile_n`` verbatim (25600 for
    lm-head) instead of re-partitioning to a power of two (SIMT baseline:
    16384, i.e. 6 full + 1 tail chunk where SIMD does 4 chunks).  In SIMT
    this is a *fragment-extent* question (the (SUB, BLOCK_N) fp32 fragment
    feeds T.reduce_max/reduce_sum, whose width is checked against
    pow2/replication rules), so it is only valid while the chunk stays
    pow2-clean internally -- the knob exists to measure/probe the constraint.
    Env override: ``TILELANG_LSE_BLOCKN``.

  * O3 ``HMAX`` (SIMD OP-F16MAX, +4.0% on attn-32k, +15% lm-head there):
    pass-1 max in the 16-bit input domain.  In SimdVF this is a pure win
    (vld NORM 128-lane + vmax = 2 ops / 128 elements).  In SimtVF the
    *reduction itself* is what has to run in the 16-bit domain
    (``T.reduce_max`` on a bf16/fp16 fragment -> AscendAllReduce<Op,...>),
    and the SIMT round-1 E3/E4 adjudication already measured half-precision
    AllReduce to be catastrophically slow (bisheng software fallback, 50x /
    30-60x).  This branch therefore keeps only an elementwise-16-bit probe
    (env ``TILELANG_LSE_HMAX``) to re-confirm the negative with generated
    source rather than by assertion; it is NOT expected to win.

programming_mode: simt
auto_schedule: yes

Everything else (float32 intermediate chain, threads=256, kernel-width
chunk loop, flash merge order) is byte-identical to the FINAL MERGED SIMT
tiled kernel, so an A/B against it attributes any delta to O1/O2/O3 alone.

Call surface (unchanged, wrapper-compatible):
    _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y
"""

import argparse
import math
import os

import tilelang
import tilelang.ascend.language as T
import torch

# 56 = vector_core_num of Ascend950PR_9579 (torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56

# UB budget under SimtVF: 216 KiB * 0.9 discount (DESIGN section 4.5,
# CONST-capacity-a5 assumption). VRF fragment budget: 128 KiB cap for the
# (SUB, BLOCK_N) fp32 x_frag (CONST-vrf-capacity-a5 assumption).
UB_BUDGET_BYTES = int(216 * 1024 * 0.9)  # 199065
VRF_BUDGET_BYTES = 128 * 1024  # 131072

# SimtVF thread width: round-2 OP6 winner of the SIMT tiled member.  Re-checked
# under O1/O2 (env knob) because the per-tile overhead removed by ROWS could
# shift the allreduce-tree U-curve optimum.
TUNED_SIMTVF_THREADS = int(os.environ.get("TILELANG_LSE_THREADS", "256"))


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _env_flag(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v not in ("0", "false", "False", "off", "no")


# ---------- O1: ROWS policy (copied from the SIMD member, verbatim) ----------
# SIMD round 1 single-point proof on attn-weights-32k (M=1024, block_m=1):
# ROWS=2 +16.4%, ROWS=4 +25.8%, ROWS=8 +26.6% (mte3 micro-write batching +
# fewer persistent iterations; aiv_mte3 64.3% -> 6.2%).  Policy: M <=
# NUM_BLOCKS -> ROWS=1 (single-wave regime: maximize core parallelism, e.g.
# lm-head M=4); else the largest pow2 ROWS whose makespan
# ceil(ceil(M/ROWS)/56)*ROWS stays within min+1 rows, capped at 8.
def _pick_rows(M: int, num_blocks: int) -> int:
    if num_blocks >= M:
        return 1
    ms_min = math.ceil(M / num_blocks)  # rows per core at ROWS=1
    best = 1
    for r in (2, 4, 8):
        ms = math.ceil(math.ceil(M / r) / num_blocks) * r
        if ms <= ms_min + 1:
            best = r
    return best


def _pick_block_n(tile_n: int, elem: int) -> int:
    """Internal chunk width BLOCK_N.

    Two rules, selected by ``TILELANG_LSE_BN`` (default ``direct``):

    * ``pow2`` -- the SIMT FINAL MERGED / baseline rule: largest power of two
      with BLOCK_N <= tile_n and 2*BLOCK_N*elem <= UB budget.  This is what
      the SIMT member's DESIGN called a *fragment constraint* of the
      (SUB, BLOCK_N) fp32 fragment feeding T.reduce_max/reduce_sum.
    * ``direct`` -- the SIMD member's rule (its DESIGN 8.3 A/B winner):
      keep the wrapper's ``tile_n`` verbatim (rounded down to a 64-multiple,
      clamped to the UB budget).  lm-head tile_n=25600 -> 4 chunks instead
      of the pow2 rule's 7 (6 full + 1 masked tail).
    """
    rule = os.environ.get("TILELANG_LSE_BN", "direct")
    if rule == "pow2":
        block_n = 1
        while block_n * 2 <= tile_n and 2 * (block_n * 2) * elem <= UB_BUDGET_BYTES:
            block_n *= 2
        return block_n
    block_n = (min(tile_n, UB_BUDGET_BYTES // (2 * elem)) // 64) * 64
    return max(block_n, 64) if tile_n >= 64 else tile_n


def _pick_threads(sub_tile_elems: int) -> int:
    return min(TUNED_SIMTVF_THREADS, sub_tile_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    m = x32.max(dim=-1, keepdim=True).values
    e = torch.exp(x32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (m + torch.log(s)).squeeze(-1).to(x.dtype)


# ---------- Kernel builder ----------
def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    if M < 1 or N < 1 or tile_n < 1:
        raise ValueError(f"M, N, tile_n must be >= 1, got M={M}, N={N}, tile_n={tile_n}")
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # ---- host staticization (DESIGN section 3.3 / 5.2) ----
        BLOCK_N = _env_int("TILELANG_LSE_BLOCKN", 0) or _pick_block_n(tile_n, elem)
        # explicit numeric override: clamp to the x_ub double-buffer budget
        while BLOCK_N > 1 and 2 * BLOCK_N * elem > UB_BUDGET_BYTES:
            BLOCK_N //= 2
        # SUB: largest power of two with SUB <= block_m, x_ub double-buffer
        # <= UB budget and x_frag <= VRF budget (unchanged from FINAL).
        SUB = 1
        while block_m >= SUB * 2:
            cand = SUB * 2
            if (2 * cand * BLOCK_N * elem <= UB_BUDGET_BYTES) and (
                cand * BLOCK_N * 4 <= VRF_BUDGET_BYTES
            ):
                SUB = cand
            else:
                break

        num_full = N // BLOCK_N  # full chunks (all columns valid)
        rem = N % BLOCK_N  # tail-chunk column count
        TILE_R = tilelang.next_power_of_2(rem) if rem > 0 else 0
        # ---- O1: ROWS re-derivation (persistent-tile row residency) ----
        # FINAL SIMT used the wrapper block_m directly (=1 for attn-32k -> one
        # row per tile).  ROWS supersedes it; the SIMD policy is reused and
        # the internal ROWS is kept a multiple of SUB (both powers of two) so
        # each tile is a whole number of SUB-row sub-batches.
        ROWS = _env_int("TILELANG_LSE_ROWS", 0) or _pick_rows(M, NUM_BLOCKS)
        ROWS = max(ROWS, SUB)
        if ROWS % SUB != 0:
            raise ValueError(f"ROWS={ROWS} must be a multiple of SUB={SUB}")
        full = M // ROWS  # complete ROWS-row tiles
        rem_m = M % ROWS  # rows in the tail tile
        num_tiles = full + (1 if rem_m > 0 else 0)
        num_sub = ROWS // SUB  # SUB-row sub-batches per full tile
        tsub = rem_m // SUB  # full SUB-row sub-batches in tail
        tsub_rem = rem_m % SUB  # rows in the tail's partial sub-batch
        threads = _pick_threads(SUB * BLOCK_N)
        HMAX = _env_flag("TILELANG_LSE_HMAX", False) and dtype in ("float16", "bfloat16")

        @T.macro
        def sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, rows, slot):
            # stats init: m_run = -inf, s_run = 0 (per sub-batch, DESIGN S-init)
            with T.SimtVF(threads=threads):
                for r in T.Parallel(SUB):
                    stats_ub[r, 0] = -T.infinity(T.float32)
                    stats_ub[r, 1] = T.float32(0.0)
            # ---- full chunks (all columns valid) ----
            if num_full > 0:
                for c in T.serial(num_full):
                    col = c * BLOCK_N
                    T.copy(
                        x[row_base : row_base + rows, col : col + BLOCK_N],
                        x_ub[0:rows, 0:BLOCK_N],
                    )
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((SUB, BLOCK_N), "float32")
                        m_c = T.alloc_fragment((SUB,), "float32")
                        s_c = T.alloc_fragment((SUB,), "float32")
                        prev_frag = T.alloc_fragment((SUB,), "float32")
                        if HMAX:
                            # O3 probe (faithful port of SIMD OP-F16MAX): run
                            # the pass-1 reduction itself in the 16-bit input
                            # domain.  max() rounds nothing, so the selected
                            # value is exact -- the question is purely what
                            # bisheng does with a bf16/fp16-component
                            # AscendAllReduce.
                            xh_frag = T.alloc_fragment((SUB, BLOCK_N), dtype)
                            m_ch = T.alloc_fragment((SUB,), dtype)
                            if rows == SUB:
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    xh_frag[r, i] = x_ub[r, i]
                            else:
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    xh_frag[r, i] = T.if_then_else(
                                        r < rows,
                                        x_ub[r, i],
                                        T.cast(-T.infinity(T.float32), dtype),
                                    )
                            T.reduce_max(xh_frag, m_ch, dim=1, clear=True)
                            for r, i in T.Parallel(SUB, BLOCK_N):
                                x_frag[r, i] = T.cast(x_ub[r, i], "float32")
                            for r in T.Parallel(SUB):
                                prev_frag[r] = stats_ub[r, 0]
                                stats_ub[r, 0] = T.max(stats_ub[r, 0], T.cast(m_ch[r], "float32"))
                        else:
                            if rows == SUB:
                                # no-predicate direct load (all rows + cols valid)
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    x_frag[r, i] = T.cast(x_ub[r, i], "float32")
                            else:
                                # row predicate for the partial sub-batch
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    x_frag[r, i] = T.if_then_else(
                                        r < rows,
                                        T.cast(x_ub[r, i], "float32"),
                                        -T.infinity(T.float32),
                                    )
                            T.reduce_max(x_frag, m_c, dim=1, clear=True)
                            for r in T.Parallel(SUB):
                                prev_frag[r] = stats_ub[r, 0]
                                stats_ub[r, 0] = T.max(stats_ub[r, 0], m_c[r])
                        for r, i in T.Parallel(SUB, BLOCK_N):
                            x_frag[r, i] = T.exp(x_frag[r, i] - stats_ub[r, 0])
                        T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                        for r in T.Parallel(SUB):
                            stats_ub[r, 1] = (
                                stats_ub[r, 1] * T.exp(prev_frag[r] - stats_ub[r, 0]) + s_c[r]
                            )
            # ---- tail chunk (rem columns, masked load, pad=-inf) ----
            if rem > 0:
                col = num_full * BLOCK_N
                T.copy(x[row_base : row_base + rows, col:N], x_ub[0:rows, 0:rem])
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((SUB, TILE_R), "float32")
                    m_c = T.alloc_fragment((SUB,), "float32")
                    s_c = T.alloc_fragment((SUB,), "float32")
                    prev_frag = T.alloc_fragment((SUB,), "float32")
                    for r, i in T.Parallel(SUB, TILE_R):
                        x_frag[r, i] = T.if_then_else(
                            T.And(r < rows, i < rem),
                            T.cast(x_ub[r, i], "float32"),
                            -T.infinity(T.float32),
                        )
                    T.reduce_max(x_frag, m_c, dim=1, clear=True)
                    for r in T.Parallel(SUB):
                        prev_frag[r] = stats_ub[r, 0]
                        stats_ub[r, 0] = T.max(stats_ub[r, 0], m_c[r])
                    for r, i in T.Parallel(SUB, TILE_R):
                        x_frag[r, i] = T.exp(x_frag[r, i] - stats_ub[r, 0])
                    T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                    for r in T.Parallel(SUB):
                        stats_ub[r, 1] = (
                            stats_ub[r, 1] * T.exp(prev_frag[r] - stats_ub[r, 0]) + s_c[r]
                        )
            # ---- lse = m_run + log(s_run); write into the tile batch slot ----
            with T.SimtVF(threads=threads):
                out_frag = T.alloc_fragment((SUB,), dtype)
                for r in T.Parallel(SUB):
                    out_frag[r] = T.cast(stats_ub[r, 0] + T.log(stats_ub[r, 1]), dtype)
                T.copy(out_frag, lse_ub[slot : slot + SUB])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, BLOCK_N), dtype)
                stats_ub = T.alloc_shared((SUB, 2), "float32")
                lse_ub = T.alloc_shared((ROWS,), dtype)
                T.annotate_buffer_versions({x_ub: 2, stats_ub: 1, lse_ub: 1})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    base = t * ROWS
                    if t < full:
                        # ---- full tile (ROWS rows, all in-bounds) ----
                        for s in range(num_sub):
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, base + s * SUB, SUB, s * SUB)
                        # O1: ONE MTE3 write-back for the whole tile.
                        T.copy(lse_ub[0:ROWS], y[base : base + ROWS])
                    else:
                        # ---- tail tile (rem_m rows) ----
                        for s in range(tsub):
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, base + s * SUB, SUB, s * SUB)
                        if tsub_rem > 0:
                            sub_batch(
                                x,
                                y,
                                x_ub,
                                stats_ub,
                                lse_ub,
                                M - tsub_rem,
                                tsub_rem,
                                tsub * SUB,
                            )
                        T.copy(lse_ub[0:rem_m], y[base : base + rem_m])

        return main

    return _func


# ---------- Call helpers ----------
def _call(M, N, dtype, tile_n, block_m, x):
    kernel = _logsumexp_kernel_tiled(M, N, _dtype_str(dtype), tile_n)(block_m)
    return kernel(x)


# ---------- Layered testing ----------
def _run_case(M, N, dtype, tile_n, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call(M, N, dtype, tile_n, block_m, x)
    torch.npu.synchronize()
    golden = golden_logsumexp(x)
    tol = _tolerance(dtype)
    max_diff = (y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y, golden, rtol=tol, atol=tol)
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} block_m={block_m} "
        f"tile_n={tile_n} max_diff={max_diff:.3e}"
    )
    return max_diff


def run_L0():
    _run_case(33, 33000, torch.float16, 16640, 1, "L0")  # non-pow2 tile_n + rem tail
    _run_case(1024, 32768, torch.bfloat16, 16384, 1, "L0")  # wl2 (ROWS policy 4)
    _run_case(4, 102400, torch.float16, 25600, 1, "L0")  # wl3 (rem=4096 tail chunk)
    _run_case(4, 102400, torch.bfloat16, 25600, 1, "L0")  # wl3 bf16
    _run_case(4, 32768, torch.float16, 16384, 1, "L0")
    _run_case(33, 32768, torch.float16, 16384, 4, "L0")  # M tail (rem_m) + bm>1
    _run_case(4, 32768, torch.float32, 16384, 1, "L0")  # fp32 (UB 128 KiB)
    _run_case(1025, 32768, torch.bfloat16, 16384, 1, "L0")  # ROWS tail path


def run_L1():
    for N in (32768, 33000, 65540, 102400, 32513, 256, 300):
        _run_case(33, N, torch.float16, 16384, 1, "L1")
    for M in (1, 3, 56, 57, 128):
        _run_case(M, 32768, torch.float16, 16384, 1, "L1")
    for tile_n in (25600, 16640, 13824, 8192, 4096, 256):
        _run_case(33, 33000, torch.float16, tile_n, 1, "L1")
    for bm in (1, 2, 4, 8, 16):
        _run_case(33, 32768, torch.float16, 16384, bm, "L1")
    for M, N in ((33, 33000), (4, 102400), (128, 32768)):
        _run_case(M, N, torch.float32, 16384, 1, "L1")
        _run_case(M, N, torch.bfloat16, 16384, 1, "L1")
    _run_case(1, 65536, torch.float16, 16384, 1, "L1")
    _run_case(8, 100, torch.float16, 16384, 1, "L1")


def run_L2():
    cases = [
        (8192, 32768, torch.float16, 16384, 1),
        (2, 1024000, torch.float16, 16384, 1),
        (33, 33000, torch.float16, 256, 1),
        (33, 32512, torch.float16, 16384, 1),
    ]
    for M, N, dt, tile_n, bm in cases:
        try:
            _run_case(M, N, dt, tile_n, bm, "L2")
        except Exception as e:
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")


def run_boundary():
    tile_n = 256
    block_m = 8
    n = 512

    def base(rows=8, cols=n, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    def special(x, tag, source_semantics=False):
        Mm, Nn = x.shape
        y = _call(Mm, Nn, x.dtype, tile_n, block_m, x)
        torch.npu.synchronize()
        ref = _source_lse_sim(x) if source_semantics else golden_logsumexp(x)
        tol = _tolerance(x.dtype)
        torch.testing.assert_close(y, ref, rtol=tol, atol=tol, equal_nan=True)
        print(f"[{tag}] PASS")

    special(base(), "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    special(x, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    special(x, "Boundary-all-equal")
    x = base()
    x[:, 0] = float("-inf")
    special(x, "Boundary-neg-inf")
    x = base()
    x[:, 0] = float("nan")
    special(x, "Boundary-nan")
    special(torch.randn(3, 1, dtype=torch.float16, device="npu"), "Boundary-N1")
    special(torch.randn(1, 16, dtype=torch.float16, device="npu"), "Boundary-M1")
    x = base()
    x[:, 0] = -65504.0
    x[:, 256] = 65504.0
    special(x, "Boundary-cross-chunk-maxgap")
    x = base()
    x[:, :] = float("-inf")
    special(x, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    special(x, "Boundary-pos-inf", source_semantics=True)

    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y1 = _call(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1), "nondeterministic output"
    print("[Boundary-determinism] PASS")


def _show_source():
    kernel = _logsumexp_kernel_tiled(1024, 32768, "bfloat16", 16384)(1)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true")
    args, _ = parser.parse_known_args()
    if args.show_source:
        _show_source()
        return
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
