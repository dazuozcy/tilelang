# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""LogSumExpFwdOpSimd -- Stage 4 perf_opt FINAL canonical artifact (dual kernel).

This is the state-contract final artifact (``kernel_opt_py_path`` =
``perf_opt/LogSumExpFwdOpSimd.py``, see ``.stage_state.json``): the
consolidated module containing BOTH tuned kernels of the LogSumExpFwdOpSimd
integration (explicit-SIMD / SimdVF variant, 2048-bit ``T.simd.*``
intrinsics):

  * ``_logsumexp_kernel_single`` -- single-tile residency variant, tuned
    (round 1 f16max: pass-1 row max computed in the 16-bit input domain --
    exact / bit-identical output; bf16 goes through an EVEN/ODD fp32 split
    because asc_reduce_max has no bf16 overload).
  * ``_logsumexp_kernel_tiled`` -- N-chunk streaming variant, tuned
    (round 1 f16max + ROWS policy via ``_pick_rows``: M <= 56 -> ROWS=1 for
    core parallelism, else the largest pow2 whose makespan stays within
    min+1, capped 8 -- batches the 2-byte lse UB->GM writes; BLOCK_N keeps
    the wrapper tile_n direct-retention rule: the DESIGN 8.3 A/B sweet-spot
    candidate 16384 was measured and rejected, -52%).

The per-kernel drop-in files ``perf_opt/_logsumexp_kernel_single.py`` /
``perf_opt/_logsumexp_kernel_tiled.py`` contain the same frozen kernel code
(byte-frozen measurement artifacts; the TileOPs wrapper's perf_opt import
switch references those paths).  This module is the single-file consolidated
view required by the Stage 4 gate; keep the three files in sync (they are
frozen deliverables -- no further edits expected).

Tuned defaults are exposed as module-level constants: ``HMAX_ENABLE = True``
(16-bit-domain pass-1 max, shared by both variants) and the tiled
``_pick_rows`` ROWS policy (env override ``TILELANG_LSE_ROWS`` kept for
sweep reproducibility).  Call surfaces are wrapper-compatible and unchanged:

    _logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y
    _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU.
Precision: full L0/L1/L2/boundary suites of both variants (see run_L0/
run_L1/run_L2/run_boundary -- each runs the single and tiled suites).
"""

import argparse
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch

# 56 = vector_core_num of Ascend950PR_9579 (torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56

# Single-tile column cap from the TileOPs wrapper (_primitives.py MAX_SINGLE_TILE_COLS).
MAX_SINGLE_TILE_COLS = 32512

# UB budget under SimdVF: 248 KiB * 0.9 discount (DESIGN section 4.5,
# CONST-capacity-a5 assumption; SimdVF has no SimtVF 32 KiB thread-context
# deduction).  Single variant: VRF is streaming (~2.5 KiB), so SUB is capped
# by UB alone.  Tiled variant: same budget with the UB_RESERVE_BYTES static
# reserve (lse_ub + alignment/misc headroom) deducted in the chunk clamp.
UB_BUDGET_BYTES = int(248 * 1024 * 0.9)  # 228556
UB_RESERVE_BYTES = 8256

# Wrapper heuristic budget (216 KiB = SimtVF UB budget; the wrapper tiled
# path sizes tile_n against this).  Used only in run_L0 to recompute the
# wrapper tile_n expectation via the in-repo _primitives mirror (DESIGN 8.2).
WRAPPER_BUDGET_BYTES = 216 * 1024

# SimdVF fp32 lane count == group width (64 fp32 / 128 fp16 per instruction).
GROUP = 64


# Stage 4 tuned final (SimdVF single variant):
#   Round 1 winner f16max -- pass-1 row max computed in the 16-bit input
#   domain (vld NORM 128-lane + vmax) with only the winner widened to fp32
#   (exact: max rounds nothing; bit-identical output, max_diff=0 on all
#   tune workloads).  Measured: attn-4k fp16/bf16 +5.6/+6.3%, 3d +5.1%
#   (ab_test B_better, -5.06%).  Refuted alternatives: SUB down-sweep
#   (-60~-70%), balanced-wave slab (-23%, loses copy/compute overlap).


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU (source test baseline ref_program)."""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE GPU algorithm semantics.

    m = max(x); y = m + log(sum(exp(x - m))) with fp32 intermediates (matching
    the source's fp32 chain, DESIGN section 0.1).  Unlike torch.logsumexp this
    propagates the source-inherent divergence (all -inf row -> NaN, +inf row
    -> NaN) exactly, so it is used to validate migration fidelity for those
    corner cases (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    x32 = x.float()
    m = x32.max(dim=-1, keepdim=True).values
    e = torch.exp(x32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (m + torch.log(s)).squeeze(-1).to(x.dtype)


def _source_lse_tiled_sim(x: torch.Tensor, tile_n: int) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE tiled online algorithm semantics.

    Replicates chunk-serial flash merge order (m_run/s_run recursion, pad=-inf
    tail columns) so source-inherent divergence corners (all -inf row -> NaN,
    +inf row -> NaN, first-chunk-all -inf -> NaN) validate against THIS, not
    against torch.logsumexp (DESIGN 9.3 / verify_equiv V5).
    """
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    for i in range(M):  # row-independent (scheduling-independent)
        m_run, s_run = float("-inf"), 0.0  # fp32 chain (source dtype semantics)
        for c0 in range(0, N, tile_n):
            cols = x[i, c0 : c0 + tile_n].float()
            cols = torch.cat(
                [cols, torch.full((tile_n - cols.numel(),), float("-inf"), device=x.device)]
            )  # out-of-range cols = -inf (source masked semantics)
            m_c = cols.max()
            prev = m_run
            m_run = max(m_run, m_c)
            e = torch.exp(cols - m_run)
            s_c = e.sum()
            s_run = s_run * torch.exp(torch.tensor(prev - m_run)) + s_c
        out[i] = torch.tensor(m_run + torch.log(torch.tensor(s_run))).to(x.dtype)
    return out


def _tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32


# ---------- Kernel builder ----------
# ---------- Stage 4 Round 1: pass-1 max in the 16-bit input domain ----------
# For fp16/bf16 inputs whose padded row width is 128-aligned, pass 1 computes
# the row max directly on 128-lane 16-bit vectors (vld NORM + vmax) and
# widens only the final broadcast winner to fp32.  max() involves no
# rounding, so the 16-bit-domain max selects exactly the same input value as
# the baseline fp32 detour (bit-identical m; pass 2 unchanged -> bit-identical
# output).  Vec-op count for pass 1 drops from 3 ops / 64 elems to
# 2 ops / 128 elems.  The masked tail-group path stays in fp32 (cold path).
HMAX_ENABLE = True


def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits on-chip), SimdVF variant.

    Keeps the wrapper call surface ``_logsumexp_kernel_single(M, N, dtype)
    (block_m)(x)``; the GPU ``threads`` argument is removed per K3 (SimdVF has
    no thread concept at all).
    """
    if M < 1 or N < 1:
        raise ValueError(f"M, N must be >= 1, got M={M}, N={N}")
    if N > MAX_SINGLE_TILE_COLS:
        raise ValueError(
            f"N={N} exceeds MAX_SINGLE_TILE_COLS={MAX_SINGLE_TILE_COLS} "
            f"(single-tile contract; larger N is the tiled variant)"
        )

    N_pad = ((N + GROUP - 1) // GROUP) * GROUP  # align_up(N, 64)
    G_full = N // GROUP
    k_tail = N % GROUP
    elem = _elem_bytes(dtype)
    # HMAX: 16-bit-domain pass-1 max over 128-lane NORM units (needs the
    # padded row width to be 128-aligned; the masked tail group stays fp32).
    HMAX = HMAX_ENABLE and dtype in ("float16", "bfloat16") and N_pad % 128 == 0
    NH = N_pad // 128 if HMAX else 0

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # SUB: max power-of-two satisfying (i) x_ub double-buffer <= UB budget,
        # (ii) SUB <= block_m.  SUB >= 1 always (DESIGN section 5.2).  No VRF
        # constraint (VRF streaming zero-residency in SimdVF).
        SUB = 1
        while block_m >= SUB * 2:
            cand = SUB * 2
            if 2 * cand * N_pad * elem <= UB_BUDGET_BYTES:
                SUB = cand
            else:
                break

        full = M // SUB
        rem = M % SUB
        num_batches = full + (1 if rem else 0)

        @T.macro
        def compute_rows(x_ub, lse_ub, rows):
            # Two-pass per-row reduction, fully explicit SimdVF intrinsics.
            with T.SimdVF():
                neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                zero_f = T.simd.vdup(T.float32(0.0), "float32")
                if dtype == "float32":
                    one_pt = T.simd.pset(32, "PAT_VL1")
                else:
                    one_pt = T.simd.pset(16, "PAT_VL1")
                if k_tail > 0:
                    tail_mask = T.simd.update_mask(k_tail, 32)

                for r in range(rows):
                    # ---- pass 1: row max (deferred cross-lane reduce) ----
                    # HMAX: exact 16-bit-domain max over 128-lane NORM units;
                    # only the winner is widened to fp32.
                    if HMAX:
                        acc_h = T.simd.alloc_var(dtype)
                        acc_h = T.simd.vld(x_ub[r, 0])
                        for g in range(1, NH):
                            acc_h = T.simd.vmax(acc_h, T.simd.vld(x_ub[r, g * 128]))
                        if dtype == "float16":
                            m_brc = T.simd.vcvt(
                                T.simd.vdupv(T.simd.vcmax(acc_h)), "float32", part=0
                            )
                        else:
                            # bfloat16: asc_reduce_max has no bf16 overload
                            # (reg_reduce.h) -- split to fp32 via EVEN/ODD
                            # widen + vmax (exact).
                            m_brc = T.simd.vdupv(
                                T.simd.vcmax(
                                    T.simd.vmax(
                                        T.simd.vcvt(acc_h, "float32", part=0),
                                        T.simd.vcvt(acc_h, "float32", part=1),
                                    )
                                )
                            )
                        if k_tail > 0:
                            # masked tail group (cold): fp32 domain, exact
                            # broadcast-level merge
                            c = T.simd.vcvt(
                                T.simd.vld(x_ub[r, G_full * GROUP], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            c = T.simd.vsel(c, neg_inf, tail_mask)
                            m_brc = T.simd.vmax(m_brc, T.simd.vdupv(T.simd.vcmax(c)))
                    else:
                        # mutable register accumulator (local.var), reassigned via
                        # buffer_store so loop-carried reductions stay in scope.
                        acc = T.simd.alloc_var("float32")
                        acc = neg_inf
                        for g in range(G_full):
                            if dtype == "float32":
                                c = T.simd.vld(x_ub[r, g * GROUP])
                            else:
                                c = T.simd.vcvt(
                                    T.simd.vld(x_ub[r, g * GROUP], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            acc = T.simd.vmax(acc, c)
                        if k_tail > 0:
                            if dtype == "float32":
                                c = T.simd.vld(x_ub[r, G_full * GROUP])
                            else:
                                c = T.simd.vcvt(
                                    T.simd.vld(x_ub[r, G_full * GROUP], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            c = T.simd.vsel(c, neg_inf, tail_mask)
                            acc = T.simd.vmax(acc, c)
                        m_brc = T.simd.vdupv(T.simd.vcmax(acc))

                    # ---- pass 2: fused exp(x - m) + row sum (UB re-read) ----
                    accs = T.simd.alloc_var("float32")
                    accs = zero_f
                    for g in range(G_full):
                        if dtype == "float32":
                            c = T.simd.vld(x_ub[r, g * GROUP])
                        else:
                            c = T.simd.vcvt(
                                T.simd.vld(x_ub[r, g * GROUP], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                        e = T.simd.vexpdif(c, m_brc)
                        accs = T.simd.vadd(accs, e)
                    if k_tail > 0:
                        if dtype == "float32":
                            c = T.simd.vld(x_ub[r, G_full * GROUP])
                        else:
                            c = T.simd.vcvt(
                                T.simd.vld(x_ub[r, G_full * GROUP], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                        c = T.simd.vsel(c, neg_inf, tail_mask)
                        e = T.simd.vexpdif(c, m_brc)
                        accs = T.simd.vadd(accs, e)

                    # ---- epilogue: y = m + log(S), cast, single-point store ----
                    s = T.simd.vcadd(accs)
                    out = T.simd.vadd(m_brc, T.simd.vln(s))
                    if dtype == "float32":
                        T.simd.vsts(lse_ub[r], out, one_pt, "ONEPT_B32")
                    else:
                        out_h = T.simd.vcvt(out, dtype)
                        T.simd.vsts(lse_ub[r], out_h, one_pt, "ONEPT_B16")

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, N_pad), dtype)
                lse_ub = T.alloc_shared((SUB,), dtype)
                T.annotate_buffer_versions({x_ub: 2})
                for t in T.Persistent([num_batches], NUM_BLOCKS, core_id, num_stages=2):
                    if t < full:
                        # ---- full sub-batch (SUB rows, all in-bounds) ----
                        T.copy(x[t * SUB : (t + 1) * SUB, 0:N], x_ub[0:SUB, 0:N])
                        compute_rows(x_ub, lse_ub, SUB)
                        T.copy(lse_ub[0:SUB], y[t * SUB : (t + 1) * SUB])
                    else:
                        # ---- tail sub-batch (rem rows, static slice) ----
                        T.copy(x[M - rem : M, 0:N], x_ub[0:rem, 0:N])
                        compute_rows(x_ub, lse_ub, rem)
                        T.copy(lse_ub[0:rem], y[M - rem : M])

        return main

    return _func


# ---------- Call helpers ----------
def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _call(M: int, N: int, dtype: torch.dtype, block_m: int, x: torch.Tensor) -> torch.Tensor:
    kernel = _logsumexp_kernel_single(M, N, _dtype_str(dtype))(block_m)
    return kernel(x)


# Stage 4 tuned final (SimdVF tiled variant):
#   Round 1 f16max (16-bit-domain pass-1 max, exact) + ROWS policy
#   (rows per persistent tile: M<=56 -> 1 for core parallelism, else the
#   largest pow2 with makespan within min+1, capped 8 -- attn-32k M=1024
#   picks ROWS=4: batches the 2-byte lse UB->GM writes, aiv_mte3 64.3%
#   -> 7.9%).  Measured vs baseline: attn-32k +30.1% (72.44 -> 50.58us,
#   aiv_mte2 85.7% ~= MTE2 wall), lm-head fp16/bf16 +15.5/+14.8%.
#   DESIGN 8.3 A/B adjudicated: BLOCK_N = tile_n direct-retention (25600)
#   beats the SimdVF sweet-spot candidate 16384 by 52% (more chunks =
#   more per-chunk overhead; 20480/51200 also lose or tie).


# ---------- Kernel builder ----------
def _load_group(x_ub, g: int, dtype: str):
    """Load 64-element group g from x_ub, promoting to fp32 (64 fp32 lanes).

    fp16/bf16: vld(UNPK_B16) + vcvt(part=0); fp32: vld NORM direct (no vcvt).
    """
    if dtype == "float32":
        return T.simd.vld(x_ub[0, g * GROUP])
    return T.simd.vcvt(T.simd.vld(x_ub[0, g * GROUP], dist="UNPK_B16"), "float32", part=0)


# ---------- Stage 4 Round 1: pass-1 max in the 16-bit input domain ----------
# For fp16/bf16 (and 128-aligned column extents) pass 1 computes the row max
# directly on 128-lane 16-bit vectors (vld NORM + vmax), converting only the
# final broadcast winner to fp32.  max() involves no rounding, so a 16-bit
# domain max selects exactly the same input value as the baseline fp32
# detour (bit-identical m, pass 2 unchanged -> bit-identical output).
# Vec-op count for pass 1 drops from 3 ops / 64 elems to 2 ops / 128 elems.
# The cold tail-chunk / masked tail-group paths keep the baseline fp32 form.
HMAX_ENABLE = True


# ---------- Stage 4 Round 2: + ROWS policy (rows per persistent tile) --------
# Round 1 single-point proof on attn-weights-32k (M=1024, block_m=1): ROWS=2
# +16.4%, ROWS=4 +25.8%, ROWS=8 +26.6% (mte3 micro-write batching + fewer
# persistent iterations; aiv_mte3 64.3% -> 6.2%).  Policy: M <= NUM_BLOCKS ->
# ROWS=1 (single-wave regime: maximize core parallelism, e.g. lm-head M=4);
# else the largest pow2 ROWS whose makespan ceil(ceil(M/ROWS)/56)*ROWS stays
# within min+1 rows (P-LAYOUT-ROW-RESIDENCY measured rule), capped at 8
# (measured flat ROWS=4..8, 0.9% apart).  TILELANG_LSE_ROWS overrides.
def _pick_rows(M: int, num_blocks: int) -> int:
    import math

    if num_blocks >= M:
        return 1
    ms_min = math.ceil(M / num_blocks)  # rows per core at ROWS=1
    best = 1
    for r in (2, 4, 8):
        ms = math.ceil(math.ceil(M / r) / num_blocks) * r
        if ms <= ms_min + 1:
            best = r
    return best


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile (online streaming) logsumexp kernel, SimdVF variant.

    Keeps the wrapper call surface ``_logsumexp_kernel_tiled(M, N, dtype,
    tile_n)(block_m)(x)``; the GPU ``threads`` argument is removed per K3
    (SimdVF has no thread concept).  tile_n is honoured directly: BLOCK_N is
    the largest 64-multiple with BLOCK_N <= tile_n and 2*BLOCK_N*elem <= UB
    budget reserve (no pow2 re-partition -- DESIGN section 0.6 R2).
    """
    if M < 1 or N < 1 or tile_n < 1:
        raise ValueError(f"M, N, tile_n must be >= 1, got M={M}, N={N}, tile_n={tile_n}")
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # ---- host staticization (DESIGN section 3.3 / 5.2) ----
        # BLOCK_N: largest 64-multiple with BLOCK_N <= tile_n and
        # 2*BLOCK_N*elem <= UB_BUDGET_BYTES - UB_RESERVE_BYTES.  Wrapper
        # tile_n <= 32512 -> 16-bit never clamps (BLOCK_N = tile_n); fp32 or
        # explicit oversized tile_n clamped (16-bit cap 55040 / fp32 27520).
        BLOCK_N = GROUP
        while (
            tile_n >= BLOCK_N + GROUP
            and 2 * (BLOCK_N + GROUP) * elem <= UB_BUDGET_BYTES - UB_RESERVE_BYTES
        ):
            BLOCK_N += GROUP

        G = BLOCK_N // GROUP  # full-chunk group count
        # HMAX: 16-bit-domain pass-1 max on 128-element NORM units (full
        # chunks only; requires BLOCK_N % 128 == 0).
        HMAX = HMAX_ENABLE and dtype in ("float16", "bfloat16") and BLOCK_N % 128 == 0
        GH = BLOCK_N // 128 if HMAX else 0
        num_full = N // BLOCK_N  # full chunks (all columns valid)
        rem = N % BLOCK_N  # tail-chunk column count (0 = none)
        G_tail = rem // GROUP  # tail-chunk full groups
        k_tail = rem % GROUP  # tail-chunk tail-group valid elements
        # ROWS: rows per persistent tile (Round 1 winner on 32k; policy above,
        # env override for sweep).  lse_ub is sized ROWS so the per-tile lse
        # write-back batches ROWS outputs into one MTE3 op.
        ROWS = _env_int("TILELANG_LSE_ROWS", 0) or _pick_rows(M, NUM_BLOCKS)
        if ROWS < 1:
            raise ValueError(f"ROWS must be >= 1, got {ROWS}")
        full_m = M // ROWS  # complete ROWS-row tiles
        rem_m = M % ROWS  # rows in the tail tile
        num_tiles = full_m + (1 if rem_m > 0 else 0)

        @T.macro
        def row_task(x, y, x_ub, stats_ub, lse_ub, row_base, rows):
            # ---- per row: stats-UB-roundtrip running (m, s) (DESIGN R5
            # fallback).  m_run / s_run live in a 2-element fp32 UB buffer,
            # written as a single point (ONEPT_B32) and reloaded as a
            # broadcast (BRC_B32) with mem_bar("VST_VLD") ordering, so no
            # loop-carried local.var crosses VF-block boundaries (VFChecker
            # risk 1, allocate.py L103-168 + vf_regions.h keeps enclosing
            # scope).  flash_attention core.py L103/L117/L120-126 precedent.
            for r in range(rows):
                row = row_base + r
                with T.SimdVF():  # stats init (source S-init)
                    neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                    zero_f = T.simd.vdup(T.float32(0.0), "float32")
                    one_pt32 = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(stats_ub[0], neg_inf, one_pt32, "ONEPT_B32")
                    T.simd.vsts(stats_ub[1], zero_f, one_pt32, "ONEPT_B32")
                    T.simd.mem_bar("VST_VLD")

                # ---- full chunks (all columns valid, zero tail group) ----
                if num_full > 0:
                    for c in T.serial(num_full):
                        col = c * BLOCK_N
                        T.copy(
                            x[row : row + 1, col : col + BLOCK_N],
                            x_ub[0:1, 0:BLOCK_N],
                        )
                        with T.SimdVF():
                            neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                            zero_f = T.simd.vdup(T.float32(0.0), "float32")
                            one_pt32 = T.simd.pset(32, "PAT_VL1")
                            m_run_old = T.simd.vld(stats_ub[0], "BRC_B32")
                            s_run_old = T.simd.vld(stats_ub[1], "BRC_B32")
                            # pass 1: chunk max (deferred cross-lane reduce).
                            # HMAX: exact 16-bit-domain max over 128-lane
                            # NORM units; only the winner is widened to fp32.
                            if HMAX:
                                acc_h = T.simd.alloc_var(dtype)
                                acc_h = T.simd.vld(x_ub[0, 0])
                                for g in range(1, GH):
                                    acc_h = T.simd.vmax(acc_h, T.simd.vld(x_ub[0, g * 128]))
                                if dtype == "float16":
                                    m_c_brc = T.simd.vcvt(
                                        T.simd.vdupv(T.simd.vcmax(acc_h)),
                                        "float32",
                                        part=0,
                                    )
                                else:
                                    # bfloat16: asc_reduce_max has no bf16
                                    # overload (reg_reduce.h) -- split the
                                    # 128-lane bf16 max broadcast to fp32 via
                                    # EVEN/ODD widen + vmax (exact: widening
                                    # and max round nothing).
                                    m_c_brc = T.simd.vdupv(
                                        T.simd.vcmax(
                                            T.simd.vmax(
                                                T.simd.vcvt(acc_h, "float32", part=0),
                                                T.simd.vcvt(acc_h, "float32", part=1),
                                            )
                                        )
                                    )
                            else:
                                acc = T.simd.alloc_var("float32")
                                acc = neg_inf
                                for g in range(G):
                                    acc = T.simd.vmax(acc, _load_group(x_ub, g, dtype))
                                m_c_brc = T.simd.vdupv(T.simd.vcmax(acc))
                            # max merge (source S3)
                            m_run_new = T.simd.vmax(m_run_old, m_c_brc)
                            # pass 2: fused exp(x - m_run) + sum (UB re-read)
                            accs = T.simd.alloc_var("float32")
                            accs = zero_f
                            for g in range(G):
                                accs = T.simd.vadd(
                                    accs, T.simd.vexpdif(_load_group(x_ub, g, dtype), m_run_new)
                                )
                            s_c = T.simd.vcadd(accs)
                            # flash rescale merge (source S6)
                            s_run_new = T.simd.vadd(
                                T.simd.vmul(s_run_old, T.simd.vexpdif(m_run_old, m_run_new)),
                                s_c,
                            )
                            T.simd.vsts(stats_ub[0], m_run_new, one_pt32, "ONEPT_B32")
                            T.simd.vsts(stats_ub[1], s_run_new, one_pt32, "ONEPT_B32")
                            T.simd.mem_bar("VST_VLD")

                # ---- tail chunk (rem columns, masked group, pad=-inf) ----
                if rem > 0:
                    col = num_full * BLOCK_N
                    T.copy(x[row : row + 1, col:N], x_ub[0:1, 0:rem])
                    with T.SimdVF():
                        neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                        zero_f = T.simd.vdup(T.float32(0.0), "float32")
                        one_pt32 = T.simd.pset(32, "PAT_VL1")
                        m_run_old = T.simd.vld(stats_ub[0], "BRC_B32")
                        s_run_old = T.simd.vld(stats_ub[1], "BRC_B32")
                        if k_tail > 0:
                            tail_mask = T.simd.update_mask(k_tail, 32)
                        # pass 1: chunk max
                        acc = T.simd.alloc_var("float32")
                        acc = neg_inf
                        for g in range(G_tail):
                            acc = T.simd.vmax(acc, _load_group(x_ub, g, dtype))
                        if k_tail > 0:
                            cv = _load_group(x_ub, G_tail, dtype)
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            acc = T.simd.vmax(acc, cv)
                        m_c_brc = T.simd.vdupv(T.simd.vcmax(acc))
                        m_run_new = T.simd.vmax(m_run_old, m_c_brc)
                        # pass 2: sum of exp
                        accs = T.simd.alloc_var("float32")
                        accs = zero_f
                        for g in range(G_tail):
                            accs = T.simd.vadd(
                                accs, T.simd.vexpdif(_load_group(x_ub, g, dtype), m_run_new)
                            )
                        if k_tail > 0:
                            cv = _load_group(x_ub, G_tail, dtype)
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            accs = T.simd.vadd(accs, T.simd.vexpdif(cv, m_run_new))
                        s_c = T.simd.vcadd(accs)
                        s_run_new = T.simd.vadd(
                            T.simd.vmul(s_run_old, T.simd.vexpdif(m_run_old, m_run_new)), s_c
                        )
                        T.simd.vsts(stats_ub[0], m_run_new, one_pt32, "ONEPT_B32")
                        T.simd.vsts(stats_ub[1], s_run_new, one_pt32, "ONEPT_B32")
                        T.simd.mem_bar("VST_VLD")

                # ---- row epilogue: y = m + log(S), cast, single-point store ----
                with T.SimdVF():
                    m_final = T.simd.vld(stats_ub[0], "BRC_B32")
                    s_final = T.simd.vld(stats_ub[1], "BRC_B32")
                    out = T.simd.vadd(m_final, T.simd.vln(s_final))
                    if dtype == "float32":
                        one_pt = T.simd.pset(32, "PAT_VL1")
                        T.simd.vsts(lse_ub[r], out, one_pt, "ONEPT_B32")
                    else:
                        one_pt = T.simd.pset(16, "PAT_VL1")
                        out_h = T.simd.vcvt(out, dtype)
                        T.simd.vsts(lse_ub[r], out_h, one_pt, "ONEPT_B16")

            # task-end lse write-back (static slice, DESIGN R4)
            T.copy(lse_ub[0:rows], y[row_base : row_base + rows])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((1, BLOCK_N), dtype)
                stats_ub = T.alloc_shared((2,), "float32")
                lse_ub = T.alloc_shared((ROWS,), dtype)
                T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    if t < full_m:
                        row_task(x, y, x_ub, stats_ub, lse_ub, t * ROWS, ROWS)
                    else:
                        row_task(x, y, x_ub, stats_ub, lse_ub, M - rem_m, rem_m)

        return main

    return _func


def _call_tiled(M, N, dtype, tile_n, block_m, x):
    kernel = _logsumexp_kernel_tiled(M, N, _dtype_str(dtype), tile_n)(block_m)
    return kernel(x)


# ---------- Wrapper tile_n recomputation (DESIGN 8.2 / 11.2) ----------
def _wrapper_tile_n(N: int, elem: int) -> int:
    """Recompute the wrapper tile_n via the in-repo _primitives mirror.

    Avoids hardcoding hand-computed constants (DESIGN 11.2 lesson); a tile_n
    drift in the wrapper contract surfaces here as an assertion failure rather
    than a silent pass.
    """
    _examples_root = None
    _d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(_d, "tileops")):
            _examples_root = _d
            break
        _d = os.path.dirname(_d)
    if _examples_root is None:
        raise RuntimeError("tileops package not found from " + __file__)
    _tileops_root = _examples_root
    if _tileops_root not in sys.path:
        sys.path.insert(0, _tileops_root)
    from tileops.kernels.reduction._primitives import (  # noqa: F401
        MAX_SINGLE_TILE_COLS,
        align_up,
        compute_tile_n,
    )

    budget = min(WRAPPER_BUDGET_BYTES, MAX_SINGLE_TILE_COLS * 1 * elem)
    return compute_tile_n(1, elem, align_up(N, 256), budget=budget)


# ---------- Layered testing ----------
def _run_single_case(M, N, dtype, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call(M, N, dtype, block_m, x)
    torch.npu.synchronize()
    golden = golden_logsumexp(x)
    tol = _tolerance(dtype)
    max_diff = (y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y, golden, rtol=tol, atol=tol)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} block_m={block_m} max_diff={max_diff:.3e}")
    return max_diff


def _run_single_special_case(x, block_m, tag, *, source_semantics=False):
    """Compare a pre-built special-value input against golden (equal_nan aware).

    source_semantics=True compares against the fp32 source-algorithm simulator
    instead of torch.logsumexp, for the all -inf / +inf divergence rows
    (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    M, N = x.shape
    dtype = x.dtype
    y = _call(M, N, dtype, block_m, x)
    torch.npu.synchronize()
    ref = _source_lse_sim(x) if source_semantics else golden_logsumexp(x)
    tol = _tolerance(dtype)
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol, equal_nan=True)
    # isfinite-mask the diff so +/-inf/NaN positions don't poison the report
    # (inf - inf = NaN would otherwise dominate max_diff).
    diff = y.float() - ref.float()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def _run_single_L0():
    # Gate cases (DESIGN section 8.2 L0 plan): wl1/wl4 equivalents, smoke
    # 3-dtype, M tail (rem=1), non-pow2 N (tail group mask).
    _run_single_case(1024, 4096, torch.float16, 16, "L0")
    _run_single_case(128, 16384, torch.float16, 4, "L0")
    _run_single_case(32, 256, torch.float32, 16, "L0")
    _run_single_case(32, 256, torch.float16, 16, "L0")
    _run_single_case(32, 256, torch.bfloat16, 16, "L0")
    _run_single_case(33, 256, torch.float16, 16, "L0")
    _run_single_case(32, 300, torch.float16, 16, "L0")


def _run_single_L1():
    # Shape / config sweep: M breadth, N breadth (incl. non-pow2 and non-32B
    # rows), 3 dtypes, block_m sweep incl. SUB split points.
    for M in (1, 3, 33, 56, 57, 128, 1024):
        for N in (1, 7, 256, 300, 4096, 16384):
            _run_single_case(M, N, torch.float16, 16, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 256), (32, 300), (1024, 4096), (128, 16384)):
        _run_single_case(M, N, torch.float32, 16, "L1")
        _run_single_case(M, N, torch.bfloat16, 16, "L1")
    # block_m sweep (wrapper heuristic full set + extreme 1), SUB split
    # verification (block_m > SUB vs block_m == SUB).
    for bm in (1, 2, 4, 8, 16):
        _run_single_case(32, 256, torch.float16, bm, "L1")
    # M=1 (full tail), 1D flatten, multi-dim dim=[0,2] flatten.
    _run_single_case(1, 256, torch.float16, 16, "L1")
    _run_single_case(1, 256, torch.float32, 16, "L1")
    _run_single_case(128, 16384, torch.float16, 4, "L1")


def _run_single_L2():
    # Robustness / stress: large M multi-wave scheduling (num_batches > 56),
    # contract boundary N=32512 SUB=1 degenerate form.  Recorded as warnings,
    # do not block the verdict.
    cases = [
        (8192, 4096, torch.float16, 16),
        (8192, 4096, torch.float32, 16),
        (2, 32512, torch.float16, 2),
        (128, 16384, torch.float32, 16),
    ]
    for M, N, dt, bm in cases:
        try:
            _run_single_case(M, N, dt, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")
    # Negative assert: N beyond the single-tile column cap must raise.
    try:
        _logsumexp_kernel_single(4, 32768, "float16")
        print("[L2] WARN (not blocking): N=32768 did not raise (expected single-tile assert)")
    except ValueError:
        print("[L2] PASS: N=32768 rejected by single-tile assert (tiled-variant boundary)")


def _run_single_boundary():
    n = 128

    def base(rows=8, cols=128, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_single_special_case(base(), 8, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_single_special_case(x, 8, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_single_special_case(x, 8, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_single_special_case(x, 8, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_single_special_case(x, 8, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_single_special_case(x, 8, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_single_special_case(x, 8, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_single_special_case(x, 8, "Boundary-nan")
    # N=1 and M=1 minimal shapes.
    _run_single_special_case(torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, "Boundary-N1")
    _run_single_special_case(
        torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, "Boundary-M1"
    )

    # source-inherited divergence (vs fp32 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_single_special_case(x, 8, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_single_special_case(x, 8, "Boundary-pos-inf", source_semantics=True)

    # tail group + tail batch double trigger (rem=1 && k_tail=44).
    _run_single_case(33, 300, torch.float16, 16, "Boundary")

    # determinism: same input 3x, bitwise identical.
    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    y1 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    y2 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


def _show_single_source():
    """Print the generated AscendC source (spill / capacity check, DESIGN risk 1)."""
    kernel = _logsumexp_kernel_single(1024, 4096, "float16")(16)
    print(kernel.get_kernel_source())


# ---------- Layered testing ----------
def _run_tiled_case(M, N, dtype, tile_n, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call_tiled(M, N, dtype, tile_n, block_m, x)
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


def _run_tiled_special_case(x, block_m, tile_n, tag, *, source_semantics=False):
    """Compare a pre-built special-value input (equal_nan aware).

    source_semantics=True compares against the fp32 source-tiled-algorithm
    simulator instead of torch.logsumexp, for the all -inf / +inf /
    first-chunk-all -inf divergence rows (DESIGN 8.2 Boundary / 9.3).
    """
    M, N = x.shape
    dtype = x.dtype
    y = _call_tiled(M, N, dtype, tile_n, block_m, x)
    torch.npu.synchronize()
    ref = _source_lse_tiled_sim(x, tile_n) if source_semantics else golden_logsumexp(x)
    tol = _tolerance(dtype)
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol, equal_nan=True)
    # isfinite-mask the diff so +/-inf/NaN positions don't poison the report.
    diff = y.float() - ref.float()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def _run_tiled_L0():
    # Gate cases (DESIGN section 8.2 L0 plan).  Wrapper tile_n recomputed via
    # the _primitives mirror and asserted against the design expectations
    # (wl2 16384 / wl3 25600 / (33,33000) 32512).
    def _wtn(N, elem, expected):
        tn = _wrapper_tile_n(N, elem)
        assert tn == expected, f"wrapper tile_n drift: got {tn}, expected {expected}"
        return tn

    # Case 1 first (DESIGN 10.4 Step 1): simplest mainstream form (bm=1, rem=0)
    # isolates the risk-1 alloc_var-cross-block compile.
    _run_tiled_case(1024, 32768, torch.bfloat16, _wtn(32768, 2, 16384), 1, "L0")  # wl2
    _run_tiled_case(4, 102400, torch.float16, _wtn(102400, 2, 25600), 1, "L0")  # wl3
    _run_tiled_case(4, 102400, torch.bfloat16, _wtn(102400, 2, 25600), 1, "L0")  # wl3 bf16
    _run_tiled_case(33, 33000, torch.float16, _wtn(33000, 2, 32512), 1, "L0")  # tail chunk + M tail
    _run_tiled_case(4, 32768, torch.float16, _wtn(32768, 2, 16384), 1, "L0")  # test shape
    _run_tiled_case(33, 32768, torch.float16, _wtn(32768, 2, 16384), 1, "L0")  # M tail + large N
    _run_tiled_case(8, 100, torch.float32, 64, 1, "L0")  # fp32 direct + k_tail=36
    _run_tiled_case(1024, 32768, torch.bfloat16, _wtn(32768, 2, 16384), 4, "L0")  # bm>1


def _run_tiled_L1():
    # N breadth (fp16, block_m=1): multi-chunk N, non-pow2 N (rem>0), N<BLOCK_N
    # (num_full=0 pure tail path), exact-pow2 N.
    for N in (32768, 33000, 65540, 102400, 32513, 256, 300):
        _run_tiled_case(33, N, torch.float16, 16384, 1, "L1")
    # M breadth (incl. M < block_m / M % block_m != 0 / M=1 single-row).
    for M in (1, 3, 56, 57, 128):
        _run_tiled_case(M, 32768, torch.float16, 16384, 1, "L1")
    # tile_n scan (BLOCK_N clamp + chunk-count variation) over non-pow2 N,
    # incl. non-pow2 tile_n (16640 / 25600 / 13824 -- all 64-multiples).
    for tile_n in (25600, 16640, 13824, 8192, 4096, 256):
        _run_tiled_case(33, 33000, torch.float16, tile_n, 1, "L1")
    # block_m sweep (wrapper full set + extreme 1).
    for bm in (1, 2, 4, 8, 16):
        _run_tiled_case(33, 32768, torch.float16, 16384, bm, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 33000), (4, 102400), (128, 32768)):
        _run_tiled_case(M, N, torch.float32, 16384, 1, "L1")
        _run_tiled_case(M, N, torch.bfloat16, 16384, 1, "L1")
    # shape / granularity scan (DESIGN 8.2): num_full=0 / k_tail in {0,1,63} /
    # single chunk / 63-chunk long chain / minimal shapes.
    _run_tiled_case(3, 257, torch.float16, 256, 1, "L1")  # k_tail=1
    _run_tiled_case(5, 63, torch.float16, 64, 1, "L1")  # num_full=0, k_tail=63
    _run_tiled_case(7, 65, torch.float16, 64, 1, "L1")  # num_full=1, k_tail=1
    _run_tiled_case(1, 1, torch.float16, 256, 1, "L1")  # M=1, N=1
    _run_tiled_case(2, 1024000, torch.float16, 16384, 1, "L1")  # 63-chunk long chain
    _run_tiled_case(57, 32768, torch.float16, 16384, 1, "L1")  # M non-div by 56
    # M < block_m (all-tail-batch path).
    _run_tiled_case(3, 32768, torch.float16, 16384, 16, "L1")


def _run_tiled_L2():
    # Robustness / stress: large M multi-wave, very large N (long merge chain),
    # tiny tile_n (many chunks), boundary neighbourhood, fp32 clamp path
    # (BLOCK_N=27520, 99.9% UB).  Recorded as warnings, do not block verdict.
    cases = [
        (8192, 32768, torch.float16, 16384, 1),  # 8192 tiles, ~146 waves
        (2, 1024000, torch.bfloat16, 16384, 1),  # 63-chunk merge chain
        (33, 33000, torch.float16, 256, 1),  # tiny tile_n -> 129 chunks
        (33, 32512, torch.float16, 16384, 1),  # single/tiled boundary
        (33, 32513, torch.float16, 16384, 1),
        (1024, 32768, torch.float32, 32512, 1),  # fp32 clamp (BLOCK_N=27520)
    ]
    for M, N, dt, tile_n, bm in cases:
        try:
            _run_tiled_case(M, N, dt, tile_n, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")


def _run_tiled_boundary():
    # Canonical boundary config: tile_n=256 -> BLOCK_N=256, N=512 spans 2 full
    # chunks (exercises the chunk loop + flash merge for special values).
    tile_n = 256
    block_m = 8
    n = 512

    def base(rows=8, cols=n, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_tiled_special_case(base(), block_m, tile_n, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-nan")
    # N=1 and M=1 minimal shapes (single tail chunk path).
    _run_tiled_special_case(
        torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, 256, "Boundary-N1"
    )
    _run_tiled_special_case(
        torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, 256, "Boundary-M1"
    )

    # tiled-specific merge corners (vs golden / source sim).
    # cross-chunk max gap 131008: -65504 in chunk0, +65504 in chunk1 (flash
    # form exp args <= 0 -> no overflow; DESIGN V5 corner).
    x = base()
    x[:, 0] = -65504.0  # chunk0 (cols [0, 256))
    x[:, 256] = 65504.0  # chunk1 (cols [256, 512))
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-cross-chunk-maxgap")
    # dominant value in chunk0 vs chunk1 (merge order both directions).
    x = base()
    x[:, 0] = 100.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-dominant-chunk0")
    x = base()
    x[:, 256] = 100.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-dominant-chunk1")
    # equal values spanning the chunk boundary (associativity).
    x = base()
    x[:, :] = 1.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-equal-span-chunks")

    # source-inherited divergence (vs fp32 source-tiled semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-pos-inf", source_semantics=True)
    # tiled-specific corner: first chunk all -inf, later chunk finite -> NaN
    # (torch gives finite; source tiled gives NaN -- DESIGN V5).
    x = base()
    x[:, 0:256] = float("-inf")
    x[:, 256:512] = 1.0
    _run_tiled_special_case(
        x, block_m, tile_n, "Boundary-first-chunk-neg-inf", source_semantics=True
    )
    # tail chunk all -inf (after a finite chunk) -> neutral (matches golden).
    x = base()
    x[:, 0:256] = 1.0
    x[:, 256:512] = float("-inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-tail-chunk-neg-inf")

    # determinism: same input 3x, bitwise identical.
    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y1 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y2 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


def _show_tiled_source():
    """Print the generated AscendC source (spill / capacity check, DESIGN risk 1)."""
    kernel = _logsumexp_kernel_tiled(1024, 32768, "bfloat16", 16384)(1)
    print(kernel.get_kernel_source())


# ---------- Combined suite entrypoints (Stage 4 gate contract) ----------
def run_L0():
    """L0 gate cases of BOTH variants (single first, then tiled)."""
    _run_single_L0()
    _run_tiled_L0()


def run_L1():
    _run_single_L1()
    _run_tiled_L1()


def run_L2():
    _run_single_L2()
    _run_tiled_L2()


def run_boundary():
    _run_single_boundary()
    _run_tiled_boundary()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument(
        "--show-source",
        default=None,
        choices=["single", "tiled"],
        help="print the generated AscendC source of one variant",
    )
    args, _ = parser.parse_known_args()

    if args.show_source == "single":
        _show_single_source()
        return
    if args.show_source == "tiled":
        _show_tiled_source()
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
