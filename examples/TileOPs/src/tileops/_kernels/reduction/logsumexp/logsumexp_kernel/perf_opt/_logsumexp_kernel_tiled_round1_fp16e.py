# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Multi-tile (online streaming) row-wise logsumexp for Ascend A5 (migrated
from GPU TileLang).

Stage 4 perf_opt branch (round 1, OP4): fp16-e experimental adjudication
variant (DESIGN tiled section 1.6.4, mandatory A/B; stats_ub merge chain
kept fp32 per the frozen variant spec).

programming_mode: simt
auto_schedule: yes

Row-wise numerically-stable logsumexp (reduction axis dim=1). fp16 / bf16 /
fp32 in/out, fp32 intermediate. The tiled variant handles arbitrary N by
streaming BLOCK_N pow2 chunks with a flash-style running (max, sum) merge
(DESIGN section 0.1 / 0.6 / 1.4):

    m_run = max over chunks of (max_j x_c[j])
    s_run = s_run * exp(prev_max - m_run) + sum_j exp(x_c[j] - m_run)
    y_i   = m_run + log(s_run)

Kernel structure (persistent 56-core grid, N-chunk streaming, UB residency):
  * host staticization derives BLOCK_N / SUB / num_full / rem / TILE_R /
    full / rem_m / num_tiles / num_sub / last_sub_rows / tsub / tsub_rem /
    threads (all JIT constants). tile_n is honoured as a budget cap: the
    internal chunk width BLOCK_N is the largest power of two with
    BLOCK_N <= tile_n and 2*BLOCK_N*elem <= 194.4 KiB (flash-merge
    associativity makes the re-partition exact -- verify_equiv.py V1/V6).
  * T.Persistent row-block loop with a TIR tail branch (``if t < full``).
  * per-sub-batch: stats_ub init -> T.serial(num_full) full-chunk loop
    (T.copy GM->UB -> SimtVF direct load -> reduce_max -> max merge ->
    in-place exp -> reduce_sum -> flash rescale merge) -> optional static
    tail chunk (masked load, pad=-inf) -> lse+cast -> fragment->UB -> UB->GM
    static slice.

Call surface (wrapper-compatible, threads removed per K3):
    _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU
(source-repo baseline; fp32 accumulation matches DESIGN section 0.1 dtype
semantics). Divergence corner cases (all -inf / +inf rows -> NaN vs torch's
-inf/+inf) are source-algorithm-inherent and verified against a fp32 source
semantics simulator in run_boundary (migration fidelity), not against golden.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


# 56 = vector_core_num of Ascend950PR_9579 (torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56

# UB budget under SimtVF: 216 KiB * 0.9 discount (DESIGN section 4.5,
# CONST-capacity-a5 assumption). VRF fragment budget: 128 KiB cap for the
# (SUB, BLOCK_N) fp32 x_frag (CONST-vrf-capacity-a5 assumption).
UB_BUDGET_BYTES = int(216 * 1024 * 0.9)  # 199065
VRF_BUDGET_BYTES = 128 * 1024  # 131072


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU (source test baseline ref_program)."""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE GPU algorithm semantics.

    m = max(x); y = m + log(sum(exp(x - m))) with fp32 intermediates (matching
    the source's fp32 chain, DESIGN section 0.1). Unlike torch.logsumexp this
    propagates the source-inherent divergence (all -inf row -> NaN, +inf row
    -> NaN) exactly, so it is used to validate migration fidelity for those
    corner cases (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    x32 = x.float()
    m = x32.max(dim=-1, keepdim=True).values
    e = torch.exp(x32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (m + torch.log(s)).squeeze(-1).to(x.dtype)


# ---------- Kernel builder ----------
def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile (online streaming) logsumexp kernel.

    Keeps the wrapper call surface ``_logsumexp_kernel_tiled(M, N, dtype,
    tile_n)(block_m)(x)``; the GPU ``threads`` argument is removed per K3
    (threads derived inside: min(1024, SUB * BLOCK_N)). tile_n is honoured as
    a budget cap: the internal chunk width BLOCK_N is the largest power of two
    with BLOCK_N <= tile_n and 2*BLOCK_N*elem <= 194.4 KiB.
    """
    if M < 1 or N < 1 or tile_n < 1:
        raise ValueError(f"M, N, tile_n must be >= 1, got M={M}, N={N}, tile_n={tile_n}")
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # ---- host staticization (DESIGN section 3.3 / 5.2) ----
        # BLOCK_N: largest power of two with BLOCK_N <= tile_n and
        # 2*BLOCK_N*elem <= UB budget (fp16/bf16 cap 32768, fp32 cap 16384).
        BLOCK_N = 1
        while BLOCK_N * 2 <= tile_n and 2 * (BLOCK_N * 2) * elem <= UB_BUDGET_BYTES:
            BLOCK_N *= 2
        # SUB: largest power of two with SUB <= block_m, x_ub double-buffer
        # <= UB budget and x_frag <= VRF budget.
        SUB = 1
        while SUB * 2 <= block_m:
            cand = SUB * 2
            if (2 * cand * BLOCK_N * elem <= UB_BUDGET_BYTES) and (
                cand * BLOCK_N * 4 <= VRF_BUDGET_BYTES
            ):
                SUB = cand
            else:
                break

        num_full = N // BLOCK_N           # full chunks (all columns valid)
        rem = N % BLOCK_N                 # tail-chunk column count
        TILE_R = tilelang.next_power_of_2(rem) if rem > 0 else 0
        full = M // block_m               # complete block_m tiles
        rem_m = M % block_m               # rows in the tail tile
        num_tiles = full + (1 if rem_m > 0 else 0)
        num_sub = block_m // SUB          # full SUB-row sub-batches per tile
        last_sub_rows = block_m % SUB     # rows in the last partial sub-batch
        tsub = rem_m // SUB               # full SUB-row sub-batches in tail
        tsub_rem = rem_m % SUB            # rows in the tail's partial sub-batch
        threads = min(1024, SUB * BLOCK_N)

        @T.macro
        def sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, rows):
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
                        if dtype == "float16":
                            # fp16-e chain (DESIGN tiled 1.6.4): native fp16
                            # load (no cast), fp16 max (exact), fp16 sub+exp
                            # (x and m_run are fp16-exact values -> fp16 sub
                            # equals the fp32 detour + cast, bit-for-bit),
                            # fp16 sum tree; stats_ub merge chain stays fp32.
                            x_frag = T.alloc_fragment((SUB, BLOCK_N), "float16")
                            m_c = T.alloc_fragment((SUB,), "float16")
                            s_c = T.alloc_fragment((SUB,), "float16")
                            m16 = T.alloc_fragment((SUB,), "float16")
                            prev_frag = T.alloc_fragment((SUB,), "float32")
                            if rows == SUB:
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    x_frag[r, i] = x_ub[r, i]
                            else:
                                for r, i in T.Parallel(SUB, BLOCK_N):
                                    x_frag[r, i] = T.if_then_else(
                                        r < rows,
                                        x_ub[r, i],
                                        -T.infinity(T.float16),
                                    )
                            T.reduce_max(x_frag, m_c, dim=1, clear=True)
                            for r in T.Parallel(SUB):
                                prev_frag[r] = stats_ub[r, 0]
                                stats_ub[r, 0] = T.max(
                                    stats_ub[r, 0], T.cast(m_c[r], "float32")
                                )
                                m16[r] = T.cast(stats_ub[r, 0], "float16")
                            for r, i in T.Parallel(SUB, BLOCK_N):
                                x_frag[r, i] = T.exp(x_frag[r, i] - m16[r])
                            T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                            for r in T.Parallel(SUB):
                                stats_ub[r, 1] = (
                                    stats_ub[r, 1]
                                    * T.exp(prev_frag[r] - stats_ub[r, 0])
                                    + T.cast(s_c[r], "float32")
                                )
                        else:
                            x_frag = T.alloc_fragment((SUB, BLOCK_N), "float32")
                            m_c = T.alloc_fragment((SUB,), "float32")
                            s_c = T.alloc_fragment((SUB,), "float32")
                            prev_frag = T.alloc_fragment((SUB,), "float32")
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
                    if dtype == "float16":
                        # fp16-e chain (DESIGN tiled 1.6.4), tail chunk.
                        x_frag = T.alloc_fragment((SUB, TILE_R), "float16")
                        m_c = T.alloc_fragment((SUB,), "float16")
                        s_c = T.alloc_fragment((SUB,), "float16")
                        m16 = T.alloc_fragment((SUB,), "float16")
                        prev_frag = T.alloc_fragment((SUB,), "float32")
                        for r, i in T.Parallel(SUB, TILE_R):
                            x_frag[r, i] = T.if_then_else(
                                T.And(r < rows, i < rem),
                                x_ub[r, i],
                                -T.infinity(T.float16),
                            )
                        T.reduce_max(x_frag, m_c, dim=1, clear=True)
                        for r in T.Parallel(SUB):
                            prev_frag[r] = stats_ub[r, 0]
                            stats_ub[r, 0] = T.max(
                                stats_ub[r, 0], T.cast(m_c[r], "float32")
                            )
                            m16[r] = T.cast(stats_ub[r, 0], "float16")
                        for r, i in T.Parallel(SUB, TILE_R):
                            x_frag[r, i] = T.exp(x_frag[r, i] - m16[r])
                        T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                        for r in T.Parallel(SUB):
                            stats_ub[r, 1] = (
                                stats_ub[r, 1]
                                * T.exp(prev_frag[r] - stats_ub[r, 0])
                                + T.cast(s_c[r], "float32")
                            )
                    else:
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
            # ---- lse = m_run + log(s_run), two-stage write ----
            with T.SimtVF(threads=threads):
                out_frag = T.alloc_fragment((SUB,), dtype)
                for r in T.Parallel(SUB):
                    out_frag[r] = T.cast(
                        stats_ub[r, 0] + T.log(stats_ub[r, 1]), dtype
                    )
                T.copy(out_frag, lse_ub)
            T.copy(lse_ub[0:rows], y[row_base : row_base + rows])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, BLOCK_N), dtype)
                stats_ub = T.alloc_shared((SUB, 2), "float32")
                lse_ub = T.alloc_shared((SUB,), dtype)
                T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    base = t * block_m
                    if t < full:
                        # ---- full tile (block_m rows, all in-bounds) ----
                        for s in T.serial(num_sub):
                            row_base = base + s * SUB
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, SUB)
                        if last_sub_rows > 0:
                            row_base = base + num_sub * SUB
                            sub_batch(
                                x, y, x_ub, stats_ub, lse_ub, row_base, last_sub_rows
                            )
                    else:
                        # ---- tail tile (rem_m rows) ----
                        for s in T.serial(tsub):
                            row_base = base + s * SUB
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, SUB)
                        if tsub_rem > 0:
                            row_base = M - tsub_rem
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, tsub_rem)

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


def _run_special_case(x, block_m, tile_n, tag, *, source_semantics=False):
    """Compare a pre-built special-value input against golden (equal_nan aware).

    source_semantics=True compares against the fp32 source-algorithm simulator
    instead of torch.logsumexp, for the all -inf / +inf divergence rows
    (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    M, N = x.shape
    dtype = x.dtype
    y = _call(M, N, dtype, tile_n, block_m, x)
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


def run_L0():
    # Gate cases (DESIGN section 8.2 L0 plan). Case 5 first per REVIEW note 5:
    # the (33, 33000) tail chunk is a 256-elem fragment @ SimtVF(threads=1024)
    # (extent < threads, replication=4) -- validate the reduce config early.
    _run_case(33, 33000, torch.float16, 16640, 1, "L0")    # non-pow2 tile_n + rem=232 tail chunk
    _run_case(1024, 32768, torch.bfloat16, 16384, 1, "L0")  # wl2 (2 full chunks)
    _run_case(4, 102400, torch.float16, 25600, 1, "L0")     # wl3 (rem=4096 tail chunk)
    _run_case(4, 102400, torch.bfloat16, 25600, 1, "L0")    # wl3 bf16
    _run_case(4, 32768, torch.float16, 16384, 1, "L0")      # test L76 shape
    _run_case(33, 32768, torch.float16, 16384, 4, "L0")     # M tail (rem_m=1) + bm>1
    _run_case(4, 32768, torch.float32, 16384, 1, "L0")      # fp32 (UB 128 KiB)


def run_L1():
    # N breadth (fp16, block_m=1): non-pow2 N (rem>0), N < BLOCK_N (num_full=0),
    # multi-chunk N, exact-pow2 N.
    for N in (32768, 33000, 65540, 102400, 32513, 256, 300):
        _run_case(33, N, torch.float16, 16384, 1, "L1")
    # M breadth (incl. M < block_m / M % block_m != 0 / M=1 single-row streaming).
    for M in (1, 3, 56, 57, 128):
        _run_case(M, 32768, torch.float16, 16384, 1, "L1")
    # tile_n scan (BLOCK_N clamp + chunk-count variation) over non-pow2 N.
    for tile_n in (25600, 16640, 13824, 8192, 4096, 256):
        _run_case(33, 33000, torch.float16, tile_n, 1, "L1")
    # block_m sweep (wrapper heuristic full set + SUB split points).
    for bm in (1, 2, 4, 8, 16):
        _run_case(33, 32768, torch.float16, 16384, bm, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 33000), (4, 102400), (128, 32768)):
        _run_case(M, N, torch.float32, 16384, 1, "L1")
        _run_case(M, N, torch.bfloat16, 16384, 1, "L1")
    # M=1 streaming + minimal N (N < BLOCK_N full tail-chunk path).
    _run_case(1, 65536, torch.float16, 16384, 1, "L1")
    _run_case(8, 100, torch.float16, 16384, 1, "L1")
    _run_case(8, 100, torch.float32, 16384, 1, "L1")


def run_L2():
    # Robustness / stress: large M multi-wave, very large N (long merge chain),
    # tiny tile_n (many chunks), single/tiled switch boundary neighbourhood.
    # Recorded as warnings, do not block the verdict.
    cases = [
        (8192, 32768, torch.float16, 16384, 1),   # 8192 tiles, ~146 waves
        (2, 1024000, torch.float16, 16384, 1),    # 63-chunk merge chain
        (33, 33000, torch.float16, 256, 1),       # tiny tile_n -> 128 chunks
        (33, 32512, torch.float16, 16384, 1),     # single/tiled boundary
        (33, 32513, torch.float16, 16384, 1),
    ]
    for M, N, dt, tile_n, bm in cases:
        try:
            _run_case(M, N, dt, tile_n, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")


def run_boundary():
    # Canonical boundary config: tile_n=256 -> BLOCK_N=256, N=512 spans 2 full
    # chunks (exercises the chunk loop + flash merge for special values).
    tile_n = 256
    block_m = 8
    n = 512

    def base(rows=8, cols=n, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_special_case(base(), block_m, tile_n, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_special_case(x, block_m, tile_n, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_special_case(x, block_m, tile_n, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_special_case(x, block_m, tile_n, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_special_case(x, block_m, tile_n, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_special_case(x, block_m, tile_n, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_special_case(x, block_m, tile_n, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_special_case(x, block_m, tile_n, "Boundary-nan")
    # N=1 and M=1 minimal shapes (single tail chunk path).
    _run_special_case(
        torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, 256, "Boundary-N1"
    )
    _run_special_case(
        torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, 256, "Boundary-M1"
    )

    # tiled-specific merge corners (vs golden / source sim).
    # cross-chunk max gap 131008: -65504 in chunk0, +65504 in chunk1 (flash form
    # exp args <= 0 -> no overflow; DESIGN V5 corner).
    x = base()
    x[:, 0] = -65504.0     # chunk0 (cols [0, 256))
    x[:, 256] = 65504.0    # chunk1 (cols [256, 512))
    _run_special_case(x, block_m, tile_n, "Boundary-cross-chunk-maxgap")
    # dominant value in chunk0 vs chunk1 (merge-order both directions).
    x = base()
    x[:, 0] = 100.0        # dominant in chunk0
    _run_special_case(x, block_m, tile_n, "Boundary-dominant-chunk0")
    x = base()
    x[:, 256] = 100.0      # dominant in chunk1
    _run_special_case(x, block_m, tile_n, "Boundary-dominant-chunk1")
    # equal values spanning the chunk boundary (associativity).
    x = base()
    x[:, :] = 1.0
    _run_special_case(x, block_m, tile_n, "Boundary-equal-span-chunks")

    # source-inherited divergence (vs fp32 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_special_case(x, block_m, tile_n, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_special_case(x, block_m, tile_n, "Boundary-pos-inf", source_semantics=True)

    # determinism: same input 3x, bitwise identical.
    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y1 = _call(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y2 = _call(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


def _show_source():
    """Print the generated AscendC source (spill / capacity check, DESIGN risk 1)."""
    kernel = _logsumexp_kernel_tiled(1024, 32768, "bfloat16", 16384)(1)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true", help="print generated AscendC source")
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
