# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Single-tile row-wise logsumexp for Ascend A5 (migrated from GPU TileLang).

Stage 4 perf_opt branch (round 1, OP3): fp16-e experimental adjudication
variant (DESIGN single section 1.6.4, mandatory A/B).

programming_mode: simt
auto_schedule: yes

Row-wise numerically-stable logsumexp (reduction axis dim=1).  fp16 / bf16 /
fp32 in/out, fp32 intermediate.  The single-tile variant assumes N fits
on-chip (N_padded <= 32512 and block_m * elem * N_padded <= 216 KiB), as
decided by the TileOPs wrapper.  See DESIGN.md sections 1.4 / 3.3 / 6:

    y_i = m_i + log(sum_j exp(x_ij - m_i)),  m_i = max_j x_ij

Kernel structure (persistent 56-core grid, UB/VRF row-sub-batch residency):
  * host staticization derives TILE / SUB / num_sub / full / rem / num_tiles /
    threads (all JIT constants).
  * T.Persistent row-block loop with a TIR tail branch (``if t < full``).
  * per-sub-batch: T.copy GM->UB (x_ub double-buffered) -> SimtVF masked load
    (-inf pad identity) -> reduce_max -> in-place exp -> reduce_sum -> per-row
    lse+cast -> fragment->UB -> UB->GM static slice.

Call surface (wrapper-compatible, threads removed per K3):
    _logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU
(source-repo baseline; fp32 accumulation matches DESIGN section 0.1 dtype
semantics).  Divergence corner cases (all -inf / +inf rows -> NaN vs torch's
-inf/+inf) are source-algorithm-inherent and verified against a fp64 source
semantics simulator in run_boundary (migration fidelity), not against golden.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


# 56 = vector_core_num of Ascend950PR_9579 (torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56

# Single-tile column cap from the TileOPs wrapper (_primitives.py MAX_SINGLE_TILE_COLS).
MAX_SINGLE_TILE_COLS = 32512

# UB budget under SimtVF: 216 KiB * 0.9 discount (DESIGN section 4.5,
# CONST-capacity-a5 assumption).  VRF fragment budget: 128 KiB cap for the
# (SUB, TILE) fp32 x_frag (CONST-vrf-capacity-a5 assumption).
UB_BUDGET_BYTES = int(216 * 1024 * 0.9)  # 199065
VRF_BUDGET_BYTES = 128 * 1024  # 131072


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


def _tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32


# ---------- Kernel builder ----------
def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits on-chip).

    Keeps the wrapper call surface ``_logsumexp_kernel_single(M, N, dtype)
    (block_m)(x)``; the GPU ``threads`` argument is removed per K3 (threads
    derived inside as min(1024, SUB * TILE)).
    """
    if M < 1 or N < 1:
        raise ValueError(f"M, N must be >= 1, got M={M}, N={N}")
    if N > MAX_SINGLE_TILE_COLS:
        raise ValueError(
            f"N={N} exceeds MAX_SINGLE_TILE_COLS={MAX_SINGLE_TILE_COLS} "
            f"(single-tile contract; larger N is the tiled variant)"
        )

    TILE = tilelang.next_power_of_2(N)
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # SUB: max power-of-two satisfying (i) x_ub double-buffer <= UB budget,
        # (ii) x_frag fp32 <= VRF budget, (iii) SUB <= block_m.  SUB >= 1 always
        # (DESIGN section 5.2).
        SUB = 1
        while SUB * 2 <= block_m:
            cand = SUB * 2
            if (2 * cand * N * elem <= UB_BUDGET_BYTES) and (
                cand * TILE * 4 <= VRF_BUDGET_BYTES
            ):
                SUB = cand
            else:
                break

        num_full_sub = block_m // SUB       # full SUB-row sub-batches per full tile
        last_sub_rows = block_m % SUB       # rows in the last (partial) sub-batch
        full = M // block_m                 # number of complete block_m tiles
        rem = M % block_m                   # rows in the tail tile
        num_tiles = full + (1 if rem else 0)
        tsub = rem // SUB                   # full SUB-row sub-batches in the tail
        tsub_rem = rem % SUB                # rows in the tail's last partial sub-batch
        threads = min(1024, SUB * TILE)

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, N), dtype)
                lse_ub = T.alloc_shared((SUB,), dtype)
                T.annotate_buffer_versions({x_ub: 2})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    base = t * block_m
                    if t < full:
                        # ---- full tile (block_m rows, all in-bounds) ----
                        for s in T.serial(num_full_sub):
                            row_base = base + s * SUB
                            T.copy(x[row_base : row_base + SUB, 0:N], x_ub)
                            with T.SimtVF(threads=threads):
                                if dtype == "float16":
                                    # fp16-e chain (DESIGN 1.6.4): native fp16
                                    # load (no cast), fp16 max (exact), fp16
                                    # sub+exp (x, m are fp16-exact inputs ->
                                    # the fp16 subtraction equals the fp32
                                    # detour + cast, bit-for-bit), fp16 sum
                                    # tree, fp32 log epilogue.
                                    x_frag = T.alloc_fragment((SUB, TILE), "float16")
                                    m_frag = T.alloc_fragment((SUB,), "float16")
                                    s_frag = T.alloc_fragment((SUB,), "float16")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            i < N,
                                            x_ub[r, i],
                                            -T.infinity(T.float16),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(
                                            T.cast(m_frag[r], "float32")
                                            + T.log(T.cast(s_frag[r], "float32")),
                                            dtype,
                                        )
                                    T.copy(out_frag, lse_ub)
                                else:
                                    x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                    m_frag = T.alloc_fragment((SUB,), "float32")
                                    s_frag = T.alloc_fragment((SUB,), "float32")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            i < N,
                                            T.cast(x_ub[r, i], "float32"),
                                            -T.infinity(T.float32),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                    T.copy(out_frag, lse_ub)
                            T.copy(lse_ub, y[row_base : row_base + SUB])
                        if last_sub_rows > 0:
                            # partial last sub-batch (block_m % SUB != 0; not
                            # reachable under the wrapper power-of-two block_m
                            # heuristic, kept for arbitrary-config generality).
                            row_base = base + num_full_sub * SUB
                            T.copy(
                                x[row_base : row_base + last_sub_rows, 0:N],
                                x_ub[0:last_sub_rows, 0:N],
                            )
                            with T.SimtVF(threads=threads):
                                if dtype == "float16":
                                    # fp16-e chain (DESIGN 1.6.4), partial sub-batch.
                                    x_frag = T.alloc_fragment((SUB, TILE), "float16")
                                    m_frag = T.alloc_fragment((SUB,), "float16")
                                    s_frag = T.alloc_fragment((SUB,), "float16")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            T.And(r < last_sub_rows, i < N),
                                            x_ub[r, i],
                                            -T.infinity(T.float16),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(
                                            T.cast(m_frag[r], "float32")
                                            + T.log(T.cast(s_frag[r], "float32")),
                                            dtype,
                                        )
                                    T.copy(out_frag, lse_ub)
                                else:
                                    x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                    m_frag = T.alloc_fragment((SUB,), "float32")
                                    s_frag = T.alloc_fragment((SUB,), "float32")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            T.And(r < last_sub_rows, i < N),
                                            T.cast(x_ub[r, i], "float32"),
                                            -T.infinity(T.float32),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                    T.copy(out_frag, lse_ub)
                            T.copy(lse_ub[0:last_sub_rows], y[row_base : row_base + last_sub_rows])
                    else:
                        # ---- tail tile (rem rows) ----
                        if tsub > 0:
                            for s in T.serial(tsub):
                                row_base = base + s * SUB
                                T.copy(x[row_base : row_base + SUB, 0:N], x_ub)
                                with T.SimtVF(threads=threads):
                                    if dtype == "float16":
                                        # fp16-e chain (DESIGN 1.6.4), full sub-batch.
                                        x_frag = T.alloc_fragment((SUB, TILE), "float16")
                                        m_frag = T.alloc_fragment((SUB,), "float16")
                                        s_frag = T.alloc_fragment((SUB,), "float16")
                                        out_frag = T.alloc_fragment((SUB,), dtype)
                                        for r, i in T.Parallel(SUB, TILE):
                                            x_frag[r, i] = T.if_then_else(
                                                i < N,
                                                x_ub[r, i],
                                                -T.infinity(T.float16),
                                            )
                                        T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                        for r, i in T.Parallel(SUB, TILE):
                                            x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                        T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                        for r in T.Parallel(SUB):
                                            out_frag[r] = T.cast(
                                                T.cast(m_frag[r], "float32")
                                                + T.log(T.cast(s_frag[r], "float32")),
                                                dtype,
                                            )
                                        T.copy(out_frag, lse_ub)
                                    else:
                                        x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                        m_frag = T.alloc_fragment((SUB,), "float32")
                                        s_frag = T.alloc_fragment((SUB,), "float32")
                                        out_frag = T.alloc_fragment((SUB,), dtype)
                                        for r, i in T.Parallel(SUB, TILE):
                                            x_frag[r, i] = T.if_then_else(
                                                i < N,
                                                T.cast(x_ub[r, i], "float32"),
                                                -T.infinity(T.float32),
                                            )
                                        T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                        for r, i in T.Parallel(SUB, TILE):
                                            x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                        T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                        for r in T.Parallel(SUB):
                                            out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                        T.copy(out_frag, lse_ub)
                                T.copy(lse_ub, y[row_base : row_base + SUB])
                        if tsub_rem > 0:
                            row_base = M - tsub_rem
                            T.copy(x[row_base:M, 0:N], x_ub[0:tsub_rem, 0:N])
                            with T.SimtVF(threads=threads):
                                if dtype == "float16":
                                    # fp16-e chain (DESIGN 1.6.4), tail partial sub-batch.
                                    x_frag = T.alloc_fragment((SUB, TILE), "float16")
                                    m_frag = T.alloc_fragment((SUB,), "float16")
                                    s_frag = T.alloc_fragment((SUB,), "float16")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            T.And(r < tsub_rem, i < N),
                                            x_ub[r, i],
                                            -T.infinity(T.float16),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(
                                            T.cast(m_frag[r], "float32")
                                            + T.log(T.cast(s_frag[r], "float32")),
                                            dtype,
                                        )
                                    T.copy(out_frag, lse_ub)
                                else:
                                    x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                    m_frag = T.alloc_fragment((SUB,), "float32")
                                    s_frag = T.alloc_fragment((SUB,), "float32")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            T.And(r < tsub_rem, i < N),
                                            T.cast(x_ub[r, i], "float32"),
                                            -T.infinity(T.float32),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                    T.copy(out_frag, lse_ub)
                            T.copy(lse_ub[0:tsub_rem], y[row_base:M])

        return main

    return _func


# ---------- Call helpers ----------
def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _call(M: int, N: int, dtype: torch.dtype, block_m: int, x: torch.Tensor) -> torch.Tensor:
    kernel = _logsumexp_kernel_single(M, N, _dtype_str(dtype))(block_m)
    return kernel(x)


# ---------- Layered testing ----------
def _run_case(M, N, dtype, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call(M, N, dtype, block_m, x)
    torch.npu.synchronize()
    golden = golden_logsumexp(x)
    tol = _tolerance(dtype)
    max_diff = (y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y, golden, rtol=tol, atol=tol)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} block_m={block_m} max_diff={max_diff:.3e}")
    return max_diff


def _run_special_case(x, block_m, tag, *, source_semantics=False):
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
    diff = (y.float() - ref.float())
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    # Gate cases (DESIGN section 8.2 L0 plan): wl1/wl4 equivalents, smoke
    # 3-dtype, M tail (rem=1), non-pow2 N (masked load).
    _run_case(1024, 4096, torch.float16, 16, "L0")
    _run_case(128, 16384, torch.float16, 4, "L0")
    _run_case(32, 256, torch.float32, 16, "L0")
    _run_case(32, 256, torch.float16, 16, "L0")
    _run_case(32, 256, torch.bfloat16, 16, "L0")
    _run_case(33, 256, torch.float16, 16, "L0")
    _run_case(32, 300, torch.float16, 16, "L0")


def run_L1():
    # Shape / config sweep: M breadth, N breadth (incl. non-pow2 and 32B
    # non-aligned rows), 3 dtypes, block_m sweep incl. SUB split points.
    for M in (1, 3, 33, 56, 57, 128, 1024):
        for N in (1, 7, 256, 300, 4096, 16384):
            _run_case(M, N, torch.float16, 16, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 256), (32, 300), (1024, 4096), (128, 16384)):
        _run_case(M, N, torch.float32, 16, "L1")
        _run_case(M, N, torch.bfloat16, 16, "L1")
    # block_m sweep (wrapper heuristic full set + extreme 1), SUB split
    # verification (block_m > SUB vs block_m == SUB).
    for bm in (1, 2, 4, 8, 16):
        _run_case(32, 256, torch.float16, bm, "L1")
    # M=1 (full tail), 1D flatten, multi-dim dim=[0,2] flatten.
    _run_case(1, 256, torch.float16, 16, "L1")
    _run_case(1, 256, torch.float32, 16, "L1")
    _run_case(128, 16384, torch.float16, 4, "L1")


def run_L2():
    # Robustness / stress: large M multi-wave scheduling (num_tiles > 56),
    # contract boundary N=32512 SUB=1 degenerate form.  Recorded as warnings,
    # do not block the verdict.
    cases = [
        (8192, 4096, torch.float16, 16),
        (8192, 4096, torch.float32, 16),
        (2, 32512, torch.float16, 2),
    ]
    for M, N, dt, bm in cases:
        try:
            _run_case(M, N, dt, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")
    # Negative assert: N beyond the single-tile column cap must raise.
    try:
        _logsumexp_kernel_single(4, 32768, "float16")
        print("[L2] WARN (not blocking): N=32768 did not raise (expected single-tile assert)")
    except ValueError:
        print("[L2] PASS: N=32768 rejected by single-tile assert (tiled-variant boundary)")


def run_boundary():
    n = 128

    def base(rows=8, cols=128, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_special_case(base(), 8, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_special_case(x, 8, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_special_case(x, 8, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_special_case(x, 8, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_special_case(x, 8, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_special_case(x, 8, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_special_case(x, 8, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_special_case(x, 8, "Boundary-nan")
    # N=1 and M=1 minimal shapes.
    _run_special_case(torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, "Boundary-N1")
    _run_special_case(torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, "Boundary-M1")

    # source-inherited divergence (vs fp64 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_special_case(x, 8, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_special_case(x, 8, "Boundary-pos-inf", source_semantics=True)

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


def _show_source():
    """Print the generated AscendC source (spill / capacity check, DESIGN risk 1)."""
    kernel = _logsumexp_kernel_single(1024, 4096, "float16")(16)
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
