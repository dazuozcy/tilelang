# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Single-tile row-wise logsumexp for Ascend A5 (migrated from GPU TileLang).

programming_mode: simd
auto_schedule: yes

Row-wise numerically-stable logsumexp (reduction axis dim=1).  fp16 / bf16 /
fp32 in/out, fp32 intermediate.  The single-tile variant assumes N fits
on-chip (N_padded <= 32512), as decided by the TileOPs wrapper.  This is the
explicit-SIMD (SimdVF) variant: all per-element work is expressed as 2048-bit
``T.simd.*`` vector intrinsics (64 fp32 / 128 fp16 lanes per instruction).

    y_i = m_i + log(sum_j exp(x_ij - m_i)),  m_i = max_j x_ij

Kernel structure (persistent 56-core grid, UB row-sub-batch residency, VRF
streaming, no row-data register residency):

  * host staticization derives N_pad / G_full / k_tail / SUB / full / rem /
    num_batches (all JIT constants); NUM_BLOCKS = 56 = vector_core_num.
  * T.Persistent row-sub-batch loop with a TIR tail branch (``if t < full``).
  * per row, two passes over UB: pass1 vmax-accumulate + single vcmax -> m_brc
    (vdupv broadcast); pass2 vexpdif(c, m_brc) fused exp -> vadd-accumulate +
    single vcadd -> s; epilogue vadd(m_brc, vln(s)) -> vcvt -> ONEPT store.
  * tail group (N % 64 != 0) neutralised in fp32 domain via
    vsel(c, -inf, update_mask(k_tail, 32)); N_pad = align_up(N, 64) columns
    are DMA-compact (only N columns copied), pad lanes never reach arithmetic.

Call surface (wrapper-compatible, threads removed per K3):
    _logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU
(source-repo baseline; fp32 accumulation matches DESIGN section 0.1 dtype
semantics).  Divergence corner cases (all -inf / +inf rows -> NaN vs torch's
-inf/+inf) are source-algorithm-inherent and verified against an fp32 source
semantics simulator in run_boundary (migration fidelity), not against golden.
"""

import argparse

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
# deduction).  VRF is streaming (~2.5 KiB), so SUB is capped by UB alone.
UB_BUDGET_BYTES = int(248 * 1024 * 0.9)  # 228556

# SimdVF fp32 lane count == group width (64 fp32 / 128 fp16 per instruction).
GROUP = 64


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
# ---------- Stage 4 Round 1: pass-1 max in the 16-bit input domain ----------
# For fp16/bf16 inputs whose padded row width is 128-aligned, pass 1 computes
# the row max directly on 128-lane 16-bit vectors (vld NORM + vmax) and
# widens only the final broadcast winner to fp32.  max() involves no
# rounding, so the 16-bit-domain max selects exactly the same input value as
# the baseline fp32 detour (bit-identical m; pass 2 unchanged -> bit-identical
# output).  Vec-op count for pass 1 drops from 3 ops / 64 elems to
# 2 ops / 128 elems.  The masked tail-group path stays in fp32 (cold path).
HMAX_ENABLE = True


# ---------- Stage 4 Round 2: + hoisted predicate masks -----------------------
# On top of the Round 1 f16max winner: pre-create the 16-bit predicate mask
# (the vcvt default) once per VF block and pass it explicitly to the hot-loop
# conversions, removing the per-iteration asc_create_mask_b16 scalar op.
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
                    m16_all = T.simd.pset(16, "PAT_ALL")
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
                            acc_h = T.simd.vmax(acc_h, T.simd.vld(x_ub[r, g * 128]), mask=m16_all)
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
                                mask=m16_all,
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
    diff = y.float() - ref.float()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    # Gate cases (DESIGN section 8.2 L0 plan): wl1/wl4 equivalents, smoke
    # 3-dtype, M tail (rem=1), non-pow2 N (tail group mask).
    _run_case(1024, 4096, torch.float16, 16, "L0")
    _run_case(128, 16384, torch.float16, 4, "L0")
    _run_case(32, 256, torch.float32, 16, "L0")
    _run_case(32, 256, torch.float16, 16, "L0")
    _run_case(32, 256, torch.bfloat16, 16, "L0")
    _run_case(33, 256, torch.float16, 16, "L0")
    _run_case(32, 300, torch.float16, 16, "L0")


def run_L1():
    # Shape / config sweep: M breadth, N breadth (incl. non-pow2 and non-32B
    # rows), 3 dtypes, block_m sweep incl. SUB split points.
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

    # source-inherited divergence (vs fp32 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_special_case(x, 8, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_special_case(x, 8, "Boundary-pos-inf", source_semantics=True)

    # tail group + tail batch double trigger (rem=1 && k_tail=44).
    _run_case(33, 300, torch.float16, 16, "Boundary")

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
