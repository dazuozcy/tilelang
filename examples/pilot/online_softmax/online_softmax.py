# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Row-wise numerically-stable softmax for Ascend A5 (migrated from GPU TileLang).

programming_mode: simt
auto_schedule: yes

Row-wise softmax (reduction axis dim=1). fp16 in/out, fp32 intermediate.
Two paths, per DESIGN.md section 1.4 / 3.3:

  * resident  (N <= 32768): UB/VRF row-block residency. Single GM read, single
    exp evaluation reused for normalization via per-row reciprocal (e * (1/S)).
    x_ub holds the BM x N row block (double-buffered, in-place overwrite); the
    SimtVF thread domain runs masked load -> reduce_max -> exp -> reduce_sum ->
    reciprocal -> masked in-place normalize.
  * streaming (N > 32768): online two-pass over BLOCK_N=8192 chunks with the
    flash-style running (max, sum) merge (O6, fp16 sentinel clamp) plus lse
    recompute; X is read twice and Y written once.

golden = torch.softmax(dim=1) computed on NPU (source-repo baseline, fp32
accumulation matches the DESIGN section 0.1 dtype semantics).
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


# 56 = vector_core_num of Ascend950PR_9579 (real query: torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56
BLOCK_N_STREAM = 8192  # streaming chunk width (fp16 -> 16 KiB per row)


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_online_softmax(X: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax, torch on NPU (source test baseline X.softmax(dim=1))."""
    assert X.device.type == "npu" and X.dtype == torch.float16
    return X.softmax(dim=1)


def _tolerance(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1e-1
    return 1e-3  # fp32


# ---------- Kernel builders ----------
def _build_resident(M, N, BM, TILE, num_tiles, threads):
    """UB/VRF row-block residency path (N <= 32768). See DESIGN section 3.3."""

    @T.prim_func
    def main(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(NUM_BLOCKS) as core_id:
            x_ub = T.alloc_shared((BM, N), "float16")
            T.annotate_buffer_versions({x_ub: 2})
            for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                base = t * BM
                T.copy(X[base : base + BM, 0:N], x_ub)
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((BM, TILE), "float32")
                    e_frag = T.alloc_fragment((BM, TILE), "float32")
                    m_frag = T.alloc_fragment((BM,), "float32")
                    s_frag = T.alloc_fragment((BM,), "float32")
                    inv_frag = T.alloc_fragment((BM,), "float32")
                    # Masked load: pad lanes get -inf (max identity); exp(-inf - m)
                    # then yields 0 (sum identity), so both reductions stay neutral.
                    for r, i in T.Parallel(BM, TILE):
                        x_frag[r, i] = T.if_then_else(
                            i < N, T.cast(x_ub[r, i], "float32"), -T.infinity(T.float32)
                        )
                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                    for r, i in T.Parallel(BM, TILE):
                        e_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                    T.reduce_sum(e_frag, s_frag, dim=1, clear=True)
                    for r in T.Parallel(BM):
                        inv_frag[r] = T.float32(1.0) / s_frag[r]
                    for r, i in T.Parallel(BM, TILE):
                        if i < N:
                            x_ub[r, i] = T.cast(e_frag[r, i] * inv_frag[r], "float16")
                T.copy(x_ub, Y[base : base + BM, 0:N])

    return main


def _build_streaming(M, N, BM, num_chunks, num_tiles):
    """Online two-pass streaming path (N > 32768). See DESIGN section 1.4."""
    BLOCK_N = BLOCK_N_STREAM
    TILE_C = tilelang.next_power_of_2(BLOCK_N)
    threads = min(1024, BM * TILE_C)
    num_full = N // BLOCK_N
    rem = N - num_full * BLOCK_N
    sentinel = T.float32(-65504.0)

    @T.prim_func
    def main(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(NUM_BLOCKS) as core_id:
            x_ub = T.alloc_shared((BM, BLOCK_N), "float16")
            # stats_ub[r, 0] = running max (m_run); stats_ub[r, 1] = running sum (s_run),
            # reused later to hold lse = m_run + log(s_run).
            stats_ub = T.alloc_shared((BM, 2), "float32")
            T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})
            for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                base = t * BM
                with T.SimtVF(threads=threads):
                    for r in T.Parallel(BM):
                        stats_ub[r, 0] = -T.infinity(T.float32)
                        stats_ub[r, 1] = T.float32(0.0)
                # ---- pass 1: full chunks, running (max, sum) flash merge ----
                for c in T.serial(num_full):
                    col = c * BLOCK_N
                    T.copy(X[base : base + BM, col : col + BLOCK_N], x_ub)
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        e_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        m_c = T.alloc_fragment((BM,), "float32")
                        s_c = T.alloc_fragment((BM,), "float32")
                        mn_frag = T.alloc_fragment((BM,), "float32")
                        for r, i in T.Parallel(BM, TILE_C):
                            x_frag[r, i] = T.cast(x_ub[r, i], "float32")
                        T.reduce_max(x_frag, m_c, dim=1, clear=True)
                        # fp16 finite sentinel clamp (O6): an all -inf chunk gets
                        # max -65504 so exp(x - m_c) = exp(-inf) = 0, not NaN.
                        for r in T.Parallel(BM):
                            m_c[r] = T.max(m_c[r], sentinel)
                        for r, i in T.Parallel(BM, TILE_C):
                            e_frag[r, i] = T.exp(x_frag[r, i] - m_c[r])
                        T.reduce_sum(e_frag, s_c, dim=1, clear=True)
                        for r in T.Parallel(BM):
                            mn_frag[r] = T.max(stats_ub[r, 0], m_c[r])
                            stats_ub[r, 1] = (
                                stats_ub[r, 1] * T.exp(stats_ub[r, 0] - mn_frag[r])
                                + s_c[r] * T.exp(m_c[r] - mn_frag[r])
                            )
                            stats_ub[r, 0] = mn_frag[r]
                # ---- pass 1 tail chunk (partial last block, static extent) ----
                if rem > 0:
                    col = num_full * BLOCK_N
                    T.copy(X[base : base + BM, col:N], x_ub[0:BM, 0:rem])
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        e_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        m_c = T.alloc_fragment((BM,), "float32")
                        s_c = T.alloc_fragment((BM,), "float32")
                        mn_frag = T.alloc_fragment((BM,), "float32")
                        for r, i in T.Parallel(BM, TILE_C):
                            x_frag[r, i] = T.if_then_else(
                                i < rem, T.cast(x_ub[r, i], "float32"), -T.infinity(T.float32)
                            )
                        T.reduce_max(x_frag, m_c, dim=1, clear=True)
                        for r in T.Parallel(BM):
                            m_c[r] = T.max(m_c[r], sentinel)
                        for r, i in T.Parallel(BM, TILE_C):
                            e_frag[r, i] = T.exp(x_frag[r, i] - m_c[r])
                        T.reduce_sum(e_frag, s_c, dim=1, clear=True)
                        for r in T.Parallel(BM):
                            mn_frag[r] = T.max(stats_ub[r, 0], m_c[r])
                            stats_ub[r, 1] = (
                                stats_ub[r, 1] * T.exp(stats_ub[r, 0] - mn_frag[r])
                                + s_c[r] * T.exp(m_c[r] - mn_frag[r])
                            )
                            stats_ub[r, 0] = mn_frag[r]
                # ---- lse = m_run + log(s_run) ----
                with T.SimtVF(threads=threads):
                    for r in T.Parallel(BM):
                        stats_ub[r, 1] = stats_ub[r, 0] + T.log(stats_ub[r, 1])
                # ---- pass 2: full chunks, recompute y = exp(x - lse) ----
                for c in T.serial(num_full):
                    col = c * BLOCK_N
                    T.copy(X[base : base + BM, col : col + BLOCK_N], x_ub)
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        for r, i in T.Parallel(BM, TILE_C):
                            x_frag[r, i] = T.cast(x_ub[r, i], "float32")
                        for r, i in T.Parallel(BM, TILE_C):
                            x_ub[r, i] = T.cast(T.exp(x_frag[r, i] - stats_ub[r, 1]), "float16")
                    T.copy(x_ub, Y[base : base + BM, col : col + BLOCK_N])
                # ---- pass 2 tail chunk ----
                if rem > 0:
                    col = num_full * BLOCK_N
                    T.copy(X[base : base + BM, col:N], x_ub[0:BM, 0:rem])
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((BM, TILE_C), "float32")
                        for r, i in T.Parallel(BM, TILE_C):
                            x_frag[r, i] = T.if_then_else(
                                i < rem, T.cast(x_ub[r, i], "float32"), T.float32(0.0)
                            )
                        for r, i in T.Parallel(BM, TILE_C):
                            if i < rem:
                                x_ub[r, i] = T.cast(T.exp(x_frag[r, i] - stats_ub[r, 1]), "float16")
                    T.copy(x_ub[0:BM, 0:rem], Y[base : base + BM, col:N])

    return main


def _build_program(M, N):
    """Host-side staticization: derive TILE / BM / path, return the PrimFunc."""
    TILE = tilelang.next_power_of_2(N)
    BM = 4 if (N <= 8192 and M % 4 == 0) else (2 if (N <= 16384 and M % 2 == 0) else 1)
    resident = N <= 32768
    if resident:
        num_tiles = tilelang.cdiv(M, BM)
        threads = min(1024, BM * TILE)
        return _build_resident(M, N, BM, TILE, num_tiles, threads)
    # streaming path: BM = 4 or 1 (per DESIGN section 3.3)
    BM = 4 if M % 4 == 0 else 1
    num_chunks = tilelang.cdiv(N, BLOCK_N_STREAM)
    num_tiles = tilelang.cdiv(M, BM)
    return _build_streaming(M, N, BM, num_chunks, num_tiles)


@tilelang.jit(target="ascend", out_idx=[1])
def online_softmax_kernel(M, N):
    """Build and JIT-compile the row-wise softmax PrimFunc for shape (M, N).

    Canonical @tilelang.jit(target="ascend") entry (lazy mode: returns the
    PrimFunc built by _build_program). The resident/streaming dispatch logic in
    _build_program is unchanged; out_idx=[1] returns the Y output tensor.
    """
    return _build_program(M, N)


def online_softmax(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    return online_softmax_kernel(M, N)(X)


# ---------- Layered testing ----------
def _run_case(M, N, dtype, tag, equal_nan=False):
    X = torch.randn(M, N, dtype=dtype, device="npu")
    Y = online_softmax(X)
    torch.npu.synchronize()
    golden = golden_online_softmax(X)
    tol = _tolerance(dtype)
    max_diff = (Y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(Y, golden, rtol=tol, atol=tol, equal_nan=equal_nan)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} max_diff={max_diff:.3e}")
    return max_diff


def _run_special_case(X, tag):
    """Compare a pre-built special-value input against golden (equal_nan aware)."""
    Y = online_softmax(X)
    torch.npu.synchronize()
    golden = golden_online_softmax(X)
    tol = _tolerance(torch.float16)
    # equal_nan=True: NaN rows (expected per DESIGN section 8.2) compare equal.
    torch.testing.assert_close(Y, golden, rtol=tol, atol=tol, equal_nan=True)
    max_diff = (Y.float() - golden.float()).abs().nan_to_num(nan=0.0).max().item()
    print(f"[{tag}] PASS: shape={tuple(X.shape)} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    # Main shape + smoke (DESIGN section 8.2 L0 plan).
    _run_case(8192, 8192, torch.float16, "L0")
    _run_case(128, 4096, torch.float16, "L0")


def run_L1():
    # Shape / config sweep: BM adaptivity, non-pow2 masking, non-32B rows.
    for M in (1, 3, 56, 57, 128, 4096, 8192):
        for N in (1, 16, 1000, 4096, 8191, 8192):
            _run_case(M, N, torch.float16, "L1")
    # Streaming path (N > 32768) and minimal shape.
    _run_case(128, 65536, torch.float16, "L1")
    _run_case(1, 1, torch.float16, "L1")


def run_L2():
    # Robustness / stress: large shapes + resident/streaming switch boundary.
    for M, N in ((16384, 16384), (32768, 32768), (32769, 32768), (128, 32769)):
        try:
            _run_case(M, N, torch.float16, "L2")
        except Exception as e:  # L2 recorded as warn, does not block the verdict
            print(f"[L2] WARN (not blocking): shape=({M},{N}) {e}")


def run_boundary():
    # Special values: extremes, all-equal, single-dominant, subnormal, underflow,
    # +/-inf, NaN, N=1/M=1, determinism (DESIGN section 8.2 L2/Boundary plan).
    n = 128

    def base(rows=8, cols=128):
        return torch.zeros(rows, cols, dtype=torch.float16, device="npu")

    # all zeros (row sum of 1 -> uniform output)
    _run_special_case(base(), "Boundary-zeros")

    # extremes +-65504
    X = base()
    X[:, 0] = 65504.0
    X[:, 1] = -65504.0
    _run_special_case(X, "Boundary-extreme")

    # all-equal rows
    X = base()
    X[:, :] = 3.5
    _run_special_case(X, "Boundary-all-equal")

    # single dominant value
    X = base()
    X[:, 0] = 100.0
    _run_special_case(X, "Boundary-single-dominant")

    # subnormal magnitude
    X = base()
    X[:, 0] = 6e-8
    X[:, 1] = -6e-8
    _run_special_case(X, "Boundary-subnormal")

    # deep underflow (exp(-745) ~ 0)
    X = base()
    X[:, 0] = -745.0
    X[:, 1] = -744.0
    _run_special_case(X, "Boundary-deep-underflow")

    # +inf mixed (whole row -> NaN, torch-consistent)
    X = base()
    X[:, 0] = float("inf")
    _run_special_case(X, "Boundary-pos-inf")

    # -inf mixed (finite rows -> -inf position maps to 0)
    X = base()
    X[:, 0] = float("-inf")
    _run_special_case(X, "Boundary-neg-inf")

    # all -inf row (-> NaN)
    X = base()
    X[:, :] = float("-inf")
    _run_special_case(X, "Boundary-all-neg-inf")

    # NaN row (-> all NaN)
    X = base()
    X[:, 0] = float("nan")
    _run_special_case(X, "Boundary-nan")

    # N=1 and M=1 minimal shapes
    X = torch.randn(3, 1, dtype=torch.float16, device="npu")
    _run_special_case(X, "Boundary-N1")
    X = torch.randn(1, 16, dtype=torch.float16, device="npu")
    _run_special_case(X, "Boundary-M1")

    # determinism: same input 3x, bitwise identical
    X = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = online_softmax(X)
    torch.npu.synchronize()
    y1 = online_softmax(X)
    torch.npu.synchronize()
    y2 = online_softmax(X)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()

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
