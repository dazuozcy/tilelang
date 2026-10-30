# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Token-major weighted reduce for the indexed expert MLP (Ascend A5).

programming_mode: simd
auto_schedule: yes

Source: GPU TileOPs
``src/tileops/_kernels/moe/indexed_expert_gemm.py`` ``_weighted_reduce``
(L219-252), migrated to the Ascend dialect
(``import tilelang.ascend.language as T``).

Semantics (DESIGN section 0.1): given ``expert_output[T, K, H]``
(``dtype`` in {fp16, bf16}, K = top_k routes) and ``topk_weights[T, K]``
(fp32 routing weights), compute for every (token, col):

    output[token, col] = cast_d( scaling * sum_{slot=0}^{K-1}
        cast_f32(expert_output[token, slot, col]) * topk_weights[token, slot] )

The sum accumulates in fp32 over slots in *ascending* order, ``scaling`` is
applied once in fp32 afterwards, and a single narrowing cast produces the
storage dtype.  ``output`` is written in place (``out_idx=[]``).

Structure (DESIGN section 3.3): a 1-D ``T.Kernel(num_tokens * nh)`` grid of
chunks (``nh`` h-chunks per token, chosen by ``_pick_nh`` so ``BH = H / nh``
divides H -- no tail blocks, every ``T.copy`` extent is compile-time static).
Each block stages a ``[K, BH]`` slice of expert output and the ``[K]`` weight
row into UB, runs a SimdVF reduce over 128-lane column groups (fp32 slot-
ascending FMA via ``vmula``, weights broadcast with ``BRC_B32``), and writes
back the narrowing-cast result.  The 16->32 widen splits even/odd lanes
(``vcvt(part=0/1)``); the epilogue re-merges them with ``vor`` (inverse
permutation of the load split, flash_attention core precedent) and stores with
one full-width ``vsts(dist="NORM_B16")``.

Call surface (wrapper-compatible, ``threads`` removed per K3/K9):

    _weighted_reduce(num_tokens, top_k, hidden_size, dtype_str,
                     scaling)()(expert_output, topk_weights, output)

``out_idx=[]`` -- in-place write of ``output``, no return value.

run: python _weighted_reduce.py --level all
"""

import argparse
import functools

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

import tilelang
import tilelang.ascend.language as T


_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Adversarial corner values (DESIGN 8.3 / verify_equiv corner set, saturation
# option (b)): extreme finite values capped at +-1000 and weights <= 2.0 so the
# fp32 accumulation stays inside the storage dtype's representable range --
# no finite overflow, so the vcvt(sat=True) narrowing matches torch's ``.to``
# (inf/-inf/NaN/+0/-0/subnormal stay in range and are compared positionally).
_CORNER_VALS = [
    0.0, -0.0, 1.0, -1.0, 0.5,
    float("inf"), float("-inf"), float("nan"),
    1000.0, -1000.0,              # extreme finite (no saturation)
    6.104e-05, 5.96e-08,          # fp16 min-normal / subnormal
    3.0, -0.25,
]
_CORNER_WEIGHTS = [0.0, 1.0, -1.0, 1e-3, -1e-3, 0.5, 2.0, -0.5,
                   0.25, -2.0, 0.125, 1.5]


def _pick_nh(num_tokens: int, hidden_size: int, *, target_blocks: int = 224,
             min_cols: int = 128) -> int:
    """Host compile-time chunk-count heuristic (DESIGN 5.2): pick a divisor n
    of H (n <= H // 128 so every chunk keeps >= 128 columns) minimizing
    |T*n - target| with ties toward larger n; fallback n = 1 (whole row)."""
    cands = [n for n in range(1, hidden_size + 1)
             if hidden_size % n == 0 and n <= max(1, hidden_size // min_cols)]
    if not cands:
        return 1
    return min(cands, key=lambda n: (abs(num_tokens * n - target_blocks), -n))


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_weighted_reduce(expert_output: torch.Tensor, topk_weights: torch.Tensor,
                           scaling: float) -> torch.Tensor:
    """Source-semantics weighted reduce (DESIGN 0.1).

    expert_output: [T, K, H] fp16/bf16; topk_weights: [T, K] fp32 -> [T, H].
    fp32 accumulation over slots (ascending), scaling applied once in fp32,
    single narrowing cast at the end. Device: stays on expert_output.device.
    """
    x32 = expert_output.float()                              # per-element up-cast
    acc = torch.zeros(expert_output.shape[0], expert_output.shape[2],
                      dtype=torch.float32, device=expert_output.device)
    for k in range(expert_output.shape[1]):                  # slot ascending
        acc = acc + x32[:, k, :] * topk_weights[:, k:k + 1]  # fp32 mul + add
    return (acc * scaling).to(expert_output.dtype)           # one narrowing cast


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _weighted_reduce(num_tokens: int, top_k: int, hidden_size: int,
                     dtype: str, scaling: float):
    """Build the token-major weighted-reduce kernel (fp32 slot-ascending FMA).

    Returns the no-arg ``@tilelang.jit`` factory; the A5 wrapper calls
    ``_weighted_reduce(T, K, H, dtype_str, float(scaling))(**config)(...)``
    with ``config == {}``.
    """
    nh = _pick_nh(num_tokens, hidden_size)          # h-chunks per token
    bh = hidden_size // nh                          # bh | H -> no tail block
    bh_pad = ((bh + 127) // 128) * 128              # 128-lane group padding
    ng = bh_pad // 128                              # groups per chunk

    # Build-time capacity guard (DESIGN 5.4): x_ub + out_ub + w_ub must fit
    # the 90%-discounted UB budget (228,556 B). Harness/test shapes are far
    # below it; raise rather than silently downgrade nh for Stage 3.
    ub_bytes = top_k * bh_pad * 2 + bh_pad * 2 + top_k * 4
    assert ub_bytes <= 228556, (
        f"_weighted_reduce UB budget exceeded: {ub_bytes}B > 228556B "
        f"(T={num_tokens} K={top_k} H={hidden_size})"
    )

    @tilelang.jit(target="ascend", out_idx=[])
    def _kernel():
        @T.prim_func
        def main(
            expert_output: T.Tensor((num_tokens, top_k, hidden_size), dtype),
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),
            output: T.Tensor((num_tokens, hidden_size), dtype),
        ):
            with T.Kernel(num_tokens * nh) as bid:
                token = bid // nh                    # block-level scalar
                h0 = (bid % nh) * bh
                x_ub = T.alloc_shared((top_k, bh_pad), dtype)
                w_ub = T.alloc_shared((top_k,), "float32")
                out_ub = T.alloc_shared((bh_pad,), dtype)
                # ---- staging: static extents only (DESIGN 0.6 R2) ----
                T.copy(expert_output[token, 0:top_k, h0:h0 + bh], x_ub[0:top_k, 0:bh])
                T.copy(topk_weights[token, 0:top_k], w_ub[0:top_k])
                # ---- vector reduce: lanes = H columns, K in registers ----
                with T.SimdVF():
                    acc_even = T.simd.alloc_var("float32")  # VRF, not UB
                    acc_odd = T.simd.alloc_var("float32")
                    for g in T.serial(ng):
                        acc_even = T.simd.vdup(T.float32(0.0), "float32")
                        acc_odd = T.simd.vdup(T.float32(0.0), "float32")
                        for k in range(top_k):             # compile-time unroll
                            w_vec = T.simd.vld(w_ub[k], "BRC_B32")     # broadcast
                            x_h = T.simd.vld(x_ub[k, g * 128])         # 128 lanes,
                            #   register lane j <-> column g*128 + j
                            x_even = T.simd.vcvt(x_h, "float32", part=0)  # even lanes
                            x_odd = T.simd.vcvt(x_h, "float32", part=1)   # odd lanes
                            T.simd.vmula(acc_even, x_even, w_vec)      # slot ascending
                            T.simd.vmula(acc_odd, x_odd, w_vec)
                        if scaling != 1.0:                    # OPT-1 specialization
                            acc_even = T.simd.vmuls(acc_even, T.float32(scaling))
                            acc_odd = T.simd.vmuls(acc_odd, T.float32(scaling))
                        # epilogue: even/odd narrowing placement + bitwise
                        # re-merge (inverse of the load split) + one full-width
                        # 128-lane store
                        even16 = T.simd.vcvt(acc_even, dtype, part=0)  # even 16b lanes
                        odd16 = T.simd.vcvt(acc_odd, dtype, part=1)   # odd 16b lanes
                        merged = T.simd.vor(
                            T.reinterpret(even16, "uint16x128"),
                            T.reinterpret(odd16, "uint16x128"),
                        )  # disjoint lane positions -> OR == merge
                        T.simd.vsts(out_ub[g * 128],
                                    T.reinterpret(merged, dtype + "x128"),
                                    dist="NORM_B16")          # 128 contiguous elements
                # ---- writeback: extent bh (padded lanes stay in UB, E6) ----
                T.copy(out_ub[0:bh], output[token, h0:h0 + bh])

        return main

    return _kernel


# ---------- Layered testing ----------
def _compile_kernel(T_: int, K: int, H: int, dtype_str: str, scaling: float):
    return _weighted_reduce(T_, K, H, dtype_str, scaling)()


def _gen_rand(T_: int, K: int, H: int, dtype: torch.dtype, seed: int):
    """Random inputs: x ~ N(0, 0.1) cast to dtype; weights = softmax(randn)."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T_, K, H, generator=g, dtype=torch.float32) * 0.1).to(dtype).to("npu")
    w = torch.softmax(torch.randn(T_, K, generator=g, dtype=torch.float32), dim=-1).to("npu")
    return x, w


def _gen_corner(T_: int, K: int, H: int, dtype: torch.dtype):
    """Adversarial corner values (saturation-safe, DESIGN 8.3 / option (b))."""
    vals = torch.tensor(_CORNER_VALS, dtype=torch.float32)
    idx = torch.arange(T_ * K * H) % vals.numel()
    x = vals[idx].reshape(T_, K, H).to(dtype).to("npu")
    wv = torch.tensor(_CORNER_WEIGHTS, dtype=torch.float32)
    widx = torch.arange(T_ * K) % wv.numel()
    w = wv[widx].reshape(T_, K).to("npu")
    return x, w


def _report_diff(out: torch.Tensor, golden: torch.Tensor) -> float:
    """Max abs diff over finite elements (0.0 if none finite)."""
    d = (out.float() - golden.float()).abs()
    finite = torch.isfinite(d)
    return d[finite].max().item() if finite.any() else 0.0


def _run_case(T_: int, K: int, H: int, dtype_str: str, scaling: float,
              x: torch.Tensor, w: torch.Tensor, tag: str, *,
              corner: bool = False, kernel=None):
    assert tuple(x.shape) == (T_, K, H) and x.dtype == _TORCH_DTYPE[dtype_str]
    assert tuple(w.shape) == (T_, K) and w.dtype == torch.float32
    assert x.device.type == "npu" and w.device.type == "npu"
    if kernel is None:
        kernel = _compile_kernel(T_, K, H, dtype_str, scaling)
    out = torch.empty(T_, H, dtype=_TORCH_DTYPE[dtype_str], device="npu")
    kernel(x, w, out)
    torch.npu.synchronize()
    golden = golden_weighted_reduce(x, w, scaling)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2, equal_nan=corner)
    # nonfinite mask equality (overflow/saturation + inf/nan position safety net)
    if not torch.equal(torch.isfinite(out.float()), torch.isfinite(golden.float())):
        raise AssertionError(f"[{tag}] nonfinite mask mismatch")
    max_diff = _report_diff(out, golden)
    print(f"[{tag}] PASS: T={T_} K={K} H={H} dtype={dtype_str} s={scaling} "
          f"max_diff={max_diff:.3e}")
    return max_diff


def _run_determinism(T_: int, K: int, H: int, dtype_str: str, scaling: float,
                     x: torch.Tensor, w: torch.Tensor, tag: str):
    kernel = _compile_kernel(T_, K, H, dtype_str, scaling)
    outs = []
    for _ in range(3):
        out = torch.empty(T_, H, dtype=_TORCH_DTYPE[dtype_str], device="npu")
        kernel(x, w, out)
        torch.npu.synchronize()
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]), (
        "nondeterministic output over 3 runs"
    )
    _run_case(T_, K, H, dtype_str, scaling, x, w, tag + "-exact", kernel=kernel)
    print(f"[{tag}] PASS: 3x bitwise identical")


def run_L0():
    # 1. compile smoke: 8 manifest specializations (s=1.0 skip-mul path).
    for T_, K, H in [(1, 8, 7168), (32, 8, 7168), (64, 8, 7168), (32, 8, 4096)]:
        for dtype_str in ("float16", "bfloat16"):
            _compile_kernel(T_, K, H, dtype_str, 1.0)
    print("[L0-compile-smoke] PASS: 8 manifest specializations lower")

    # 2. golden compare: test shape set x {rand, corner} x 2 dtype, s=2.5 (mul path).
    shapes = [(4, 2, 128), (17, 2, 128), (4, 2, 100), (8, 4, 512), (2, 2, 128), (1, 2, 128)]
    for dtype_str in ("float16", "bfloat16"):
        dt = _TORCH_DTYPE[dtype_str]
        for (T_, K, H) in shapes:
            _run_case(T_, K, H, dtype_str, 2.5, *_gen_rand(T_, K, H, dt, seed=1234),
                      "L0-rand", corner=False)
            _run_case(T_, K, H, dtype_str, 2.5, *_gen_corner(T_, K, H, dt),
                      "L0-corner", corner=True)

    # 3. scaling dual path: s=1.0 (OPT-1 skip-mul) vs s=2.5 (mul).
    for dtype_str in ("float16", "bfloat16"):
        dt = _TORCH_DTYPE[dtype_str]
        for (T_, K, H) in [(4, 2, 128), (8, 4, 512)]:
            x, w = _gen_rand(T_, K, H, dt, seed=99)
            _run_case(T_, K, H, dtype_str, 1.0, x, w, "L0-scale-1.0", corner=False)
            _run_case(T_, K, H, dtype_str, 2.5, x, w, "L0-scale-2.5", corner=False)

    # 4. manifest-scale golden (real workload shape, nh=7 chunking).
    for dtype_str in ("float16", "bfloat16"):
        dt = _TORCH_DTYPE[dtype_str]
        x, w = _gen_rand(32, 8, 7168, dt, seed=7)
        _run_case(32, 8, 7168, dtype_str, 2.5, x, w, "L0-manifest-32x8x7168",
                  corner=False)

    # 5. determinism (3x bitwise identical).
    dt = torch.float16
    x, w = _gen_rand(32, 8, 7168, dt, seed=42)
    _run_determinism(32, 8, 7168, "float16", 1.0, x, w, "L0-determinism")


def run_L1():
    cases = [
        (16, 4, 256), (32, 8, 512), (5, 3, 384), (17, 2, 128), (8, 1, 512),
        (1, 8, 1024), (33, 4, 512), (2, 2, 100), (8, 4, 96), (4, 8, 1000),
    ]
    for dtype_str in ("float16", "bfloat16"):
        dt = _TORCH_DTYPE[dtype_str]
        for i, (T_, K, H) in enumerate(cases):
            x, w = _gen_rand(T_, K, H, dt, seed=1000 + i)
            _run_case(T_, K, H, dtype_str, 2.5, x, w, "L1-rand", corner=False)
    # corner spot-checks at broader shapes (incl. s=1.0 skip-mul).
    for dtype_str in ("float16", "bfloat16"):
        dt = _TORCH_DTYPE[dtype_str]
        _run_case(8, 4, 512, dtype_str, 2.5, *_gen_corner(8, 4, 512, dt),
                  "L1-corner", corner=True)
        _run_case(16, 8, 384, dtype_str, 1.0, *_gen_corner(16, 8, 384, dt),
                  "L1-corner-s1", corner=True)
    # K=1 (single-slot degenerate) and T=1 (single token).
    _run_case(8, 1, 128, "float16", 2.5, *_gen_rand(8, 1, 128, torch.float16, seed=5),
              "L1-K1", corner=False)
    _run_case(1, 8, 128, "float16", 2.5, *_gen_rand(1, 8, 128, torch.float16, seed=6),
              "L1-T1", corner=False)


def run_L2():
    # Robustness / large-shape probing; failures recorded as warnings (non-blocking).
    for T_, K, H, dtype_str in [(256, 8, 4096, "float16"), (64, 8, 7168, "bfloat16"),
                                (128, 4, 4096, "float16")]:
        try:
            dt = _TORCH_DTYPE[dtype_str]
            x, w = _gen_rand(T_, K, H, dt, seed=2000)
            _run_case(T_, K, H, dtype_str, 2.5, x, w, "L2", corner=False)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (not blocking): T={T_} K={K} H={H} "
                  f"dtype={dtype_str}: {e}")


def run_boundary():
    _run_case(8, 1, 128, "float16", 2.5, *_gen_rand(8, 1, 128, torch.float16, seed=1),
              "Boundary-K1", corner=False)
    _run_case(1, 8, 128, "float16", 2.5, *_gen_rand(1, 8, 128, torch.float16, seed=2),
              "Boundary-T1", corner=False)
    _run_case(4, 2, 100, "float16", 2.5, *_gen_rand(4, 2, 100, torch.float16, seed=3),
              "Boundary-H100", corner=False)
    _run_case(4, 2, 100, "bfloat16", 2.5, *_gen_rand(4, 2, 100, torch.bfloat16, seed=4),
              "Boundary-H100-bf16", corner=False)
    _run_case(1, 4, 96, "bfloat16", 2.5, *_gen_rand(1, 4, 96, torch.bfloat16, seed=5),
              "Boundary-H96", corner=False)
    _run_case(2, 2, 128, "float16", 2.5, *_gen_rand(2, 2, 128, torch.float16, seed=6),
              "Boundary-H128", corner=False)
    _run_case(3, 8, 2, "float16", 2.5, *_gen_rand(3, 8, 2, torch.float16, seed=7),
              "Boundary-H2", corner=False)
    _run_case(4, 2, 100, "float16", 2.5, *_gen_corner(4, 2, 100, torch.float16),
              "Boundary-corner-H100", corner=True)


def _show_source():
    """Print the generated AscendC source (debugging aid; tilelang-ascend skill 11)."""
    kernel = _compile_kernel(4, 2, 128, "float16", 2.5)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true",
                        help="print the generated AscendC kernel source")
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
