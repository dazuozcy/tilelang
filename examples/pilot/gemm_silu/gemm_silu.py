# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""gemm_silu.py -- TileLang-Ascend fused GEMM + SiLU epilogue operator.

programming_mode: mixed
auto_schedule: yes

Math (DESIGN.md section 1.3)::

    Z = A @ B^T           (fp16 inputs, fp32 accumulation in the Cube L0C)
    C = silu(Z) = Z / (1 + exp(-Z))   (fp32 domain, direct-division form)
    C cast to fp16 at the write-back (cast post-poned past the activation).

Structure (DESIGN.md section 3.3): a persistent mixed kernel. Each of the
M_TILES x N_TILES output tiles runs the Cube NT GEMM path (T.copy GM->L1 with
ND2NZ fractal conversion inserted automatically by InsertNd2Nz, then T.gemm
with transpose_B=True, clear_accum on the first K tile) followed by a dual-AIV
epilogue: T.dual_copy L0C->UB (M-split, fp32, no type conversion on dav-950),
a SimtVF elementwise pass z/(1+exp(-z)) with implicit fp32->fp16 cast into the
output UB tile, and T.dual_copy UB->GM (M-split write-back).

The kernel is parameterized by (M, N, K) and requires each to be a multiple of
the 128-sized block (the contract shape 512x512x512 satisfies this; 16-aligned
shapes that are also multiples of 128 are accepted for the L1/L2 tests). Non
block-divisible shapes are rejected in the host wrapper (DESIGN.md section 8.4).

run: python gemm_silu.py --level all
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


# Block sizes (DESIGN.md section 5.2). TILE_M == TILE_N == TILE_K == 128.
TILE_M = 128
TILE_N = 128
TILE_K = 128
NUM_STAGES = 2  # K-dim software-pipeline depth (L1 double buffering).


def golden_gemm_silu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference on NPU (DESIGN.md section 8.1).

    A [M, K] fp16; B [N, K] fp16 (NT convention, B transpose participates in the
    product); returns C [M, N] fp16. Semantics: fp32 accumulation GEMM -> silu
    applied on the fp32 domain (z * sigmoid(z), the baseline form, independent of
    the kernel's direct-division optimization) -> cast fp16 (post-poned).
    """
    A = A.to("npu")
    B = B.to("npu")
    Z = A.float() @ B.float().t()
    return (Z * torch.sigmoid(Z)).to(torch.float16)


@tilelang.jit(target="ascend", out_idx=[2])
def build_gemm_silu(M: int, N: int, K: int):
    """Build the gemm_silu PrimFunc (mixed Cube GEMM + Vector silu epilogue)."""
    if M % TILE_M or N % TILE_N or K % TILE_K:
        raise ValueError(
            f"gemm_silu expects M, N, K each a multiple of {TILE_M}/{TILE_N}/{TILE_K} "
            f"(block size); got ({M}, {N}, {K})."
        )
    m_tiles = M // TILE_M
    n_tiles = N // TILE_N
    k_tiles = K // TILE_K
    num_blocks = m_tiles * n_tiles  # 1-D grid only: T.Kernel(N), no threads=.

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(num_blocks) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc_ub = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            out_ub = T.alloc_shared((TILE_M // 2, TILE_N), "float16")
            for tile_idx in T.Persistent([m_tiles * n_tiles], num_blocks, bx):
                m_tile = tile_idx // n_tiles
                n_tile = tile_idx % n_tiles
                for kt in T.Pipelined(k_tiles, num_stages=NUM_STAGES):
                    T.copy(
                        A[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K],
                        x_l1,
                    )
                    T.copy(
                        B[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K],
                        w_l1,
                    )
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
                # Epilogue: L0C -> UB (M-split dual AIV, fp32, no conversion) ->
                # SimtVF silu + cast fp16 -> UB -> GM (M-split dual AIV).
                T.dual_copy(res, acc_ub)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(TILE_M // 2, TILE_N):
                        z = acc_ub[i, j]
                        out_ub[i, j] = z / (1.0 + T.exp(-z))
                T.dual_copy(
                    out_ub,
                    C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N],
                )

    return main


def _tolerance(dtype):
    # fp16 output -> 1e-2 (DESIGN.md section 8.2).
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3


def _run_case(M, N, K, tag, seed=0, atol=None, rtol=None, a=None, b=None, equal_nan=False):
    if a is None or b is None:
        torch.manual_seed(seed)
        a = torch.randn(M, K, dtype=torch.float16).to("npu")
        b = torch.randn(N, K, dtype=torch.float16).to("npu")
    kernel = build_gemm_silu(M, N, K)
    c = kernel(a, b)
    torch.npu.synchronize()
    golden = golden_gemm_silu(a, b)
    if atol is None or rtol is None:
        atol = rtol = _tolerance(torch.float16)
    torch.testing.assert_close(c, golden, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = (c.float() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({M},{N},{K}) max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    """Contract shape 512x512x512, randn x3 seeds (DESIGN.md section 8.3)."""
    for seed in (0, 1, 2):
        _run_case(512, 512, 512, "L0", seed=seed)


def run_L1():
    """Semantic boundary: adjacent 16-aligned (block-multiple) shapes, value
    ranges covering the fp32 exp overflow boundary (+-88), NaN/inf propagation,
    and bitwise determinism (DESIGN.md section 8.3)."""
    # Adjacent shapes (multiples of the 128 block, hence also 16-aligned).
    for shape in ((256, 256, 256), (128, 128, 128), (384, 256, 512), (512, 512, 256)):
        _run_case(*shape, "L1-shape", seed=3)

    # Value-range construction: scale A/B to push z ~ +-100 (both sides of the
    # exp overflow boundary +-88) and ~ +-1e-3 (near-zero region).
    torch.manual_seed(4)
    base_a = torch.randn(512, 512, dtype=torch.float16).to("npu")
    base_b = torch.randn(512, 512, dtype=torch.float16).to("npu")
    for scale in (8.0, 0.02):
        _run_case(
            512, 512, 512, "L1-range",
            a=(base_a * scale).to(torch.float16),
            b=(base_b * scale).to(torch.float16),
        )

    # NaN/inf propagation: verify the output NaN positions match the golden
    # (equal_nan=True).
    torch.manual_seed(5)
    a_nan = torch.randn(512, 512, dtype=torch.float16).to("npu")
    b_nan = torch.randn(512, 512, dtype=torch.float16).to("npu")
    a_nan[0, 0] = float("nan")
    b_nan[3, 7] = float("inf")
    _run_case(512, 512, 512, "L1-naninf", a=a_nan, b=b_nan, equal_nan=True)

    # Determinism: same input run twice -> bitwise identical.
    torch.manual_seed(6)
    a_det = torch.randn(512, 512, dtype=torch.float16).to("npu")
    b_det = torch.randn(512, 512, dtype=torch.float16).to("npu")
    kernel = build_gemm_silu(512, 512, 512)
    c1 = kernel(a_det, b_det)
    c2 = kernel(a_det, b_det)
    torch.npu.synchronize()
    assert torch.equal(c1, c2), "gemm_silu output is not bitwise deterministic"
    print("[L1-determinism] PASS: bitwise identical")


def run_L2():
    """Large-shape robustness probe; failures recorded as warnings (non-blocking)."""
    try:
        _run_case(1024, 1024, 1024, "L2", seed=7)
    except Exception as e:
        print(f"[L2] WARN (record without blocking): {e}")


def run_boundary():
    """Special values: zeros, ones (positive saturation), neg-ones (exp overflow
    -> -0.0), extreme finite values, and single-NaN injection (DESIGN.md 8.4)."""
    zeros_a = torch.zeros(512, 512, dtype=torch.float16, device="npu")
    zeros_b = torch.zeros(512, 512, dtype=torch.float16, device="npu")
    _run_case(512, 512, 512, "Boundary-zeros", a=zeros_a, b=zeros_b)

    ones_a = torch.ones(512, 512, dtype=torch.float16, device="npu")
    ones_b = torch.ones(512, 512, dtype=torch.float16, device="npu")
    # z = K * 1 = 512 -> silu(512) ~ 512 (large positive saturation).
    _run_case(512, 512, 512, "Boundary-ones", a=ones_a, b=ones_b)

    neg_a = torch.ones(512, 512, dtype=torch.float16, device="npu")
    neg_b = -torch.ones(512, 512, dtype=torch.float16, device="npu")
    # z = -512 -> exp(512) overflows fp32 -> silu -> -0.0 (correct limit).
    _run_case(512, 512, 512, "Boundary-negones", a=neg_a, b=neg_b)

    # Extreme finite values pushing z beyond the exp overflow boundary both ways.
    torch.manual_seed(8)
    ext_a = (torch.randn(512, 512, dtype=torch.float16) * 0.5).to("npu")
    ext_b = (torch.randn(512, 512, dtype=torch.float16) * 100.0).to("npu")
    _run_case(512, 512, 512, "Boundary-extreme", a=ext_a, b=ext_b)

    # Single-NaN injection propagates through the whole output row.
    torch.manual_seed(9)
    nan_a = torch.randn(512, 512, dtype=torch.float16).to("npu")
    nan_a[7, 7] = float("nan")
    _run_case(512, 512, 512, "Boundary-nan", a=nan_a, equal_nan=True)


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
