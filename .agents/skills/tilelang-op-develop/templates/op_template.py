# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""{op}.py template -- TileLang-Ascend operator skeleton (target="ascend").

This template takes elementwise_add as the example operator. When generating
an actual operator, replace the kernel/golden/test cases per the frozen
DESIGN.md, then remove every TODO marker (gate check: no placeholders).

Fill-in guide (every fill point is marked with TODO below):
  1. Kernel: always `import tilelang.ascend.language as T` (Ascend dialect).
     Record the verified programming mode here after L0 passes:
       programming_mode: simt  (T.SimtVF(threads=N) + T.Parallel; default)
                         | simd (T.SimdVF() + T.simd.* 2048-bit intrinsics)
                         | mixed (T.MixedKernel(sids=1|2) + T.gemm epilogue)
       auto_schedule: yes (default) | no (fully manual flags; compile needs
         pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
     Selection authority: tilelang-ascend-opdev skill family table and the
     tilelang-ascend skill (section references in AGENTS.md).
  2. Golden: torch reference computed on NPU tensors (device="npu").
  3. Tolerance by dtype: fp8 -> 1e-1, bf16/fp16 -> 1e-2, fp32 -> 1e-3
     (5e-3 for accumulation-heavy operators); DESIGN.md may override.
  4. Layered tests: run_L0 / run_L1 / run_L2 / run_boundary + --level entry.

run: python {op}.py --level all
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_add(x, y):
    # TODO: Replace with the reference semantics from DESIGN.md section 8.
    # The golden must be computable with torch on NPU tensors.
    return x + y


# ---------- Tolerance by dtype ----------
def _tolerance(dtype):
    # TODO: Tighten per the DESIGN.md precision standard when it specifies one.
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1e-1
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32; use 5e-3 for accumulation-heavy operators


# ---------- Kernel ----------
def add_kernel(M, N, block_M=32, block_N=32, in_dtype="float32", out_dtype="float32"):
    """Build the PrimFunc of the operator (programming_mode: simt).

    TODO: For simd mode use T.SimdVF() + T.simd.* intrinsics (hot spots only);
    for mixed (Cube+Vector) kernels follow examples/ascend/example_gemm.py:
    T.copy GM->L1 (ND2NZ fractal conversion is inserted automatically by the
    InsertNd2Nz pass -- never hand-write NZ staging) -> T.gemm(A_l1, B_l1,
    C_l0c, transpose_B=True) -> T.dual_copy L0C->UB. Weights live in GM as
    [N, K] for this NT path.
    """
    m_tiles = (M + block_M - 1) // block_M
    n_tiles = (N + block_N - 1) // block_N
    num_blocks = m_tiles * n_tiles  # 1-D grid only: T.Kernel(N), no threads=

    @T.prim_func
    def _main(
        A: T.Tensor((M, N), in_dtype),
        B: T.Tensor((M, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            # TODO: Replace with the DESIGN.md grid plan (grid size /
            # AIC-AIV partition / sync strategy). Ascend supports 1-D grids
            # only, so decompose 2-D tiles by div/mod as below.
            by = bx // n_tiles
            bi = bx % n_tiles
            # TODO: Allocate per the DESIGN.md memory plan. T.alloc_shared
            # maps to UB (248KB physical, 216KB with SimtVF); T.alloc_l1 to
            # CBuf; T.alloc_l0a/l0b/l0c to the Cube buffers.
            a_ub = T.alloc_shared((block_M, block_N), in_dtype)
            b_ub = T.alloc_shared((block_M, block_N), in_dtype)
            c_ub = T.alloc_shared((block_M, block_N), out_dtype)
            # TODO: If DESIGN.md specifies software pipelining, use
            # T.Pipelined(..., num_stages=...) with T.annotate_buffer_versions
            # instead of the plain copies below.
            T.copy(A[by * block_M : (by + 1) * block_M, bi * block_N : (bi + 1) * block_N], a_ub)
            T.copy(B[by * block_M : (by + 1) * block_M, bi * block_N : (bi + 1) * block_N], b_ub)
            # TODO: Replace with the DESIGN.md loop structure. Synchronization
            # relies on AutoSchedule by default (auto_schedule: yes); manual
            # T.ascend_set_flag/T.ascend_wait_flag is only for auto_schedule:
            # no kernels and must be paired and commented.
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(block_M, block_N):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
            T.copy(c_ub, C[by * block_M : (by + 1) * block_M, bi * block_N : (bi + 1) * block_N])

    return _main


# ---------- Layered testing ----------
def _run_case(M, N, dtype, tag):
    a = torch.randn(M, N, dtype=dtype, device="npu")
    b = torch.randn(M, N, dtype=dtype, device="npu")
    # TODO: Set tile sizes and staging per the DESIGN.md tiling strategy.
    program = add_kernel(
        M,
        N,
        block_M=32,
        block_N=32,
        in_dtype=str(dtype).replace("torch.", ""),
        out_dtype=str(dtype).replace("torch.", ""),
    )
    # Canonical entry: @tilelang.jit(target="ascend"). tilelang.compile(...,
    # target="ascend", out_idx=-1) is the equivalent explicit form; both are
    # accepted by the Stage 3 gate.
    # TODO: For a fully manual kernel (auto_schedule: no) also pass
    # pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}.
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    c = kernel(a, b)
    torch.npu.synchronize()
    golden = golden_add(a, b)
    tol = _tolerance(dtype)
    torch.testing.assert_close(c, golden, rtol=tol, atol=tol)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype}")


def run_L0():
    # TODO: Cover the minimal shapes from the DESIGN.md L0 test plan.
    for M, N in [(128, 128), (256, 256)]:
        _run_case(M, N, torch.float32, "L0")


def run_L1():
    # TODO: Replace with the DESIGN.md representative shapes. Non-divisible
    # tails are exercised here (compiler OOB handling); keep them only if the
    # DESIGN.md tiling/tail strategy permits.
    for M, N in [(130, 130), (64, 200)]:
        _run_case(M, N, torch.float32, "L1")


def run_L2():
    # Robustness / large-shape probing; failures are recorded as warnings and
    # do not block the [PRECISION_PASS] verdict.
    try:
        _run_case(1024, 1024, torch.float32, "L2")
        # TODO: Add more dtypes per the DESIGN.md I/O spec, e.g.:
        # _run_case(128, 128, torch.bfloat16, "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): {e}")


def run_boundary():
    # TODO: Replace with the DESIGN.md special-value cases (zeros, extremes,
    # denormals, empty dims where legal).
    a = torch.zeros(128, 128, dtype=torch.float32, device="npu")
    b = torch.zeros(128, 128, dtype=torch.float32, device="npu")
    program = add_kernel(128, 128, block_M=32, block_N=32)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    c = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(c, golden_add(a, b), rtol=1e-3, atol=1e-3)
    print("[Boundary] PASS: zeros")


def _show_source():
    """Print the generated AscendC source (debugging aid, see tilelang-ascend
    skill section 11 and docs/tools/ for lower traces and pass diffs)."""
    program = add_kernel(128, 128, block_M=32, block_N=32)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true", help="print the generated AscendC kernel source")
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
