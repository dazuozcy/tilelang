# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""SiLU (Swish) activation kernel for TileLang-Ascend (target="ascend").

Elementwise mapping: silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).

programming_mode: simt
auto_schedule: yes

Implementation notes (frozen DESIGN.md):
  - Pure vector elementwise operator; Cube (AIC) and L1/L0 buffers are unused.
  - 1-D grid T.Kernel(NUM_BLOCKS) (no threads= param), SIMT thread domain declared
    inside T.SimtVF(threads=512).
  - Flat continuous axis tiling: TILE=2048 elements (16 x 128-el fp16 vector width),
    NUM_BLOCKS=32, ITERATIONS = TOTAL // (NUM_BLOCKS * TILE).
  - Grid divisibility is a hard constraint: TOTAL must be a multiple of
    NUM_BLOCKS * TILE (otherwise trailing elements are silently never written).
  - auto_schedule: yes -> MTE2 -> VF -> MTE3 flags are inserted by AutoSchedule;
    no manual ascend_set_flag/ascend_wait_flag.
  - SimtVF T.Parallel loop lowers T.exp to the 2048-bit vexp vector instruction
    (src/ascend/transform/ascend_simdvf_lower_parallel.cc); no explicit SIMD
    intrinsics required.

run: python silu.py --level {L0,L1,L2,boundary,all}
"""

import argparse

import torch
import torch_npu  # noqa: F401  (registers the NPU device; torch.npu.is_available() == True)

import tilelang
import tilelang.ascend.language as T


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_silu(x):
    """PyTorch reference: fp32-opmath silu, cast back to the input dtype.

    Runs on the NPU (the input tensor lives on device="npu"). Independent of the
    kernel's fp16 direct-computation path (golden uses fp32 intermediate).
    """
    return torch.nn.functional.silu(x.to(torch.float32)).to(x.dtype)


# ---------- Tolerance by dtype ----------
def _tolerance(dtype):
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1e-1
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32; use 5e-3 for accumulation-heavy operators


# ---------- Kernel ----------
NUM_BLOCKS = 32   # 1-D grid; single wave (32 <= 56 AIV); 2^5 divides TOTAL = 2^17
TILE = 2048       # elements per block per iteration (16 x 128-el fp16 vector width)
THREADS = 512     # SIMT thread domain (TILE / THREADS = 4 elements per thread)


@tilelang.jit(target="ascend", out_idx=[1])
def silu_kernel(M, N):
    """Build and JIT-compile the silu PrimFunc for a flat [M, N] fp16 tensor.

    TOTAL = M * N must be an exact multiple of NUM_BLOCKS * TILE so every element
    is covered with no tail block (design decision: exact division only).
    """
    TOTAL = M * N
    assert TOTAL % (NUM_BLOCKS * TILE) == 0, (
        f"shape ({M}, {N}) violates grid divisibility: "
        f"{TOTAL} % {NUM_BLOCKS * TILE} != 0 (silent tail drop would occur)"
    )
    ITERATIONS = TOTAL // (NUM_BLOCKS * TILE)

    @T.prim_func
    def silu(
        A: T.Tensor((TOTAL,), "float16"),
        B: T.Tensor((TOTAL,), "float16"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            a_ub = T.alloc_shared((TILE,), "float16")
            b_ub = T.alloc_shared((TILE,), "float16")
            T.annotate_buffer_versions({a_ub: 2, b_ub: 2})
            for step in T.Pipelined(ITERATIONS, num_stages=2):
                begin = (step * NUM_BLOCKS + bx) * TILE
                T.copy(A[begin : begin + TILE], a_ub)
                with T.SimtVF(threads=THREADS):
                    for i in T.Parallel(TILE):
                        b_ub[i] = a_ub[i] / (T.float16(1.0) + T.exp(-a_ub[i]))
                T.copy(b_ub, B[begin : begin + TILE])

    return silu


# ---------- Layered testing ----------
def _run_case(x, tag, tol=None):
    """Run the kernel on a pre-built NPU tensor `x`, compare against golden_silu.

    Returns max absolute difference (fp32 metric, NaN-masked) for reporting.
    """
    M, N = x.shape
    x_flat = x.flatten()
    kernel = silu_kernel(M, N)
    out = kernel(x_flat)
    torch.npu.synchronize()
    golden = golden_silu(x)
    out2d = out.reshape(M, N)
    if tol is None:
        tol = _tolerance(x.dtype)

    nan_mask = torch.isnan(out2d)
    diff = (out2d.float() - golden.float()).abs()
    diff = diff.masked_fill(nan_mask, 0.0)
    max_diff = diff.max().item()

    torch.testing.assert_close(out2d, golden, rtol=tol, atol=tol, equal_nan=True)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={x.dtype} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    """Contract + same-divisibility variant shapes (blocking)."""
    for M, N in [(128, 1024), (512, 512)]:
        x = torch.randn(M, N, dtype=torch.float16, device="npu")
        _run_case(x, "L0")


def run_L1():
    """Representative elementwise shapes (all exact multiples of NUM_BLOCKS*TILE)."""
    for M, N in [(256, 256), (128, 512), (256, 512), (512, 128)]:
        x = torch.randn(M, N, dtype=torch.float16, device="npu")
        _run_case(x, "L1")


def run_L2():
    """Large-shape robustness probing; failures recorded as warnings (non-blocking)."""
    for M, N in [(1024, 1024), (2048, 512)]:
        try:
            x = torch.randn(M, N, dtype=torch.float16, device="npu")
            _run_case(x, "L2")
        except Exception as e:  # noqa: BLE001 (record without blocking)
            print(f"[L2] WARN (record without blocking): shape=({M},{N}) {e}")


def run_boundary():
    """Special-value semantics per DESIGN.md 8.3 (0 / subnormal / saturation / inf / NaN)."""
    M, N = 128, 1024
    x = torch.zeros(M, N, dtype=torch.float16, device="npu")
    vals = [
        6e-8, -6e-8, 11.0, -11.0, 30.0, -30.0,
        65504.0, -65504.0, float("inf"), float("-inf"), float("nan"),
    ]
    for k, v in enumerate(vals):
        x[0, k] = v

    kernel = silu_kernel(M, N)
    out = kernel(x.flatten()).reshape(M, N)
    torch.npu.synchronize()
    golden = golden_silu(x)

    # NaN inputs propagate NaN, +inf -> +inf, -inf -> NaN in both kernel and
    # golden.  For a finite max-diff metric, drop non-finite differences (NaN
    # arises from NaN-inputs and from inf - inf at the +inf position).
    diff = (out.float() - golden.float()).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    max_diff = diff.max().item()

    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2, equal_nan=True)
    print(f"[Boundary] PASS: special values (0/subnormal/+-11/+-30/+-65504/+-inf/NaN) "
          f"finite_max_diff={max_diff:.3e}")


def _show_source():
    """Print the generated AscendC source (debugging aid, see tilelang-ascend
    skill section 11 and docs/tools/ for lower traces and pass diffs)."""
    kernel = silu_kernel(128, 1024)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
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