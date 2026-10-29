"""Design probe: verify T.exp + T.Parallel lowering under SimtVF on Ascend.

Silu: y = x / (1 + exp(-x)). Checks that `T.exp` inside a SimtVF T.Parallel
loop compiles on the ascend backend and matches the PyTorch reference.
Grid arithmetic: TOTAL = 131072 = 2^17 has no factor 7, so 28/56-block grids
cannot divide it evenly; use 32 blocks x 2048 x 2 iterations (exact division).
Run: PYTHONPATH=<repo> python3 examples/pilot/silu/history_version/design_probe_simt_exp.py
"""
import torch
import tilelang
import tilelang.ascend.language as T

M, N = 128, 1024
TOTAL = M * N  # 131072
NUM_BLOCKS = 32
TILE = 2048
THREADS = 512
assert TOTAL % (NUM_BLOCKS * TILE) == 0


@T.prim_func
def silu_kernel(A: T.Tensor((TOTAL,), "float16"), B: T.Tensor((TOTAL,), "float16")):
    with T.Kernel(NUM_BLOCKS) as bx:
        a_ub = T.alloc_shared((TILE,), "float16")
        b_ub = T.alloc_shared((TILE,), "float16")
        T.annotate_buffer_versions({a_ub: 2, b_ub: 2})
        for step in T.Pipelined(TOTAL // (NUM_BLOCKS * TILE), num_stages=2):
            begin = (step * NUM_BLOCKS + bx) * TILE
            T.copy(A[begin : begin + TILE], a_ub)
            with T.SimtVF(threads=THREADS):
                for i in T.Parallel(TILE):
                    b_ub[i] = a_ub[i] / (T.float16(1.0) + T.exp(-a_ub[i]))
            T.copy(b_ub, B[begin : begin + TILE])


def main():
    a = torch.randn(M, N, dtype=torch.float16, device="npu").flatten()
    kernel = tilelang.compile(silu_kernel, out_idx=-1)
    out = kernel(a)
    # Golden candidates: fp32-upcast silu (cast back) and direct fp16 silu.
    ref32 = torch.nn.functional.silu(a.float()).to(torch.float16)
    ref16 = torch.nn.functional.silu(a)
    torch.testing.assert_close(out, ref32, rtol=1e-2, atol=1e-2)
    print("[PROBE] vs fp32-upcast golden (rtol=atol=1e-2): PASS")
    torch.testing.assert_close(out, ref16, rtol=1e-2, atol=1e-2)
    print("[PROBE] vs direct-fp16 golden (rtol=atol=1e-2): PASS")
    d32 = (out.float() - ref32.float()).abs().max().item()
    d16 = (out.float() - ref16.float()).abs().max().item()
    print(f"[PROBE] max|out-ref32| = {d32:.3e}, max|out-ref16| = {d16:.3e}")
    print("[PROBE] silu via T.exp + SimtVF T.Parallel (32 blocks x 2048 x 2 iters): PASS")


if __name__ == "__main__":
    main()
