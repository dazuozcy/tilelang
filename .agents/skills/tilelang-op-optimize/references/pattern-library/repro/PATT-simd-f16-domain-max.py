"""PATT-simd-f16-domain-max -- SimdVF pass-1 在 16-bit 输入域做 max.

条目: elementwise.md D-LOGSUMEXP-SIMD-F16MAX
快形态: 两遍 row-reduction 的 pass1 对 fp16/bf16 用 128-lane 16-bit 向量 max
      （vld NORM + vmax，2 ops/128 elems），仅 winner 单次 vcvt 升 fp32；max 无
      舍入 -> 与 fp32 detour 位级一致。bf16 例外: asc_reduce_max 无 bf16 重载
      （CANN 9.2.0 reg_reduce.h），须 EVEN/ODD 拆分 fp32 归约。
形态: before/after 完整对照（可运行，断言位级一致；--bench 附 msprof 时延）。
首次验证: tilelang dev（fc32f47c 后）/ CANN 9.2.0 + dav-3510 + Ascend950PR_9579,
      2026-10-17（LogSumExpFwdOpSimd Stage 4，5 workload +4.0~+15.5%，max_abs=0）
origin_task: logsumexp-simd-harness
重验记录: （追加式）
"""

import sys

import torch
import tilelang
import tilelang.ascend.language as T

GROUP = 64


def build(M: int, N: int, dtype: str, hmax: bool):

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        NH = N // 128

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), "float32")):
            with T.Kernel(56) as core_id:
                x_ub = T.alloc_shared((block_m, N), dtype)
                y_ub = T.alloc_shared((block_m,), "float32")
                for t in T.Persistent([(M + block_m - 1) // block_m], 56, core_id, num_stages=2):
                    r0 = t * block_m
                    rows = block_m if r0 + block_m <= M else M - r0
                    T.copy(x[r0 : r0 + rows, 0:N], x_ub[0:rows, 0:N])
                    with T.SimdVF():
                        neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                        one_pt = T.simd.pset(32, "PAT_VL1")
                        for r in range(rows):
                            if hmax:
                                acc_h = T.simd.alloc_var(dtype)
                                acc_h = T.simd.vld(x_ub[r, 0])
                                for g in range(1, NH):
                                    acc_h = T.simd.vmax(acc_h, T.simd.vld(x_ub[r, g * 128]))
                                if dtype == "float16":
                                    m_brc = T.simd.vcvt(T.simd.vdupv(T.simd.vcmax(acc_h)), "float32", part=0)
                                else:  # bf16: no asc_reduce_max overload -> EVEN/ODD split
                                    m_brc = T.simd.vdupv(T.simd.vcmax(T.simd.vmax(
                                        T.simd.vcvt(acc_h, "float32", part=0),
                                        T.simd.vcvt(acc_h, "float32", part=1))))
                            else:
                                acc = T.simd.alloc_var("float32")
                                acc = neg_inf
                                for g in range(N // GROUP):
                                    acc = T.simd.vmax(acc, T.simd.vcvt(
                                        T.simd.vld(x_ub[r, g * GROUP], dist="UNPK_B16"),
                                        "float32", part=0))
                                m_brc = T.simd.vdupv(T.simd.vcmax(acc))
                            T.simd.vsts(y_ub[r], m_brc, one_pt, "ONEPT_B32")
                    T.copy(y_ub[0:rows], y[r0 : r0 + rows])

        return main

    return _func


def main() -> int:
    M, N = 64, 4096
    for dtype in ("float16", "bfloat16"):
        x = torch.randn(M, N, dtype=getattr(torch, dtype), device="npu")
        ref = x.float().max(dim=-1).values
        outs = {}
        for hmax in (False, True):
            k = build(M, N, dtype, hmax)(8)
            y = k(x)
            torch.npu.synchronize()
            outs[hmax] = y
            assert torch.equal(y, ref), f"{dtype} hmax={hmax}: wrong max"
        assert torch.equal(outs[False], outs[True]), f"{dtype}: not bitwise identical"
        print(f"[{dtype}] baseline-f32detour == 16-bit-domain max (bitwise) OK")

    if "--bench" in sys.argv:
        from tilelang.profiler import do_bench

        for dtype in ("float16", "bfloat16"):
            x = torch.randn(M, N, dtype=getattr(torch, dtype), device="npu")
            for hmax in (False, True):
                k = build(M, N, dtype, hmax)(8)
                ms = do_bench(lambda: k(x), backend="msprof", early_stop_baseline=None)
                print(f"[{dtype}] hmax={hmax}: {ms * 1000:.2f} us")  # full-kernel gain: +4~+15.5%
    print("PATT-simd-f16-domain-max PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
