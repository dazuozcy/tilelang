"""TRAP-fp16-fragment-exp-slow -- fp16 fragment 超越函数链的软件化 lowering（性能陷阱）.

条目: elementwise.md D-LOGSUMEXP-MIGRATION-FIDELITY（fp16-e 备选 Stage 4 实测否决子项）
现象: SimtVF 内对 fp16 dtype fragment 直接做 T.exp/reduce 链（省一次 cast 的"优化"），
      生成 AscendC 与 fp32 版结构完全相同，唯一差异是逐元素标量 hexp（fp32 为 expf）
      + half fragment 上的 AscendAllReduce——实测 50-60x 慢（4k 52.97→2579.8us；
      3d 19.1→350.2us；lm-head 20.4→594.2us）。归因：bisheng 对半精度标量超越
      函数/半精度归约的软件化 lowering，非 kernel 结构。
绕法: 中间计算链保持 fp32 fragment（cast 一次输入、cast 一次输出），只有输入输出
      buffer 用窄 dtype——数值位级一致且不付出 50x 代价。
断言: 默认（可运行部分）两种 kernel vs torch.logsumexp 精度均通过（fp16-e 数值
      本身合规，max_rel <= 1e-2，慢的是时延）；--bench 时 do_bench(msprof) 断言
      trap(fp16-fragment) 中位数 >= 10 x good(fp32-fragment) 中位数。
首次验证: tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579, 2026-10-17
origin_task: TileOPs LogSumExpFwdOp Stage 4（opt_log.md Round 1 E3/E4 + profiles/source/*__r1_fp16e.c；溯源允许失效）
重验记录: 2026-10-17 双跑 --bench 实测 good(fp32-frag,t256)=21.38us / trap(fp16-frag,t256)=439.7us
          -> 20.6x（E3 原始结构 t1024/SUB=8 下 50x；首跑 good 误读 181us 系崩溃进程残留设备态）
"""

import sys

import torch
import tilelang
import tilelang.ascend.language as T

M, N, ROWS, SUB, TILE, DTYPE = 1024, 4096, 4, 4, 4096, "float16"


def build(frag_dtype: str):
    """frag_dtype='float32' -> 绕法(好)；'float16' -> 陷阱形态（exp 链窄 dtype）。"""
    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        @T.prim_func
        def main(x: T.Tensor((M, N), DTYPE), y: T.Tensor((M,), DTYPE)):
            with T.Kernel(56) as core_id:
                x_ub = T.alloc_shared((SUB, N), DTYPE)
                lse_ub = T.alloc_shared((SUB,), DTYPE)
                T.annotate_buffer_versions({x_ub: 2})
                for t in T.Persistent([M // ROWS], 56, core_id, num_stages=2):
                    base = t * ROWS
                    for s in T.serial(ROWS // SUB):
                        rb = base + s * SUB
                        T.copy(x[rb : rb + SUB, 0:N], x_ub)
                        # ---- 陷阱 vs 绕法：fragment dtype ----
                        # 陷阱(慢): xf/mf/sf 用 "float16"，exp 直接吃窄 dtype
                        # 绕法(好): 中间链 "float32"，仅输入 cast 进/输出 cast 出
                        with T.SimtVF(threads=256):
                            xf = T.alloc_fragment((SUB, TILE), frag_dtype)
                            mf = T.alloc_fragment((SUB,), frag_dtype)
                            sf = T.alloc_fragment((SUB,), frag_dtype)
                            of = T.alloc_fragment((SUB,), DTYPE)
                            for r, i in T.Parallel(SUB, TILE):
                                xf[r, i] = T.cast(x_ub[r, i], frag_dtype)
                            T.reduce_max(xf, mf, dim=1, clear=True)
                            for r, i in T.Parallel(SUB, TILE):
                                xf[r, i] = T.exp(xf[r, i] - mf[r])
                            T.reduce_sum(xf, sf, dim=1, clear=True)
                            for r in T.Parallel(SUB):
                                of[r] = T.cast(mf[r] + T.log(sf[r]), DTYPE)
                            T.copy(of, lse_ub)
                        T.copy(lse_ub[0:SUB], y[rb : rb + SUB])

        return main

    return _func(ROWS)


def main() -> int:
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ref = torch.logsumexp(x.float(), dim=-1).to(torch.float16)
    kernels = {}
    for tag, fd in (("good_fp32frag", "float32"), ("trap_fp16frag", "float16")):
        k = build(fd)
        y = k(x)
        torch.npu.synchronize()
        rel = ((y.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-6)).max().item()
        assert rel <= 1e-2, f"[{tag}] numerically invalid: max_rel={rel:.3e}"
        kernels[tag] = k
        print(f"[{tag}] correctness PASS (max_rel={rel:.3e} — fp16-e 数值合规, 慢的是时延)")
    if "--bench" in sys.argv:
        from tilelang.profiler import do_bench

        med = {}
        for tag, k in kernels.items():
            fn = (lambda kk: (lambda: kk(x)))(k)  # closure: do_bench calls fn()
            ms = sorted(float(do_bench(fn, backend="msprof", warmup=25.0, rep=100.0,
                                       early_stop_baseline=None)) * 1000.0 for _ in range(3))
            med[tag] = ms[1]
            print(f"[{tag}] do_bench(msprof) median = {med[tag]:.3f} us")
        assert med["trap_fp16frag"] >= 10.0 * med["good_fp32frag"], f"expected >=10x: {med}"
        print(f"[bench] slowdown = {med['trap_fp16frag']/med['good_fp32frag']:.1f}x")
    else:
        print("[hint] run with --bench on NPU for the do_bench(msprof) slowdown assertion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
