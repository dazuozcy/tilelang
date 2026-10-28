"""PATT-simtvf-reduce-threads-width -- SimtVF row-reduction 线程宽度 U 曲线（慢→快优化点）.

条目: elementwise.md D-REDUCE-SIMTVF-API（threads 宽度 sweep 子项）
现象: SimtVF 内 T.reduce_max/reduce_sum 降级为 AscendAllReduce<Op, threads, 1, 0>
      ——树宽 = T.SimtVF(threads=...) 参数。1024（默认满宽）并非最优：收窄到 256
      每行归约树深/同步规模减 2 档，换每线程串行元素数 ×4；实测 U 曲线
      1024→512→256 单调改善，128 串行反超回退。
断言: 默认（可运行部分）双 kernel vs torch.logsumexp 精度通过；
      --bench 时 do_bench(msprof) 断言 fast(256) 中位数 <= slow(1024) 中位数
      （相对关系断言，绝对收益仅打印：logsumexp 单 tile 变体实测 +16%）。
首次验证: tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579, 2026-10-17
origin_task: TileOPs LogSumExpFwdOp Stage 4（opt_log.md Round 2 E5/E8/E10；溯源允许失效）
重验记录: 2026-10-17 --bench 实测 slow(t1024)=25.4us / fast(t256)=21.4us -> +15.8%
          （与 opt_log E8 的 +16.1% 一致；E10 t128 回退佐证 U 曲线右支）
"""

import sys

import torch
import tilelang
import tilelang.ascend.language as T

M, N, ROWS, SUB, TILE, DTYPE = 1024, 4096, 4, 4, 4096, "float16"


def build(threads: int):
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
                        # ---- 慢→快关键更改：threads 宽度 ----
                        # before(慢): with T.SimtVF(threads=1024):
                        # after (快): with T.SimtVF(threads=256):
                        with T.SimtVF(threads=threads):
                            xf = T.alloc_fragment((SUB, TILE), "float32")
                            mf = T.alloc_fragment((SUB,), "float32")
                            sf = T.alloc_fragment((SUB,), "float32")
                            of = T.alloc_fragment((SUB,), DTYPE)
                            for r, i in T.Parallel(SUB, TILE):
                                xf[r, i] = T.cast(x_ub[r, i], "float32")
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
    for tag, threads in (("slow_t1024", 1024), ("fast_t256", 256)):
        k = build(threads)
        y = k(x)
        torch.npu.synchronize()
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
        kernels[tag] = k
        print(f"[{tag}] correctness PASS (max_diff={(y.float()-ref.float()).abs().max():.3e})")
    if "--bench" in sys.argv:
        from tilelang.profiler import do_bench

        med = {}
        for tag, k in kernels.items():
            fn = (lambda kk: (lambda: kk(x)))(k)  # closure: do_bench calls fn()
            ms = [
                float(do_bench(fn, backend="msprof", warmup=25.0, rep=100.0,
                               early_stop_baseline=None)) * 1000.0
                for _ in range(3)
            ]
            med[tag] = sorted(ms)[1]
            print(f"[{tag}] do_bench(msprof) median = {med[tag]:.3f} us")
        assert med["fast_t256"] <= med["slow_t1024"], (
            f"threads=256 expected <= threads=1024: {med}"
        )
        print(f"[bench] gain = {(med['slow_t1024']-med['fast_t256'])/med['slow_t1024']*100:+.1f}%")
    else:
        print("[hint] run with --bench on NPU for the do_bench(msprof) A/B assertion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
