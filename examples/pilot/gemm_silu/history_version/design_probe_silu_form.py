# design_probe_silu_form.py — D-3 设计期探针：SimtVF 元素级 silu 三公式向量段代价对比
#
# 未知常数：标量 tanh / exp / div intrinsic 在 SimtVF（编译器自动向量化）路径上的相对代价
#   （CONST-valu-model 只给 special=8 / div=13 / 常规=51 ops/cycle 的 assumption 档位；
#    tanh 无 simd 域 intrinsic〔tilelang/ascend/language/simd.py 无 vtanh〕，标量路径
#    lowering 代价无实测——本探针直接实测三公式在 UB 驻留数据上的迭代吞吐）。
#
# 方法：128x128 fp32 tile 驻留 UB，SimtVF 内原地迭代应用 silu 公式 REPEAT=64 次（读 UB→算→写 UB，
#   每次迭代含 AutoSchedule RAW 同步，三形态同构，差值隔离公式本身代价），GM 只进出各一次。
#   三公式：
#     form 0（基线）: z * (1/(1+exp(-z)))
#     form 1（直除）: z / (1+exp(-z))
#     form 2（tanh）: 0.5*z*(1+tanh(0.5*z))
#
# 断言：三 kernel 均编译通过；各自与 torch 同公式 64 次迭代参照 max_diff < 1e-4（同式对照，
#   非跨式等价——跨式等价见 verify_equiv.py）。
# 测量：do_bench(msprof) 延迟与相对比值（量级判断，不锁绝对值）。
#
# 首次验证：tilelang-fork fc32f47c / CANN 9.x / Ascend950PR_9579, 2026-10-16
# origin_task: pilot/gemm_silu Stage 1 first_design（裁决 §1.6.1 主选公式 B vs 备选 C）
import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

M = N = 128
REPEAT = 64
THREADS = 128


def make_kernel(form: int):
    @T.prim_func
    def silu_repeat(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((M, N), "float32")
            T.copy(X, x_ub)
            for r in range(REPEAT):
                with T.SimtVF(threads=THREADS):
                    for i, j in T.Parallel(M, N):
                        z = x_ub[i, j]
                        if form == 0:
                            x_ub[i, j] = z * (1.0 / (1.0 + T.exp(-z)))
                        if form == 1:
                            x_ub[i, j] = z / (1.0 + T.exp(-z))
                        if form == 2:
                            t = 0.5 * z
                            x_ub[i, j] = t * (1.0 + T.tanh(t))
            T.copy(x_ub, Y)

    return silu_repeat


def torch_ref(x, form):
    for _ in range(REPEAT):
        if form == 0:
            x = x * (1.0 / (1.0 + torch.exp(-x)))
        elif form == 1:
            x = x / (1.0 + torch.exp(-x))
        else:
            t = 0.5 * x
            x = t * (1.0 + torch.tanh(t))
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(M, N, device="npu", dtype=torch.float32)
    results = {}
    for form, name in ((0, "base: z*(1/(1+e^-z))"), (1, "div : z/(1+e^-z))"), (2, "tanh: 0.5z(1+tanh(z/2))")):
        prog = make_kernel(form)
        kernel = tilelang.compile(prog, target="ascend", out_idx=[1])
        y = kernel(x)
        ref = torch_ref(x.clone(), form)
        max_diff = (y - ref).abs().max().item()
        assert max_diff < 1e-4, f"form {form} max_diff={max_diff:.2e} >= 1e-4"
        print(f"form {form} ({name}): correctness passed (max_diff={max_diff:.2e})")
        if args.bench:
            latency = do_bench(lambda: kernel(x), backend="msprof", _n_warmup=10, _n_repeat=30)
            results[form] = latency
            print(f"form {form} ({name}): {latency * 1000:.2f} us  ({REPEAT} iters over {M}x{N} fp32)")

    if args.bench and len(results) == 3:
        base = results[0]
        print("\n=== 相对比值（form0=1.00） ===")
        for form in (0, 1, 2):
            print(f"  form {form}: {results[form] / base:.3f}")
        print(
            "\n判读：form2/form1 < 1 显著（如 <=0.9）→ tanh 消除除法有实测收益（§1.6.1 备选 C 上调）；"
            "\n      form2/form1 >= 1 → tanh 标量路径代价高于省下的除法（备选 C 维持不采纳，主选 B）。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
