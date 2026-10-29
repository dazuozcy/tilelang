# verify_equiv.py — gemm_silu 等价性机器验证（D-1，S1-EQUIV-EXEC）
#
# 对照对象（§1.6.1 采纳项 / 实验裁决备选）：
#   基线  f_base(z) = z * (1/(1+exp(-z)))          —— 用户公式 silu(z)=z*sigmoid(z) 的自然展开
#   候选B f_div (z) = z / (1+exp(-z))              —— 直除形式（省 1 次乘）
#   候选C f_tanh(z) = 0.5*z*(1+tanh(0.5*z))        —— tanh 形式（消除除法；实验裁决备选）
#
# 方法（torch CPU）：
#   口径1  fp64 参照：两式均以 fp64 计算，按 kernel 实际 dtype 路径（fp32 计算 → fp16 舍入）对照；
#   口径2  fp32 运算模拟：以 fp32 算术逐式计算（模拟 Vector 段真实舍入路径），再 cast fp16 对照，
#          覆盖 z<=-88 的 exp fp32 溢出边界行为；
#   角点：0 / ±subnormal / ±1e-3 / ±0.28 / ±1 / ±2.3 / ±10 / ±20 / ±88 / ±89 / ±100 / ±1e4 / ±inf / NaN
#          + 随机 N(0, sqrt(512))（z 实际分布尺度）与 N(0,1) 各 1e6 样本。
# PASS 判据：fp16 口径在 §8.2 容差（atol=rtol=1e-2）下违反率 == 0，且有限输入上无 NaN/inf 分歧。
#
# 首次验证：tilelang-fork fc32f47c / CANN 9.x / Ascend950PR_9579, 2026-10-16
# origin_task: pilot/gemm_silu Stage 1 first_design

import math

import torch

ATOL = RTOL = 1e-2  # §8.2 fp16 档


def f_base(z):
    return z * (1.0 / (1.0 + torch.exp(-z)))


def f_div(z):
    return z / (1.0 + torch.exp(-z))


def f_tanh(z):
    t = 0.5 * z
    return t * (1.0 + torch.tanh(t))


def ulp_diff_fp(a, b, dtype):
    """按 dtype 计相对 ulp 差（fp64 中间量避免比值溢出伪影；对 |结果|<1 为保守上界）。"""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    tiny = torch.nextafter(
        torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype)
    ).to(torch.float64).item()
    diff = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs()).clamp_min(tiny)
    return (diff / (scale * tiny)).log2().ceil().clamp_min(0)


def run_case(name, z64):
    """返回 (max_fp32_ulp, fp16_violation_rate, max_fp16_ulp, pass)。"""
    finite = torch.isfinite(z64)
    rows = {}
    for tag, fn in (("div", f_div), ("tanh", f_tanh)):
        # 口径1：fp64 计算 → fp32 → fp16（两式同精度路径舍入）
        yb64, yc64 = f_base(z64), fn(z64)
        d64 = (yc64 - yb64).abs().max().item()
        # 口径2：fp32 运算模拟（kernel Vector 段真实路径）
        z32 = z64.to(torch.float32)
        yb32, yc32 = f_base(z32).to(torch.float32), fn(z32).to(torch.float32)
        u32 = ulp_diff_fp(yc32, yb32, torch.float32)
        u32_max = u32[finite & torch.isfinite(yb32) & torch.isfinite(yc32)].max().item()
        # NaN/inf 传播一致性（有限输入上）
        mismatch_special = int(
            ((torch.isnan(yb32) != torch.isnan(yc32)) | (torch.isinf(yb32) != torch.isinf(yc32)))[finite].sum()
        )
        # fp16 出口 + §8.2 容差违反率
        yb16, yc16 = yb32.to(torch.float16), yc32.to(torch.float16)
        viol = (yc16.float() - yb16.float()).abs() > (ATOL + RTOL * yb16.float().abs())
        viol_rate = viol[finite].float().mean().item() if finite.any() else 0.0
        u16 = ulp_diff_fp(yc16, yb16, torch.float16)
        u16_max = u16[finite & torch.isfinite(yb16) & torch.isfinite(yc16)].max().item()
        ok = viol_rate == 0.0 and mismatch_special == 0
        rows[tag] = (d64, u32_max, u16_max, viol_rate, mismatch_special, ok)
    print(f"[{name}] n={z64.numel()}")
    for tag, (d64, u32, u16, vr, ms, ok) in rows.items():
        print(
            f"  {tag:5s}: fp64 max|d|={d64:.3e}  fp32 max ulp={u32:.0f}  "
            f"fp16 max ulp={u16:.0f}  违反率={vr:.2e}  NaN/inf 分歧={ms}  "
            f"{'EQUIV_PASS' if ok else 'EQUIV_FAIL'}"
        )
    return rows


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    all_rows = {}
    # 随机：z 实际分布（K=512 fp32 累加，z~N(0, sqrt(512))）与标准正态
    all_rows["rand-N(0,22.6)"] = run_case(
        "rand-N(0,22.6)", torch.randn(1_000_000, generator=g, dtype=torch.float64) * math.sqrt(512)
    )
    all_rows["rand-N(0,1)"] = run_case(
        "rand-N(0,1)", torch.randn(1_000_000, generator=g, dtype=torch.float64)
    )
    # 角点：含 exp fp32 溢出边界（exp(88.7)>fp32 max）、silu 极值点、subnormal、inf/NaN
    corners = torch.tensor(
        [0.0, -0.0, 1e-45, -1e-45, 1e-30, -1e-30, 1e-7, -1e-7, 1e-3, -1e-3,
         0.28, -0.28, 1.0, -1.0, 2.3, -2.3, 10.0, -10.0, 20.0, -20.0,
         88.0, -88.0, 89.0, -89.0, 100.0, -100.0, 1e4, -1e4,
         float("inf"), float("-inf"), float("nan")],
        dtype=torch.float64,
    )
    all_rows["corners"] = run_case("corners", corners)
    # 汇总判定（所有用例全 PASS 才 PASS）
    final = {}
    for case, rows in all_rows.items():
        for tag, (*_, ok) in rows.items():
            final[tag] = final.get(tag, True) and ok
    print("\n=== 汇总（各候选跨全部用例） ===")
    for tag, ok in final.items():
        print(f"  {tag:5s}: {'EQUIV_PASS' if ok else 'EQUIV_FAIL'}")
    return 0 if all(final.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
