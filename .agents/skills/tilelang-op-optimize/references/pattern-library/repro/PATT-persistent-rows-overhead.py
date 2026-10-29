"""PATT-persistent-rows-overhead -- persistent 行块粒度：makespan 模型缺 per-tile 开销项（delta 骨架）.

条目引用: layout.md P-LAYOUT-ROW-RESIDENCY + elementwise.md D-SIMD-PERSISTENT-ROWS-LSE-BATCH（流式变体 lse 微写批量化同型 host 侧重推导）.

条目: layout.md P-LAYOUT-ROW-RESIDENCY（驻留变体 tile 粒度选择子项）
现象: T.Persistent 的 56 核调度按 ceil(num_tiles/56) 波次分派 tile；纯 makespan 模型
      min ceil(ceil(M/ROWS)/56)*ROWS 选出的 ROWS=1（M=1024: 1024 tiles、makespan 19 行）
      实测 43.6us，远慢于 ROWS=4（256 tiles、makespan 20 行）的 25.6us——每 tile 固定
      开销（lse 微型 MTE3 写出握手、vf_call 发射、MTE2 issue）随 tile 数线性放大，
      makespan-only 模型未建模该项。
绕法/快形态: 候选限 pow2 ROWS（保持 SUB=ROWS 单子批干净形态），取
      makespan <= min_makespan+1 行内的最大 ROWS（M=1024->4 实测最优 +50.9%；
      M=128->4 = baseline 等价 tie；M=4->2）。
形态: delta 骨架（效应须完整 persistent kernel 规模显现，不要求可运行；
      py_compile 通过）。慢/快差异只在 host 侧 _pick_rows 规则，kernel 体不变。
首次验证: tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579, 2026-10-17
origin_task: TileOPs LogSumExpFwdOp Stage 4（opt_log.md Round 1 E1 sweep r∈{1,2,4}；
             生成源码 8-tile 组×7 slot 结构见 profiles/source/；溯源允许失效）
重验记录: （追加式）
"""

NUM_BLOCKS = 56


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------- before（慢）：纯 makespan 最小化 ----------------
def _pick_rows_slow(M: int) -> int:
    """E1 原始启发式：ROWS ∈ [1,16]，最小化 (makespan_rows, tiles)——
    tie-break 更少 tiles。M=1024: r=1 的 makespan 19 行唯一最优 -> 选 1。"""
    best_rows, best_key = 1, None
    for rows in range(1, 17):
        tiles = _ceil_div(M, rows)
        key = (_ceil_div(tiles, NUM_BLOCKS) * rows, tiles)
        if best_key is None or key < best_key:
            best_rows, best_key = rows, key
    return best_rows  # M=1024 -> 1（实测 43.6us，劣于 ROWS=4 的 25.6us）


# ---------------- after（快）：makespan+1 行内取最大 pow2 ROWS ----------------
def _pick_rows_fast(M: int) -> int:
    """pow2 候选 {1,2,4,8,16} 中取 makespan <= min+1 行的最大者（per-tile 开销项的
    启发式补偿：tile 数减半的固定开销节省优先于 1 行内的 makespan 损失）。"""
    cands = [r for r in (1, 2, 4, 8, 16) if r <= max(1, M)]
    makespans = {r: _ceil_div(_ceil_div(M, r), NUM_BLOCKS) * r for r in cands}
    min_ms = min(makespans.values())
    return max(r for r in cands if makespans[r] <= min_ms + 1)  # M=1024 -> 4 (+50.9%)


# ---------------- kernel 侧接入点（不变量的说明，非可运行体） ----------------
# ROWS = _pick_rows_slow(M)   # before：persistent tile 行数 = 纯 makespan 最优
# ROWS = _pick_rows_fast(M)   # after ：同上，但经 per-tile 开销补偿规则
# 其余（SUB 行子批、num_tiles、T.Persistent([num_tiles], 56, ...)）全部由 ROWS
# 派生，kernel 体逐字不变——本优化点是纯 host 静态化参数重推导。

if __name__ == "__main__":
    for M in (1024, 128, 4):
        s, f = _pick_rows_slow(M), _pick_rows_fast(M)
        print(f"M={M}: slow(pure-makespan) ROWS={s}  fast(revised) ROWS={f}")
    assert _pick_rows_fast(1024) == 4 and _pick_rows_slow(1024) == 1
    print("PATT-persistent-rows-overhead skeleton OK")
