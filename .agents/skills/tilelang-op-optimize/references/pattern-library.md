# Ascend A5 性能模式库（已拆分——见 pattern-library/ 目录）

> **本文件已按渐进披露原则拆分为主题文件**（K-2）——本存根仅保留入口指引与兼容重定向，**不再更新内容**；所有条目位于：
>
> ```text
> .agents/skills/tilelang-op-optimize/references/pattern-library/
> ├── INDEX.md            # 唯一入口：维护规则 + 证伪协议 + 检索规则 + 条目索引（必读）
> ├── constants.md        # 硬件常数表（A5 assumption 级条目，设计期 roofline 口径，D-2）
> ├── layout.md           # 布局与搬运模式
> ├── elementwise.md      # 多输入搬运效率
> ├── attention.md        # attention / 混合核族
> ├── traps-compiler.md   # 编译器/解析器陷阱
> ├── traps-runtime.md    # 运行时/数值/语义陷阱
> ├── cases.md            # 案例索引与参考实现集
> └── repro/              # 最小可复现代码（ED-B，知识回归测试）
> ```
>
> **A5 冷启动状态（2026-10-16）**：本库自旧栈（A2A3，MLIR 路线，旧设备实测）整套复制后内容清零——原条目绑定旧工具链与旧设备实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。仅 `constants.md` 以 assumption 级 A5 条目起步（源码/文档证据标注），其余主题文件为空骨架，由 A5 调优任务回写积累（维护规则与预算见 INDEX.md §1）。调优手段种子知识源：`tilelang-ascend` skill §12（profiling recipes）与 `maint/scripts/ascend_perf_regression.py`。
>
> **检索**：`python3 .agents/tools/kb_search.py "<算子族/症状/API/dtype>"`（覆盖本库全部主题 + bottleneck-patterns + algorithm-candidates + capability-gaps + queue pending）；**写入**（optimizer 回写 / evolver 合入）一律作用于主题文件，五种 delta 与预算规则不变（INDEX.md §1）。
