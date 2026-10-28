# TileOPs（Ascend A5 侧集成工程）

TileLang Ascend A5（dav-3510，`target="ascend"`）算子基准框架——自 tilelang-mlir-fork 的
`examples/TileOPs/` 移植的**框架层**，作为 tilelang-op-conductor 自动算子生成系统的
Stage 5 集成落位（适配方案 §2.7，Phase 3）。

## 目录结构与各层职责

```
examples/TileOPs/
├── tileops/                    # tileops 包
│   ├── device.py               # 设备后端抽象（torch_npu / CUDA / CPU），A5 UB 容量常量唯一事实源
│   ├── kernels/                # Kernel 层：TileLang kernel 抽象基类 + 5 个算子族目录
│   │   ├── kernel_base.py      #   Kernel ABC（supported_archs / init_config / config 选择）
│   │   └── reduction/_primitives.py  # 族内共享助手（align_up / compute_tile_n / T.macro 工厂）
│   ├── ops/                    # Op 层：硬件感知 kernel 分发、roofline、多维规整
│   │   ├── op_base.py          #   Op ABC（arch 兼容检查走设备后端名匹配）
│   │   ├── _multidim.py        #   多维规约 shape 工具（设备无关）
│   │   └── reduction/reduce.py #   共享 ReduceOp 基类（validate → transpose → reshape → pad → dispatch）
│   ├── workloads/              # Workload 层：输入生成（shape + dtype）
│   ├── manifest/               # manifest 层：YAML 算子注册表（规格唯一事实源）
│   ├── testing/                # Testing 层：TestBase 正确性基类（vs PyTorch 参考实现）
│   └── utils/                  # 通用工具（str2dtype 等）
├── tests/                      # 正确性测试（conftest 已就位，算子测试随算子生成）
├── benchmarks/                 # 性能基准（链路由基准移植任务维护）
├── reports/                    # Stage 5 报告落位目录
└── pyproject.toml
```

端到端数据流：`manifest`（YAML 规格）→ `workloads`（输入生成）→ `ops`（校验/整形/分发）
→ `kernels`（TileLang kernel）→ `testing`（正确性）/ `benchmark`（时延）→ `reporting`（报告）。

## 空注册表说明

**910B kernel 实现一律不移植**。所有算子注册表（`tileops/manifest/*.yaml`）、算子族
目录（`kernels/`、`ops/`、`workloads/` 下各族）以空脚手架形式落位，schema 与示例条目
见各 yaml 文件头注释。

A5 kernel 由 conductor 端到端生成：

1. **Stage 0**：add-ascend-op skill 流程生成算子脚手架（manifest 条目 + 目录 + 测试骨架）；
2. **Stage 1-3**：生成 DESIGN.md → kernel + golden 实现 → 精度验证；
3. **Stage 5**：集成回本工程（kernel/op/workload/test/bench 各就各位）。

## 技术口径

- 唯一规范入口：`@tilelang.jit(target="ascend")`，npu-arch `dav-3510`
  （Ascend950：cube_core_num=28 / vector_core_num=56）。
- 编程模式：simt / simd / mixed（kernel 内 `T.SimtVF` / `T.SimdVF` / `T.MixedKernel`
  声明），与 auto_schedule 正交；`Kernel` 基类不持有模式字段。
- 内存层级：`T.alloc_shared` = UB（A5 物理 248KB、SimtVF 216KB，常量唯一事实源在
  `tileops/device.py`）、`T.alloc_l1` = CBuf（512KB）、L0A/L0B/L0C 为 Cube 缓冲。
- 性能口径：`tilelang.profiler.do_bench(fn, backend="msprof")` 为唯一端到端时延权威。
- 报告 `Ratio(%)`：msprof 工具 roofline 性能占比（`msprof op --aic-metrics=Roofline`
  子进程归因通道，`GM Read + Write` 点，SoC 峰值模型在 CANN 工具内部；op 按 do_bench
  时延匹配选择）。不可得时（如闭包 functor）回退瓶颈管线占用率
  （`do_bench(backend="msprof_detail")`，最忙 pipe busy ratio）。开关：
  `TILEOPS_MSPROF_ROOFLINE=0` / `TILEOPS_MSPROF_DETAIL=0`。
- 精度验证：golden 用 NPU 上的 torch 计算，`torch.testing.assert_close`，容差按 dtype
  （fp8→1e-1，bf16/fp16→1e-2，fp32→1e-3）。

## 使用

```bash
cd examples/TileOPs
export PYTHONPATH=/home/z00875089/tilelang-fork:/home/z00875089/tilelang-fork/examples/TileOPs

# 列出已注册算子（当前为空注册表）
python -m tileops.reporting.cli list

# 运行算子的正确性测试与基准（<op_name> 需已注册）
python -m tileops.reporting.cli run <op_name>

# 运行 pytest 正确性套件
python -m pytest tests/ -x
```

## 未移植内容

- 910B 全部算子实现（kernels/ops/workloads 下具体算子、tests/ops/test_*.py、
  benchmarks/ops/bench_*.py、reports/ 历史报告）。
- 源仓 `docs/roofline_*.md` 两篇 910B 历史 roofline 分析文档。
- 源仓的编程模式环境变量机制（模式由进程级环境变量切换的一套旧机制）整体删除，
  编程模式改为 kernel 内声明（simt / simd / mixed），详见上文技术口径。

