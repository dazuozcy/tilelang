# TileOPs 迁移来源与阶段衔接契约

本标准适用于 migration-harness；plain 的单文件来源契约保持不变。
当前 GPU 源仓为 `/home/zuo/code/TileOPs`，仅支持 `src/tileops` 布局。Ascend 目标是本仓 `examples/TileOPs/`，
同样采用 `src/tileops/{_ops,_kernels,manifest/spec}`、根 `workloads/` 与 tests/benchmarks。
不提供旧目录或旧元数据兼容分支。转换步骤权威见
`examples/TileOPs/.agents/skills/add-ascend-op/references/tileops-src-layout.md`。

## 1. 发现与元数据

用 `extract_tl_kernel.py --inspect` 做 AST/YAML 发现，不导入源 package。
源 manifest 无 `source` 时从 `_ops` / `_kernels`、继承的 `kernel_types`
和实际 imports/calls 定位实现。非 TileOPs 才回落 plain；已识别 TileOPs
的发现错误是规格/环境问题，不得静默切换场景。

新源仓 `.migration_meta.json` 必须有 `source_layout: "src"`、绝对路径
`gpu_repo_root` 和 `gpu_source` 对象。已有产物结构迁移后，缺少新来源字段的 meta 必须重新脚手架化，不能直接续跑。
`gpu_source` 的路径全部相对 GPU 根，字段如下：

- `manifest_path`、`op_path`：源规格及 public Op 实现文件。
- `kernel_bindings`：非空列表，每项 `{path, class}`，覆盖迁移范围内实现。
- `function_sources`：`{函数名: 源文件路径}`，键与 `extracted_functions`
  完全相同；必须指向该函数原始定义文件，不得用合成提取文件替代。
- `workload_paths`、`test_paths`、`bench_paths`：非空路径列表，含实际依赖
  的用例和 helper 文件；多个源文件的目标用例不得遗漏。
- `revision`：源代码版本标识；源码有本地修改时附 dirty 状态或内容指纹，
  不得仅凭 HEAD 声称完全可复现。

`conversion_path` 是新增的顶层字段，路径相对 A5 根，指向 Stage 0
转换记录。其内容须有源/目标参数和返回值、manifest/workload 展开对照、
Kernel/Op 调用映射、完整测试/benchmark 案例清单、分支覆盖/排除依据。
保留原有目标字段 `wrapper_path`、`extracted_file`、`test_path`、`bench_path`；
wrapper/extracted 路径使用目标 `src/tileops/_kernels/`，test/bench 使用根目录路径；
不得把源仓绝对路径误当目标路径。多 Kernel 的 A5 exports 用 `kernel_class_names`
列表表示；`kernel_class_name` 保留为其中一个真实类名，不能拼接多个名字。

Stage 0 门禁对新 meta 检查来源路径、逐函数映射、转换记录和目标文件；
它只是机械下界，不替代人工语义核对或完整 Tier 1 的 import/collect 验证。

## 2. Stage 1–3 dispatch

conductor 为每个函数组装并向 designer/reviewer/developer 透传：

- `source_op_path`：`gpu_repo_root / gpu_source.function_sources[func]`
  的绝对路径，保证原始函数可定位；映射缺失即停止，不从旧布局猜测。
- `source_context`：`meta_path`、当前函数名、GPU 根、`gpu_source`、A5
  `conversion_path` 的绝对路径、提取参考文件及目标 test/bench 路径。

designer 不只读 kernel：连同 Op host dispatch、call-spec/interface 和源
manifest/workloads 读取，区分整个 public Op 的契约与当前内部函数的契约。
多阶段 kernel 的中间 buffer、返回 tuple、workspace 和调用顺序必须入设计。
reviewer 逐项核对转换记录及当前函数的来源；developer 按冻结设计保留 wrapper
消费的符号、参数和返回结构，golden 对应当前函数语义，不能把 public Op 的
最终输出当成 partial/reduction 内核的 golden。设计修订必须继续传完整上下文。
源码不应在这些阶段被 import 或执行。发现转换错误返回明确问题给 conductor，
不得在后续阶段静默改写规格、测试或丢掉实现分支。

## 3. Stage 4–5、报告与恢复

Stage 4 只消费迁移后的 A5 benchmark 入口及其实际分派，不能用 GPU
`benchmarks.api` 或源性能数值替代 A5 workload/测量。Stage 5 只消费目标
wrapper/Kernel/Op/tests/bench；多类 exports 全部检查。报告与 session timing
继续使用 A5 reporting CLI；不依赖 GPU reporting package。

续跑时按 meta 恢复来源上下文，核对源 revision/指纹；源文件变化必须重新
验证受影响函数和转换记录，不能把旧设计/精度报告视为新源码的验证。
终态蒸馏附 meta、转换记录、源版本与静态/运行时验证范围，绝不把静态提取
成功蒸馏为 NPU 精度或性能实证。

## 4. 无设备环境

用户禁止设备操作或明确无 NPU 时，只做 AST/YAML/路径与文件集成验证，
不执行 import 冒烟、pytest collection、compile_only、设备可用性探测、
correctness、benchmark 或 msprof。允许 `integrate_kernel.py --skip-smoke`。
静态检查不产生 `[PRECISION_PASS]` 或 `INTEGRATE_COMPLETED`，不伪造 run.json；
真实迁移按既有 `BLOCKED_ENVIRONMENT` 处理，不新增成功信号或放宽门禁。
维护规则文件的任务无需启动迁移状态机。
