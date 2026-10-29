# RETROSPECTIVE — examples/pilot/silu

## Stage 1 (design) Retrospective — first_design — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 按脚本路径直接执行探针时 `import tilelang` 阴影解析到 tilelang-mlir-fork（无 `tilelang.ascend`），报 ModuleNotFoundError；须以本仓 root 为 cwd 或显式 PYTHONPATH | `examples/pilot/silu/history_version/design_probe_simt_exp.py` 文件头 Run 注释；DESIGN.md §9.2「环境阴影导入」行 | tilelang-op-design skill D-3 探针选项补一句：「探针脚本须以本仓 root 为 cwd 或 `PYTHONPATH=<repo>` 执行，防跨仓阴影导入」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SimtVF `T.Parallel` 内 `T.exp` 自动向量化实证：`tir.exp` lower 为 2048-bit `vexp`（`src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`），fp16 输入可用，vs fp32-opmath golden max abs diff 3.906e-3（< 1e-2 容差）——激活类算子（sigmoid/gelu/tanh 合成）simt 模式无需显式 intrinsic | D | `examples/pilot/silu/history_version/design_probe_simt_exp.py`（2026-10-16 全量 PASS） + `src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412` | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/silu/history_version/design_probe_simt_exp.py` | tilelang-fork dev build（repo fc32f47c 时期源码）+ CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16 | tilelang-op-optimize skill pattern-library（elementwise 模式条目候选） |
| 网格整除陷阱（负面发现）：元素总量 2^17 不含因子 7，含 7 因子的核数网格（28/56）无法整除——违例不报编译错误，表现为尾部静默漏写（探针 v1：28 blocks × 4096，12.5% 输出未初始化，max abs diff 4.8 伪装成精度问题） | D | `examples/pilot/silu/DESIGN.md` §5.3 / §9.2；探针 v1 运行记录（28-blocks 配置） | 复现条件：`NUM_BLOCKS=28, TILE=4096, TOTAL=131072` 的同构 kernel（`131072 % (28×4096) != 0`，`T.Pipelined(1)` 只覆盖 114688 元素） | 同上 | pattern-library traps（编译器/运行时陷阱条目候选） |
| silu/sigmoid 公式候选否决表：vexpdif 融合仅 SimdVF+fp32（simd.py:1182-1195）fp16 需双向 cast 净亏；A5 有直连自然指数 vexp 使 exp2 换底无益；tanh 无直连 intrinsic；基线 `x/(1+exp(-x))` 4 向量 op 已最小 | P | `examples/pilot/silu/DESIGN.md` §1.6.0 R1 表 | none | 同上 | tilelang-op-design skill references/algorithm-candidates.md（ALG-elementwise 或新增 activation 条目候选） |

### Transferable Lessons（可迁移教训）

- 运行算子目录下的 python 脚本须以本仓 root 为 cwd（或 `PYTHONPATH` 指向本仓），否则 `import tilelang` 阴影解析到 tilelang-mlir-fork，报 `No module named 'tilelang.ascend'`。
- grid 设计前先做元素总量因子分解：2^n 总量与 7 因子核数（28 AIC / 56 AIV）天然互斥；整除违例的症状是**尾部静默漏写**（输出含未初始化内存），极易被误判为精度问题——host 侧 `assert TOTAL % (NUM_BLOCKS*TILE) == 0` 是必要防御。
- SimtVF `T.Parallel` 循环内的标准数学 intrinsic（`T.exp`/`T.log`/`T.sqrt`/`T.fabs`，即 `tir.*` 名）由 `ascend_simdvf_lower_parallel` 自动向量化为 2048-bit SIMD 指令，fp16 可用；激活类算子优先 simt 模式直写公式，SimdVF 显式 intrinsic 仅在需要掩码/舍入模式控制时介入。

## Stage 2 (design-review) Retrospective — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 机械复核工具 | `design_calc_check.py` 对纯 Vector 扁平算子设计的解析覆盖率低：§4.5 有规整的 per-buffer 预算表、§5.5 有完整分核三要素，但 `ub_budget`/`l0c_budget`/`core_split` 三项均 skip（表格形态/维度声明格式不在 parser 预期内），仅 `r3_metrics` 机检生效——机械拦截位对逐元素类设计退化为全人工复核 | 本次 REVIEW.md 机械复核段 JSON：3 skip + 1 pass；DESIGN.md §4.5/§5.5 表格实际存在且格式规整 | `.agents/tools/design_calc_check.py` 扩展 parser：识别「Buffer/Shape/dtype/大小」列式预算表与 §5.5 分核三要素的 TOTAL/NUM_BLOCKS/TILE 常量声明（纯 Vector 形态） | R |
| 检索工具口径 | `kb_search.py "silu sigmoid elementwise"` 返回 no entries matched，而直接 grep `algorithm-candidates.md` 可命中 ALG-elementwise（family: [elementwise, activation]）——多词查询未做 family 关键词回退，设计文档若只依赖 kb_search 会误判「候选库无命中行」；本次检视以直接 grep 候选库全文为准完成复核 | 检视过程两次检索输出对比（kb_search no match vs grep 命中 line 100-106） | `tilelang-op-design` skill references/algorithm-research.md §5 检索入口说明补一句：「kb_search 未命中时须直接 grep 候选库 family 字段确认，不得据 kb_search 空结果断言无命中行」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 检视严重程度标定规则（P 类）：边界/特殊值表中的期望值常数易出现「接近但不精确」的手算偏差（本次 §8.3 两处：-1.66e-4 vs 精算 -1.84e-4、-6.9e-12 vs 精算 -2.8e-12）——检视须独立重算并按「偏差相对测试容差的量级」定性：偏差 ≪ atol（≥2 个数量级）且判定时以 golden 实算为准 → 建议级；偏差可能翻转判定 → 阻塞级 | P | 本次 REVIEW.md 维度 6「建议级小疵」条目；DESIGN.md §8.3 边界表 | none | tilelang-fork fc32f47c 时期，2026-10-16 | tilelang-design-review skill §3 维度 6（检查项补注） |
| API 负向断言的检视查证套路（P 类）：对「方言无 X intrinsic」类断言，① 全目录大小写不敏感 grep（`tilelang/ascend/`）② simd.py `def` 清单枚举 ③ 探针/升降低径源码旁证，三步即可确定性裁定，无需真机（本次据此复核 vtanh 不存在、vexpdif fp32-only、exp2 无快档三处负向断言全部成立） | P | 本次 REVIEW.md 维度 8「负向断言复核」；simd.py:860/1052/1067/1182-1195、ascend_simdvf_lower_parallel.cc:405-414 | `grep -ri tanh tilelang/ascend/`（零命中）等 | 同上 | tilelang-design-review skill §3 维度 8（检视方式补注） |

### Transferable Lessons（可迁移教训）

- 检视设计文档中的量化断言时，「复算数字」与「核对常数出处」必须同时做：本次 D-2 估算（988/694 cyc、1.4×）复算全对，且逐值回溯到 constants.md 条目（VALU 51/13/8、MTE2 100B+90、MTE3 115B+180）才确认非凑数——任一侧缺失都可能放过口径错误（VP-2026-0009 类）。
- 机械复核 skip ≠ 免检：design_calc_check 3 项 skip 时，人工复核须在 REVIEW.md 中显式留下算术过程（16KB≤194.4KB、2 iter 精确、32≤56 单波），否则下游无法区分「机检通过」与「未检」。
- 探针类实证（编译通过/max diff 数值）检视时只能核对「探针源码存在 + 逻辑与声明一致 + 记录自洽（日期/配置/双 golden 口径）」，运行数值本身不可离线复验——结论中应如实区分「实证核对」与「数值采信」两级置信度。

## Stage 3 (develop) Retrospective — first_impl — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 逐元素激活算子做 Boundary 特殊值用例时，若输入含 ±inf，`diff = (out - golden).abs()` 在 +inf 位置产生 `inf - inf = NaN`，`-inf → NaN` 与 `NaN → NaN` 位置也产生 NaN——朴素 `max_diff = diff.max().item()` 会报告 `nan`，即便 `assert_close(equal_nan=True)` 已 PASS；op-develop skill/template 的 Boundary 用例指导未提示此点 | `examples/pilot/silu/silu.py` run_boundary()（初版 max_diff=nan，修复后改为 isfinite 掩码，见 144-160 行）；模板 `op_template.py` run_boundary() 仅对比 zeros | `.agents/skills/tilelang-op-develop` Phase 4 或模板 run_boundary() 补一句：「特殊值用例对比指标须先 `torch.where(torch.isfinite(diff), diff, 0)` 再取 max，避免 inf-inf=NaN 污染报告」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| fp16 逐元素激活边界修正：silu(±6e-8 次正规输入) 实际输出 ±0（而非 DESIGN.md §8.3 预期的 ≈±3e-8），因 x·sigmoid(x)≈3e-8 < fp16 最小次正规 5.96e-8 下溢 flush；kernel 与 fp32-opmath→fp16 golden 位级一致（finite diff=0），属设计边界表精度不精而非正确性缺陷 | D | `examples/pilot/silu/silu.py` run_boundary()（idx 0/1：in=5.96e-8 → out=0）；DESIGN.md §8.3「±6e-8 → ≈±3e-8」行 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/silu/silu.py --level boundary`（观察 idx 0/1 out=0） | tilelang-fork dev build 48345cd7 + CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16 | tilelang-op-design skill references/algorithm-candidates.md（激活边界语义条目候选，或 pattern-library elementwise） |
| 激活类 Boundary 精度报告方法：对含 ±inf/NaN 输入的特殊值用例，`|out-golden|` 在 inf-inf / NaN 位置产生 NaN，报告 finite max_diff 须先 `torch.where(torch.isfinite(diff), diff, 0)` 掩码，且 assert_close 需 `equal_nan=True` | P | `examples/pilot/silu/silu.py:144-160`（run_boundary 掩码逻辑） | `torch.tensor([float('inf')]) - torch.tensor([float('inf')])` 返回 nan；同形状 out/golden 均 inf 时 diff=nan | 同上 | tilelang-op-develop skill（Boundary 用例写法） |

### Transferable Lessons（可迁移教训）

- 扁平 1D 逐元素算子用「@tilelang.jit 工厂函数参数化 (M, N)」即可一个 kernel 覆盖分层测试多 shape（各 shape 编译一次并缓存），无需为每个测试 shape 手写独立 kernel。
- 元素级激活 kernel 做特殊值（±inf/NaN）对比时，`assert_close` 必须传 `equal_nan=True`；报告 max_diff 前须把非有限差值掩码为 0，否则 `inf - inf = NaN` 会让 PASS 用例的 max_diff 显示为 nan。
- 次正规输入（< fp16 最小次正规 5.96e-8 量级）经激活后输出会下溢 flush 为 0，且 kernel 与 fp32-opmath→fp16 golden 双方同 flush，位级一致——设计边界表若只写「≈x/2」需意识到 fp16 回 cast 会把它压成 0。
