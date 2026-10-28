"""Phase 0 probe 2: run op forward per workload, dump kernel + config."""
import torch

from tileops.workloads.reduction import LogSumExpWorkload
from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

cases = [
    ((32, 32, 4096), torch.float16, -1, "attn-weights-4k-float16"),
    ((32, 32, 4096), torch.bfloat16, -1, "attn-weights-4k-bfloat16"),
    ((32, 32, 32768), torch.bfloat16, -1, "attn-weights-32k-bfloat16"),
    ((4, 102400), torch.float16, -1, "lm-head-logits-float16"),
    ((4, 102400), torch.bfloat16, -1, "lm-head-logits-bfloat16"),
    ((4, 128, 4096), torch.float16, [0, 2], "3d-multidim-reduce-float16"),
]
for shape, dtype, dim, wid in cases:
    test = LogSumExpWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = LogSumExpFwdOp(dtype=dtype, dim=dim)
    y = op.forward(*inputs)
    torch.npu.synchronize()
    kern = op.kernel
    print(f"{wid}: M={kern.M} N={kern.N} dtype={kern.dtype_str} "
          f"config={kern.config} tile_n={kern._tile_n} y_shape={tuple(y.shape)}")
