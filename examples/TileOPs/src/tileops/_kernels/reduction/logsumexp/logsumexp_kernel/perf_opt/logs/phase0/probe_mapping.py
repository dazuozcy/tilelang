"""Phase 0 probe: confirm workload -> kernel mapping + wrapper default configs."""
import torch

from tileops.workloads.reduction import LogSumExpWorkload
from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

print("npu available:", torch.npu.is_available())
props = torch.npu.get_device_properties()
print("device:", props.name, "cube:", props.cube_core_num, "vector:", props.vector_core_num)

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
    # inspect the op's kernel wrapper (M, N, config)
    kern = op._kernel if hasattr(op, "_kernel") else None
    print(f"--- {wid}: shape={shape} dtype={dtype} dim={dim}")
    for attr in ("_kernel", "_kernel_cache", "kernel", "config"):
        if hasattr(op, attr):
            v = getattr(op, attr)
            if attr == "_kernel_cache" and isinstance(v, dict):
                for key, kk in v.items():
                    print(f"    _kernel_cache[{key}]: M={getattr(kk,'M',None)} N={getattr(kk,'N',None)} "
                          f"dtype={getattr(kk,'dtype',None)} config={getattr(kk,'config',None)} "
                          f"tile_n={getattr(kk,'_tile_n',None)}")
            elif attr == "config" and v:
                print(f"    op.config = {v}")
            elif v is not None and not callable(v):
                print(f"    op.{attr} = {type(v)}")
    # derive M,N like the op does (movedim+reshape): reduced dims to the end
    import math
    x = inputs[0]
    dims = [dim] if isinstance(dim, int) else list(dim)
    dims = sorted(d % x.ndim for d in dims)
    keep = [i for i in range(x.ndim) if i not in dims]
    N = math.prod(x.shape[i] for i in dims)
    M = math.prod(x.shape[i] for i in keep)
    print(f"    derived M={M} N={N}")
    if hasattr(op, "_kernel_cache") and isinstance(op._kernel_cache, dict):
        for key, kk in op._kernel_cache.items():
            print(f"    kernel[{key}]: M={getattr(kk,'M',None)} N={getattr(kk,'N',None)} "
                  f"dtype_str={getattr(kk,'dtype_str',None)} config={getattr(kk,'config',None)}")
EOF_MARKER_NOT_NEEDED = None
