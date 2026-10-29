"""Dump full AscendC sources for baseline configs to profiles/source/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

OUT = os.path.join(bc.PROFILES_DIR, "source")
os.makedirs(OUT, exist_ok=True)

specs = [
    ("_logsumexp_kernel_single", "attn-weights-4k-float16"),
    ("_logsumexp_kernel_single", "3d-multidim-reduce-float16"),
    ("_logsumexp_kernel_tiled", "attn-weights-32k-bfloat16"),
    ("_logsumexp_kernel_tiled", "lm-head-logits-float16"),
]
for kernel_id, wid in specs:
    w = [x for x in bc.tune_workloads(kernel_id) if x["workload_id"] == wid][0]
    mod = bc.load_impl(bc.BASELINE_IMPLS[kernel_id])
    p = w["params"]
    if kernel_id.endswith("single"):
        k = getattr(mod, "_logsumexp_kernel_single")(p["M"], p["N"], w["dtype"])(p["block_m"])
    else:
        k = getattr(mod, "_logsumexp_kernel_tiled")(p["M"], p["N"], w["dtype"], p["tile_n"])(p["block_m"])
    src = k.get_kernel_source()
    path = os.path.join(OUT, f"{kernel_id}__{wid}__baseline.c")
    with open(path, "w") as f:
        f.write(src)
    print(f"wrote {path} ({len(src.splitlines())} lines)")
