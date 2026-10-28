"""Inspect the fp16-e branch lowering: why is it 50x slower (exp software path?)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

os.environ.setdefault("TILELANG_LSE_ROWS", "4")
w = [x for x in bc.tune_workloads("_logsumexp_kernel_single")
     if x["workload_id"] == "attn-weights-4k-float16"][0]
mod = bc.load_impl("_logsumexp_kernel_single_round1_fp16e.py")
p = w["params"]
k = getattr(mod, "_logsumexp_kernel_single")(p["M"], p["N"], w["dtype"])(p["block_m"])
src = k.get_kernel_source()
out = os.path.join(bc.PROFILES_DIR, "source", "_logsumexp_kernel_single__attn-weights-4k-float16__r1_fp16e.c")
with open(out, "w") as f:
    f.write(src)
print(f"wrote {out} ({len(src.splitlines())} lines)")
# show the exp / reduce lowering region
import re
for i, ln in enumerate(src.splitlines()):
    if re.search(r"exp|Exp|log|Log|AllReduce|Reduce|half|hexp|expf", ln):
        print(f"[{i}] {ln.strip()[:170]}")
