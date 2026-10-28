"""Phase 1 source inspection: check (a) single masked-load folding at N==TILE,
(b) tiled lse write structure (MTE3 tiny copies)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

# --- single (1024, 4096) fp16 block_m=16 (attn-weights-4k) ---
w = [x for x in bc.tune_workloads("_logsumexp_kernel_single")
     if x["workload_id"] == "attn-weights-4k-float16"][0]
mod = bc.load_impl(bc.BASELINE_IMPLS["_logsumexp_kernel_single"])
p = w["params"]
k = getattr(mod, "_logsumexp_kernel_single")(p["M"], p["N"], w["dtype"])(p["block_m"])
src = k.get_kernel_source()
print("=== single (1024,4096,fp16,bm=16) source: load section ===")
# find the SimtVF load: look for select / vcmp / -inf patterns
for pat in ("select", "Select", "cmp", "Cmp", "inf", "Exp", "exp"):
    hits = [ln.strip() for ln in src.splitlines() if pat in ln]
    print(f"pattern {pat!r}: {len(hits)} lines")
    for h in hits[:6]:
        print("   ", h[:160])
print("\ncopy/DataCopy lines:")
for ln in src.splitlines():
    if "DataCopy" in ln or "data_move" in ln or "copy" in ln.lower():
        print("   ", ln.strip()[:170])

print("\n\n=== tiled (1024,32768,bf16,bm=1,tile_n=16384) source: tail/lse section ===")
w2 = [x for x in bc.tune_workloads("_logsumexp_kernel_tiled")
      if x["workload_id"] == "attn-weights-32k-bfloat16"][0]
mod2 = bc.load_impl(bc.BASELINE_IMPLS["_logsumexp_kernel_tiled"])
p2 = w2["params"]
k2 = getattr(mod2, "_logsumexp_kernel_tiled")(p2["M"], p2["N"], w2["dtype"], p2["tile_n"])(p2["block_m"])
src2 = k2.get_kernel_source()
lines = src2.splitlines()
for i, ln in enumerate(lines):
    if "DataCopy" in ln or "data_move" in ln:
        print(f"   [{i}] {ln.strip()[:170]}")
# count loop structure around the persistent body
print("\nscalar exp / log calls:")
for i, ln in enumerate(lines):
    if re.search(r"\b(Exp|Log|exp|log)\s*\(", ln):
        print(f"   [{i}] {ln.strip()[:160]}")
print("\nTOTAL lines:", len(lines))
