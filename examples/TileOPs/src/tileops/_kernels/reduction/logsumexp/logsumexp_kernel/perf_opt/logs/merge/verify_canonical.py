"""verify_canonical.py -- measure perf_opt/LogSumExpFwdOp.py on all 6 tune workloads.

Purpose: the state-contract final artifact (kernel_opt_py_path) is the
consolidated LogSumExpFwdOp.py; the phase=final records were collected via the
per-kernel drop-in files (byte-frozen same code).  This script re-measures the
canonical artifact with the identical do_bench(msprof) protocol and verifies
each workload is within the 1% reconciliation tolerance of the recorded final
value (gate: S4-PERF-RECORDS duration reconciliation basis).  NOT appended to
perf_records.jsonl (the final records already exist; this is verification
evidence quoted in opt_log.md).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as bc  # noqa: E402

RECORDED_FINAL_US = {
    ("_logsumexp_kernel_single", "attn-weights-4k-float16"): 21.495,
    ("_logsumexp_kernel_single", "attn-weights-4k-bfloat16"): 21.575,
    ("_logsumexp_kernel_single", "3d-multidim-reduce-float16"): 13.799,
    ("_logsumexp_kernel_tiled", "attn-weights-32k-bfloat16"): 100.226,
    ("_logsumexp_kernel_tiled", "lm-head-logits-float16"): 17.757,
    ("_logsumexp_kernel_tiled", "lm-head-logits-bfloat16"): 17.714,
}

impl = os.path.join(bc.PERF_OPT_DIR, "LogSumExpFwdOp.py")
print("| kernel_id | workload_id | recorded_final_us | canonical_us | diff | <=1% |")
print("|---|---|---:|---:|---:|---|")
ok_all = True
for w in bc.tune_workloads():
    key = (w["kernel_id"], w["workload_id"])
    run_fn, _ = bc.make_runner(impl, w)
    medians = [bc.median_us(bc.msprof_samples_us(run_fn, 3)) for _ in range(3)]
    canon = bc.median_us(medians)
    rec = RECORDED_FINAL_US[key]
    diff = abs(canon - rec) / rec * 100.0
    ok = diff <= 1.0
    ok_all = ok_all and ok
    print(f"| {w['kernel_id']} | {w['workload_id']} | {rec:.3f} | {canon:.3f} | {diff:.2f}% | {'yes' if ok else 'NO'} |")
print("VERDICT:", "ALL_WITHIN_1PCT" if ok_all else "OUT_OF_TOLERANCE")
