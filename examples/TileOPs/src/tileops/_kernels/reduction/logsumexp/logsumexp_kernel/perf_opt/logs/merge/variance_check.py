"""variance_check.py -- Phase 3 final-record reconciliation (append-only safe).

Gate: "同 workload 复测方差 <=1%" for the final do_bench(msprof) records.
Protocol per workload (skill '先确认测量分辨力' section): 5 INDEPENDENT
medians, each the median of 5 do_bench(backend='msprof') samples, run
sequentially on the SAME final kernel object; report per-run medians, max
pairwise spread, and the pooled median.  No records are appended here --
this is the reconciliation evidence quoted in opt_log.md Final Summary.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as bc  # noqa: E402

RUNS = 5
SAMPLES = 5

FINAL_IMPLS = {
    "_logsumexp_kernel_single": "_logsumexp_kernel_single.py",
    "_logsumexp_kernel_tiled": "_logsumexp_kernel_tiled.py",
}

results = {}
for kernel_id, impl in FINAL_IMPLS.items():
    for w in bc.tune_workloads(kernel_id):
        run_fn, _ = bc.make_runner(os.path.join(bc.PERF_OPT_DIR, impl), w)
        medians = []
        for i in range(RUNS):
            samples = bc.msprof_samples_us(run_fn, SAMPLES)
            medians.append(bc.median_us(samples))
        pooled = bc.median_us([m for m in medians])
        spread = (max(medians) - min(medians)) / min(medians) * 100.0
        results[f"{kernel_id}|{w['workload_id']}"] = (medians, pooled, spread)
        print(
            f"[{w['workload_id']}] medians(us)={[f'{m:.3f}' for m in medians]} "
            f"pooled={pooled:.3f} max_spread={spread:.2f}%",
            flush=True,
        )

print("\n## variance reconciliation (final files, same kernel object)")
print("| kernel_id | workload_id | pooled_us | max_spread | <=1% |")
print("|---|---|---:|---:|---|")
for key, (medians, pooled, spread) in results.items():
    kernel_id, wid = key.split("|")
    print(f"| {kernel_id} | {wid} | {pooled:.3f} | {spread:.2f}% | {'yes' if spread <= 1.0 else 'NO'} |")
