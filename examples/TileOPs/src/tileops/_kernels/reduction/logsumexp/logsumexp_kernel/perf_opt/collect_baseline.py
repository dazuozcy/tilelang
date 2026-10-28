"""collect_baseline.py -- Phase 1: per-kernel tune-workload do_bench(msprof) baseline.

For every tune workload in workload_inventory.json (grouped by kernel):
  1. golden sanity of the baseline impl on this workload (callable-boundary check);
  2. two independent measurement batches of 5 do_bench(msprof) samples each
     (median per batch; variance between batches must be <= 1%);
  3. msprof_detail deep dive (pipe ratios) archived under profiles/baseline/;
  4. one perf_records.jsonl line (phase=baseline, round=0, candidate=baseline).

Usage: python collect_baseline.py [--workload <id>] [--samples 5]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_common as bc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=None)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for kernel_id in ("_logsumexp_kernel_single", "_logsumexp_kernel_tiled"):
        for w in bc.tune_workloads(kernel_id):
            wid = w["workload_id"]
            if args.workload and wid != args.workload:
                continue
            impl = bc.BASELINE_IMPLS[kernel_id]
            print(f"\n=== {kernel_id} / {wid} ===", flush=True)
            maxdiff = bc.golden_check(impl, w)
            print(f"golden sanity PASS (max_diff={maxdiff:.3e})", flush=True)

            run_fn, _x = bc.make_runner(impl, w)
            batch1 = bc.msprof_samples_us(run_fn, args.samples)
            batch2 = bc.msprof_samples_us(run_fn, args.samples)
            med1, med2 = bc.median_us(batch1), bc.median_us(batch2)
            var = bc.variance_pct(med1, med2)
            med = bc.median_us(batch1 + batch2)
            status = "valid" if var <= 1.0 else "variance_over_threshold"
            print(
                f"do_bench(msprof): batch medians {med1:.3f} / {med2:.3f} us, "
                f"pooled median {med:.3f} us, batch variance {var:.2f}% -> {status}",
                flush=True,
            )

            detail = bc.msprof_detail(run_fn)
            arch = bc.archive_detail("baseline", kernel_id, wid, "baseline", detail)
            print(
                f"msprof_detail: dur={detail['dur_us']:.3f}us bottleneck={detail['bottleneck_pipe']} "
                f"({detail['bottleneck_ratio'] * 100:.1f}%) aiv_cycles={detail['aiv_total_cycles']} "
                f"pipes={ {k: round(v, 3) for k, v in detail['pipes'].items() if v > 0.001} }",
                flush=True,
            )

            bc.append_record(
                {
                    "round": 0,
                    "candidate_id": "baseline",
                    "parent_id": "none",
                    "kernel_id": kernel_id,
                    "workload_id": wid,
                    "phase": "baseline",
                    "artifact_path": impl,
                    "artifact_sha256": bc.file_sha256(impl),
                    "duration_us": round(med, 3),
                    "l0_pass": True,
                    "profiler_output": arch,
                    "timestamp": bc.timestamp_utc(),
                }
            )
            rows.append((kernel_id, wid, med, var, detail))
            print(f"appended perf_records.jsonl baseline line: {wid} = {med:.3f} us", flush=True)

    print("\n## Baseline Performance Test Data (do_bench msprof, kernel-level boundary)")
    print("| kernel_id | workload_id | candidate_id | duration_us | samples | variance_pct | bottleneck | profiler_output |")
    print("|---|---|---|---:|---:|---:|---|---|")
    for kernel_id, wid, med, var, detail in rows:
        print(
            f"| {kernel_id} | {wid} | baseline | {med:.3f} | {args.samples * 2} | {var:.2f} | "
            f"{detail['bottleneck_pipe']} {detail['bottleneck_ratio'] * 100:.1f}% | "
            f"profiles/baseline/{kernel_id}/{wid}_baseline |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
