"""run_experiments.py -- Stage 4 A/B experiment runner (T-4, do_bench(msprof) based).

Adapted from references/run_experiments_template.py for the LogSumExpFwdOp
dual-kernel task. Core protocol (signal-registry.md section 5): the SAME
workload's two kernel objects (A = baseline / current best, B = candidate)
are measured with INTERLEAVED repeated do_bench(backend="msprof") runs
(A/B/A/B order), early_stop_baseline=None. Per experiment this runner:

  1. runs the L0 precision regression for the implementation(s)
     (``python impl.py --level L0`` subprocess; --skip-l0-a skips the A side
     once the baseline L0 has been confirmed in this stage);
  2. interleaves do_bench(msprof) measurement rounds for A and B on every
     requested tune workload of the kernel (plus optional --custom shapes
     for DESIGN 1.6.4 adjudication evidence, which are NOT recorded);
  3. appends one perf_records.jsonl line per recorded tune workload
     (append-only; duration_us = ms * 1000);
  4. collects max_rel / max_abs vs the torch golden for the B side
     (fp16-e adjudication guard);
  5. optionally collects msprof_detail for the B side (--detail) and archives
     it under profiles/;
  6. prints a candidate-vs-current-best summary table.

Usage (inside perf_opt/):
  python run_experiments.py --kernel single --b _logsumexp_kernel_single_round1_rows.py --b-id r1_rows
  python run_experiments.py --kernel tiled --b _logsumexp_kernel_tiled_round1_blockn.py \
      --b-id r1_blockn --skip-l0-a
  python run_experiments.py --kernel tiled --b ... --custom 1024,32768,float16,16384,1
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_common as bc  # noqa: E402

KERNEL_IDS = {
    "single": "_logsumexp_kernel_single",
    "tiled": "_logsumexp_kernel_tiled",
}


def run_l0(impl_path: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, impl_path, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=os.path.dirname(os.path.abspath(impl_path)),
    )
    tail = (proc.stdout or "").strip().splitlines()[-3:] + (proc.stderr or "").strip().splitlines()[
        -3:
    ]
    return proc.returncode == 0, " | ".join(tail)[-400:]


def bench_batch(run_fn, samples: int) -> list[float]:
    return bc.msprof_samples_us(run_fn, samples)


def rel_err_stats(impl_path: str, workload: dict) -> tuple[float, float]:
    """(max_abs, max_rel) of the impl vs the fp32-domain torch golden."""
    import torch

    module = bc.load_impl(impl_path)
    p = workload["params"]
    torch_dtype = bc._DTYPES[workload["dtype"]]
    gen = torch.Generator(device="npu").manual_seed(1234)
    x = torch.randn(p["M"], p["N"], dtype=torch_dtype, device="npu", generator=gen)
    kernel = bc.build_kernel(module, workload)
    y = kernel(x)
    torch.npu.synchronize()
    ref = torch.logsumexp(x.float(), dim=-1).to(torch_dtype).float()
    diff = (y.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    return float(diff.max().item()), float((diff / denom).max().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", required=True, choices=list(KERNEL_IDS))
    ap.add_argument("--a", default=None, help="A side impl (default: baseline kernel file)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-id", default="baseline")
    ap.add_argument("--b-id", required=True)
    ap.add_argument("--b-parent", default="baseline")
    ap.add_argument(
        "--workloads",
        default=None,
        help="comma-separated tune workload ids (default: all tune of the kernel)",
    )
    ap.add_argument(
        "--custom",
        default=None,
        help="extra adjudication shape M,N,dtype[,tile_n,block_m] (measured, NOT recorded)",
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--skip-l0-a", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--record-phase", default="candidate", choices=["candidate", "merged", "final"])
    ap.add_argument("--round-num", type=int, default=1)
    ap.add_argument(
        "--b-env",
        action="append",
        default=[],
        help="KEY=VAL env applied before the B side is loaded (sweep support)",
    )
    ap.add_argument(
        "--a-env",
        action="append",
        default=[],
        help="KEY=VAL env applied ONLY around the A-side kernel build "
        "(B-side env keys are removed during A builds so an "
        "A=B-same-file sweep isolates the knob to B)",
    )
    ap.add_argument(
        "--detail", action="store_true", help="collect msprof_detail for B per workload"
    )
    args = ap.parse_args()

    kernel_id = KERNEL_IDS[args.kernel]
    a_impl = args.a or bc.BASELINE_IMPLS[kernel_id]
    for impl in (a_impl, args.b):
        if not os.path.isfile(impl):
            print(f"missing implementation file: {impl}")
            return 2

    # B-side env (sweeps): applied before load_impl picks up the branch module
    # (also inherited by the B-side L0 subprocess so the swept config is what
    # the precision regression exercises).
    b_env_keys = []
    for kv in args.b_env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
        b_env_keys.append(k)
        print(f"b-env: {k}={v}")

    def _push_env(kvs):
        """Apply KEY=VAL list; return the saved prior state."""
        saved = {}
        for kv in kvs:
            k, _, v = kv.partition("=")
            saved[k] = os.environ.get(k)
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return saved

    def _pop_env(saved):
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def make_runner_isolated(impl_path, workload, side):
        """make_runner with A/B env isolation.

        A side: b-env keys removed, a-env keys applied (default: no env).
        B side: b-env keys present (already set), a-env keys removed.
        """
        if side == "A":
            saved_b = {k: os.environ.get(k) for k in b_env_keys}
            for k in b_env_keys:
                os.environ.pop(k, None)
            saved_a = _push_env(args.a_env)
            try:
                return bc.make_runner(impl_path, workload)
            finally:
                _pop_env(saved_a)
                _pop_env(saved_b)
        saved_a = _push_env(args.a_env)
        try:
            return bc.make_runner(impl_path, workload)
        finally:
            _pop_env(saved_a)

    all_tune = [w["workload_id"] for w in bc.tune_workloads(kernel_id)]
    if args.workloads:
        wids = [w.strip() for w in args.workloads.split(",")]
    else:
        wids = all_tune
    for wid in wids:
        if wid not in all_tune:
            print(f"workload {wid!r} is not a tune workload of {kernel_id} (tune: {all_tune})")
            return 2

    if not args.skip_l0:
        ok, tail = run_l0(args.b)
        print(f"L0 B ({args.b}): {'pass' if ok else 'FAIL'} {tail if not ok else ''}")
        if not ok:
            return 3
        if not args.skip_l0_a:
            ok_a, tail_a = run_l0(a_impl)
            print(f"L0 A ({a_impl}): {'pass' if ok_a else 'FAIL'} {tail_a if not ok_a else ''}")
            if not ok_a:
                return 3

    results = {}
    for wid in wids:
        workload = [w for w in bc.tune_workloads(kernel_id) if w["workload_id"] == wid][0]
        run_a, _ = make_runner_isolated(a_impl, workload, "A")
        run_b, _ = make_runner_isolated(args.b, workload, "B")
        round_medians = {"A": [], "B": []}
        all_samples = {"A": [], "B": []}
        for rnd in range(1, args.rounds + 1):
            for label, fn in (("A", run_a), ("B", run_b)):
                samples = bench_batch(fn, args.samples)
                all_samples[label].extend(samples)
                round_medians[label].append(bc.median_us(samples))
        a_us = bc.median_us(all_samples["A"])
        b_us = bc.median_us(all_samples["B"])
        rel_pct = (a_us - b_us) / a_us * 100.0
        b_faster = sum(1 for a, b in zip(round_medians["A"], round_medians["B"]) if b < a)
        max_abs, max_rel = rel_err_stats(args.b, workload)
        detail_note = ""
        if args.detail:
            detail = bc.msprof_detail(run_b)
            arch = bc.archive_detail(f"round{args.round_num}", kernel_id, wid, args.b_id, detail)
            detail_note = (
                f"; B detail: dur={detail['dur_us']:.2f}us bottleneck={detail['bottleneck_pipe']} "
                f"{detail['bottleneck_ratio'] * 100:.1f}% -> {arch}"
            )
        results[wid] = (a_us, b_us, rel_pct, b_faster, max_abs, max_rel)
        print(
            f"[{wid}] A({args.a_id})={a_us:.3f}us B({args.b_id})={b_us:.3f}us "
            f"rel={rel_pct:+.1f}% paired_B_faster={b_faster}/{args.rounds} "
            f"B_max_abs={max_abs:.3e} B_max_rel={max_rel:.3e}{detail_note}",
            flush=True,
        )
        if not args.no_record:
            bc.append_record(
                {
                    "round": args.round_num,
                    "candidate_id": args.b_id,
                    "parent_id": args.b_parent,
                    "kernel_id": kernel_id,
                    "workload_id": wid,
                    "phase": args.record_phase,
                    "artifact_path": os.path.abspath(args.b),
                    "artifact_sha256": bc.file_sha256(args.b),
                    "duration_us": round(b_us, 3),
                    "l0_pass": True,
                    "profiler_output": "none",
                    "timestamp": bc.timestamp_utc(),
                }
            )

    if args.custom:
        parts = args.custom.split(",")
        M, N, dtype = int(parts[0]), int(parts[1]), parts[2]
        tile_n = int(parts[3]) if len(parts) > 3 and parts[3] else 16384
        block_m = int(parts[4]) if len(parts) > 4 and parts[4] else 1
        workload = {
            "kernel_id": kernel_id,
            "workload_id": f"custom-{M}x{N}-{dtype}",
            "dtype": dtype,
            "params": {"M": M, "N": N, "block_m": block_m, "tile_n": tile_n, "dim": -1},
        }
        run_a, _ = make_runner_isolated(a_impl, workload, "A")
        run_b, _ = make_runner_isolated(args.b, workload, "B")
        sa = bc.msprof_samples_us(run_a, args.samples)
        sb = bc.msprof_samples_us(run_b, args.samples)
        max_abs, max_rel = rel_err_stats(args.b, workload)
        a_us, b_us = bc.median_us(sa), bc.median_us(sb)
        print(
            f"[{workload['workload_id']}] (adjudication evidence, NOT recorded) "
            f"A={a_us:.3f}us B={b_us:.3f}us rel={(a_us - b_us) / a_us * 100:+.1f}% "
            f"B_max_abs={max_abs:.3e} B_max_rel={max_rel:.3e}",
            flush=True,
        )

    print(f"\n## candidate vs current best (kernel {kernel_id}, round {args.round_num})")
    print(
        "| workload_id | A_us | B_us | rel_pct | paired_B_faster | B_max_abs | B_max_rel | verdict |"
    )
    print("|---|---:|---:|---:|---|---:|---:|---|")
    for wid, (a_us, b_us, rel_pct, b_faster, max_abs, max_rel) in results.items():
        if rel_pct > 5.0 and b_faster == args.rounds:
            verdict = "improved"
        elif rel_pct > 5.0:
            verdict = "inconsistent"
        elif rel_pct < -5.0:
            verdict = "regression"
        else:
            verdict = "tie(<=5% band)"
        print(
            f"| {wid} | {a_us:.3f} | {b_us:.3f} | {rel_pct:+.1f} | "
            f"{b_faster}/{args.rounds} | {max_abs:.3e} | {max_rel:.3e} | {verdict} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
