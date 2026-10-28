"""run_experiments.py -- Stage 4 A/B experiment runner template (T-4, do_bench based).

Copied into perf_opt/ by the optimizer on its FIRST Phase 2 round and adapted
(this file is a TEMPLATE -- the adaptation points are the CONFIG constants and
the `load_runner` function below). After the first round, each round just
updates the A/B implementation files and re-runs this script; the agent only
reads the summary table it prints. This turns "agent round-trips drive
experiments" into "script executes, agent reads the table" (>=60% fewer agent
interaction rounds; also immune to the large-artifact session-overflow failure
class, see queue VP-2026-0021).

Core protocol (single metric contract, see _shared/standards/signal-registry.md
#5): the SAME workload's two kernel objects (A = baseline / current best,
B = candidate) are measured with INTERLEAVED repeated do_bench(backend="msprof")
runs (A/B/A/B order). Each measurement is several do_bench samples whose median
is taken; the per-round medians, the pooled medians and the relative comparison
are printed at the end. Per experiment this runner, in order:
  1. runs the L0 precision regression for both implementations (`--level L0`);
  2. interleaves do_bench(msprof) measurement rounds for A and B
     (early_stop_baseline=None: controlled comparisons must not mix the
     event-based early-stop estimate with msprof samples);
  3. appends one line per measured tune workload to perf_records.jsonl
     (append-only; duration_us = ms * 1000; profiler_output = deep-dive archive
     dir or "none");
  4. prints a candidate-vs-current-best summary table with the paired
     direction-consistency verdict (improved / tie / regression / inconclusive).

do_bench returns milliseconds; all latencies here are reported in microseconds
(ms * 1000). Deep-dive data (msprof_detail pipe ratios / npusim traces) is
collected separately per profile-collection.md and archived under profiles/.

Adaptation points (marked "ADAPT"):
  - CONFIG: operator identity, workload id, A/B implementation files;
  - load_runner(): how an implementation file becomes a zero-arg callable that
    launches the target kernel once on the given workload (keep the callable
    boundary identical between A and B: allocations, output init, sync).

Usage (inside perf_opt/):
  python run_experiments.py                      # A/B: interleaved, defaults from CONFIG
  python run_experiments.py --b my_op_round2_op1.py --workload case_fp16_4096
  python run_experiments.py --rounds 5 --samples 5
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys

# ----------------------------------------------------------------- CONFIG (ADAPT)
OP = "my_op"  # ADAPT: operator name (file stem)
KERNEL_ID = "path/to/kernel.py::factory::main"  # ADAPT: unique implementation identity
WORKLOAD_ID = "case_fp16_4096"  # ADAPT: the tune workload being measured
WORKLOAD_INVENTORY = "workload_inventory.json"  # ADAPT: Phase 1 inventory path
A_IMPL = "my_op.py"  # ADAPT: A side (baseline / current best)
B_IMPL = "my_op_round2_op1.py"  # ADAPT: B side (candidate branch)
A_ID = "baseline"  # ADAPT: candidate_id of the A side (current merged best)
B_ID = "round2_op1"  # ADAPT: candidate_id of the B side
B_PARENT_ID = "baseline"  # ADAPT: parent_id of the B side (its derivation base)
N_ROUNDS = 3  # independent interleaved A/B rounds (>=3 for the <5% protocol)
SAMPLES_PER_ROUND = 3  # do_bench(msprof) samples per measurement, median taken
WARMUP_MS = 25.0  # do_bench warmup budget in milliseconds
REP_MS = 100.0  # do_bench repeat budget in milliseconds
TIMEOUT_S = 900
PERF_RECORDS = "perf_records.jsonl"
PROFILES_DIR = "profiles"
LOGS_DIR = "logs"
# ---------------------------------------------------------------------------------


def tune_workload_ids() -> list[str]:
    """Tune workload IDs of this kernel from the Phase 1 inventory.

    Only inventory items with kind == "tune" for this KERNEL_ID may be
    measured and recorded (signal-registry.md §5: smoke/skipped must not have
    performance records).
    """
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as f:
        inventory = json.load(f)
    return sorted(
        item["workload_id"]
        for item in inventory.get("workloads", [])
        if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"
    )


def safe_component(raw: str) -> str:
    """Reject path/injection-bearing workload IDs (defense in depth)."""
    if "/" in raw or "\\" in raw or ";" in raw or raw != raw.strip():
        raise ValueError(f"unsafe workload id: {raw!r}")
    return raw


def load_runner(impl_path: str, workload_id: str):
    """ADAPT: build the measured callable for one implementation and workload.

    Must return a zero-arg callable that launches the target kernel exactly once
    on the given workload (steady-state; first JIT/module load happens before
    timing inside do_bench's warmup). Keep the callable boundary identical
    between A and B. The default expects the implementation file to expose:

        def make_runner(workload_id): ...  # -> zero-arg callable

    Replace this function if the entry needs other arguments.
    """
    module_name = os.path.splitext(os.path.basename(impl_path))[0]
    spec = importlib.util.spec_from_file_location(f"_ab_{module_name}", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import implementation file: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    maker = getattr(module, "make_runner", None)
    if not callable(maker):
        raise AttributeError(
            f"{impl_path} does not expose make_runner(workload_id); adapt load_runner()"
        )
    return maker(workload_id)


def run_l0(impl_path: str) -> bool:
    proc = subprocess.run(
        [sys.executable, impl_path, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    return proc.returncode == 0


def bench_samples_ms(run_fn, samples: int) -> list[float]:
    """One measurement = several do_bench(msprof) samples (returns milliseconds).

    early_stop_baseline is disabled: the event-based estimate is a different
    measurement path and must not be mixed with profiler samples.
    """
    from tilelang.profiler import do_bench

    return [
        do_bench(
            run_fn,
            backend="msprof",
            warmup=WARMUP_MS,
            rep=REP_MS,
            early_stop_baseline=None,
        )
        for _ in range(samples)
    ]


def median_us(samples_ms: list[float]) -> float:
    return statistics.median(samples_ms) * 1000.0


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def timestamp_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default=A_IMPL, help="A side implementation file")
    ap.add_argument("--b", default=B_IMPL, help="B side implementation file")
    ap.add_argument("--a-id", default=A_ID, help="candidate_id of the A side")
    ap.add_argument("--b-id", default=B_ID, help="candidate_id of the B side")
    ap.add_argument("--workload", default=WORKLOAD_ID, help="tune workload ID")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS, help="interleaved A/B rounds")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_ROUND, help="do_bench samples per measurement")
    ap.add_argument("--skip-l0", action="store_true", help="skip the L0 precision regression")
    ap.add_argument("--no-record", action="store_true", help="do not append to perf_records.jsonl")
    args = ap.parse_args()
    if args.rounds < 1 or args.samples < 1:
        ap.error("--rounds and --samples must be >= 1")
    try:
        safe_component(args.workload)
    except ValueError as exc:
        print(str(exc))
        return 2
    # Inventory discipline: only tune workloads of this kernel may be measured
    try:
        tune_ids = tune_workload_ids()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read workload inventory {WORKLOAD_INVENTORY}: {exc}")
        return 2
    if args.workload not in tune_ids:
        print(
            f"workload {args.workload!r} is not a tune workload of {KERNEL_ID} "
            f"in the inventory (tune: {tune_ids}); smoke/skipped workloads "
            "must not have performance records"
        )
        return 2

    sides = (("A", args.a, args.a_id), ("B", args.b, args.b_id))
    for _, impl, _ in sides:
        if not os.path.isfile(impl):
            print(f"missing implementation file: {impl}")
            return 2

    if not args.skip_l0:
        for label, impl, _ in sides:
            ok = run_l0(impl)
            print(f"L0 {label} ({impl}): {'pass' if ok else 'FAIL'}")
            if not ok:
                return 3

    runners = {}
    for label, impl, _ in sides:
        try:
            runners[label] = load_runner(impl, args.workload)
        except Exception as exc:  # noqa: BLE001 - report and stop, do not guess
            print(f"load_runner failed for {label} ({impl}): {exc}")
            return 4

    # Interleaved measurement: A/B/A/B... one measurement per side per round.
    round_medians = {"A": [], "B": []}
    all_samples = {"A": [], "B": []}
    for rnd in range(1, args.rounds + 1):
        for label, _, _ in sides:
            samples = bench_samples_ms(runners[label], args.samples)
            all_samples[label].extend(samples)
            round_medians[label].append(median_us(samples))
            print(
                f"round {rnd}/{args.rounds} {label}: median = {round_medians[label][-1]:.3f} us "
                f"(samples: {', '.join(f'{s * 1000:.3f}' for s in samples)} us)"
            )

    a_us = statistics.median(all_samples["A"])
    b_us = statistics.median(all_samples["B"])
    rel_pct = (a_us - b_us) / a_us * 100.0
    # Paired direction consistency: B faster than A in every interleaved round?
    paired_b_faster = sum(1 for a, b in zip(round_medians["A"], round_medians["B"]) if b < a)
    consistent = paired_b_faster == args.rounds
    if not consistent and rel_pct > 5.0:
        verdict = "inconsistent"  # pooled median favors B but pairing is not stable
    elif rel_pct > 5.0:
        verdict = "improved"
    elif rel_pct < -5.0:
        verdict = "regression"
    else:
        verdict = "tie"  # inside the noise band: needs the interleaved multi-run protocol

    print(
        f"\n| side | impl | candidate_id | rounds | median_us | min_round_us | max_round_us |"
    )
    print("|---|---|---|---:|---:|---:|---:|")
    for label, impl, cid in sides:
        meds = round_medians[label]
        print(
            f"| {label} | {impl} | {cid} | {len(meds)} | "
            f"{statistics.median(all_samples[label]):.3f} | {min(meds):.3f} | {max(meds):.3f} |"
        )
    print(
        f"\nB vs A: {rel_pct:+.1f}% (A {a_us:.3f} us -> B {b_us:.3f} us); "
        f"paired direction: B faster in {paired_b_faster}/{args.rounds} rounds; "
        f"verdict: {verdict}"
    )
    print(
        "note: |diff| <= 5% is the noise band -- adjudicate with the interleaved "
        "multi-run protocol (ab_test.py) before recording a winner"
    )

    if not args.no_record:
        record = {
            "round": 0,
            "candidate_id": args.b_id,
            "parent_id": B_PARENT_ID,
            "kernel_id": KERNEL_ID,
            "workload_id": args.workload,
            "phase": "candidate",
            "artifact_path": args.b,
            "artifact_sha256": file_sha256(args.b),
            "duration_us": round(b_us, 3),
            "l0_pass": True,
            "profiler_output": "none",
            "timestamp": timestamp_utc(),
        }
        append_record(record)
        print(f"appended perf_records.jsonl: {args.b_id} @ {args.workload} = {b_us:.3f} us")
        print("(adjust round/phase/parent_id when reusing this template for baseline/merged/final)")

    os.makedirs(LOGS_DIR, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
