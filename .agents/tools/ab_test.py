"""ab_test.py -- interleaved A/B measurement protocol on do_bench (T-2, A5).

Implements the run-state bimodality countermeasure: same-kernel cross-run
bimodality makes single measurements unreliable for <5% candidate
differences. This tool runs an interleaved A/B/A/B... sequence (>= 3 pairs,
alternating execution order per pair to break slow drift) where every
measurement is a ``tilelang.profiler.do_bench(fn, backend="msprof")`` call
(the sole end-to-end latency authority on the ascend target — see
``.agents/skills/tilelang-ascend/references/profiling.md``), and reports
the merged median difference plus an exact paired sign test.

Each arm is a Python module that defines ``make_fn() -> Callable[[], None]``
returning the callable to benchmark (same workload for both arms; the
callable must launch the target kernel — first JIT / module load / input
setup must happen inside ``make_fn()``, not in ``fn``, so that only steady
state is measured). Arms are imported in-process so the two kernel objects
share one environment.

Usage:
  python3 .agents/tools/ab_test.py \
      --a-script perf_opt/ab_arms/baseline_arm.py \
      --b-script perf_opt/ab_arms/candidate_arm.py \
      --workdir perf_opt/ab_test [--pairs 3] [--warmup-ms 25] [--rep-ms 100] \
      [--noise 0.03] [--backend msprof]

Raw msprof output is redirected under ``--workdir/profiles/`` (via
``ASCEND_WORK_PATH``) for later msprof_detail deep-dives. Exit code 0 on
success (verdict printed as JSON), 2 on usage/environment error, 1 on
measurement failure.
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys


def _load_arm(script_path: str, arm: str):
    """Import an arm module and return its make_fn() callable."""
    if not os.path.isfile(script_path):
        raise RuntimeError(f"arm script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(f"ab_arm_{arm}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import arm script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make_fn = getattr(module, "make_fn", None)
    if not callable(make_fn):
        raise RuntimeError(f"arm script {script_path} must define make_fn() -> callable")
    fn = make_fn()
    if not callable(fn):
        raise RuntimeError(f"make_fn() in {script_path} did not return a callable")
    return fn


def _measure(fn, backend: str, warmup_ms: float, rep_ms: float) -> float:
    """One do_bench measurement; returns milliseconds."""
    from tilelang.profiler import do_bench

    return float(
        do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            backend=backend,
            return_mode="median",
            early_stop_baseline=None,
        )
    )


def sign_test_paired(diffs: list) -> float:
    """Exact two-sided paired sign test p-value over non-zero diffs."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    p = 2.0 * tail / (2**n)
    return min(1.0, p)


def summarize_pairs(runs: dict, noise: float = 0.03) -> dict:
    """Adjudicate already collected, alternating-order msprof pairs."""
    if not 0 <= noise < 1:
        raise ValueError("noise must be in [0, 1)")
    if len(runs["A"]) != len(runs["B"]) or len(runs["A"]) < 3:
        raise ValueError("at least three complete pairs are required")
    if any(not math.isfinite(v) or v <= 0 for arm in ("A", "B") for v in runs[arm]):
        raise ValueError("samples must be finite and positive")
    pair_diffs = [b - a for a, b in zip(runs["A"], runs["B"])]
    med_a = statistics.median(runs["A"])
    med_b = statistics.median(runs["B"])
    rel = (med_a - med_b) / med_a if med_a else 0.0
    p_value = sign_test_paired(pair_diffs)
    # consistent direction = every pair agrees with the merged median sign
    consistent = all((d > 0) == (sum(pair_diffs) > 0) and d != 0 for d in pair_diffs) if pair_diffs else False
    # For small n the exact sign test cannot reach p<0.1 (n=3 floor is
    # 0.25), so direction consistency is the deciding criterion there
    # (documented protocol: merged median + stable pairing order); the
    # p-value gates the verdict only once n >= 5.
    significance = consistent and (len(pair_diffs) < 5 or p_value < 0.1)
    if abs(rel) < noise:
        verdict = "tie"
    elif rel > 0 and significance:
        verdict = "B_better"
    elif rel < 0 and significance:
        verdict = "A_better"
    else:
        verdict = "inconclusive"

    return {
        "a_merged_median_ms": med_a,
        "b_merged_median_ms": med_b,
        "b_rel_change": -rel,
        "sign_test_p": p_value,
        "direction_consistent": consistent,
        "noise_threshold": noise,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--a-script",
        required=True,
        help="baseline arm: python module defining make_fn() -> callable",
    )
    ap.add_argument(
        "--b-script",
        required=True,
        help="candidate arm: python module defining make_fn() -> callable",
    )
    ap.add_argument("--workdir", default="perf_opt/ab_test", help="output dir for profiles and logs")
    ap.add_argument("--pairs", type=int, default=3, help="number of A/B pairs (>= 3 recommended)")
    ap.add_argument(
        "--backend",
        default="msprof",
        choices=("msprof", "event", "wall"),
        help="do_bench backend (msprof = sole end-to-end latency authority)",
    )
    ap.add_argument("--warmup-ms", type=float, default=25.0, help="do_bench warmup (ms)")
    ap.add_argument("--rep-ms", type=float, default=100.0, help="do_bench repeat window (ms)")
    ap.add_argument(
        "--noise",
        type=float,
        default=0.03,
        help="noise threshold for the tie/adopt verdict",
    )
    args = ap.parse_args()

    if args.pairs < 3 or not 0 <= args.noise < 1:
        ap.error("--pairs must be >=3 and --noise in [0,1)")
    os.makedirs(args.workdir, exist_ok=True)
    profiles_dir = os.path.join(args.workdir, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    # Redirect msprof raw output (profiling.md: ASCEND_WORK_PATH redirection).
    os.environ["ASCEND_WORK_PATH"] = os.path.abspath(profiles_dir)

    try:
        import tilelang  # noqa: F401  (environment sanity check)
        import torch

        if args.backend == "msprof" and not torch.npu.is_available():
            print(json.dumps({"error": "torch.npu not available (CANN/torch_npu environment)"}))
            return 2
        fn_a = _load_arm(args.a_script, "A")
        fn_b = _load_arm(args.b_script, "B")
    except (ImportError, RuntimeError) as e:
        print(json.dumps({"error": str(e)}))
        return 2

    runs = {"A": [], "B": []}
    pair_diffs = []
    try:
        for i in range(args.pairs):
            # alternate which arm runs first within each pair (drift guard)
            order = ("A", "B") if i % 2 == 0 else ("B", "A")
            medians = {}
            for arm in order:
                fn = fn_a if arm == "A" else fn_b
                ms = _measure(fn, args.backend, args.warmup_ms, args.rep_ms)
                medians[arm] = ms
                runs[arm].append(ms)
                print(
                    f"pair{i + 1} {arm}: do_bench({args.backend}) = {ms:.6f} ms",
                    file=sys.stderr,
                )
            pair_diffs.append(medians["B"] - medians["A"])
    except Exception as e:  # measurement failure (msprof launch, NPU error)
        print(json.dumps({"error": f"measurement failed: {e}"}))
        return 1

    summary = summarize_pairs(runs, args.noise)
    result = {
        "protocol": "interleaved A/B multi-run on do_bench(msprof)",
        "metric": "do_bench_msprof_latency",
        "pairs": args.pairs,
        "backend": args.backend,
        "a_medians_ms": runs["A"],
        "b_medians_ms": runs["B"],
        "a_merged_median_ms": round(summary["a_merged_median_ms"], 6),
        "b_merged_median_ms": round(summary["b_merged_median_ms"], 6),
        "b_rel_change": round(summary["b_rel_change"], 4),
        "pair_diffs_ms": [round(d, 6) for d in pair_diffs],
        "sign_test_p": round(summary["sign_test_p"], 4),
        "direction_consistent": summary["direction_consistent"],
        "noise_threshold": args.noise,
        "profiler_output": profiles_dir,
        "verdict": summary["verdict"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
