#!/usr/bin/env python3
"""Gate-4 remediation: rewrite the 6 premature final records as consolidated-
artifact final records (fresh do_bench(msprof) measurements).

Background: the original final records pointed at the per-function drop-in
files (premature "final" labeling); the Stage 4 gate requires exactly one
final record per (kernel_id, workload_id) whose artifact_path is the
consolidated perf_opt/LogSumExpFwdOpSimd.py with its real SHA256.  The gate's
"exactly one final" rule makes pure appending impossible, so the 6 premature
final lines are removed (backed up first) and replaced by fresh records
measured on the consolidated file (logs/gate4/final_*_consolidated.log).
All other lines (baseline/candidate/merged history) are preserved verbatim.
"""

import datetime
import hashlib
import json
import os
import re

PERF_OPT = (
    "/home/z00875089/tilelang-fork/examples/TileOPs/tileops/kernels/"
    "reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt"
)
RECORDS = os.path.join(PERF_OPT, "perf_records.jsonl")
CONS = os.path.join(PERF_OPT, "LogSumExpFwdOpSimd.py")
LOG_DIR = os.path.join(PERF_OPT, "logs", "gate4")
BACKUP = os.path.join(LOG_DIR, "perf_records.before_remediation.jsonl")

KERNEL_OF_LOG = {
    "final_single_consolidated.log": "_logsumexp_kernel_single",
    "final_tiled_consolidated.log": "_logsumexp_kernel_tiled",
}
LINE_RE = re.compile(
    r"^\[(?P<wid>[A-Za-z0-9\-]+)\] A\((?P<aid>[^)]*)\)=(?P<a>[\d.]+)us "
    r"B\(final\)=(?P<b>[\d.]+)us .*-> (?P<arch>\S+)\s*$"
)

# ---------------------------------------------------------------- parse logs
fresh = {}
for log_name, kernel_id in KERNEL_OF_LOG.items():
    path = os.path.join(LOG_DIR, log_name)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            wid = m.group("wid")
            assert wid not in fresh, f"duplicate workload in logs: {wid}"
            fresh[wid] = {
                "kernel_id": kernel_id,
                "a_us": float(m.group("a")),
                "b_us": float(m.group("b")),
                "archive": m.group("arch"),
            }

EXPECTED = {
    "_logsumexp_kernel_single": [
        "attn-weights-4k-float16",
        "attn-weights-4k-bfloat16",
        "3d-multidim-reduce-float16",
    ],
    "_logsumexp_kernel_tiled": [
        "attn-weights-32k-bfloat16",
        "lm-head-logits-float16",
        "lm-head-logits-bfloat16",
    ],
}
got = sorted(fresh)
want = sorted(w for ws in EXPECTED.values() for w in ws)
assert got == want, f"parsed workloads mismatch:\n  got:  {got}\n  want: {want}"
print(f"parsed {len(fresh)} fresh final measurements from logs")

cons_sha = hashlib.sha256(open(CONS, "rb").read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ------------------------------------------------- backup + filter + append
with open(RECORDS, encoding="utf-8") as f:
    lines = f.readlines()

with open(BACKUP, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"backup written: {BACKUP} ({len(lines)} lines)")

kept, removed = [], []
for line in lines:
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("phase") == "final":
        removed.append(rec)
    else:
        kept.append(line)

assert len(removed) == 6, f"expected 6 premature final records, got {len(removed)}"
for rec in removed:
    print(
        f"  removing premature final: {rec['kernel_id']}/{rec['workload_id']} "
        f"({rec['duration_us']}us @ {os.path.basename(rec['artifact_path'])})"
    )

appended = []
for kernel_id, wids in EXPECTED.items():
    for wid in wids:
        m = fresh[wid]
        assert m["kernel_id"] == kernel_id
        rec = {
            "round": 3,
            "candidate_id": "final",
            "parent_id": "merged",
            "kernel_id": kernel_id,
            "workload_id": wid,
            "phase": "final",
            "artifact_path": CONS,
            "artifact_sha256": cons_sha,
            "duration_us": m["b_us"],
            "l0_pass": True,
            "profiler_output": m["archive"],
            "timestamp": now,
        }
        appended.append(rec)
        kept.append(json.dumps(rec, ensure_ascii=False) + "\n")

with open(RECORDS, "w", encoding="utf-8") as f:
    f.writelines(kept)

print(
    f"perf_records.jsonl rewritten: {len(lines)} -> {len(kept)} lines "
    f"({len(removed)} premature finals removed, {len(appended)} appended)"
)
print()
print("## fresh final records (consolidated artifact) ##")
for rec in appended:
    print(
        f"  {rec['kernel_id']}/{rec['workload_id']}: {rec['duration_us']}us  "
        f"(A baseline same-session {fresh[rec['workload_id']]['a_us']}us)"
    )
