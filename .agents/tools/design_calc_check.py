"""design_calc_check.py -- mechanical arithmetic checks on DESIGN.md (D-5).

Best-effort mechanical re-computation of the machine-checkable parts of a
Stage 1 design document. Output is an advisory report for the Stage 2
reviewer (dimension 2/3/8 support) -- it is NOT a gate: any check that
cannot be parsed from the document is reported as "skip" with the reason.

Checks:
  1. ub_budget   -- sum the buffer sizes declared in the memory plan section
                    (mixed B/KB/MB units are normalized) and compare against
                    the per-level on-chip capacity (A5 assumption values,
                    pattern-library/constants.md CONST-capacity-a5, applied
                    with the 90% assumption discount until measured).
  2. l0c_budget  -- for Cube-path designs, block_M * block_N * accum_bytes
                    must fit the L0C budget.
  3. core_split  -- recompute the 1-D grid size ceil(M/bm) * ceil(N/bn) from
                    the numbers the design declares and cross-check the
                    declared block/grid count (A5: T.Kernel(N) is 1-D only).
  4. r3_metrics  -- the research complexity table must cover the four
                    required metrics (FLOPs / access bytes / scan passes /
                    intermediate buffer peak).

Capacity constants are configurable (A5 values are assumption-grade):
  --constants '{"ub_kb": 248, "simt_ub_kb": 216, "l1_kb": 512, ...}'
  or env DESIGN_CALC_CONSTANTS (same JSON). Defaults follow
  pattern-library/constants.md CONST-capacity-a5 (evidence:
  src/ascend/transform/auto_schedule.cc:106-140).

Usage:
  python3 .agents/tools/design_calc_check.py --design examples/{p}/{op}/DESIGN.md
Exit code 0 always (advisory); parse the JSON "checks" list.
"""

import argparse
import json
import math
import os
import re

# A5 assumption-grade defaults (CONST-capacity-a5; 90% discount applied in
# check_ub_budget until the entries turn `measured`).
DEFAULT_CONSTANTS = {
    "ub_kb": 248.0,        # physical UnifiedBuffer (auto_schedule.cc)
    "simt_ub_kb": 216.0,   # 248KB - 32KB SIMT_VF thread context
    "l1_kb": 512.0,        # CBuf
    "l0c_kb": 256.0,
    "l0a_kb": 64.0,
    "l0b_kb": 64.0,
    "assumption_discount": 0.9,
    "status": "assumption",
}
CONSTANTS_SOURCE = "pattern-library/constants.md CONST-capacity-a5"
UNIT_KB = {"b": 1.0 / 1024, "kb": 1.0, "mb": 1024.0}


def _load_constants(override_json: str | None) -> dict:
    constants = dict(DEFAULT_CONSTANTS)
    raw = override_json or os.environ.get("DESIGN_CALC_CONSTANTS")
    if raw:
        try:
            constants.update(json.loads(raw))
        except json.JSONDecodeError:
            pass  # keep defaults; malformed override is advisory
    return constants


def _section(text: str, marker: str) -> str:
    """Return the body of the first heading containing the marker."""
    lines = text.splitlines()
    body, inside, depth = [], False, 0
    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            if inside and len(m.group(1)) <= depth:
                break
            if marker in m.group(2):
                inside, depth = True, len(m.group(1))
                continue
        if inside:
            body.append(ln)
    return "\n".join(body)


SIZE_CHAIN_RE = re.compile(r"((?:\d+(?:\.\d+)?\s*\+\s*)+\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
TOTAL_MARK = re.compile(r"合计|总占用|total", re.I)
CAPACITY_MARK = re.compile(r"容量|佐证|参考|≤")
# SimtVF usage inside the design text -> the tighter 216KB budget applies.
SIMTVF_RE = re.compile(r"T\.SimtVF|SimtVF\(")


def _parse_sizes(s: str) -> list:
    """Parse size expressions in a string; chains like '16 + 16 + 8KB'
    count as one summed entry (mixed byte-unit trap)."""
    out = []
    covered = []
    for m in SIZE_CHAIN_RE.finditer(s):
        total = sum(float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1)))
        out.append(total * UNIT_KB[m.group(2).lower()])
        covered.append(m.span())
    for m in SIZE_RE.finditer(s):
        if any(a <= m.start() < b for a, b in covered):
            continue
        out.append(float(m.group(1)) * UNIT_KB[m.group(2).lower()])
    return out


def check_ub_budget(text: str, constants: dict) -> dict | list:
    """Sum per-level buffer tables in the memory plan and compare against
    the discounted level capacity; recompute any declared 合计 row."""
    discount = float(constants.get("assumption_discount", 0.9))
    level_capacity_kb = {
        "UB": float(constants["ub_kb"]),
        "L1": float(constants["l1_kb"]),
        "L0C": float(constants["l0c_kb"]),
        "L0A": float(constants["l0a_kb"]),
        "L0B": float(constants["l0b_kb"]),
    }
    simt = bool(SIMTVF_RE.search(text))
    if simt:
        level_capacity_kb["UB"] = float(
            constants.get("simt_ub_kb", constants["ub_kb"])
        )
    sec = _section(text, "4.5") or _section(text, "内存") or ""
    level = None
    buckets = {}
    declared_totals = {}
    for ln in sec.splitlines():
        m = re.search(r"\*\*\s*(UB|L1|L0C|L0A|L0B)\b", ln) or re.search(
            r"^#{2,6}.*\b(UB|L1|L0C)\b", ln
        )
        if m:
            level = m.group(1).upper()
            buckets.setdefault(level, [])
            continue
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        joined = " ".join(cells)
        if set(joined.replace(" ", "")) <= {"-", " "}:
            continue
        if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
            continue  # header rows
        if level is None:
            continue
        if TOTAL_MARK.search(joined) and not CAPACITY_MARK.search(joined):
            vals = _parse_sizes(joined)
            if vals:
                declared_totals[level] = sum(vals)
            continue
        vals = _parse_sizes(joined)
        if vals:
            buckets[level].extend(vals)
    checks = []
    for lv, sizes in buckets.items():
        if not sizes:
            continue
        total = sum(sizes)
        cap = level_capacity_kb.get(lv)
        budget = cap * discount if constants.get("status") == "assumption" else cap
        detail = (
            f"{lv}: {len(sizes)} buffer entries sum = {total:.1f} KB"
            + (
                f" (declared 合计 {declared_totals.get(lv):.1f} KB, "
                f"recompute matches: "
                f"{abs(declared_totals.get(lv, total) - total) < 1.0})"
                if lv in declared_totals
                else ""
            )
            + f"; capacity = {cap:.0f} KB"
            + (f" (SimtVF thread-context budget {constants['simt_ub_kb']:.0f} KB)" if lv == "UB" and simt else "")
            + (
                f"; assumption discount x{discount} -> budget {budget:.0f} KB"
                if budget != cap
                else ""
            )
            + "; NOTE: AutoSchedule multibuffering adds buffer versions on "
            "top of the declared shapes (tilelang-ascend skill §4.3) -- A5 "
            "inflation factor not yet measured"
        )
        status = "fail" if budget and total > budget else "pass"
        checks.append(
            {
                "check": f"{lv.lower()}_budget",
                "status": status,
                "detail": detail,
            }
        )
    if not checks:
        return {
            "check": "ub_budget",
            "status": "skip",
            "detail": "no per-level buffer tables parsed from the memory "
            "plan (advisory checker; verify manually)",
        }
    return checks


# Cube-path affirmative API tokens (A5): T.gemm / l1-targeting T.copy /
# l0 allocations / MixedKernel. Negations like "无 matmul" don't count.
CUBE_PATH_RE = re.compile(
    r"T\.gemm|alloc_l0[abc]|MixedKernel|T\.copy\([^)]*l1|T\.dual_copy"
)


def check_l0c_budget(text: str, constants: dict) -> dict:
    # L0C only exists on the Cube path: pure Vector/elementwise designs have
    # no GEMM tiles, and any MxN pair we parse there is a tensor shape.
    if not CUBE_PATH_RE.search(text):
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": "no affirmative Cube-path API (T.gemm/alloc_l0*/"
            "MixedKernel/T.copy->l1/T.dual_copy) in this design",
        }
    sec = _section(text, "Tiling") or text
    bm = bn = None
    m = re.search(r"(?:block_?[MN]|bm)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bm = int(m.group(1))
    m = re.search(r"(?:block_?[NM]|bn)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bn = int(m.group(1))
    if bm is None or bn is None:
        # try "bm x bn" adjacent pattern
        m = re.search(
            r"\b(\d{2,4})\s*[x×*]\s*(\d{2,4})\b.*?(?:L0C|累加|accum|fp32)", text
        )
        if m:
            bm, bn = int(m.group(1)), int(m.group(2))
    if bm is None or bn is None:
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": "no block_M x block_N values parsed",
        }
    accum_bytes = 4  # fp32 accumulation is the design default
    need_kb = bm * bn * accum_bytes / 1024.0
    cap = float(constants["l0c_kb"])
    discount = float(constants.get("assumption_discount", 0.9))
    budget = cap * discount if constants.get("status") == "assumption" else cap
    ok = need_kb <= budget
    align_ok = bm % 16 == 0 and bn % 16 == 0  # CONST-gemm-dim-constraints
    detail = (
        f"block {bm}x{bn} x fp32 accum = {need_kb:.1f} KB vs "
        f"L0C {cap:.0f} KB"
        + (f" (assumption budget {budget:.0f} KB)" if budget != cap else "")
        + ("" if align_ok else "; WARN: block_M/block_N not multiples of 16 "
            "(C0 fractal alignment, CONST-gemm-dim-constraints)")
    )
    status = "pass" if ok and align_ok else ("fail" if not ok else "warn")
    return {
        "check": "l0c_budget",
        "status": status,
        "detail": detail,
    }


def check_core_split(text: str) -> dict:
    """A5: kernel launch is a 1-D grid (T.Kernel(N)); recompute the block
    count from the tiling and cross-check the declared grid size."""
    sec = _section(text, "Tiling") or _section(text, "5") or text
    nums = {}
    for key, pats in (
        ("M", [r"\bM\s*=\s*(\d+)", r"seq.*?(\d{3,6})\b"]),
        ("N", [r"\bN\s*=\s*(\d+)"]),
        ("bm", [r"\b(?:bm|block_?m)\s*=\s*(\d+)"]),
        ("bn", [r"\b(?:bn|block_?n)\s*=\s*(\d+)"]),
    ):
        for p in pats:
            m = re.search(p, sec, re.I)
            if m:
                nums[key] = int(m.group(1))
                break
    declared = None
    m = (
        re.search(r"(?:num_blocks?|grid\s*(?:大小|size)?)\s*[=＝:：]?\s*(\d+)", sec)
        or re.search(r"T\.Kernel\(\s*(\d+)\s*\)", sec)
        or re.search(r"grid\s*[=＝:：]\s*(\d+)\s*(?:个|blocks?)", sec)
    )
    if m:
        declared = int(m.group(1))
    if not all(k in nums for k in ("M", "bm")):
        return {
            "check": "core_split",
            "status": "skip",
            "detail": f"insufficient parsed dims {nums}; declared grid "
            f"size = {declared}",
        }
    n = nums.get("N", 1)
    bn = nums.get("bn", 1)
    blocks = math.ceil(nums["M"] / nums["bm"]) * math.ceil(n / bn)
    if declared is None:
        return {
            "check": "core_split",
            "status": "pass",
            "detail": f"recomputed 1-D grid = ceil({nums['M']}/"
            f"{nums['bm']}) x ceil({n}/{bn}) = {blocks} blocks "
            f"(design declares no explicit count)",
        }
    ok = declared == blocks
    return {
        "check": "core_split",
        "status": "pass" if ok else "fail",
        "detail": f"recomputed ceil({nums['M']}/{nums['bm']}) x "
        f"ceil({n}/{bn}) = {blocks} blocks vs declared grid {declared} "
        f"(1-D T.Kernel(N))",
    }


def check_r3_metrics(text: str) -> dict:
    sec = _section(text, "1.6.0") or text
    metrics = {
        "FLOPs": bool(re.search(r"FLOPs?|浮点", sec, re.I)),
        "access_bytes": bool(re.search(r"访存|Bytes|字节", sec)),
        "scan_passes": bool(re.search(r"扫描遍数|遍数|扫描次数", sec)),
        "buffer_peak": bool(re.search(r"中间缓冲|缓冲峰值|buffer", sec, re.I)),
    }
    missing = [k for k, v in metrics.items() if not v]
    if missing:
        return {
            "check": "r3_metrics",
            "status": "fail",
            "detail": f"complexity table missing metrics: {missing}",
        }
    return {
        "check": "r3_metrics",
        "status": "pass",
        "detail": "all four R3 metrics present "
        "(FLOPs/access bytes/scan passes/buffer peak)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True, help="path to DESIGN.md")
    ap.add_argument(
        "--constants",
        default=None,
        help="JSON overriding capacity constants (defaults: A5 assumption "
        "values from pattern-library/constants.md CONST-capacity-a5)",
    )
    args = ap.parse_args()
    constants = _load_constants(args.constants)
    try:
        with open(args.design, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(json.dumps({"error": str(e)}))
        return 0
    ub = check_ub_budget(text, constants)
    if not isinstance(ub, list):
        ub = [ub]
    report = {
        "design": args.design,
        "capacity_constants": {
            "ub_kb": constants["ub_kb"],
            "simt_ub_kb": constants.get("simt_ub_kb"),
            "l1_kb": constants["l1_kb"],
            "l0c_kb": constants["l0c_kb"],
            "l0a_kb": constants["l0a_kb"],
            "l0b_kb": constants["l0b_kb"],
            "assumption_discount": constants.get("assumption_discount"),
            "status": constants.get("status"),
            "source": CONSTANTS_SOURCE,
        },
        "checks": [
            *ub,
            check_l0c_budget(text, constants),
            check_core_split(text),
            check_r3_metrics(text),
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
