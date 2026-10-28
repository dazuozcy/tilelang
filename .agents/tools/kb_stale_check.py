"""kb_stale_check.py -- toolchain-staleness detection for KB entries (E-5).

"Entries are auto-downgraded to pending re-verification after toolchain
changes" was a prose-only convention; this tool makes it mechanical. It never
modifies library files -- it only reports.

Current stamp detection (best effort, override with --stamp):
  tilelang: import tilelang and read __version__ + repo HEAD if this repo is
            a dev root (git rev-parse --short HEAD)
  CANN:     $ASCEND_HOME_PATH/compiler/version.info (Version= line) or
            $ASCEND_HOME_PATH/x86_64-linux/ascend_toolkit_install.info
            (version= line) or /usr/local/Ascend/ascend-toolkit/latest
            fallbacks
  bisheng:  the compiler binary located like tilelang.contrib.bisheng does
            ($BISHENG_HOME/bin/bisheng, then PATH); `bisheng --version`
            yields `clang version X.Y.Z (clang-<hash> ...)` — version +
            build hash form the stamp component
  npu-arch: $ASCEND_NPU_ARCH or default dav-3510 (A5/950,
            tilelang/contrib/bisheng.py default)

Classification (two-tier, since 2026-10-17 — P4):
  stale           strong signal: the entry's CANN (major.minor) or bisheng
                  (version / build hash) differs from the current toolchain.
                  These components live outside this repo, so a mismatch
                  means a genuine toolchain upgrade → re-verify before reuse.
  commit_shifted  weak signal: only the tilelang repo commit moved (every
                  repo commit used to flip all entries to "stale", which
                  drowned the pre-task re-verification checklist). Reported
                  separately as advisory; compiler source may have changed.
  fresh           all comparable components match.
  unknown         nothing comparable parsed (manual check, --include-unknown).

Usage:
  python3 .agents/tools/kb_stale_check.py [--stamp "tilelang X + CANN Y + bisheng Z"]
                                          [--include-unknown] [--json]
Exit 0 always (advisory); consumes the JSON "stale" list in conductor start /
optimizer Phase 0 as the pre-task re-verification checklist. Repro-backed
entries can be mechanically re-verified with:
  python3 .agents/tools/repro_runner.py --filter stale
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))
KB_GLOBS = [
    os.path.join(
        REPO_ROOT,
        ".agents",
        "skills",
        "tilelang-op-optimize",
        "references",
        "pattern-library",
        "*.md",
    ),
    os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md"),
]
COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
CANN_RE = re.compile(r"\bCANN\s+(\S+)", re.I)
BISHENG_RE = re.compile(r"\bbisheng\s+(\d+\.\d+\.\d+)(?:\s*\(([^)]*)\))?", re.I)


def _cann_major_minor(s: str):
    """Return (major, minor) ints from a CANN version string, else None.

    ``9.2.0`` / ``9.2`` -> (9, 2); ``9.x`` / ``dev`` -> None (unparsable,
    treated as unknown rather than mismatched).
    """
    m = re.match(r"(\d+)\.(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _detect_bisheng() -> tuple:
    """Locate the bisheng compiler and read its version identity.

    Mirrors tilelang/contrib/bisheng.py lookup order ($BISHENG_HOME/bin,
    then PATH, then the tilelang helper) and parses
    ``clang version X.Y.Z (clang-<hash> ...)`` from ``bisheng --version``.

    Returns (version, build_hash) — either may be None when unavailable.
    """
    bin_path = None
    home = os.environ.get("BISHENG_HOME", "").rstrip("/")
    for cand in (os.path.join(home, "bin", "bisheng") if home else None,
                 shutil.which("bisheng")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            bin_path = cand
            break
    if bin_path is None:
        try:
            from tilelang.contrib.bisheng import find_bisheng_path  # noqa: PLC0415

            bin_path = find_bisheng_path()
        except Exception:  # noqa: BLE001 -- advisory tool
            return None, None
    try:
        out = subprocess.run(
            [bin_path, "--version"], capture_output=True, text=True, timeout=15, check=False
        )
        text = (out.stdout or "") + (out.stderr or "")
    except Exception:  # noqa: BLE001
        return None, None
    m = re.search(r"clang\s+version\s+(\d+\.\d+\.\d+)(?:\s*\(([^)]*)\))?", text)
    if not m:
        return None, None
    build_hash = None
    if m.group(2):
        hm = re.search(r"\b([0-9a-f]{7,40})\b", m.group(2))
        build_hash = hm.group(1) if hm else None
    return m.group(1), build_hash


def detect_current_stamp() -> dict:
    stamp = {
        "tilelang": None,
        "tilelang_commit": None,
        "cann": None,
        "bisheng": None,
        "bisheng_hash": None,
        "npu_arch": None,
        "raw": "",
    }
    try:
        import io
        import contextlib

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            import tilelang  # noqa: PLC0415
        stamp["tilelang"] = getattr(tilelang, "__version__", None)
    except Exception:  # noqa: BLE001 -- advisory tool
        pass
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode == 0:
            stamp["tilelang_commit"] = head.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    home = os.environ.get("ASCEND_HOME_PATH", "").rstrip("/")
    for cand in (
        home + "/compiler/version.cfg",
        home + "/compiler/version.info",
        home + "/x86_64-linux/ascend_toolkit_install.info",
        home + "/../version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info",
        "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    ):
        try:
            with open(cand) as f:
                text = f.read()
            m = re.search(r"^[Vv]ersion\s*=\s*(\S+)", text, re.M)
            if m:
                stamp["cann"] = m.group(1)
                break
        except OSError:
            continue
    # bisheng compiler version identity (outside this repo -> strong signal)
    stamp["bisheng"], stamp["bisheng_hash"] = _detect_bisheng()
    # npu-arch: env override or dav-3510 default (bisheng.py:92-94)
    stamp["npu_arch"] = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
    bisheng_part = stamp["bisheng"] or "?"
    if stamp["bisheng_hash"]:
        bisheng_part += f" ({stamp['bisheng_hash']})"
    stamp["raw"] = "tilelang {} (commit {}) + CANN {} + bisheng {} + {}".format(
        stamp["tilelang"] or "?",
        stamp["tilelang_commit"] or "?",
        stamp["cann"] or "?",
        bisheng_part,
        stamp["npu_arch"],
    )
    return stamp


def parse_entries(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, REPO_ROOT)
    entries = []
    parts = re.split(r"(?m)^---\s*$", text)
    for i in range(1, len(parts) - 1, 2):
        fm_raw = parts[i]
        m = re.search(r"^toolchain:\s*(.+)$", fm_raw, re.M)
        mid = re.search(r"^id:\s*(\S+)", fm_raw, re.M)
        if m and mid:
            entries.append(
                {"id": mid.group(1), "toolchain": m.group(1).strip(), "path": rel}
            )
    # queue.md VP entries carry '- toolchain_stamp:'
    for m in re.finditer(r"(?m)^## (VP-\d{4}-\d{4})\s*$([\s\S]*?)(?=^## |\Z)", text):
        mm = re.search(r"^- toolchain_stamp:\s*(.+)$", m.group(2), re.M)
        if mm:
            entries.append(
                {"id": m.group(1), "toolchain": mm.group(1).strip(), "path": rel}
            )
    return entries


def classify(entry_stamp: str, current: dict) -> str:
    """Two-tier staleness classification (see module docstring).

    Priority: stale (CANN/bisheng toolchain mismatch) > commit_shifted
    (tilelang repo commit moved only) > fresh > unknown.
    """
    e = entry_stamp or ""

    # --- strong: toolchain components outside this repo -------------------
    cann_stale = cann_known = False
    m = CANN_RE.search(e)
    if m and current.get("cann"):
        a = _cann_major_minor(m.group(1))
        b = _cann_major_minor(current["cann"])
        if a and b:
            cann_known = True
            cann_stale = a != b
    bish_stale = bish_known = False
    m = BISHENG_RE.search(e)
    if m and current.get("bisheng"):
        bish_known = True
        if m.group(1) != current["bisheng"]:
            bish_stale = True
        elif current.get("bisheng_hash") and m.group(2):
            hm = re.search(r"\b([0-9a-f]{7,40})\b", m.group(2))
            if hm and hm.group(1) != current["bisheng_hash"]:
                bish_stale = True
    if cann_stale or bish_stale:
        return "stale"

    # --- weak: tilelang repo commit (moves on every commit) ---------------
    e_commits = COMMIT_RE.findall(e)
    if e_commits:
        cur = current.get("tilelang_commit")
        if (
            cur
            and cur not in e_commits
            and not any(c.startswith(cur) or cur.startswith(c) for c in e_commits)
        ):
            return "commit_shifted"
        return "fresh"

    # --- no commit: fresh only when a toolchain component confirmed equal --
    if cann_known or bish_known:
        return "fresh"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stamp",
        default=None,
        help="override current stamp, e.g. 'tilelang 0.1.2+abcdef123 + CANN 8.5.0'",
    )
    ap.add_argument("--include-unknown", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.stamp:
        cann_m = CANN_RE.search(args.stamp)
        bish_m = BISHENG_RE.search(args.stamp)
        bish_hash = None
        if bish_m and bish_m.group(2):
            hm = re.search(r"\b([0-9a-f]{7,40})\b", bish_m.group(2))
            bish_hash = hm.group(1) if hm else None
        current = {
            "raw": args.stamp,
            "tilelang_commit": (COMMIT_RE.findall(args.stamp) or [None])[0],
            "cann": cann_m.group(1) if cann_m else None,
            "bisheng": bish_m.group(1) if bish_m else None,
            "bisheng_hash": bish_hash,
            "tilelang": None,
        }
    else:
        current = detect_current_stamp()

    stale, commit_shifted, unknown = [], [], []
    for pattern in KB_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            for e in parse_entries(path):
                verdict = classify(e["toolchain"], current)
                if verdict == "stale":
                    stale.append({**e, "verdict": "stale"})
                elif verdict == "commit_shifted":
                    commit_shifted.append({**e, "verdict": "commit_shifted"})
                elif verdict == "unknown":
                    unknown.append({**e, "verdict": "unknown"})

    report = {
        "current_stamp": current["raw"],
        "checked": "pattern-library topic files + queue.md",
        "stale_count": len(stale),
        "stale": stale,
        "commit_shifted_count": len(commit_shifted),
        "commit_shifted": commit_shifted,
        "unknown_count": len(unknown),
        "note": "stale = CANN/bisheng toolchain version differs (re-verify "
                "before reuse; repro-backed entries: repro_runner.py --filter "
                "stale); commit_shifted = only the tilelang repo commit moved "
                "(advisory); unknown = nothing comparable parsed "
                "(--include-unknown to list)",
    }
    if args.include_unknown:
        report["unknown"] = unknown
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"current stamp: {current['raw']}")
        print(f"stale entries (toolchain mismatch): {len(stale)}")
        for e in stale:
            print(f"  [stale] {e['id']}  ({e['path']})  stamp: {e['toolchain'][:80]}")
        print(
            f"commit-shifted entries (tilelang commit moved, advisory): "
            f"{len(commit_shifted)}"
        )
        for e in commit_shifted:
            print(f"  [commit_shifted] {e['id']}  ({e['path']})")
        print(
            f"unknown (unparsable stamp, manual check): {len(unknown)} "
            f"(--include-unknown to list)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
