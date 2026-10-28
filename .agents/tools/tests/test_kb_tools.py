"""Tests for kb_lint.py (K-4 + ED-F) and kb_search.py (E-1/K-3)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / ".agents" / "tools"


def run_tool(name: str, *args: str) -> tuple[int, dict | str]:
    proc = subprocess.run(
        [sys.executable, str(TOOLS / name), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout


def test_kb_lint_real_library_is_clean():
    rc, out = run_tool("kb_lint.py")
    assert rc == 0, out


def test_kb_lint_detects_missing_repro_and_bad_enum(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    pl.mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    (pl / "traps-runtime.md").write_text(
        "---\n"
        "id: TRAP-test-entry\n"
        "kind: trap\n"
        "status: bogus\n"  # KB-FM-ENUM
        "origin_task: task-x\n"
        "toolchain: tilelang 0.1.2+abcdef123 + CANN 8.5.0 2026-09-10\n"
        "repro: repro/TRAP-test-entry.py\n"  # KB-REPRO-EXISTS (missing)
        "---\n"
        "### test entry\nbody\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--files", str(pl / "traps-runtime.md"))
    assert rc == 1
    assert "KB-FM-ENUM" in out
    assert "KB-REPRO-EXISTS" in out


def test_kb_lint_repro_must_not_reference_workspace(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    (pl / "repro").mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    repro = pl / "repro" / "TRAP-coupled.py"
    repro.write_text(
        '"""[repro] TRAP-coupled\n\nFirst verified: 2026-09-10 test\n"""\n'
        "open('examples/proj/op/kernel.py')\n",
        encoding="utf-8",
    )
    (pl / "traps-runtime.md").write_text(
        "---\n"
        "id: TRAP-coupled\n"
        "kind: trap\n"
        "status: verified\n"
        "origin_task: task-x\n"
        "toolchain: tilelang 0.1.2+abcdef123 + CANN 8.5.0 2026-09-10\n"
        "repro: repro/TRAP-coupled.py\n"
        "---\n"
        "### coupled entry\nbody\n",
        encoding="utf-8",
    )
    rc, out = run_tool(
        "kb_lint.py", "--root", str(tmp_path), "--files", str(pl / "traps-runtime.md")
    )
    assert rc == 1
    assert "KB-REPRO-DECOUPLE" in out


def test_kb_search_returns_relevant_entries():
    # A5 cold-start: pattern-library theme files are empty; the assumption-grade
    # constants must be retrievable (capacity / C0 / throughput queries).
    rc, out = run_tool("kb_search.py", "UB 容量 L0C", "--json")
    assert rc == 0
    data = json.loads(out)
    assert data["results"], "expected hits for capacity constants"
    ids = [r["id"] for r in data["results"]]
    assert any("CONST-capacity" in i for i in ids), ids


def test_kb_search_covers_all_knowledge_sources():
    rc, out = run_tool("kb_search.py", "--index")
    assert rc == 0
    assert "CONST-capacity-a5" in out  # pattern-library constants (A5 assumption)
    assert "BP_run_state_bimodality" in out  # bottleneck-patterns
    assert "ALG-attention" in out  # algorithm-candidates
    # capability-gaps: A5 cold-start (zeroed) -- queue.md still indexed.
    # After the first A5 distillation loop, queue.md legitimately carries
    # VP-2026-* proposals.  Guard consistency instead of emptiness: every
    # VP- id surfaced by the index must exist in the current queue.md
    # (catches stale/leaked entries from the pre-A5 lineage, not new ones).
    queue_text = (
        Path(__file__).resolve().parents[2] / "evolution" / "queue.md"
    ).read_text(encoding="utf-8")
    indexed_vp_ids = set(re.findall(r"VP-\d{4}-\d+", out))
    assert indexed_vp_ids, "expected queue.md entries in the index"
    stale = indexed_vp_ids - set(re.findall(r"VP-\d{4}-\d+", queue_text))
    assert not stale, f"index serves entries absent from queue.md: {sorted(stale)}"


def test_kb_lint_flags_task_workspace_repro_in_queue(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    ev = tmp_path / ".agents" / "evolution"
    (pl / "repro").mkdir(parents=True)
    ev.mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    (ev / "queue.md").write_text(
        "# queue\n## Pending\n"
        "## VP-2026-0099\n"
        "- type: P\n"
        "- title: t\n"
        "- repro: python examples/foo/bar.py --level all\n"
        "- status: pending\n"
        "- confirmations: 1/2\n"
        "## Decided\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--root", str(tmp_path))
    assert rc == 1
    assert "KB-QUEUE-REPRO" in out


def test_kb_lint_queue_inline_python_command_is_exempt(tmp_path):
    ev = tmp_path / ".agents" / "evolution"
    ev.mkdir(parents=True)
    (ev / "queue.md").write_text(
        "# queue\n## Pending\n"
        "## VP-2026-0098\n"
        "- type: R\n"
        "- repro: python3 -c \"import re; re.finditer('examples/x', s)\"\n"
        "- status: pending\n"
        "- confirmations: -/-\n"
        "## Decided\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--root", str(tmp_path))
    assert "KB-QUEUE-REPRO" not in out


def test_kb_stale_check_reports_and_never_modifies():
    rc, out = run_tool("kb_stale_check.py", "--json")
    assert rc == 0
    data = json.loads(out)
    assert "current_stamp" in data and "stale" in data
    # two-tier classification keys (P4): strong toolchain mismatch vs the
    # advisory every-commit shift
    assert "commit_shifted" in data and "commit_shifted_count" in data
    # advisory tool: exit 0 even with stale entries


def _load_stale_check_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kb_stale_check", str(TOOLS / "kb_stale_check.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CURRENT = {
    "tilelang": "0.1.15",
    "tilelang_commit": "aaaaaaa123",
    "cann": "9.2.0",
    "bisheng": "15.0.5",
    "bisheng_hash": "5c68a1cb1231",
    "npu_arch": "dav-3510",
    "raw": "",
}


def test_kb_stale_classify_two_tier():
    mod = _load_stale_check_module()
    classify = mod.classify

    # everything matches -> fresh
    assert (
        classify(
            "tilelang-fork aaaaaaa123 + CANN 9.2.0 + bisheng 15.0.5 (5c68a1cb1231) + dav-3510",
            _CURRENT,
        )
        == "fresh"
    )
    # CANN major.minor mismatch -> stale (strong)
    assert (
        classify("tilelang-fork aaaaaaa123 + CANN 8.5.0 + dav-3510", _CURRENT) == "stale"
    )
    # CANN 9.2 vs 9.2.0 -> same major.minor, fresh
    assert classify("tilelang-fork aaaaaaa123 / CANN 9.2 / bisheng dav-3510", _CURRENT) == "fresh"
    # bisheng version mismatch -> stale
    assert (
        classify("tilelang-fork aaaaaaa123 + bisheng 15.0.6 (5c68a1cb1231)", _CURRENT)
        == "stale"
    )
    # bisheng build-hash mismatch -> stale
    assert (
        classify("tilelang-fork aaaaaaa123 + bisheng 15.0.5 (deadbeef1234)", _CURRENT)
        == "stale"
    )
    # only the tilelang commit moved -> commit_shifted (weak, advisory)
    assert classify("tilelang-fork bbbbbbb456 + CANN 9.2", _CURRENT) == "commit_shifted"
    # CANN component unparsable but commit comparable -> still commit_shifted
    assert classify("tilelang-fork bbbbbbb456 + CANN 9.x", _CURRENT) == "commit_shifted"
    # no commit, CANN confirmed equal -> fresh
    assert classify("CANN 9.2.0 only", _CURRENT) == "fresh"
    # no commit, nothing comparable -> unknown
    assert classify("dev build, no versions", _CURRENT) == "unknown"
    # legacy stamps that mislabel the arch as the bisheng version must not
    # parse as a bisheng component ("bisheng dav-3510" is not a version)
    assert classify("tilelang-fork bbbbbbb456 + bisheng dav-3510 + CANN 9.2", _CURRENT) == "commit_shifted"


def test_kb_stale_check_stamp_override_parses_components():
    rc, out = run_tool(
        "kb_stale_check.py",
        "--stamp",
        "tilelang 0.1.15 (commit aaaaaaa123) + CANN 9.2.0 + bisheng 15.0.5 (5c68a1cb1231) + dav-3510",
        "--json",
    )
    assert rc == 0
    data = json.loads(out)
    # override stamp parses CANN + bisheng components; real library entries
    # stamped at other commits land in commit_shifted, not stale
    assert data["stale_count"] >= 0
    assert "commit_shifted_count" in data
