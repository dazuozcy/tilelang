"""CPU-only Stage 0 validation, including missing helpers and non-execution."""

import json
from pathlib import Path
import sys
import subprocess

import pytest
import yaml

from test_statectl import _make_harness_repo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage0_check


def scaffold(tmp_path):
    repo, _, meta_path = _make_harness_repo(tmp_path)
    root = repo / "examples/TileOPs"
    meta = json.loads(meta_path.read_text())
    for rel in [
        meta["wrapper_path"],
        "workloads/reduction.py",
        "src/tileops/_ops/reduction/mop.py",
        "src/tileops/_ops/reduction/__init__.py",
        "src/tileops/_kernels/reduction/__init__.py",
        "src/tileops/_kernels/reduction/mop/__init__.py",
        "src/tileops/reduction.py",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('must not execute target code')\n")
    (root / meta["extracted_file"]).write_text("def funcA(x):\n    return x\ndef funcB(x):\n    return funcA(x)\n")
    manifest = root / "src/tileops/manifest/spec/reduction.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "MopOp": {
                    "workloads": [{"shape": [16]}],
                    "source": {
                        "kernel": meta["wrapper_path"],
                        "test": meta["test_path"],
                        "bench": meta["bench_path"],
                    },
                }
            }
        )
    )
    return repo, meta_path, meta


def test_static_check_does_not_execute_target(tmp_path):
    repo, meta_path, _ = scaffold(tmp_path)
    result = stage0_check.check(meta_path, repo, lint=False)
    assert result["failures"] == []
    assert result["scope"] == "static_only"
    assert result["runtime"] == "unverified"
    assert result["checks"]["ruff"] == "skipped"


@pytest.mark.parametrize(
    "defect,rule",
    [
        ("helper", "S0-HELPERS"),
        ("function", "S0-FUNCTION"),
        ("syntax", "S0-AST"),
        ("manifest", "S0-MANIFEST-SOURCE"),
    ],
)
def test_static_defects_are_rejected(tmp_path, defect, rule):
    repo, meta_path, meta = scaffold(tmp_path)
    extracted = repo / "examples/TileOPs" / meta["extracted_file"]
    if defect == "helper":
        extracted.write_text("def funcA(x):\n    return missing_helper(x)\ndef funcB(x):\n    return x\n")
    elif defect == "function":
        extracted.write_text("def funcA(x):\n    return x\n")
    elif defect == "syntax":
        extracted.write_text("def funcA(:\n")
    else:
        manifest = repo / "examples/TileOPs/src/tileops/manifest/spec/reduction.yaml"
        data = yaml.safe_load(manifest.read_text())
        data["MopOp"]["source"]["bench"] = "wrong.py"
        manifest.write_text(yaml.safe_dump(data))
    assert rule in {f["rule_id"] for f in stage0_check.check(meta_path, repo, lint=False)["failures"]}


def test_scope_analysis_accepts_closure_and_local_callable():
    source = """
import math
FACTOR = 2
def outer(fn, xs):
    def inner(x):
        return fn(math.sqrt(x)) * FACTOR
    return [inner(x) for x in xs]
"""
    assert stage0_check.unresolved_globals(source, "example.py") == []


def test_invalid_meta_is_structured_failure(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("{broken")
    assert stage0_check.check(path, tmp_path, lint=False)["failures"][0]["rule_id"] == "S0-META"


def test_cli_writes_static_report_without_changing_state(tmp_path):
    repo, meta_path, _ = scaffold(tmp_path)
    before = meta_path.read_bytes()
    output = tmp_path / "checks.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(stage0_check.__file__)),
            "--meta",
            str(meta_path),
            "--repo-root",
            str(repo),
            "--skip-lint",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["verdict"] == "static_pass"
    assert report["runtime"] == "unverified"
    full_report = json.loads(output.read_text())
    assert all(full_report[key] == value for key, value in report.items())
    assert full_report["artifact_hashes"]
    assert "artifact_hashes" not in report
    assert meta_path.read_bytes() == before


def test_checker_catches_invalid_compile_context(tmp_path):
    repo, meta_path, meta = scaffold(tmp_path)
    (repo / "examples/TileOPs" / meta["wrapper_path"]).write_text("return 1\n")
    assert "S0-AST" in {f["rule_id"] for f in stage0_check.check(meta_path, repo, lint=False)["failures"]}


def test_prompt_renderer_covers_each_source_function(tmp_path):
    repo, meta_path, meta = scaffold(tmp_path)
    output = stage0_check.emit_prompts(meta_path, repo)
    text = output.read_text()
    assert text.count("编程模式由 Stage 1") == len(meta["extracted_functions"])
    for name in meta["extracted_functions"]:
        assert name in text
        assert str(Path(meta["gpu_repo_root"]) / meta["gpu_source"]["function_sources"][name]) in text


def test_reuse_requires_unchanged_source_and_target(tmp_path):
    import shutil

    if not shutil.which("ruff"):
        pytest.skip("reuse requires lint evidence")
    repo, meta_path, meta = scaffold(tmp_path)
    output = tmp_path / "checks.json"
    command = [
        sys.executable,
        str(Path(stage0_check.__file__)),
        "--meta",
        str(meta_path),
        "--repo-root",
        str(repo),
        "--reuse-report",
        str(output),
        "--output",
        str(output),
    ]

    def run():
        proc = subprocess.run(command, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return json.loads(proc.stdout)

    assert run()["reuse_scaffold"] is False
    assert run()["reuse_scaffold"] is True
    # An inherited helper not selected into gpu_source must invalidate reuse.
    shared = Path(meta["gpu_repo_root"]) / "src/tileops/new_shared_helper.py"
    shared.write_text("LIMIT = 32\n")
    assert run()["reuse_scaffold"] is False
    assert run()["reuse_scaffold"] is True
    wrapper = repo / "examples/TileOPs" / meta["wrapper_path"]
    wrapper.write_text(wrapper.read_text() + "# changed\n")
    assert run()["reuse_scaffold"] is False
