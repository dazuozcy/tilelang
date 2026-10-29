"""Non-NPU checks for the Stage 4 experiment runner template."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TEMPLATE = Path(__file__).resolve().parents[2] / "skills" / "tilelang-op-optimize" / "references" / "run_experiments_template.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("stage4_runner_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_only_inventory_tune_workloads(tmp_path, monkeypatch):
    runner = _load_runner()
    inventory = {
        "workloads": [
            {"kernel_id": "k1", "workload_id": "large", "kind": "tune"},
            {"kernel_id": "k1", "workload_id": "smoke-1m", "kind": "smoke"},
            {"kernel_id": "k2", "workload_id": "other", "kind": "tune"},
        ],
    }
    path = tmp_path / "workload_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    assert runner.tune_workload_ids() == ["large"]
    with pytest.raises(ValueError):
        runner.safe_component("../smoke;rm -rf")


def test_runner_rejects_non_tune_or_unknown_workload(tmp_path, monkeypatch):
    runner = _load_runner()
    path = tmp_path / "workload_inventory.json"
    path.write_text(
        json.dumps(
            {
                "workloads": [
                    {"kernel_id": "k1", "workload_id": "shape-a", "kind": "tune"},
                    {"kernel_id": "k1", "workload_id": "smoke-1m", "kind": "smoke"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    # a smoke workload must be rejected before any measurement (exit 2)
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--workload", "smoke-1m"])
    assert runner.main() == 2
    # an unknown workload (not in this kernel's inventory) must be rejected too
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--workload", "shape-unknown"])
    assert runner.main() == 2


def test_shared_pair_adjudicator_rejects_invalid_samples():
    runner = _load_runner()
    protocol, _ = runner.workflow_tools()
    assert protocol.summarize_pairs({"A": [10, 10, 10], "B": [9.6, 9.6, 9.6]})["verdict"] == "B_better"
    assert protocol.summarize_pairs({"A": [10, 10, 10], "B": [9.9, 10, 10.1]})["verdict"] == "tie"
    assert protocol.summarize_pairs({"A": [10, 10, 10], "B": [8, 11, 8]})["verdict"] == "inconclusive"
    for values in ([0, 1, 1], [float("nan"), 1, 1], [1, 1]):
        with pytest.raises(ValueError):
            protocol.summarize_pairs({"A": [1, 1, 1], "B": values})


def test_runner_resume_reuses_complete_pairs_and_invalidates_on_context_change(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("# fake kernel")
    inventory = tmp_path / "workload_inventory.json"
    inventory.write_text(json.dumps({"workloads": [{"kernel_id": "k1", "workload_id": "shape", "kind": "tune"}]}))
    stamp = tmp_path / "context.json"
    stamp.write_text('{"toolchain": "v1"}')
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(inventory))
    monkeypatch.setattr(runner, "load_runner", lambda impl, _: impl)
    monkeypatch.setattr(runner, "run_l0", lambda _: True)
    measured = []

    def measure(impl, samples):
        measured.append(impl)
        return [1 if impl == "a.py" else 0.9] * samples

    monkeypatch.setattr(runner, "bench_samples_ms", measure)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--a", "a.py", "--b", "b.py", "--workload", "shape", "--resume", "--context-stamp", str(stamp), "--no-record"],
    )
    assert runner.main() == 0
    assert measured == ["a.py", "b.py", "b.py", "a.py", "a.py", "b.py"]
    measured.clear()
    assert runner.main() == 0
    assert measured == []
    stamp.write_text('{"toolchain": "v2"}')
    assert runner.main() == 0
    assert len(measured) == 6


def test_runner_interrupted_half_pair_is_repeated(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)
    for name in ("a.py", "b.py", "stamp"):
        (tmp_path / name).write_text("test")
    monkeypatch.setattr(runner, "tune_workload_ids", lambda: ["shape"])
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(inventory))
    monkeypatch.setattr(runner, "load_runner", lambda impl, _: impl)
    monkeypatch.setattr(runner, "run_l0", lambda _: True)
    calls = []

    def measure(impl, samples):
        calls.append(impl)
        if len(calls) == 4:
            raise RuntimeError("interrupted in second pair")
        return [1] * samples

    monkeypatch.setattr(runner, "bench_samples_ms", measure)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--a", "a.py", "--b", "b.py", "--workload", "shape", "--resume", "--context-stamp", "stamp", "--no-record"]
    )
    with pytest.raises(RuntimeError):
        runner.main()
    calls.clear()
    monkeypatch.setattr(runner, "bench_samples_ms", lambda impl, samples: calls.append(impl) or [1] * samples)
    assert runner.main() == 0
    assert calls == ["b.py", "a.py", "a.py", "b.py"]
