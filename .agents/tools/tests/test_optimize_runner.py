"""Non-NPU checks for the Stage 4 experiment runner template."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "tilelang-op-optimize"
    / "references"
    / "run_experiments_template.py"
)


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
    monkeypatch.setattr(
        sys, "argv", ["run_experiments.py", "--workload", "shape-unknown"]
    )
    assert runner.main() == 2
