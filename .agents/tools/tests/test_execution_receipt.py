"""Execution evidence must invalidate on drift and never reuse failures."""

import json
from pathlib import Path
import subprocess
import sys

TOOL = Path(__file__).resolve().parents[1] / "execution_receipt.py"


def cli(*args):
    proc = subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, timeout=10)
    return proc.returncode, json.loads(proc.stdout)


def invocation(tmp_path, code="print('tested')"):
    script = tmp_path / "test.py"
    script.write_text(code)
    return [
        "--receipt",
        tmp_path / "receipt.json",
        "--cwd",
        tmp_path,
        "--suite",
        "stage3-all",
        "--input",
        script,
        "--",
        sys.executable,
        str(script),
    ]


def test_receipt_reuse_and_drift(tmp_path):
    args = invocation(tmp_path)
    assert cli("run", *args)[0] == 0
    assert cli("verify", *args)[0] == 0
    assert cli("run", "--reuse", *args)[1]["reused"]
    (tmp_path / "test.py").write_text("print('changed')")
    assert cli("verify", *args)[0] == 1
    assert not cli("run", "--reuse", *args)[1]["reused"]
    (tmp_path / "receipt.json.log").write_text("modified log")
    assert cli("verify", *args)[0] == 1


def test_receipt_suite_and_environment_are_part_of_identity(tmp_path, monkeypatch):
    args = invocation(tmp_path)
    assert cli("run", *args)[0] == 0
    changed = list(args)
    changed[changed.index("stage3-all")] = "integration-full"
    assert cli("verify", *changed)[0] == 1
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7")
    assert cli("verify", *args)[0] == 1


def test_failure_and_timeout_never_reuse(tmp_path):
    args = invocation(tmp_path, "raise SystemExit(1)")
    assert cli("run", *args)[0] == 1
    assert not cli("run", "--reuse", *args)[1]["reused"]
    args = invocation(tmp_path, "import time; time.sleep(2)")
    rc, out = cli("run", "--timeout", ".05", *args)
    assert rc == 1 and out["result"]["timed_out"]
    assert cli("verify", *args)[0] == 1


def test_budget_preserves_use_and_counts_failures(tmp_path):
    budget = tmp_path / "budget.json"
    assert cli("budget", "--budget", budget, "--max-experiments", 1)[0] == 0
    args = invocation(tmp_path, "raise SystemExit(1)")
    assert cli("run", "--budget", budget, "--experiment", *args)[0] == 1
    rc, out = cli("run", "--budget", budget, "--experiment", *args)
    assert rc == 1 and "budget exhausted" in out["error"]
    start = json.loads(budget.read_text())["started_at"]
    assert cli("budget", "--budget", budget, "--max-experiments", 2)[0] == 0
    data = json.loads(budget.read_text())
    assert data["experiments_used"] == 1 and data["started_at"] == start
    assert cli("run", "--budget", budget, "--experiment", *args)[1]["result"]["returncode"] == 1


def test_wallclock_budget_stops_child_and_reuse_is_free(tmp_path):
    budget = tmp_path / "budget.json"
    cli("budget", "--budget", budget, "--max-experiments", 1, "--max-wallclock-s", 10)
    args = invocation(tmp_path)
    assert cli("run", "--budget", budget, "--experiment", *args)[0] == 0
    assert cli("run", "--reuse", "--budget", budget, "--experiment", *args)[1]["reused"]
    data = json.loads(budget.read_text())
    data["started_at"] -= 20
    budget.write_text(json.dumps(data))
    assert "max_wallclock_s" in cli("run", "--budget", budget, *args)[1]["error"]


def test_input_change_during_execution_is_not_success(tmp_path):
    args = invocation(tmp_path, "from pathlib import Path; Path(__file__).write_text('print(1)')")
    rc, out = cli("run", *args)
    assert rc == 1 and not out["result"]["inputs_unchanged"]
    assert cli("verify", *args)[0] == 1


def test_device_lock_wait_obeys_deadline(tmp_path):
    import fcntl

    lock_path = tmp_path / "device.lock"
    args = invocation(tmp_path)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        rc, out = cli("run", "--timeout", ".05", "--device-lock", lock_path, *args)
    assert rc == 1 and "lock deadline" in out["error"]
    assert not (tmp_path / "receipt.json.log").exists()


def test_summary_keeps_attempts_and_does_not_double_count_reuse(tmp_path):
    args = invocation(tmp_path)
    assert cli("run", *args)[0] == 0
    assert cli("run", "--reuse", *args)[0] == 0
    rc, out = cli("summary", "--dir", tmp_path)
    assert rc == 0
    assert out["executions"] == 1 and out["reuses"] == 1
    assert out["execution_s"] > 0 and out["failures"] == 0


def test_shared_experiment_reservation_is_atomic(tmp_path):
    budget = tmp_path / "budget.json"
    cli("budget", "--budget", budget, "--max-experiments", 1)
    script = tmp_path / "test.py"
    script.write_text("print('run')")
    processes = []
    for unit in ("a", "b"):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(TOOL),
                    "run",
                    "--budget",
                    str(budget),
                    "--experiment",
                    "--receipt",
                    str(tmp_path / f"{unit}.json"),
                    "--cwd",
                    str(tmp_path),
                    "--suite",
                    "candidate",
                    "--input",
                    str(script),
                    "--",
                    sys.executable,
                    str(script),
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
        )
    results = [json.loads(proc.communicate(timeout=10)[0]) for proc in processes]
    assert sum(row["ok"] for row in results) == 1
    assert json.loads(budget.read_text())["experiments_used"] == 1


def test_generated_outputs_are_required_and_drift_invalidates_reuse(tmp_path):
    args = invocation(tmp_path, "from pathlib import Path; Path('generated.py').write_text('valid')")
    output = tmp_path / "generated.py"
    args = ["--output", output, *args]
    assert cli("run", *args)[0] == 0
    assert cli("run", "--reuse", *args)[1]["reused"]
    output.write_text("target changed")
    assert cli("verify", *args)[0] == 1
    assert not cli("run", "--reuse", *args)[1]["reused"]
    assert output.read_text() == "valid"
    output.unlink()
    assert cli("verify", *args)[0] == 1


def test_missing_declared_output_is_not_success(tmp_path):
    args = invocation(tmp_path)
    rc, out = cli("run", "--output", tmp_path / "missing", *args)
    assert rc == 1 and out["result"]["returncode"] == 125


def test_dispatch_budget_is_shared_across_function_stages(tmp_path):
    budget = tmp_path / "op-budget.json"
    assert cli("budget", "--budget", budget, "--max-dispatches", 1)[0] == 0
    assert cli("reserve", "--budget", budget, "--kind", "dispatch")[0] == 0
    rc, out = cli("reserve", "--budget", budget, "--kind", "dispatch")
    assert rc == 1 and "max_dispatches" in out["error"]
    assert cli("budget", "--budget", budget, "--max-dispatches", 2)[0] == 0
    assert cli("reserve", "--budget", budget, "--kind", "dispatch")[0] == 0
    assert json.loads(budget.read_text())["dispatches_used"] == 2
