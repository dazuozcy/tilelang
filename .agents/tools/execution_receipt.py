"""Run resumable workflow units with input fingerprints and a shared budget.

This tool never reads or writes conductor state. Receipts attest command
execution, not numerical correctness: the caller must supply a test command
whose exit status enforces the required suite. All imported helpers, compiler
artifacts and environment stamps relevant to reuse must be declared as inputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, suppress
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(path, deadline=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        if deadline is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("execution lock deadline exceeded") from None
                    time.sleep(min(0.05, remaining))
        yield


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def identity(args):
    inputs = {str(Path(p).resolve()): digest(p) for p in args.input}
    if not args.argv:
        raise ValueError("a command after -- is required")
    return {
        "runner_sha256": digest(__file__),
        "argv": args.argv,
        "cwd": str(Path(args.cwd).resolve()),
        "suite": args.suite,
        "outputs": sorted(str(Path(p).resolve()) for p in args.output),
        "inputs": inputs,
        "environment": {
            "python": sys.executable,
            "version": sys.version,
            "variables": {key: os.environ.get(key) for key in sorted(set(args.env_key))},
        },
    }


def reusable(path, expected):
    try:
        record = json.loads(Path(path).read_text())
        return (
            record.get("schema") == 1
            and record.get("identity") == expected
            and record.get("returncode") == 0
            and record.get("timed_out") is False
            and record.get("inputs_unchanged") is True
            and record.get("outputs") == {path: digest(path) for path in expected["outputs"]}
            and digest(record["log_path"]) == record["log_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def event(receipt, payload):
    path = Path(receipt).parent / "events.jsonl"
    with locked(str(path) + ".lock"), path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"receipt": str(receipt), "time": time.time(), **payload}) + "\n")


def summary(args):
    events = []
    for path in Path(args.dir).rglob("events.jsonl"):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    return {
        "ok": True,
        "executions": sum(not row["reused"] for row in events),
        "reuses": sum(row["reused"] for row in events),
        "execution_s": sum(row.get("elapsed_s", 0) for row in events),
        "lock_wait_s": sum(row.get("lock_wait_s", 0) for row in events),
        "failures": sum(not row["ok"] for row in events),
    }


def budget_init(args):
    path = Path(args.budget)
    with locked(str(path) + ".lock"):
        if path.exists():
            record = json.loads(path.read_text())
            # Updating limits preserves consumption and original task start.
        else:
            record = {"schema": 1, "started_at": time.time(), "experiments_used": 0, "dispatches_used": 0}
        if args.max_dispatches is not None:
            if args.max_dispatches <= 0:
                raise ValueError("max-dispatches must be positive")
            record["max_dispatches"] = args.max_dispatches
        if args.max_experiments is not None:
            if args.max_experiments <= 0:
                raise ValueError("max-experiments must be positive")
            record["max_experiments"] = args.max_experiments
        if args.max_wallclock_s is not None:
            record["max_wallclock_s"] = args.max_wallclock_s
        atomic_json(path, record)
    return {"ok": True, "budget": record}


def reserve(path, experiment=False, dispatch=False):
    """Reserve before starting work; failed experiments still consume budget."""
    with locked(str(path) + ".lock"):
        record = json.loads(Path(path).read_text())
        remaining = None
        if record.get("max_wallclock_s") is not None:
            remaining = record["started_at"] + record["max_wallclock_s"] - time.time()
            if remaining <= 0:
                raise ValueError("budget exhausted: max_wallclock_s")
        if dispatch:
            maximum = record.get("max_dispatches")
            if maximum is not None and record.get("dispatches_used", 0) >= maximum:
                raise ValueError("budget exhausted: max_dispatches")
            record["dispatches_used"] = record.get("dispatches_used", 0) + 1
        if experiment:
            maximum = record.get("max_experiments")
            if maximum is not None and record["experiments_used"] >= maximum:
                raise ValueError("budget exhausted: max_experiments")
            record["experiments_used"] += 1
        if dispatch or experiment:
            atomic_json(path, record)
        return remaining


def run(args):
    if args.experiment and not args.budget:
        raise ValueError("--experiment requires --budget")
    receipt = Path(args.receipt).resolve()
    # Serialize identical units; callers additionally serialize device work.
    queued_at = time.time()
    deadline = queued_at + args.timeout
    if args.budget:
        budget = json.loads(Path(args.budget).read_text())
        if budget.get("max_wallclock_s") is not None:
            deadline = min(deadline, budget["started_at"] + budget["max_wallclock_s"])
    device_lock = locked(args.device_lock, deadline) if args.device_lock else nullcontext()
    with locked(str(receipt) + ".lock", deadline), device_lock:
        expected = identity(args)
        if args.reuse and reusable(receipt, expected):
            event(receipt, {"ok": True, "reused": True, "lock_wait_s": time.time() - queued_at})
            return {"ok": True, "reused": True, "receipt": str(receipt)}
        remaining = reserve(args.budget, args.experiment) if args.budget else None
        timeout = min(args.timeout, remaining) if remaining is not None else args.timeout
        timeout = min(timeout, deadline - time.time())
        if timeout <= 0:
            raise TimeoutError("execution deadline exceeded")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        log = receipt.with_suffix(receipt.suffix + ".log")
        started = time.time()
        timed_out = False
        # Separate process group so timeout also stops compiler/test children.
        with log.open("wb") as output:
            proc = subprocess.Popen(
                args.argv,
                cwd=args.cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
        try:
            unchanged = identity(args) == expected
        except OSError:
            unchanged = False
        outputs = {}
        try:
            outputs = {path: digest(path) for path in expected["outputs"]}
        except OSError:
            unchanged = False
        record = {
            "outputs": outputs,
            "schema": 1,
            "identity": expected,
            "returncode": returncode,
            "timed_out": timed_out,
            "started_at": started,
            "elapsed_s": time.time() - started,
            "lock_wait_s": started - queued_at,
            "inputs_unchanged": unchanged,
            "log_path": str(log),
            "log_sha256": digest(log),
        }
        # Changing an input during execution must invalidate the receipt.
        if not unchanged:
            record["returncode"] = 125
        atomic_json(receipt, record)
        event(
            receipt,
            {"ok": record["returncode"] == 0, "reused": False, "elapsed_s": record["elapsed_s"], "lock_wait_s": record["lock_wait_s"]},
        )
        return {"ok": returncode == 0 and not timed_out and unchanged, "reused": False, "receipt": str(receipt), "result": record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("budget", help="create/update a task budget without resetting use")
    init.add_argument("--budget", required=True)
    init.add_argument("--max-experiments", type=int)
    init.add_argument("--max-dispatches", type=int)
    init.add_argument("--max-wallclock-s", type=positive)
    init.set_defaults(func=budget_init)
    reservation = sub.add_parser("reserve", help="reserve one task-wide dispatch or experiment")
    reservation.add_argument("--budget", required=True)
    reservation.add_argument("--kind", choices=["dispatch", "experiment"], required=True)
    reservation.set_defaults(
        func=lambda args: {
            "ok": True,
            "remaining_s": reserve(args.budget, experiment=args.kind == "experiment", dispatch=args.kind == "dispatch"),
        }
    )
    report = sub.add_parser("summary", help="summarize execution/reuse events under a task directory")
    report.add_argument("--dir", required=True)
    report.set_defaults(func=summary)
    for name in ("run", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", required=True)
        command.add_argument("--cwd", required=True)
        command.add_argument("--suite", required=True, help="e.g. stage3-all, integration-full")
        command.add_argument("--input", action="append", required=True)
        command.add_argument("--output", action="append", default=[], help="generated file to fingerprint after execution")
        command.add_argument(
            "--env-key",
            action="append",
            default=[
                "PATH",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "ASCEND_HOME_PATH",
                "BISHENG_HOME",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
            ],
        )
        command.add_argument("--device-lock", help="shared lock path for serialized device execution")
        command.add_argument("--budget")
        command.add_argument("--experiment", action="store_true")
        command.add_argument("--timeout", type=positive, default=900)
        command.add_argument("--reuse", action="store_true")
        command.add_argument("argv", nargs=argparse.REMAINDER)
        command.set_defaults(func=run)
    args = parser.parse_args(argv)
    if hasattr(args, "argv") and args.argv[:1] == ["--"]:
        args.argv = args.argv[1:]
    try:
        if args.command == "verify":
            result = {"ok": reusable(args.receipt, identity(args))}
        else:
            result = args.func(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
