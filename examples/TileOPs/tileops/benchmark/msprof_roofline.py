"""Roofline metric collection via the ``msprof op`` CLI (attribution pass).

Restores the pre-A5-port report semantics of ``Ratio(%)``: the roofline
performance ratio **computed by the CANN msprof tool itself** and written
into ``visualize_data.bin`` — not a tileops-side formula.  The SoC peak
model (frequency, computility, bandwidth roofs) lives inside the msprof
tool; this module only generates a runner script, launches the CLI, and
parses the binary output.

Scope note (performance convention): this pass is an **attribution**
channel only.  End-to-end latency still comes exclusively from
``do_bench(fn, backend="msprof")`` in :mod:`tileops.benchmark.msprof`;
nothing here reports latency.

Workflow of :func:`collect_roofline_msprof`:

  1. Generate a standalone Python script that reconstructs *functor*
     (Op instance via constructor introspection, or a module-level
     callable imported by name) and runs it.
  2. Execute::

         msprof op --aic-metrics=Roofline --dump=off \
                   --output=<dir> --launch-count=N --warm-up=M \
                   [ --kernel-name=<kn> ] python <script>

  3. Parse every ``visualize_data.bin`` under the output directory
     (block type 13 = roofline graph), select the op that matches the
     do_bench latency, and return the ``GM Read + Write`` roofline point.

Configuration (environment variables):

- ``TILEOPS_MSPROF_ROOFLINE``      — "0" disables the roofline pass
  (handled in :mod:`tileops.benchmark.benchmark_base`).
- ``TILEOPS_MSPROF_ROOFLINE_LAUNCH_COUNT`` — kernel launches collected
  (default 3).
- ``TILEOPS_MSPROF_ROOFLINE_WARM_UP``     — warm-up runs (default 2).
- ``TILEOPS_MSPROF_KERNEL_NAME`` — optional ``--kernel-name`` filter.
"""

from __future__ import annotations

import glob
import inspect
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any, Callable, Optional

import torch

from tileops.device import get_device_backend

__all__ = ["collect_roofline_msprof", "parse_bin_file", "parse_bin_blocks"]

_logger = logging.getLogger("tileops.benchmark.msprof_roofline")

# Block types inside visualize_data.bin (format shared across CANN
# generations; verified on CANN 9.2.0 / Ascend950PR_9579).
DATA_TYPE_BASE_INFO = 5
DATA_TYPE_ROOFLINE = 13


# ---------------------------------------------------------------------------
# visualize_data.bin parsing (binary format, backend-agnostic)
# ---------------------------------------------------------------------------


def parse_bin_blocks(data: bytes) -> dict[int, list[bytes]]:
    """Split *data* into typed blocks.

    Block layout: ``uint64 data_size | uint8 data_type | uint8 padding_len``
    followed by *data_size* payload bytes (last *padding_len* bytes are
    padding to strip).
    """
    blocks: dict[int, list[bytes]] = defaultdict(list)
    pos = 0
    while pos + 12 <= len(data):
        data_size = struct.unpack_from("<Q", data, pos)[0]
        data_type = data[pos + 8]
        padding_len = data[pos + 9]
        data_start = pos + 12
        if data_size > len(data):
            break
        data_end = data_start + data_size
        blocks[data_type].append(data[data_start : data_end - padding_len])
        pos = data_end
    return dict(blocks)


def _safe_float(val: Any) -> float:
    if val is None or val == "NA" or val == "":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def parse_bin_file(bin_path: str) -> dict[str, Any]:
    """Parse one ``visualize_data.bin`` into base info + roofline entries."""
    with open(bin_path, "rb") as f:
        data = f.read()
    blocks = parse_bin_blocks(data)

    base_info: dict[str, Any] = {}
    if DATA_TYPE_BASE_INFO in blocks:
        j = json.loads(blocks[DATA_TYPE_BASE_INFO][0].decode("utf-8"))
        base_info = {
            "duration": _safe_float(j.get("duration")),
            "freq": _safe_float(j.get("cur_freq")),
            "soc": j.get("soc", ""),
            "block_dim": int(_safe_float(j.get("block_dim", 0))),
            "op_type": j.get("op_type", ""),
        }

    roofline_entries: list[dict[str, Any]] = []
    if DATA_TYPE_ROOFLINE in blocks:
        j = json.loads(blocks[DATA_TYPE_ROOFLINE][0].decode("utf-8"))
        for graph in j.get("multiple_rooflines", []):
            title = graph.get("title", "")
            for r in graph.get("rooflines", []):
                try:
                    entry = {
                        "title": title,
                        "bw_name": r.get("bw_name", ""),
                        "bw": float(r["bw"]),
                        "AI": float(r["point"][0]),
                        "performance": float(r["point"][1]),
                        "computility": float(r["computility"]),
                        "ratio": float(r["ratio"]),
                        "computility_name": r.get("computility_name", ""),
                    }
                except (KeyError, IndexError, TypeError, ValueError):
                    # An inapplicable pipe metric (for example a cube stub's
                    # ratio="NA") must not discard valid later roofline points.
                    continue
                roofline_entries.append(entry)

    return {
        "base_info": base_info,
        "roofline_entries": roofline_entries,
    }


# ---------------------------------------------------------------------------
# Runner-script generation (Op reconstruction / callable import)
# ---------------------------------------------------------------------------


def _is_serializable(val: Any) -> bool:
    """Check if *val* can be serialized as a Python literal in generated code."""
    if val is None or isinstance(val, (bool, int, float, str, torch.dtype)):
        return True
    if isinstance(val, (list, tuple)):
        return all(_is_serializable(x) for x in val)
    return False


def _val_repr(val: Any) -> str:
    """Convert *val* to its Python source repr for code generation."""
    if isinstance(val, torch.dtype):
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_val_repr(x) for x in val) + "]"
    return repr(val)


def _is_op_instance(functor: Any) -> bool:
    """Check if *functor* is an Op-like instance (has ``forward``)."""
    if inspect.isfunction(functor) or inspect.ismethod(functor):
        return False
    return (
        hasattr(functor, "forward") and callable(functor.forward) and hasattr(functor, "__class__")
    )


def _is_importable_callable(functor: Callable) -> bool:
    """Check if *functor* is a module-level callable importable by name."""
    if not callable(functor):
        return False
    mod = getattr(functor, "__module__", None)
    name = getattr(functor, "__name__", None)
    if not mod or not name:
        return False
    qual = getattr(functor, "__qualname__", "")
    return "<locals>" not in qual and "<lambda>" not in qual


def _extract_op_init_args(op: Any) -> dict:
    """Extract serializable constructor args from an Op instance."""
    cls = type(op)
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    init_args: dict = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if hasattr(op, name):
            val = getattr(op, name)
            if _is_serializable(val):
                init_args[name] = val
    return init_args


def _generate_runner_script(
    functor: Any,
    inputs_path: str,
    launch_count: int,
    device_str: str,
    device_index: int,
) -> str:
    """Generate a Python script that reconstructs *functor* and runs it."""
    lines = [
        "# Auto-generated msprof roofline profiling script (tileops)",
        "import warnings",
        'warnings.filterwarnings("ignore")',
        "",
        "import torch",
    ]
    if device_str == "npu":
        lines.append("import torch_npu  # noqa: F401")
        lines.append(f"torch.npu.set_device({device_index})")

    if _is_op_instance(functor):
        cls = type(functor)
        init_args = _extract_op_init_args(functor)
        args_src = ", ".join(f"{k}={_val_repr(v)}" for k, v in init_args.items())
        lines.extend(
            [
                "",
                f"from {cls.__module__} import {cls.__name__}",
                "",
                "inputs = torch.load(%r)" % inputs_path,
                "if not isinstance(inputs, (list, tuple)):",
                "    inputs = (inputs,)",
                "inputs = [x.to(%r) if isinstance(x, torch.Tensor) else x for x in inputs]" % device_str,
                "",
                "op = %s(%s)" % (cls.__name__, args_src),
                "torch.npu.synchronize()",
                "",
                "for _ in range(%d):" % max(launch_count, 1),
                "    op(*inputs)",
                "torch.npu.synchronize()",
            ]
        )
    else:  # importable module-level callable (validated by caller)
        lines.extend(
            [
                "",
                "import %s as _mod" % functor.__module__,
                "_fn = getattr(_mod, %r)" % functor.__name__,
                "",
                "inputs = torch.load(%r)" % inputs_path,
                "if not isinstance(inputs, (list, tuple)):",
                "    inputs = (inputs,)",
                "inputs = [x.to(%r) if isinstance(x, torch.Tensor) else x for x in inputs]" % device_str,
                "",
                "for _ in range(%d):" % max(launch_count, 1),
                "    _fn(*inputs)",
                "torch.npu.synchronize()",
            ]
        )
    return "\n".join(lines) + "\n"


def _resolve_msprof_bin() -> str:
    """Locate the msprof CLI binary."""
    msprof_bin = shutil.which("msprof")
    if msprof_bin:
        return msprof_bin
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidate = os.path.join(ascend_home, "tools", "profiler", "bin", "msprof")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "msprof not found (PATH or $ASCEND_HOME_PATH/tools/profiler/bin/msprof). "
        "Roofline collection requires the Ascend profiler CLI."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def collect_roofline_msprof(
    functor: Callable,
    args: tuple = (),
    *,
    kernel_name: Optional[str] = None,
    launch_count: Optional[int] = None,
    warm_up: Optional[int] = None,
    latency_us: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Collect the msprof-tool-computed roofline ratio for *functor*.

    Runs one ``msprof op --aic-metrics=Roofline`` pass in a subprocess and
    returns the ``GM Read + Write`` roofline point of the op whose captured
    duration best matches *latency_us* (the do_bench measurement of the
    same workload).  This is an attribution pass; latency authority stays
    with do_bench(msprof).

    Args:
        functor: Op instance (reconstructed via constructor introspection)
            or a module-level callable.  Closures/lambdas raise TypeError.
        args: Input tensors (saved via torch.save, moved to CPU).
        kernel_name: Optional ``--kernel-name`` filter; env override
            ``TILEOPS_MSPROF_KERNEL_NAME``.  ``None`` profiles all kernels
            and relies on duration matching to pick the op under test.
        launch_count: Kernel launches collected (default env
            ``TILEOPS_MSPROF_ROOFLINE_LAUNCH_COUNT`` or 3).
        warm_up: Warm-up runs (default env ``TILEOPS_MSPROF_ROOFLINE_WARM_UP``
            or 2).
        latency_us: The do_bench(msprof) latency of the same workload, used
            to select among multiple captured ops.

    Returns:
        ``{"Ratio(%)", "AI(Ops/Byte)", "Perf(TOps/s)", "Computility(TOps/s)",
        "GM_Roofline_BW(TB/s)", "roofline_op", "roofline_freq_mhz"}``
        or ``None`` when no roofline data was produced.

    Raises:
        TypeError: If *functor* is neither an Op instance nor an importable
            module-level callable (e.g. a closure).
        RuntimeError: If the msprof binary is unavailable.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"collect_roofline_msprof expects a tuple of args, got {type(args).__name__}."
        )

    is_op = _is_op_instance(functor)
    is_callable = _is_importable_callable(functor) if not is_op else False
    if not is_op and not is_callable:
        raise TypeError(
            "collect_roofline_msprof: functor must be an Op instance or a "
            f"module-level callable; got {type(functor).__name__} "
            f"(qualname={getattr(functor, '__qualname__', '?')}). "
            "Closures and lambdas are not supported — the pipe-utilization "
            "fallback applies."
        )

    msprof_bin = _resolve_msprof_bin()
    backend = get_device_backend()
    device_str = backend.name
    try:
        device_index = backend.current_device()
    except Exception:  # noqa: BLE001 - device query is best-effort
        device_index = 0

    if launch_count is None:
        launch_count = int(os.environ.get("TILEOPS_MSPROF_ROOFLINE_LAUNCH_COUNT", "3"))
    if warm_up is None:
        warm_up = int(os.environ.get("TILEOPS_MSPROF_ROOFLINE_WARM_UP", "2"))
    if kernel_name is None:
        kernel_name = os.environ.get("TILEOPS_MSPROF_KERNEL_NAME") or None

    tmp_dir = tempfile.mkdtemp(prefix="tileops_roofline_")
    try:
        inputs_path = os.path.join(tmp_dir, "inputs.pt")
        script_path = os.path.join(tmp_dir, "prof_script.py")
        prof_output_dir = os.path.join(tmp_dir, "msprof_output")

        saved_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        torch.save(saved_args, inputs_path)
        with open(script_path, "w") as f:
            f.write(_generate_runner_script(functor, inputs_path, launch_count, device_str, device_index))

        os.makedirs(prof_output_dir, exist_ok=True)
        # msprof refuses to profile into a group-writable directory (and
        # still exits 0 producing no output); force owner-only permissions.
        os.chmod(prof_output_dir, 0o700)

        env = dict(os.environ)
        env.setdefault("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH", ""))
        env["PYTHONPATH"] = (
            f"{os.getcwd()}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else os.getcwd()
        )

        cmd = [msprof_bin, "op"]
        if kernel_name is not None:
            cmd.append(f"--kernel-name={kernel_name}")
        cmd.extend(
            [
                f"--output={prof_output_dir}",
                f"--launch-count={launch_count}",
                f"--warm-up={warm_up}",
                "--aic-metrics=Roofline",
                "--dump=off",
                sys.executable,
                script_path,
            ]
        )
        _logger.info("msprof roofline command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, env=env, check=False
        )
        if result.returncode != 0:
            _logger.warning(
                "msprof op (roofline) exited %d; stderr: %s",
                result.returncode,
                (result.stderr or "")[-500:],
            )

        return _select_roofline_entry(prof_output_dir, latency_us=latency_us)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Op / roofline-point selection (ported selection semantics)
# ---------------------------------------------------------------------------


def _op_key(bin_path: str) -> str:
    # .../OPPROF_x/<OpName>/<launch>/visualize_data.bin -> OpName
    # .../OPPROF_x/<OpName>/visualize_data.bin          -> OpName
    # .../OPPROF_x/visualize_data.bin                   -> OPPROF_x
    parts = os.path.normpath(bin_path).split(os.sep)[:-1]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return parts[-1] if parts else bin_path


def _bin_duration(bin_data: dict) -> float:
    try:
        return float(bin_data.get("base_info", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _select_roofline_entry(
    prof_output_dir: str, *, latency_us: Optional[float] = None
) -> Optional[dict[str, Any]]:
    """Pick the op under test and return its ``GM Read + Write`` roofline point."""
    bin_files = sorted(
        glob.glob(os.path.join(prof_output_dir, "**", "visualize_data.bin"), recursive=True)
    )
    if not bin_files:
        _logger.warning("visualize_data.bin not found under %s", prof_output_dir)
        return None

    candidates: list[tuple[str, dict]] = []
    for bin_path in bin_files:
        try:
            candidates.append((bin_path, parse_bin_file(bin_path)))
        except Exception as e:  # noqa: BLE001 - one corrupt bin must not kill the pass
            _logger.warning("parse_bin_file failed for %s: %s", bin_path, e)
    if not candidates:
        return None

    op_groups: dict[str, list[tuple[str, dict]]] = {}
    for bin_path, data in candidates:
        op_groups.setdefault(_op_key(bin_path), []).append((bin_path, data))

    if len(op_groups) > 1:
        summary = ", ".join(
            f"{op}(duration~{_bin_duration(grp[0][1]):.1f}us)"
            for op, grp in sorted(op_groups.items())
        )
        _logger.warning(
            "msprof captured %d distinct ops: [%s]; selecting by %s.",
            len(op_groups),
            summary,
            "duration match to the reported latency" if latency_us else "longest duration",
        )
        if latency_us:
            chosen_op = min(op_groups, key=lambda op: abs(_bin_duration(op_groups[op][0][1]) - latency_us))
        else:
            chosen_op = max(op_groups, key=lambda op: _bin_duration(op_groups[op][0][1]))
        bin_path, bin_data = op_groups[chosen_op][0]
    else:
        bin_path, bin_data = candidates[0]

    entries = bin_data.get("roofline_entries", [])
    gm_entries = [
        e for e in entries if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", "")
    ]
    if not gm_entries:
        gm_entries = [
            e
            for e in entries
            if "GM" in e.get("bw_name", "")
            and ("GM/L2" in e.get("title", "") or "Memory Unit" in e.get("title", ""))
        ]
    if not gm_entries:
        _logger.warning("No 'GM Read + Write' roofline entry found in %s", bin_path)
        return None

    gm = gm_entries[0]
    base = bin_data.get("base_info", {})
    return {
        "Ratio(%)": round(gm["ratio"] * 100.0, 2),
        "AI(Ops/Byte)": round(gm["AI"], 4),
        "Perf(TOps/s)": round(gm["performance"], 4),
        "Computility(TOps/s)": round(gm["computility"], 4),
        # Renamed from the 910B "Bandwidth(TB/s)": the .bin `bw` field is the
        # roofline's bandwidth ceiling (tool SoC model), NOT the achieved
        # bandwidth — keep the report's Bandwidth column on the achieved
        # (manifest-bytes / measured-latency) value.
        "GM_Roofline_BW(TB/s)": round(gm["bw"], 4),
        "roofline_op": _op_key(bin_path),
        "roofline_freq_mhz": base.get("freq"),
    }
