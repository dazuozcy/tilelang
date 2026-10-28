import json
from pathlib import Path

from tileops.benchmark.benchmark_base import (
    BenchmarkBase,
    BenchmarkReport,
    _bench_with_mode,
)
from tileops.benchmark.msprof import MSPROF_RESULT_METADATA, bench_kernel_msprof
from tileops.benchmark.msprof import profile_kernel_msprof_detail
from tileops.benchmark.msprof_roofline import (
    DATA_TYPE_BASE_INFO,
    DATA_TYPE_ROOFLINE,
    parse_bin_blocks,
    parse_bin_file,
)
from tileops.reporting.analyzer import analyze_run
from tileops.reporting.cli import _parser
from tileops.reporting.collector import parse_junit_report
from tileops.reporting.report import write_reports
from tileops.reporting.runner import resolve_operator, run_operator
from tileops.reporting.session_timing import attach_session_timing, parse_session_timing


def test_parse_junit_report_preserves_properties(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.ops.test_x" name="test_ok" time="0.1">
    <properties><property name="max_abs_err" value="1.00e-04"/></properties>
  </testcase>
  <testcase classname="tests.ops.test_x" name="test_bad" time="0.2">
    <failure message="not close">details</failure>
  </testcase>
</testsuite></testsuites>""",
        encoding="utf-8",
    )

    parsed = parse_junit_report(report)

    assert parsed["tests"] == 2
    assert parsed["passed"] == 1
    assert parsed["failed"] == 1
    assert parsed["cases"][0]["properties"]["max_abs_err"] == "1.00e-04"


def test_parse_junit_report_counts_suite_level_errors(tmp_path):
    report = tmp_path / "collection_error.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="1" skipped="0"/></testsuites>',
        encoding="utf-8",
    )

    parsed = parse_junit_report(report)

    assert parsed["status"] == "failed"
    assert parsed["tests"] == 1
    assert parsed["errors"] == 1
    assert parsed["passed"] == 0


def test_bench_kernel_msprof_wraps_functor_with_partial(monkeypatch):
    calls = {"count": 0}
    functor_calls = []

    def fake_do_bench(fn, **kwargs):
        calls["kwargs"] = kwargs
        calls["count"] += 1
        fn()
        fn()
        return 0.42

    def functor(a, b, c):
        functor_calls.append((a, b, c))
        return a + b + c

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)

    latency_ms, prof_output_dir = bench_kernel_msprof(functor, (1, 2, 3))

    assert latency_ms == 0.42
    assert prof_output_dir is None
    assert functor_calls == [(1, 2, 3), (1, 2, 3)]
    assert calls["count"] == 1
    assert calls["kwargs"]["backend"] == "msprof"
    assert calls["kwargs"]["return_mode"] == "mean"
    assert calls["kwargs"]["_n_warmup"] == 0
    assert calls["kwargs"]["_n_repeat"] == 0


def test_bench_kernel_msprof_uses_bare_functor_without_args(monkeypatch):
    seen = []

    def fake_do_bench(fn, **kwargs):
        seen.append(fn)
        return 1.0

    def functor():
        return "ok"

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)

    latency_ms, prof_output_dir = bench_kernel_msprof(functor)

    assert latency_ms == 1.0
    assert prof_output_dir is None
    # No partial wrapper: the functor itself is what do_bench receives.
    assert seen == [functor]


def test_bench_kernel_msprof_maps_launch_env_to_do_bench(monkeypatch):
    captured = {}

    def fake_do_bench(fn, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)
    monkeypatch.setenv("TILEOPS_MSPROF_WARM_UP", "3")
    monkeypatch.setenv("TILEOPS_MSPROF_LAUNCH_COUNT", "7")

    bench_kernel_msprof(lambda: None)

    assert captured["backend"] == "msprof"
    assert captured["_n_warmup"] == 3
    assert captured["_n_repeat"] == 7


def test_bench_kernel_msprof_env_defaults_are_auto(monkeypatch):
    captured = {}

    def fake_do_bench(fn, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.delenv("TILEOPS_MSPROF_WARM_UP", raising=False)
    monkeypatch.delenv("TILEOPS_MSPROF_LAUNCH_COUNT", raising=False)
    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)

    bench_kernel_msprof(lambda: None)

    assert captured["_n_warmup"] == 0
    assert captured["_n_repeat"] == 0


def test_bench_kernel_msprof_env_rejects_garbage_integers(monkeypatch):
    captured = {}

    def fake_do_bench(fn, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)
    monkeypatch.setenv("TILEOPS_MSPROF_WARM_UP", "not-a-number")

    bench_kernel_msprof(lambda: None)

    assert captured["_n_warmup"] == 0


def test_bench_kernel_msprof_logs_and_ignores_deprecated_env(monkeypatch, caplog):
    captured = {}

    def fake_do_bench(fn, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)
    monkeypatch.setenv("TILEOPS_MSPROF_KERNEL_NAME", "some_kernel")
    monkeypatch.setenv("TILEOPS_MSPROF_OUTPUT_DIR", "/tmp/obsolete")
    monkeypatch.setenv("TILEOPS_MSPROF_KEEP_OUTPUT", "1")

    with caplog.at_level("WARNING", logger="tileops.benchmark.msprof"):
        latency_ms, prof_output_dir = bench_kernel_msprof(lambda: None)

    assert latency_ms == 0.0
    assert prof_output_dir is None
    message = caplog.text
    assert "TILEOPS_MSPROF_KERNEL_NAME" in message
    assert "TILEOPS_MSPROF_OUTPUT_DIR" in message
    assert "TILEOPS_MSPROF_KEEP_OUTPUT" in message
    assert "do_bench(msprof)" in message
    # The deprecated values are ignored: do_bench only receives the
    # latency-related kwargs, nothing kernel-name or output-dir related.
    assert set(captured) == {
        "warmup",
        "rep",
        "_n_warmup",
        "_n_repeat",
        "backend",
        "return_mode",
    }


def test_bench_kernel_msprof_rejects_non_tuple_args():
    try:
        bench_kernel_msprof(lambda: None, [1, 2])
    except TypeError:
        pass
    else:
        raise AssertionError("non-tuple args must be rejected")


def test_bench_kernel_msprof_forwards_do_bench_failure(monkeypatch):
    def failing_do_bench(fn, **kwargs):
        raise RuntimeError("msprof unavailable")

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", failing_do_bench)

    try:
        bench_kernel_msprof(lambda: None)
    except RuntimeError as exc:
        assert "msprof unavailable" in str(exc)
    else:
        raise AssertionError("do_bench failures must propagate to the events fallback layer")


def _fake_roofline(ratio: float = 77.6, op: str = "main_kernel"):
    return {
        "Ratio(%)": ratio,
        "AI(Ops/Byte)": 1.77,
        "Perf(TOps/s)": 2.20,
        "Computility(TOps/s)": 11.83,
        "GM_Roofline_BW(TB/s)": 1.60,
        "roofline_op": op,
        "roofline_freq_mhz": 1650,
    }


def _fake_detail(ratio: float = 0.87, pipe: str = "aiv_vec"):
    return {
        "dur_us": 110.0,
        "aiv_total_cycles": 9_200_000,
        "aic_total_cycles": 0,
        "pipes": {pipe: ratio, "aiv_mte3": 0.44},
        "bottleneck_pipe": pipe,
        "bottleneck_ratio": ratio,
    }


def test_profile_attaches_msprof_roofline_ratio(monkeypatch):
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel_msprof",
        lambda functor, args=(): (0.5, None),
    )
    monkeypatch.delenv("TILEOPS_MSPROF_ROOFLINE", raising=False)
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.collect_roofline_msprof",
        lambda functor, args=(), **kw: _fake_roofline(),
    )

    class DemoBenchmark(BenchmarkBase):
        def calculate_flops(self):
            return 1e6

        def calculate_memory(self):
            return 2e6

    bench = DemoBenchmark(workload={})
    result = bench.profile(lambda: None, mode="msprof")

    assert result["prof_mode"] == "msprof"
    assert result["Ratio(%)"] == 77.6
    assert result["AI(Ops/Byte)"] == 1.77
    assert result["Perf(TOps/s)"] == 2.20
    assert result["Computility(TOps/s)"] == 11.83
    assert result["msprof"]["ratio_source"] == "msprof_roofline_bin"
    assert result["msprof"]["roofline_op"] == "main_kernel"
    # Roofline path wins: no pipe-utilization fields on the result.
    assert "bottleneck_pipe" not in result
    assert result["latency_us"] == 500.0


def test_profile_roofline_unsupported_functor_falls_back_to_pipe(monkeypatch):
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel_msprof",
        lambda functor, args=(): (0.5, None),
    )
    monkeypatch.delenv("TILEOPS_MSPROF_ROOFLINE", raising=False)
    monkeypatch.delenv("TILEOPS_MSPROF_DETAIL", raising=False)

    def _raise_type_error(functor, args=(), **kw):
        raise TypeError("closure not supported")

    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.collect_roofline_msprof", _raise_type_error
    )
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.profile_kernel_msprof_detail",
        lambda functor, args=(): _fake_detail(),
    )

    class DemoBenchmark(BenchmarkBase):
        def calculate_flops(self):
            return 1e6

        def calculate_memory(self):
            return 2e6

    bench = DemoBenchmark(workload={})
    result = bench.profile(lambda: None, mode="msprof")

    assert result["Ratio(%)"] == 87.0
    assert result["bottleneck_pipe"] == "aiv_vec"
    assert result["msprof"]["ratio_source"] == "msprof_detail_pipe"
    assert "Perf(TOps/s)" not in result  # roofline-only fields stay absent


def test_profile_roofline_none_falls_back_to_pipe(monkeypatch):
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel_msprof",
        lambda functor, args=(): (0.5, None),
    )
    monkeypatch.delenv("TILEOPS_MSPROF_ROOFLINE", raising=False)
    monkeypatch.delenv("TILEOPS_MSPROF_DETAIL", raising=False)
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.collect_roofline_msprof",
        lambda functor, args=(), **kw: None,
    )
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.profile_kernel_msprof_detail",
        lambda functor, args=(): _fake_detail(0.51, "aiv_mte2"),
    )

    class DemoBenchmark(BenchmarkBase):
        def calculate_flops(self):
            return None

        def calculate_memory(self):
            return None

    bench = DemoBenchmark(workload={})
    result = bench.profile(lambda: None, mode="msprof")

    assert result["Ratio(%)"] == 51.0
    assert result["bottleneck_pipe"] == "aiv_mte2"
    assert "AI(Ops/Byte)" not in result  # no manifest formulas available


def test_profile_both_metric_passes_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel_msprof",
        lambda functor, args=(): (0.5, None),
    )
    monkeypatch.setenv("TILEOPS_MSPROF_ROOFLINE", "0")
    monkeypatch.setenv("TILEOPS_MSPROF_DETAIL", "0")

    def _fail(functor, args=(), **kw):
        raise AssertionError("metric passes must not run when disabled")

    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.collect_roofline_msprof", _fail
    )
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.profile_kernel_msprof_detail", _fail
    )

    class DemoBenchmark(BenchmarkBase):
        def calculate_flops(self):
            return 1e6

        def calculate_memory(self):
            return 2e6

    bench = DemoBenchmark(workload={})
    result = bench.profile(lambda: None, mode="msprof")

    assert result["prof_mode"] == "msprof"
    assert result["msprof"] == MSPROF_RESULT_METADATA
    assert "Ratio(%)" not in result
    # Latency metrics survive untouched.
    assert result["latency_us"] == 500.0


def test_profile_kernel_msprof_detail_selects_bottleneck_pipe(monkeypatch):
    class _FakeProfile:
        dur_ns = 110_000.0
        aic_total_cycles = 0
        aic_mad = 0.0
        aic_scalar = 0.0
        aic_mte1 = 0.0
        aic_mte2 = 0.0
        aic_mte3 = 0.0
        aic_fixpipe = 0.0
        aiv_total_cycles = 9_200
        aiv_vec = 0.2611
        aiv_scalar = 0.0
        aiv_mte2 = 0.8708
        aiv_mte3 = 0.4408

    captured = {}

    def fake_do_bench(fn, **kwargs):
        captured["kwargs"] = kwargs
        captured["fn"] = fn
        return _FakeProfile()

    monkeypatch.setattr("tileops.benchmark.msprof.do_bench", fake_do_bench)
    monkeypatch.setenv("TILEOPS_MSPROF_DETAIL_REPEAT", "9")

    detail = profile_kernel_msprof_detail(lambda a, b: None, (1, 2))

    assert captured["kwargs"]["backend"] == "msprof_detail"
    assert captured["kwargs"]["_n_repeat"] == 9
    assert captured["fn"].func is not None  # packaged as functools.partial
    assert detail["dur_us"] == 110.0
    assert detail["aiv_total_cycles"] == 9_200
    assert detail["bottleneck_pipe"] == "aiv_mte2"
    assert detail["bottleneck_ratio"] == 0.8708
    assert detail["pipes"]["aiv_vec"] == 0.2611


def test_profile_events_mode_has_no_msprof_metadata(monkeypatch):
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel", lambda functor, args=(): 0.2
    )

    class DemoBenchmark(BenchmarkBase):
        def calculate_flops(self):
            return None

        def calculate_memory(self):
            return None

    bench = DemoBenchmark(workload={})
    result = bench.profile(lambda: None, mode="events")

    assert result["prof_mode"] == "events"
    assert "msprof" not in result
    assert result["latency_us"] == 200.0


def test_bench_with_mode_falls_back_to_events_and_is_visible(monkeypatch):
    def failing_msprof(functor, args=()):
        raise RuntimeError("msprof failed")

    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel_msprof", failing_msprof
    )
    monkeypatch.setattr(
        "tileops.benchmark.benchmark_base.bench_kernel", lambda functor, args=(): 0.1
    )

    latency, prof_mode = _bench_with_mode(lambda: None, mode="msprof")

    assert prof_mode == "events"
    assert latency == 0.1


def test_analyzer_flags_events_fallback_for_msprof_request():
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "case-1",
                "tag": "tileops",
                "params": {},
                "result": {"latency_us": 10.0, "prof_mode": "events"},
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    assert run["status"] == "partial"
    assert run["summary"]["profiler_fallback_count"] == 1
    assert any("requested msprof but recorded events" in warning for warning in run["warnings"])


def test_analyzer_preserves_per_shape_roofline_ratio():
    benchmark = {
        "status": "present",
        "path": "benchmark.json",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "case-1",
                "label": "full-16",
                "tag": "tileops",
                "params": {"shape": [16], "dtype": "float16"},
                "result": {
                    "latency_us": 10.0,
                    "prof_mode": "msprof",
                    "Ratio(%)": 62.5,
                },
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    case = run["performance"]["cases"][0]
    assert case["roofline_utilization_percent"] == 62.5
    assert case["label"] == "full-16"
    assert "baselines" not in case
    assert "average_roofline_utilization_percent" not in run["summary"]
    assert "speedup_range" not in run["operators"][0]
    assert run["operators"][0]["ratio_range"] == {"min": 62.5, "max": 62.5}


def test_analyzer_passes_through_minimal_msprof_metadata():
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "case-1",
                "label": "full-16",
                "tag": "tileops",
                "params": {"shape": [16], "dtype": "float16"},
                "result": {
                    "latency_us": 10.0,
                    "prof_mode": "msprof",
                    "msprof": {"backend": "msprof", "latency_source": "do_bench(msprof)"},
                },
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    case = run["performance"]["cases"][0]
    assert case["msprof"] == {"backend": "msprof", "latency_source": "do_bench(msprof)"}
    assert run["status"] == "passed"
    assert run["summary"]["msprof_case_count"] == 1
    assert run["summary"]["profiler_fallback_count"] == 0


def test_analyzer_ignores_non_tileops_performance_records():
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "tileops-case",
                "label": "full-16",
                "tag": "tileops",
                "params": {"shape": [16], "dtype": "float16"},
                "result": {"latency_us": 10.0, "prof_mode": "msprof"},
            },
            {
                "operator": "DemoOp",
                "case_id": "torch-case-with-different-params-hash",
                "label": "full-16",
                "tag": "torch",
                "params": {
                    "shape": [16],
                    "dtype": "float16",
                    "result_bl": {"latency_us": 8.0},
                },
                "result": {"latency_us": 8.0, "prof_mode": "events"},
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    assert run["status"] == "passed"
    assert run["summary"]["case_count"] == 1
    assert [case["case_id"] for case in run["performance"]["cases"]] == ["tileops-case"]
    assert not any("missing tileops candidate record" in warning for warning in run["warnings"])


def test_analyzer_excludes_legacy_smoke_performance_records():
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "smoke-case",
                "label": "demo-smoke",
                "tag": "tileops",
                "params": {},
                "result": {"latency_us": 1.0, "prof_mode": "msprof"},
            },
            {
                "operator": "DemoOp",
                "case_id": "full-case",
                "label": "demo-full",
                "tag": "tileops",
                "params": {},
                "result": {"latency_us": 2.0, "prof_mode": "msprof"},
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    assert run["summary"]["case_count"] == 1
    assert [case["label"] for case in run["performance"]["cases"]] == ["demo-full"]


def test_benchmark_report_writes_structured_json(tmp_path, monkeypatch):
    BenchmarkReport.clear()
    BenchmarkReport.set_prof_mode("events")
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "benchmarks/ops/bench_demo.py::test_demo[smoke-8x16-float16] (call)",
    )
    BenchmarkReport.record(
        "DemoOp",
        {"shape": (8, 16), "dtype": "float16"},
        {"latency_us": 12.5, "Ratio(%)": 62.5, "prof_mode": "events"},
        tag="tileops",
    )

    output = BenchmarkReport.dump_json(tmp_path / "benchmark.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["profiling_mode_requested"] == "events"
    assert data["records"][0]["operator"] == "DemoOp"
    assert data["records"][0]["case_id"].startswith("DemoOp-")
    assert data["records"][0]["label"] == "smoke-8x16"

    markdown = tmp_path / "session.md"
    monkeypatch.setenv("TILEOPS_BENCHMARK_REPORT_PATH", str(markdown))
    BenchmarkReport.dump()
    assert markdown.is_file()
    assert markdown.with_suffix(".json").is_file()
    header = next(
        line
        for line in markdown.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Label |")
    )
    assert header.startswith("| Label | Latency (us) | Ratio (%) |")
    BenchmarkReport.clear()


def test_write_reports_creates_all_formats(tmp_path):
    run = {
        "operator": "DemoOp",
        "status": "passed",
        "summary": {
            "correctness_passed": True,
            "correctness_tests": 1,
            "correctness_failed": 0,
            "operator_count": 1,
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "pass_rate": 1.0,
            "benchmark_passed": True,
            "case_count": 1,
            "msprof_case_count": 1,
            "profiler_fallback_count": 0,
        },
        "warnings": [],
        "setup": {
            "metadata": {
                "framework": "TileOPs Reporting 0.1.0",
                "date": "2026-09-17T10:00:00+08:00",
                "evaluation_scope": "DemoOp",
                "profiler": "msprof",
            },
            "environment": {
                "npu": "Ascend NPU x 1",
                "cann": "9.0.0",
                "pytorch": "2.6.0",
                "python": "3.11.0",
                "os": "Linux",
            },
        },
        "correctness": {"cases": []},
        "operators": [
            {
                "operator": "DemoOp",
                "cases": 1,
                "passed": 1,
                "failed": 0,
                "pass_rate": 1.0,
                "avg_max_abs_err": 1e-4,
                "max_abs_err": 1e-4,
                "performance_cases": 1,
                "ratio_range": {"min": 52.25, "max": 52.25},
            }
        ],
        "performance": {
            "cases": [
                {
                    "operator": "DemoOp",
                    "case_id": "DemoOp-0123456789ab",
                    "label": "full-16x32",
                    "params": {"shape": [16, 32], "dtype": "float16"},
                    "prof_mode": "msprof",
                    "latency_us": 8.5,
                    "roofline_utilization_percent": 52.25,
                    "bandwidth_tb_s": 1.25,
                    "arithmetic_intensity_ops_per_byte": 4.0,
                    "performance_tops": 5.0,
                    "artifact_dir": "/tmp/artifacts/main_abc",
                    "msprof": {
                        "backend": "msprof",
                        "latency_source": "do_bench(msprof)",
                    },
                }
            ]
        },
    }

    paths = write_reports(run, tmp_path)

    assert all(path.exists() for path in paths.values())
    markdown = paths["markdown"].read_text(encoding="utf-8")
    html = paths["html"].read_text(encoding="utf-8")
    assert "DemoOp" in markdown
    assert "[16, 32]" in markdown
    assert "| DemoOp | 100.0% (1/1) | 1.00e-04 |" in markdown
    assert "| Label | Latency (us) | Ratio (%) |" in markdown
    assert "N/Ax" not in markdown
    assert "Shape / Parameters" in html
    assert "<th>Label</th><th>Latency (us)</th><th>Ratio</th>" in html
    assert "full-16x32" in html
    assert "[16, 32]" in html
    assert "<th>Operator</th><th>Correctness</th><th>Avg Max Abs Error</th>" in html
    assert "100.0% (1/1)" in html
    assert "Metadata / 元信息" in html
    assert "Environment / 运行环境" in html
    assert "Operator Analysis / 算子分析" in html
    assert "Avg Max Abs Error" in html
    assert "1.00e-04" in html
    assert "Performance Shapes" in html
    assert "Speedup" not in html
    assert "52.25%" in html
    assert "Avg Roofline" not in html
    assert "geometric mean" not in html
    assert "DemoOp-0123456789ab" in html
    assert "main_abc" in html
    assert "do_bench(msprof)" in html
    assert "{{" not in html


def test_runner_gates_benchmark_and_writes_report(tmp_path, monkeypatch):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run_pytest(**kwargs):
        target = kwargs["target"]
        calls.append((target, kwargs["pytest_args"]))
        Path(kwargs["junit_path"]).write_text(
            '<testsuite tests="1"><testcase classname="demo" name="ok"/></testsuite>',
            encoding="utf-8",
        )
        if target == "bench_demo.py":
            benchmark_json = Path(kwargs["env"]["TILEOPS_BENCHMARK_REPORT_PATH"]).with_suffix(
                ".json"
            )
            benchmark_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiling_mode_requested": "events",
                        "records": [
                            {
                                "operator": "ExternalOp",
                                "case_id": "case-1",
                                "tag": "tileops",
                                "params": {"shape": [16]},
                                "result": {
                                    "latency_us": 5.0,
                                    "prof_mode": "events",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return 0, ["pytest", target]

    monkeypatch.setattr("tileops.reporting.runner._run_pytest", fake_run_pytest)

    run, run_dir, exit_code = run_operator(
        "ExternalOp",
        test_file="test_demo.py",
        benchmark_file="bench_demo.py",
        prof_mode="events",
        reports_dir="reports",
        root=tmp_path,
    )

    assert calls == [
        ("test_demo.py", []),
        ("bench_demo.py", ["-m", "not smoke"]),
    ]
    assert exit_code == 0
    assert run["status"] == "passed"
    assert run["summary"]["benchmark_tests"] == 1
    assert run["summary"]["benchmark_failed"] == 0
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_runner_skips_benchmark_after_correctness_failure(tmp_path, monkeypatch):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run_pytest(**kwargs):
        calls.append(kwargs["target"])
        Path(kwargs["junit_path"]).write_text(
            '<testsuite tests="1" failures="1"><testcase classname="demo" '
            'name="bad"><failure message="not close"/></testcase></testsuite>',
            encoding="utf-8",
        )
        return 1, ["pytest", kwargs["target"]]

    monkeypatch.setattr("tileops.reporting.runner._run_pytest", fake_run_pytest)

    run, _run_dir, exit_code = run_operator(
        "ExternalOp",
        test_file="test_demo.py",
        benchmark_file="bench_demo.py",
        reports_dir="reports",
        root=tmp_path,
    )

    assert calls == ["test_demo.py"]
    assert exit_code == 1
    assert run["status"] == "failed"
    assert run["summary"]["benchmark_requested"] is False


def test_runner_combines_user_marker_expression_with_smoke_exclusion(tmp_path, monkeypatch):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run_pytest(**kwargs):
        calls.append((kwargs["target"], kwargs["pytest_args"]))
        Path(kwargs["junit_path"]).write_text(
            '<testsuite tests="1"><testcase classname="demo" name="ok"/></testsuite>',
            encoding="utf-8",
        )
        if kwargs["target"] == "bench_demo.py":
            Path(kwargs["env"]["TILEOPS_BENCHMARK_REPORT_PATH"]).with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiling_mode_requested": "events",
                        "records": [
                            {
                                "operator": "ExternalOp",
                                "case_id": "case-1",
                                "label": "full-case",
                                "tag": "tileops",
                                "params": {},
                                "result": {"latency_us": 1.0, "prof_mode": "events"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return 0, ["pytest", kwargs["target"]]

    monkeypatch.setattr("tileops.reporting.runner._run_pytest", fake_run_pytest)

    run_operator(
        "ExternalOp",
        test_file="test_demo.py",
        benchmark_file="bench_demo.py",
        prof_mode="events",
        pytest_args=["-m", "full or nightly", "-q"],
        root=tmp_path,
    )

    assert calls == [
        ("test_demo.py", ["-m", "full or nightly", "-q"]),
        ("bench_demo.py", ["-q", "-m", "(full or nightly) and not smoke"]),
    ]


def test_all_mode_resolves_pytest_directories():
    assert resolve_operator("all") == ("tests/ops", "benchmarks/ops")


def test_runner_rejects_xdist_for_benchmark_collection(tmp_path):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")

    try:
        run_operator(
            "ExternalOp",
            test_file="test_demo.py",
            benchmark_file="bench_demo.py",
            pytest_args=["-n", "2"],
            root=tmp_path,
        )
    except ValueError as exc:
        assert "pytest-xdist" in str(exc)
    else:
        raise AssertionError("xdist arguments must be rejected")


def test_analyzer_builds_multi_operator_summary_without_cross_shape_averages():
    correctness = {
        "status": "failed",
        "tests": 3,
        "passed": 2,
        "failed": 1,
        "errors": 0,
        "cases": [
            {
                "classname": "shared.tests",
                "nodeid": "shared.tests::a",
                "outcome": "passed",
                "properties": {"op": "AlphaOp", "max_abs_err": "1.0e-4"},
            },
            {
                "classname": "tests.ops.test_alpha",
                "nodeid": "tests.ops.test_alpha::b",
                "outcome": "failed",
            },
            {
                "classname": "shared.tests",
                "nodeid": "shared.tests::b",
                "outcome": "passed",
                "properties": {"op": "BetaOp", "max_abs_err": "2.0e-4"},
            },
        ],
    }
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "AlphaOp",
                "case_id": "alpha-1",
                "tag": "tileops",
                "params": {"shape": [16]},
                "result": {"latency_us": 10.0, "prof_mode": "msprof", "Ratio(%)": 40.0},
            },
            {
                "operator": "BetaOp",
                "case_id": "beta-1",
                "tag": "tileops",
                "params": {"shape": [32]},
                "result": {"latency_us": 10.0, "prof_mode": "msprof", "Ratio(%)": 70.0},
            },
        ],
    }
    catalog = [
        {"name": "AlphaOp", "family": "demo", "test": "tests/ops/test_alpha.py"},
        {"name": "BetaOp", "family": "demo", "test": "tests/ops/test_beta.py"},
    ]

    run = analyze_run(
        operator="all",
        correctness=correctness,
        correctness_exit_code=1,
        benchmark=benchmark,
        benchmark_exit_code=0,
        operator_catalog=catalog,
    )

    assert run["summary"]["operator_count"] == 2
    assert run["summary"]["pass_rate"] == 2 / 3
    assert run["operators"][0]["cases"] == 2
    assert run["operators"][0]["pass_rate"] == 0.5
    assert run["operators"][0]["avg_max_abs_err"] == 1e-4
    assert run["operators"][0]["performance_cases"] == 1
    assert "speedup_range" not in run["operators"][0]
    assert run["operators"][1]["ratio_range"] == {"min": 70.0, "max": 70.0}


def test_render_upgrades_legacy_run_json(tmp_path):
    legacy = {
        "schema_version": 1,
        "operator": "DemoOp",
        "status": "passed",
        "metadata": {
            "generated_at": "2026-09-17T10:00:00+08:00",
            "python": "3.11.0",
            "platform": "Linux",
            "prof_mode_requested": "events",
        },
        "summary": {"correctness_tests": 1, "correctness_failed": 0, "case_count": 1},
        "correctness": {"tests": 1, "passed": 1, "failed": 0, "errors": 0, "cases": []},
        "performance": {
            "cases": [
                {
                    "operator": "DemoOp",
                    "params": {"shape": [16]},
                    "latency_us": 5.0,
                    "prof_mode": "events",
                }
            ]
        },
        "warnings": [],
    }

    paths = write_reports(legacy, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")

    assert "DemoOp" in html
    assert "[16]" in html
    assert "Python" in html
    assert "No operator records" not in html


def test_benchmark_failures_are_preserved_in_diagnostics(tmp_path):
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }
    benchmark_tests = {
        "status": "failed",
        "tests": 1,
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "skipped": 0,
        "cases": [
            {
                "nodeid": "benchmarks.ops.test_demo::test_shape",
                "outcome": "failed",
                "message": "device error",
            }
        ],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark={"status": "present", "records": []},
        benchmark_exit_code=1,
        benchmark_tests=benchmark_tests,
    )
    paths = write_reports(run, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")

    assert run["summary"]["benchmark_failed"] == 1
    assert run["status"] == "partial"
    assert "Benchmark" in html
    assert "device error" in html


def _write_session_timing(path: Path, operator: str, total: float = 3661.25):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!-- TILEOPS_SESSION_TIMING_V1",
                json.dumps(
                    {
                        "schema_version": 1,
                        "operator": operator,
                        "total_duration_s": total,
                        "breakdown": [
                            {
                                "name": "Stage 5 integration",
                                "duration_s": 61.25,
                                "ratio_percent": 1.67,
                                "detail": "OpenCode session log",
                            }
                        ],
                    }
                ),
                "-->",
                "# Session timing analysis",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_session_timing_validates_operator_and_values(tmp_path):
    source = tmp_path / "SESSION_TIMING_ANALYSIS.md"
    _write_session_timing(source, "DemoOp")

    parsed = parse_session_timing(source, "DemoOp")
    mismatch = parse_session_timing(source, "OtherOp")

    assert parsed["status"] == "available"
    assert parsed["total_duration_s"] == 3661.25
    assert parsed["breakdown"][0]["name"] == "Stage 5 integration"
    assert mismatch["status"] == "invalid"
    assert "operator mismatch" in mismatch["reason"]


def test_session_timing_auto_discovery_supports_multi_operator_reports(tmp_path):
    root = tmp_path / "TileOPs"
    root.mkdir()
    metadata = root / "tileops" / "kernels" / "demo" / "alpha" / ".migration_meta.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"op_name": "AlphaOp", "op_slug": "alpha_task"}), encoding="utf-8"
    )
    _write_session_timing(tmp_path / "alpha_task" / "SESSION_TIMING_ANALYSIS.md", "AlphaOp")
    run = {
        "operators": [{"operator": "AlphaOp"}, {"operator": "BetaOp"}],
    }
    catalog = [
        {"name": "AlphaOp", "kernel": "tileops/kernels/demo/wrong/wrong.py"},
        {"name": "BetaOp", "kernel": "tileops/kernels/demo/beta/beta.py"},
    ]

    attach_session_timing(run, root=root, operator_catalog=catalog, enabled=True)

    records = run["session_timing"]["operators"]
    assert records["AlphaOp"]["status"] == "available"
    assert records["BetaOp"]["status"] == "missing"
    assert records["BetaOp"]["total_duration_s"] is None


def test_optional_session_timing_rendering_and_na_fallback(tmp_path):
    run = {
        "operator": "all",
        "status": "passed",
        "metadata": {},
        "setup": {},
        "summary": {
            "operator_count": 2,
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "case_count": 0,
        },
        "operators": [
            {
                "operator": "AlphaOp",
                "cases": 0,
                "passed": 0,
                "pass_rate": None,
                "performance_cases": 0,
            },
            {
                "operator": "BetaOp",
                "cases": 0,
                "passed": 0,
                "pass_rate": None,
                "performance_cases": 0,
            },
        ],
        "performance": {"cases": []},
        "correctness": {"cases": []},
        "benchmark_tests": {"cases": []},
        "warnings": [],
        "session_timing": {
            "enabled": True,
            "operators": {
                "AlphaOp": {
                    "status": "available",
                    "total_duration_s": 3661.25,
                    "breakdown": [
                        {
                            "name": "Integration",
                            "duration_s": 61.25,
                            "ratio_percent": 1.67,
                            "detail": "session log",
                        }
                    ],
                },
                "BetaOp": {
                    "status": "missing",
                    "total_duration_s": None,
                    "breakdown": [],
                    "reason": "SESSION_TIMING_ANALYSIS.md not found",
                },
            },
        },
    }

    paths = write_reports(run, tmp_path)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    html = paths["html"].read_text(encoding="utf-8")

    assert "| Operator | Total Time |" in markdown
    assert "1h01m01s" in markdown
    assert "Timing data unavailable: SESSION_TIMING_ANALYSIS.md not found" in markdown
    assert "Total Time / 总时间" in html
    assert "Session Timing / Session 耗时" in html
    assert "1h01m01s" in html
    assert "SESSION_TIMING_ANALYSIS.md not found" in html


def test_session_timing_disabled_preserves_current_report_layout(tmp_path):
    run = {
        "operator": "DemoOp",
        "status": "passed",
        "metadata": {},
        "setup": {},
        "summary": {
            "operator_count": 1,
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "case_count": 0,
        },
        "operators": [
            {
                "operator": "DemoOp",
                "cases": 0,
                "passed": 0,
                "pass_rate": None,
                "performance_cases": 0,
            }
        ],
        "performance": {"cases": []},
        "correctness": {"cases": []},
        "benchmark_tests": {"cases": []},
        "warnings": [],
    }

    paths = write_reports(run, tmp_path)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    html = paths["html"].read_text(encoding="utf-8")

    assert "Total Time" not in markdown
    assert "Session Timing" not in markdown
    assert "Total Time / 总时间" not in html
    assert "Session Timing / Session 耗时" not in html


def test_session_timing_flag_is_available_for_run_and_render():
    parser = _parser()

    run_args = parser.parse_args(["run", "--all", "--with-session-timing"])
    render_args = parser.parse_args(
        ["render", "reports/tileops/demo/run.json", "--with-session-timing"]
    )

    assert run_args.with_session_timing is True
    assert render_args.with_session_timing is True


# ---------------------------------------------------------------------------
# msprof_roofline: visualize_data.bin parsing + op/point selection
# ---------------------------------------------------------------------------


def _pack_bin_block(data_type: int, payload: bytes, padding: int = 0) -> bytes:
    import struct as _struct

    # Real visualize_data.bin blocks carry a 12-byte header: uint64 size,
    # uint8 type, uint8 padding_len, 2 reserved bytes (ignored by the parser).
    return (
        _struct.pack("<Q", len(payload) + padding)
        + bytes([data_type, padding, 0, 0])
        + payload
        + b"\x00" * padding
    )


def _roofline_payload(ratio: float, ai: float = 1.77, perf: float = 2.2, bw: float = 1.6) -> bytes:
    import json as _json

    return _json.dumps(
        {
            "multiple_rooflines": [
                {
                    "title": "GM/L2",
                    "rooflines": [
                        {
                            "bw_name": "GM Read + Write",
                            "bw": bw,
                            "point": [ai, perf],
                            "computility": 11.83,
                            "ratio": ratio,
                            "computility_name": "SIMT Float",
                        },
                        {
                            "bw_name": "L2 Read + Write",
                            "bw": 8.0,
                            "point": [ai, perf],
                            "computility": 11.83,
                            "ratio": ratio / 4,
                            "computility_name": "SIMT Float",
                        },
                        # Inapplicable stub point: ratio="NA" must not kill the block.
                        {
                            "bw_name": "GM Read + Write",
                            "bw": bw,
                            "point": [0.2, 0.3],
                            "computility": 5.9,
                            "ratio": "NA",
                            "computility_name": "Cube",
                        },
                    ],
                }
            ]
        }
    ).encode("utf-8")


def _write_bin(path, duration_us: float, ratio: float) -> None:
    import json as _json

    base = _json.dumps(
        {"duration": duration_us, "cur_freq": "1650", "soc": "Ascend950PR_9579", "op_type": "vector"}
    ).encode("utf-8")
    blob = _pack_bin_block(DATA_TYPE_BASE_INFO, base, padding=1) + _pack_bin_block(
        DATA_TYPE_ROOFLINE, _roofline_payload(ratio), padding=2
    )
    path.write_bytes(blob)


def test_parse_bin_file_extracts_roofline_entries(tmp_path):
    bin_path = tmp_path / "visualize_data.bin"
    _write_bin(bin_path, duration_us=100.5, ratio=0.776)

    data = parse_bin_file(str(bin_path))

    assert data["base_info"]["soc"] == "Ascend950PR_9579"
    assert data["base_info"]["duration"] == 100.5
    entries = data["roofline_entries"]
    # The NA-ratio stub is skipped; both valid points survive.
    assert len(entries) == 2
    gm = entries[0]
    assert gm["bw_name"] == "GM Read + Write"
    assert gm["ratio"] == 0.776
    assert gm["AI"] == 1.77
    assert gm["performance"] == 2.2
    assert gm["computility"] == 11.83


def test_parse_bin_blocks_handles_padding_and_termination(tmp_path):
    bin_path = tmp_path / "visualize_data.bin"
    _write_bin(bin_path, duration_us=1.0, ratio=0.5)
    blocks = parse_bin_blocks(bin_path.read_bytes())
    assert DATA_TYPE_BASE_INFO in blocks and DATA_TYPE_ROOFLINE in blocks
    # Padding bytes were stripped from the payload.
    assert blocks[DATA_TYPE_BASE_INFO][0].endswith(b"}")


def _make_op_tree(root, op_name: str, launches: int, duration_us: float, ratio: float):
    import os as _os

    op_dir = root / op_name
    for i in range(launches):
        launch_dir = op_dir / str(i)
        launch_dir.mkdir(parents=True)
        _write_bin(launch_dir / "visualize_data.bin", duration_us, ratio)


def test_select_roofline_entry_matches_duration(tmp_path, monkeypatch):
    from tileops.benchmark.msprof_roofline import _select_roofline_entry

    # Two captured ops: an auxiliary ACL kernel (long, low ratio) and the op
    # under test (~100us, ratio 0.78).  Duration matching must pick the latter.
    _make_op_tree(tmp_path, "ZerosLike", launches=1, duration_us=378.0, ratio=0.05)
    _make_op_tree(tmp_path, "main_kernel", launches=2, duration_us=100.0, ratio=0.78)

    picked = _select_roofline_entry(str(tmp_path), latency_us=99.6)
    assert picked is not None
    assert picked["roofline_op"] == "main_kernel"
    assert picked["Ratio(%)"] == 78.0
    assert picked["AI(Ops/Byte)"] == 1.77
    assert picked["Perf(TOps/s)"] == 2.2
    assert picked["Computility(TOps/s)"] == 11.83
    assert picked["GM_Roofline_BW(TB/s)"] == 1.6
    assert picked["roofline_freq_mhz"] == 1650

    # Without a latency hint the longest-duration op wins (legacy rule).
    picked_long = _select_roofline_entry(str(tmp_path))
    assert picked_long["roofline_op"] == "ZerosLike"


def test_select_roofline_entry_no_bin_returns_none(tmp_path):
    from tileops.benchmark.msprof_roofline import _select_roofline_entry

    assert _select_roofline_entry(str(tmp_path)) is None
