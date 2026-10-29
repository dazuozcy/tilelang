"""CPU-only source discovery/extraction regression; never imports device code."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "examples/TileOPs/.agents/skills/add-ascend-op/scripts/extract_tl_kernel.py"
spec = importlib.util.spec_from_file_location("tileops_extract_layout", SCRIPT)
extract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extract
spec.loader.exec_module(extract)


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def modern_repo(root):
    write(root, "src/tileops/manifest/spec/reduction.yaml", "ExampleOp:\n  family: reduction\n  status: implemented\n")
    write(root, "src/tileops/manifest/spec/types.yaml", "ExampleOp: shared_type\n")
    write(root, "src/tileops/_ops/example.py", "from .base import Base\nclass ExampleOp(Base):\n    pass\n")
    write(
        root,
        "src/tileops/_ops/base.py",
        """
from tileops._kernels.first import First as Alias
from tileops._kernels.second import Second
class Base:
    kernel_types: dict = {"first": Alias, "second": Second}
""",
    )
    for module, cls, name in [("first", "First", "first_builder"), ("second", "Second", "second_builder")]:
        write(
            root,
            f"src/tileops/_kernels/{module}.py",
            f"""
import tilelang
@tilelang.jit
def {name}(n):
    return n
class {cls}:
    def build(self):
        return {name}(1)
""",
        )


def test_src_layout_inherited_aliases_multiple_files(tmp_path):
    modern_repo(tmp_path)
    entry = extract.load_manifest_entry("ExampleOp", tmp_path)
    bindings = extract.resolve_kernel_bindings(entry, "ExampleOp", tmp_path)
    assert [(p.name, n) for p, n in bindings] == [("first.py", "First"), ("second.py", "Second")]
    tool = extract.Extractor(tmp_path)
    output = tool.extract(tmp_path / entry["source"]["kernel"], [], "ExampleOp", bindings=bindings)
    functions = {n.name for n in ast.parse(output).body if isinstance(n, ast.FunctionDef)}
    assert {"first_builder", "second_builder"} <= functions
    assert not tool.warnings
    assert len(tool.targets) == 2


def test_old_directory_layout_is_rejected(tmp_path):
    write(tmp_path, "tileops/manifest/reduction.yaml", "ExampleOp: {}\n")
    with pytest.raises(FileNotFoundError, match="manifest directory not found"):
        extract.load_manifest_entry("ExampleOp", tmp_path)


def test_dynamic_registry_is_explicit_failure(tmp_path):
    modern_repo(tmp_path)
    write(tmp_path, "src/tileops/_ops/base.py", "class Base:\n    kernel_types = make_registry()\n")
    entry = extract.load_manifest_entry("ExampleOp", tmp_path)
    with pytest.raises(ValueError, match="inspect manually"):
        extract.resolve_kernel_bindings(entry, "ExampleOp", tmp_path)


def test_ambiguous_op_is_explicit_failure(tmp_path):
    modern_repo(tmp_path)
    write(tmp_path, "src/tileops/_ops/duplicate.py", "class ExampleOp: pass\n")
    with pytest.raises(ValueError, match="expected one Op definition"):
        extract.load_manifest_entry("ExampleOp", tmp_path)


def test_colliding_builder_names_are_rejected(tmp_path):
    modern_repo(tmp_path)
    path = tmp_path / "src/tileops/_kernels/second.py"
    path.write_text(path.read_text().replace("second_builder", "first_builder"))
    entry = extract.load_manifest_entry("ExampleOp", tmp_path)
    bindings = extract.resolve_kernel_bindings(entry, "ExampleOp", tmp_path)
    with pytest.raises(ValueError, match="conflicting extracted name"):
        extract.Extractor(tmp_path).extract(tmp_path / entry["source"]["kernel"], [], "ExampleOp", bindings=bindings)


def test_inspect_cli_does_not_write_extraction(tmp_path, capsys):
    modern_repo(tmp_path)
    output = tmp_path / "must_not_exist.py"
    assert (
        extract.main(
            [
                "--gpu-repo-root",
                str(tmp_path),
                "--op-name",
                "ExampleOp",
                "--inspect",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    import json

    result = json.loads(capsys.readouterr().out)
    assert len(result["kernel_bindings"]) == 2
    assert not output.exists()


def test_provenance_gate_and_static_integration(tmp_path, capsys):
    """Exercise extraction → new meta → gate 0 → file integration, with no device imports."""
    import json
    import subprocess

    source = tmp_path / "gpu"
    modern_repo(source)
    base = tmp_path / "examples/TileOPs"
    wrapper_dir = base / "src/tileops/_kernels/reduction/example"
    extracted = wrapper_dir / "_reference.py"
    provenance = tmp_path / "discovery.json"
    assert (
        extract.main(
            [
                "--gpu-repo-root",
                str(source),
                "--op-name",
                "ExampleOp",
                "--out",
                str(extracted),
                "--source-info-out",
                str(provenance),
            ]
        )
        == 0
    )
    meta = json.loads(provenance.read_text())
    assert meta["gpu_source"]["function_sources"] == {
        "first_builder": "src/tileops/_kernels/first.py",
        "second_builder": "src/tileops/_kernels/second.py",
    }
    for key, rel in [
        ("workload_paths", "workloads/reduction.py"),
        ("test_paths", "tests/ops/test_example.py"),
        ("bench_paths", "benchmarks/ops/bench_example.py"),
    ]:
        write(source, rel, "# source case\n")
        meta["gpu_source"][key] = [rel]
    meta.update(
        {
            "op_name": "ExampleOp",
            "op_slug": "example",
            "family": "reduction",
            "extracted_functions": ["first_builder", "second_builder"],
            "wrapper_path": "src/tileops/_kernels/reduction/example/example.py",
            "extracted_file": "src/tileops/_kernels/reduction/example/_reference.py",
            "test_path": "tests/ops/test_example.py",
            "test_slug": "example",
            "bench_path": "benchmarks/ops/bench_example.py",
            "bench_slug": "example",
            "conversion_path": "src/tileops/_kernels/reduction/example/conversion.md",
            "kernel_class_name": "First",
            "kernel_class_names": ["First", "Second"],
        }
    )
    for rel in [
        meta["conversion_path"],
        meta["test_path"],
        meta["bench_path"],
        "src/tileops/manifest/spec/reduction.yaml",
        "workloads/reduction.py",
    ]:
        write(base, rel, "# static fixture\n")
    # If the integration script imports this package, the test must fail.
    write(base, "src/tileops/__init__.py", "raise RuntimeError('device package imported')\n")
    write(base, meta["wrapper_path"], "from ._reference import first_builder, second_builder\n")
    meta_file = wrapper_dir / ".migration_meta.json"
    meta_file.write_text(json.dumps(meta))
    tools_dir = SCRIPT.parents[6] / ".agents/tools"
    gate_spec = importlib.util.spec_from_file_location("tileops_layout_gate", tools_dir / "gate_lint.py")
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    failures, _ = gate.meta_lint(str(meta_file), str(tmp_path))
    assert not failures
    incomplete = json.loads(json.dumps(meta))
    del incomplete["gpu_source"]["function_sources"]["second_builder"]
    assert gate.migration_source_lint(incomplete, str(meta_file), str(tmp_path))
    mixed = dict(meta, wrapper_path="src/tileops/_kernels/first.py")
    assert gate.migration_source_lint(mixed, str(meta_file), str(tmp_path))
    escaped = json.loads(json.dumps(meta))
    escaped["gpu_source"]["function_sources"]["first_builder"] = "../outside.py"
    assert gate.migration_source_lint(escaped, str(meta_file), str(tmp_path))
    # Keep valid metadata on disk for the integration step.
    for func in meta["extracted_functions"]:
        product = f"examples/example/{func}"
        write(tmp_path, f"{product}/{func}.py", f'"""programming_mode: simt"""\ndef {func}(): pass\n')
        write(tmp_path, f"{product}/DESIGN.md", "# Frozen design fixture\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT.with_name("integrate_kernel.py")),
            "--meta",
            str(meta_file),
            "--tileops-root",
            str(base),
            "--examples-root",
            str(tmp_path / "examples"),
            "--skip-smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    package = wrapper_dir / "example_kernel"
    for func in meta["extracted_functions"]:
        assert (package / f"{func}.py").exists()
        assert (package / f"{func}_DESIGN.md").exists()
    assert "perf_opt" in (base / meta["wrapper_path"]).read_text()
    assert not list(base.rglob("run.json"))


def test_refreshed_target_manifest_and_package_discovery():
    """The installed package must use src and load YAML without device modules."""
    import os
    import subprocess

    base = SCRIPT.parents[4]
    code = """
import builtins
original = builtins.__import__
def no_devices(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'tilelang'}:
        raise AssertionError('device module imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = no_devices
from tileops.manifest import load_manifest, manifest_files
assert load_manifest()['LogSumExpFwdOp']['family'] == 'reduction'
assert all('/manifest/spec/' in str(path) for path in manifest_files())
from setuptools import find_packages
packages = find_packages('src')
assert 'tileops._ops' in packages and 'tileops._kernels' in packages
assert not {'tileops.ops', 'tileops.kernels', 'tileops.workloads'} & set(packages)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=base, env={**os.environ, "PYTHONPATH": str(base / "src")}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (base / "tileops").exists()
    assert (base / "workloads/workload_base.py").is_file()


def test_refreshed_reporting_project_root():
    base = SCRIPT.parents[4]
    source_file = base / "src/tileops/reporting/runner.py"
    tree = ast.parse(source_file.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "project_root")
    namespace = {"Path": Path, "__file__": str(source_file)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_file), "exec"), namespace)
    assert namespace["project_root"]() == base


@pytest.mark.parametrize("metadata", [{}, {"source_layout": "src", "refresh_status": "requires_rescaffold"}])
def test_integrator_requires_refreshed_metadata(tmp_path, metadata):
    import json

    module_spec = importlib.util.spec_from_file_location("layout_integrator", SCRIPT.with_name("integrate_kernel.py"))
    integrator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(integrator)
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="regenerat"):
        integrator.load_meta(path)


@pytest.mark.parametrize(
    "name,slug",
    [
        ("IndexedExpertMLPFwdOp", "indexed_expert_mlp"),
        ("SharedExpertMLPFwdOp", "shared_expert_mlp"),
        ("LogSumExpFwdOp", "log_sum_exp"),
        ("CastFP8Op", "cast_fp8"),
    ],
)
def test_stage0_identity_preserves_acronyms(name, slug):
    plan = extract.stage0_identity(name, "moe")
    assert plan["op_slug"] == slug
    assert plan["task_dir"] == f"examples/{slug}"
    assert plan["wrapper_path"] == f"src/tileops/_kernels/moe/{slug}/{slug}.py"


def test_transitive_helpers_keep_decorators_constants_and_imports(tmp_path):
    modern_repo(tmp_path)
    write(
        tmp_path,
        "src/tileops/_kernels/helpers.py",
        """
import math
import tilelang.language as T
FACTOR = 16
@T.macro
def align(n):
    return math.ceil(n / FACTOR) * FACTOR

def metadata_size(n):
    return align(n) + 1
""",
    )
    write(
        tmp_path,
        "src/tileops/_kernels/first.py",
        """
import tilelang
from .helpers import metadata_size
@tilelang.jit
def first_builder(n):
    return metadata_size(n)
class First:
    def build(self):
        return first_builder(1)
""",
    )
    entry = extract.load_manifest_entry("ExampleOp", tmp_path)
    bindings = extract.resolve_kernel_bindings(entry, "ExampleOp", tmp_path)
    tool = extract.Extractor(tmp_path)
    output = tool.extract(tmp_path / entry["source"]["kernel"], [], "ExampleOp", bindings=bindings)
    tree = ast.parse(output)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"metadata_size", "align", "first_builder", "second_builder"} <= funcs.keys()
    assert ast.unparse(funcs["align"].decorator_list[0]) == "T.macro"
    assert any(isinstance(n, ast.Import) and n.names[0].name == "math" for n in tree.body)
    assert "FACTOR = 16" in output
    assert {t.node.name for t in tool.targets} == {"first_builder", "second_builder"}
    assert not any(isinstance(n, ast.ClassDef) for n in tree.body)


def test_stage0_plan_generates_draft_without_claiming_completion(tmp_path, monkeypatch, capsys):
    import json

    source = tmp_path / "gpu"
    modern_repo(source)
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.chdir(target)
    assert extract.main(["--gpu-repo-root", str(source), "--op-name", "ExampleOp", "--stage0-plan"]) == 0
    plan = json.loads((target / "src/tileops/_kernels/reduction/example/stage0_plan.json").read_text())
    assert plan["op_slug"] == "example"
    assert set(plan["extracted_functions"]) == {"first_builder", "second_builder"}
    assert (target / plan["extracted_file"]).is_file()
    assert plan["metadata_draft"]["gpu_source"]["test_paths"] == []
    assert not (target / plan["meta_path"]).exists()
    assert plan["requires_review"]
