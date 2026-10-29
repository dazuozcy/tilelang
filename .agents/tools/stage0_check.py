"""CPU-only Stage 0 checks; never import source or target operator packages.

Run from repository root with Python >= 3.10 and PyYAML. Exit 0 means static
checks passed, NOT full Tier 1 or device correctness. State remains conductor-owned.
"""

import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import symtable
import sys

if sys.version_info < (3, 10):  # noqa: UP036 - diagnose an unconfigured Stage 0 interpreter
    raise SystemExit("stage0_check requires Python >= 3.10; reuse stage0_python from preflight")

import yaml

import gate_lint


def unresolved_globals(source, filename):
    """Use Python scopes so arguments, closures and comprehensions are not helpers."""
    table = symtable.symtable(source, filename, "exec")
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    known = defined | set(dir(builtins)) | {"__name__", "__file__", "__annotations__"}
    missing = set()

    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            visit(child)

    visit(table)
    return sorted(missing)


def check(meta_path, repo_root, lint=True):
    root = repo_root / "examples/TileOPs"
    failures, warnings = gate_lint.meta_lint(str(meta_path), str(repo_root))
    result = {"scope": "static_only", "runtime": "unverified", "failures": failures, "warnings": warnings, "checks": {}}
    # Invalid metadata must be repaired before using its paths or schema.
    if failures:
        return result
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def fail(rule, path, message):
        failures.append({"rule_id": rule, "file": str(path), "message": message})

    def target(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"target path escapes A5 root: {relative}")
        return path

    family, slug = meta["family"], meta["op_slug"]
    manifest = target(f"src/tileops/manifest/spec/{family}.yaml")
    try:
        entry = yaml.safe_load(manifest.read_text(encoding="utf-8"))[meta["op_name"]]
        if not isinstance(entry, dict):
            raise TypeError("manifest operator entry must be a mapping")
        if not isinstance(entry.get("workloads"), list) or not entry["workloads"]:
            fail("S0-WORKLOADS", manifest, "nonempty workload list required")
        for field, expected in [("kernel", meta["wrapper_path"]), ("test", meta["test_path"]), ("bench", meta["bench_path"])]:
            if not isinstance(entry.get("source"), dict):
                raise TypeError("manifest source must be a mapping")
            if entry["source"].get(field) != expected:
                fail("S0-MANIFEST-SOURCE", manifest, f"source.{field} differs from metadata")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        fail("S0-MANIFEST", manifest, str(exc))

    paths = {target(meta[key]) for key in ("wrapper_path", "extracted_file", "test_path", "bench_path")}
    paths.update(
        target(p)
        for p in [
            f"workloads/{family}.py",
            f"src/tileops/_ops/{family}/{slug}.py",
            f"src/tileops/_ops/{family}/__init__.py",
            f"src/tileops/_kernels/{family}/__init__.py",
            f"src/tileops/_kernels/{family}/{slug}/__init__.py",
            f"src/tileops/{family}.py",
        ]
    )
    trees = {}
    for path in sorted(paths):
        try:
            source = path.read_text(encoding="utf-8")
            trees[path] = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            fail("S0-AST", path, str(exc))
    extracted = target(meta["extracted_file"])
    if extracted in trees:
        names = {node.name for node in trees[extracted].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in meta["extracted_functions"]:
            if name not in names:
                fail("S0-FUNCTION", extracted, f"missing top-level function: {name}")
        missing = unresolved_globals(extracted.read_text(encoding="utf-8"), str(extracted))
        if missing:
            fail("S0-HELPERS", extracted, f"unresolved globals: {', '.join(missing)}")
    # Fingerprint all relevant source inputs, not only the agent-selected
    # provenance list: inherited helpers and shared case files may change too.
    source_root = Path(meta["gpu_repo_root"])
    source_files = set()
    for directory in ("src/tileops", "workloads", "tests", "benchmarks", "docs/design"):
        source_files.update(p for p in (source_root / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".md"})
    result["source_hashes"] = {p.relative_to(source_root).as_posix(): digest(p) for p in sorted(source_files)}
    paths.update({manifest, meta_path, target(meta["conversion_path"])})
    result["artifact_hashes"] = {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}
    target_files = set()
    for directory in ("src/tileops", "workloads", "tests", "benchmarks"):
        target_files.update(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml"})
    result["target_hashes"] = {p.relative_to(root).as_posix(): digest(p) for p in sorted(target_files)}
    tool_root = Path(__file__).resolve().parents[2]
    policy_files = [
        Path(__file__),
        Path(gate_lint.__file__),
        tool_root / "examples/TileOPs/.agents/skills/add-ascend-op/scripts/extract_tl_kernel.py",
        tool_root / "examples/TileOPs/.agents/skills/add-ascend-op/SKILL.md",
        tool_root / ".opencode/agents/conductor-scenarios/harness.md",
        tool_root / ".opencode/agents/tileops-scaffolder.md",
        tool_root / "pyproject.toml",
        root / "pyproject.toml",
    ]
    result["checker_hashes"] = {str(p.resolve()): digest(p) for p in policy_files if p.is_file()}
    result["checks"]["python_files"] = len(trees)
    result["checks"]["ruff"] = "skipped"
    if lint:
        ruff = shutil.which("ruff")
        if ruff:
            proc = subprocess.run([ruff, "check", *map(str, sorted(trees))], capture_output=True, text=True, timeout=120)
            result["checks"]["ruff"] = "pass" if proc.returncode == 0 else "fail"
            if proc.returncode:
                fail("S0-RUFF", root, proc.stdout + proc.stderr)
        else:
            warnings.append({"rule_id": "S0-RUFF", "message": "ruff unavailable; lint unverified"})
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit_prompts(meta_path, repo_root):
    """Render transport-only prompts from verified metadata; no model generation."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root = repo_root / "examples/TileOPs"
    blocks = []
    for name in meta["extracted_functions"]:
        source = Path(meta["gpu_repo_root"]) / meta["gpu_source"]["function_sources"][name]
        blocks.append(
            f"迁移 {meta['op_slug']} 算子 {meta['op_name']}。其 TileLang 实现函数 {name} 在 {root / meta['extracted_file']}。\n"
            f"原始定义：{source}；meta：{meta_path}；转换记录：{root / meta['conversion_path']}。\n"
            f"{name} 是对外接口，保留参数与返回契约，后端相关参数按冻结设计适配。\n"
            f"规格：{root / 'src/tileops/manifest/spec' / (meta['family'] + '.yaml')}，键 {meta['op_name']}。\n"
            f"精度用例：{root / meta['test_path']}；性能用例：{root / meta['bench_path']}。\n"
            "编程模式由 Stage 1 裁定（默认 simt，auto_schedule yes）；模块 docstring 须声明 programming_mode。"
        )
    output = (root / meta["wrapper_path"]).parent / "migration_prompts.md"
    output.write_text("\n\n---\n\n".join(blocks) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--reuse-report", type=Path, help="compare with a previous static report; unchanged verified scaffolds need no agent re-port"
    )
    parser.add_argument("--emit-prompts", action="store_true", help="after static pass, render migration_prompts.md from verified meta")
    parser.add_argument("--skip-lint", action="store_true", help="report lint skipped, not passed")
    parser.add_argument("--output", type=Path, help="optional static verification JSON; does not modify meta/state")
    args = parser.parse_args()
    try:
        result = check(args.meta.resolve(), args.repo_root.resolve(), not args.skip_lint)
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        result = {"scope": "static_only", "runtime": "unverified", "failures": [{"rule_id": "S0-INPUT", "message": str(exc)}]}
    result["verdict"] = "static_pass" if not result["failures"] else "fail"
    result["next_action"] = "return_to_conductor" if not result["failures"] else "repair_reported_failures"
    result["covered_checks"] = [
        "gate0",
        "target_paths",
        "manifest_source_paths",
        "nonempty_workloads",
        "ast",
        "compile_without_execution",
        "functions",
        "helpers",
        "scoped_lint",
    ]
    result["runtime"] = "unverified"
    result["reuse_scaffold"] = False
    if args.reuse_report and not result["failures"]:
        try:
            previous = json.loads(args.reuse_report.read_text(encoding="utf-8"))
            result["reuse_scaffold"] = (
                previous.get("verdict") == "static_pass"
                and previous.get("checks", {}).get("ruff") == result.get("checks", {}).get("ruff") == "pass"
                and all(
                    previous.get(key) == result.get(key) and result.get(key)
                    for key in ("source_hashes", "artifact_hashes", "target_hashes", "checker_hashes")
                )
            )
        except (OSError, ValueError, TypeError):
            pass  # Missing/old reports are cache misses, not migration failures.
    if args.emit_prompts and not result["failures"]:
        try:
            result["prompt_path"] = str(emit_prompts(args.meta.resolve(), args.repo_root.resolve()))
        except (OSError, KeyError, TypeError) as exc:
            result["failures"].append({"rule_id": "S0-PROMPTS", "message": str(exc)})
            result["verdict"] = "fail"
            result["next_action"] = "repair_reported_failures"
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    if args.output:
        summary = {key: value for key, value in result.items() if not key.endswith("_hashes")}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(payload, end="")
    return int(bool(result["failures"]))


if __name__ == "__main__":
    sys.exit(main())
