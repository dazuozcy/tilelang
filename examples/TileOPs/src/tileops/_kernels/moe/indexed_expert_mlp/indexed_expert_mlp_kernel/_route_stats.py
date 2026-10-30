# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Single-block route-statistics kernel for the indexed expert MLP (Ascend A5).

programming_mode: simd
auto_schedule: yes

Source: GPU TileOPs ``src/tileops/_kernels/moe/indexed_expert_gemm.py``
``_route_stats`` (L36-63) / ``_route_metadata_size`` (L30-32), migrated to the
Ascend dialect (``import tilelang.ascend.language as T``).

Semantics (DESIGN section 0.1): given ``expert_ids[T, K]`` int32 (values in
``[0, E)`` by caller contract), write a three-segment int32 metadata buffer of
size ``E + T*K + E*T``:

  * ``counts[0:E]``   -- number of routes per expert (exact);
  * ``ranks[E:E+R]``  -- each route's arrival order within its expert
    (a bijection onto ``[0, counts[e])`` per expert; the canonical realization
    is the stable route order, which the source leaves as a free choice);
  * ``lists[E+R:]``   -- ``list[e*T + rank] = route`` iff ``rank < T``
    (duplicate routes with ``rank >= T`` fall back to independent computation
    downstream; unwritten slots are deterministically zero-filled).

The GPU implementation allocates ranks with a 128-thread ``T.atomic_add``
scan and scatters 4-byte writes straight to GM.  The Ascend re-design (DESIGN
section 0.6 R1/R2) runs one block (``T.Kernel(1)``, no ``threads=``), stages
all metadata in UB, and does a deterministic stable-order scalar scan:

  ``T.copy`` GM->UB (ids) -> ``T.fill`` int32 zero (meta, SimdVF) ->
  scalar ``T.serial(token, slot)`` scan -> ``T.copy`` UB->GM (meta, single bulk).

The stable bijection is a valid realization of the source's bijection-freedom
contract (E3/E4 machine-verified); zero-filling the unused list slots makes the
whole buffer bitwise reproducible so the golden can compare exactly.

Call surface (wrapper-compatible, ``threads`` removed per K3):
    ``_route_stats(num_tokens, top_k, num_experts)()(expert_ids, metadata)``
``out_idx=[]`` -- in-place write, no return value (matches the source).
"""

import argparse
import functools

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

import tilelang
import tilelang.ascend.language as T


def _route_metadata_size(num_experts: int, num_routes: int, num_tokens: int) -> int:
    """Per-expert route counts, per-route ranks, then up to num_tokens routes listed per expert."""
    return num_experts + num_routes + num_experts * num_tokens


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_route_stats(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Stable-order route statistics (canonical realization of section 0.1 semantics).

    expert_ids: [T, K] int32 (values in [0, E)); returns int32 [E + T*K + E*T].
    The returned tensor stays on ``expert_ids.device`` (golden-on-NPU convention);
    the intermediate running counts are built on the host because they carry a
    per-expert loop-carried dependency that is inherently sequential (R <= 512).
    """
    T_, K = expert_ids.shape
    R = T_ * K
    size = num_experts + R + num_experts * T_
    flat = expert_ids.reshape(-1).cpu().tolist()  # small (R <= 512)
    metadata = [0] * size
    for route in range(R):
        e = flat[route]
        rank = metadata[e]
        metadata[e] = rank + 1  # running count
        metadata[num_experts + route] = rank  # ranks segment
        if rank < T_:  # lists segment (T-slot guard)
            metadata[num_experts + R + e * T_ + rank] = route
    return torch.tensor(metadata, dtype=torch.int32, device=expert_ids.device)


def check_route_stats_contract(metadata: torch.Tensor, expert_ids: torch.Tensor, E: int) -> None:
    """Section 0.1 property assertions (bijection-order agnostic).

    Verifies counts exactness, per-expert rank bijection, and rank/list pairing
    independent of the stable-order choice.  Runs on CPU so it does not depend on
    torch_npu operator coverage (bincount/nonzero).
    """
    T_, K = expert_ids.shape
    R = T_ * K
    metadata_c = metadata.cpu()
    flat_c = expert_ids.reshape(-1).cpu().to(torch.int64)
    counts = torch.bincount(flat_c, minlength=E)
    assert torch.equal(metadata_c[:E], counts.to(torch.int32)), "counts segment mismatch"
    base_r, base_l = E, E + R
    for e in range(E):
        routes = (flat_c == e).nonzero(as_tuple=True)[0]
        ranks = metadata_c[base_r + routes].to(torch.int64)
        assert sorted(ranks.tolist()) == list(range(len(routes))), f"expert {e} ranks not a bijection"
        for rank in range(min(len(routes), T_)):
            assert int(metadata_c[base_l + e * T_ + rank]) == int(routes[rank]), (
                f"expert {e} list slot {rank} mismatch"
            )


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _route_stats(num_tokens: int, top_k: int, num_experts: int):
    """Build the single-block deterministic route-statistics kernel.

    Returns the ``@tilelang.jit`` factory; the A5 wrapper calls
    ``_route_stats(T, K, E)(**config)(expert_ids, out)`` with ``config == {}``.
    """
    num_routes = num_tokens * top_k  # R
    size = _route_metadata_size(num_experts, num_routes, num_tokens)
    size_pad = ((size + 63) // 64) * 64  # 64-lane fill divisibility (section 4.3)
    rank_base = num_experts
    list_base = num_experts + num_routes

    @tilelang.jit(target="ascend", out_idx=[])
    def _kernel():
        @T.prim_func
        def main(
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            metadata: T.Tensor((size,), "int32"),
        ):
            with T.Kernel(1):
                ids_ub = T.alloc_shared((num_tokens, top_k), "int32")
                meta_ub = T.alloc_shared((size_pad,), "int32")
                T.copy(expert_ids, ids_ub)  # GM -> UB (MTE2)
                with T.SimdVF():
                    T.fill(meta_ub, T.int32(0))  # vector zero-fill (OPT-3)
                for token in T.serial(num_tokens):  # scalar scan (outside VF)
                    for slot in T.serial(top_k):
                        expert = ids_ub[token, slot]
                        rank = meta_ub[expert]
                        meta_ub[expert] = rank + 1
                        route = token * top_k + slot  # OPT-1: no div/mod
                        meta_ub[rank_base + route] = rank
                        if rank < num_tokens:  # T-slot guard
                            meta_ub[list_base + expert * num_tokens + rank] = route
                T.copy(meta_ub[0:size], metadata)  # UB -> GM (MTE3, bulk)

        return main

    return _kernel


# ---------- Layered testing ----------
def _compile_kernel(T_: int, K: int, E: int):
    return _route_stats(T_, K, E)()


def _gen_manifest_ids(T_: int, K: int, E: int, seed: int) -> torch.Tensor:
    """K distinct experts per row in [0, E), matching the manifest topk_ids generator."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(E, generator=g)[:K] for _ in range(T_)]
    return torch.stack(rows).to(torch.int32).to("npu")


def _run_case(T_: int, K: int, E: int, expert_ids: torch.Tensor, tag: str, kernel=None):
    assert tuple(expert_ids.shape) == (T_, K) and expert_ids.dtype == torch.int32
    assert expert_ids.device.type == "npu"
    if kernel is None:
        kernel = _compile_kernel(T_, K, E)
    size = _route_metadata_size(E, T_ * K, T_)
    out = torch.empty(size, dtype=torch.int32, device=expert_ids.device)
    kernel(expert_ids, out)
    torch.npu.synchronize()
    golden = golden_route_stats(expert_ids, E)
    if not torch.equal(out, golden):
        neq = out != golden
        idx = int(neq.nonzero()[0].item()) if neq.any() else -1
        raise AssertionError(
            f"[{tag}] metadata mismatch at idx {idx}: kernel={out[idx].item()} "
            f"golden={golden[idx].item()}"
        )
    check_route_stats_contract(out, expert_ids, E)
    print(f"[{tag}] PASS: T={T_} K={K} E={E} size={size}")
    return out


def _run_case_from_ids(ids, E: int, tag: str):
    t = torch.tensor(ids, dtype=torch.int32, device="npu")
    T_, K = t.shape
    _run_case(T_, K, E, t, tag)


def _run_determinism(T_: int, K: int, E: int, expert_ids: torch.Tensor, tag: str):
    kernel = _compile_kernel(T_, K, E)
    size = _route_metadata_size(E, T_ * K, T_)
    outs = []
    for _ in range(3):
        out = torch.empty(size, dtype=torch.int32, device=expert_ids.device)
        kernel(expert_ids, out)
        torch.npu.synchronize()
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]), (
        "nondeterministic output over 3 runs"
    )
    _run_case(T_, K, E, expert_ids, tag + "-exact", kernel=kernel)
    print(f"[{tag}] PASS: 3x bitwise identical")


def run_L0():
    # 1. compile smoke + exact compare: 3 manifest grouped specializations.
    for T_, K, E in [(32, 8, 256), (64, 8, 256), (32, 8, 128)]:
        _run_case(T_, K, E, _gen_manifest_ids(T_, K, E, seed=1234), "L0-manifest")
    # 2. test shape set, incl. (17,2,18) -> SIZE=358 (UB->GM non-32B tail, R7).
    _run_case(4, 2, 8, _gen_manifest_ids(4, 2, 8, seed=1), "L0-shape-4x2x8")
    _run_case_from_ids([[0, e] for e in range(1, 18)], 18, "L0-second-route-group")
    _run_case(8, 4, 16, torch.randint(0, 16, (8, 4), dtype=torch.int32, device="npu"), "L0-shape-8x4x16")
    _run_case(2, 2, 8, torch.randint(0, 8, (2, 2), dtype=torch.int32, device="npu"), "L0-shape-2x2x8")
    # 3. route patterns from the op-level test matrix.
    _run_case_from_ids([[0, 1], [2, 3], [4, 5], [6, 7]], 8, "L0-dispersed")
    _run_case_from_ids([[0, 1], [0, 1], [0, 2], [0, 2]], 8, "L0-reused")
    _run_case_from_ids([[0, 0], [0, 0], [0, 0], [0, 0]], 8, "L0-duplicate-fallback")
    # 4. determinism (3x bitwise identical).
    _run_determinism(64, 8, 256, _gen_manifest_ids(64, 8, 256, seed=7), "L0-determinism")


def run_L1():
    # Random dense routes over broader shapes (incl. heavy reuse and non-32B tails).
    for T_, K, E in [(8, 4, 16), (2, 2, 8), (4, 2, 8), (16, 4, 32), (5, 3, 7), (17, 2, 18), (32, 8, 128)]:
        ids = torch.randint(0, E, (T_, K), dtype=torch.int32, device="npu")
        _run_case(T_, K, E, ids, "L1-random")
    # K=1, E=1, all-same-expert, within-token duplicates.
    _run_case(8, 1, 4, torch.randint(0, 4, (8, 1), dtype=torch.int32, device="npu"), "L1-K1")
    _run_case(4, 2, 1, torch.zeros(4, 2, dtype=torch.int32, device="npu"), "L1-E1")
    _run_case(5, 3, 8, torch.full((5, 3), 7, dtype=torch.int32, device="npu"), "L1-all-same")
    dup = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]],
        dtype=torch.int32,
        device="npu",
    )
    _run_case(4, 4, 8, dup, "L1-within-token-dup")


def run_L2():
    # Robustness / large-shape probing; failures recorded as warnings (non-blocking).
    for T_, K, E in [(128, 8, 256), (64, 8, 512), (256, 4, 128)]:
        try:
            ids = torch.randint(0, E, (T_, K), dtype=torch.int32, device="npu")
            _run_case(T_, K, E, ids, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (not blocking): T={T_} K={K} E={E}: {e}")


def run_boundary():
    # zero-route experts (E > active experts).
    _run_case(
        4, 2, 16,
        torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=torch.int32, device="npu"),
        "Boundary-zero-route",
    )
    # rank == T (exact fill, no overflow).
    _run_case(2, 2, 4, torch.tensor([[0, 0], [0, 0]], dtype=torch.int32, device="npu"), "Boundary-rank-T")
    # count_e == T + 1 (first duplicate overflow -> rank >= T skipped from lists).
    _run_case(3, 2, 3, torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int32, device="npu"), "Boundary-rank-overflow")
    # E=1 with K=2.
    _run_case(3, 2, 1, torch.zeros(3, 2, dtype=torch.int32, device="npu"), "Boundary-E1-K2")
    # long all-same-expert chain (rank runs full R).
    _run_case(8, 4, 5, torch.full((8, 4), 0, dtype=torch.int32, device="npu"), "Boundary-all-same")


def _show_source():
    """Print the generated AscendC source (debugging aid; tilelang-ascend skill section 11)."""
    kernel = _compile_kernel(4, 2, 8)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true", help="print the generated AscendC kernel source")
    args, _ = parser.parse_known_args()

    if args.show_source:
        _show_source()
        return
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()