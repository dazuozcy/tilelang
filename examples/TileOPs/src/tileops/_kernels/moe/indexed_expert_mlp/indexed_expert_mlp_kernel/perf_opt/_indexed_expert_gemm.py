# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Indexed small-route expert GEMM (Ascend A5) -- route-major, expert-grouped.

programming_mode: mixed
auto_schedule: yes

STAGE 4 FINAL (merged winner: OP-D round3_leaderload structure + tuned
config bn=128/bk=512/ns=2, uniform for both instances and every benchmark
workload -- all 8 are on-grid at this config).  This file is the perf_opt
final of the indexed-expert-GEMM kernel (replaces the baseline
``_indexed_expert_gemm.py``); the wrapper's baseline/perf_opt import switch
references the TUNED_*_CONFIG module constants below pairwise.  Structure:
on-grid W loads run on leader blocks only.  Phase 1 measured the non-leader blocks' redundant K
pipeline (the PrepareMultiBuffer guard-trap workaround) at a W traffic
amplification of 1.58x (ds-v3-t32) / 2.32x (t64) / 2.27x (qwen) while the
AIC MTE2 pipe sits 82-89% busy at ~2 B/cyc -- the ND2NZ fractal/short-segment
path is the wall (round-2 ubstage AIV-channel variant measured -8%, and
block_k 256/512/1024 all plateau at the same ~95GB/s aggregate, so neither
the channel nor the segment length moves it; only traffic volume does).
This branch therefore guards ONLY the two on-grid W GM->L1 copies with
``is_leader`` -- ``T.gemm`` stays unguarded (the res L0C write/read guard
mismatch is the known PrepareMultiBuffer crash); non-leader blocks feed the
Cube whatever stale w_l1 content is present, which is harmless because their
result is never written back (the epilogue guard is unchanged).  The
off-grid path (test shapes only) is kept verbatim from the baseline.

Source: GPU TileOPs
``src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py``
``_indexed_expert_gemm`` (+ ``_route_metadata_size``), migrated to the Ascend
dialect (``import tilelang.ascend.language as T``).

Semantics (DESIGN section 0.1): for every route ``r = (token, slot)`` pick the
expert ``e = expert_ids[token, slot]`` and compute the row

    out[token, slot, :] = silu_and_mul(a_row @ W_e^T)   (gated, 2N cols split)
                        | a_row @ W_e^T                 (none)

where ``a_row`` is ``a[token, slot]`` when ``route_input`` (down projection) or
``a[token]`` otherwise (gate/up).  The K-dim dot product accumulates in fp32;
the gated activation is fused into the epilogue in the fp32 domain and the
result is cast to the storage dtype on write-back.  Grouping only decides
*which* block computes *which* rows and never changes a row's math, so a
correct golden does not consume ``route_metadata`` (DESIGN E2).

Structure (DESIGN section 3.3): one 1-D ``T.MixedKernel(grid, sids=2)`` grid of
``R x n_blocks`` blocks (``R = num_tokens * top_k``).  The AIC sub-core runs the
Cube NT GEMM path (``T.copy`` GM->L1 with ND2NZ fractal conversion inserted
automatically, then ``T.gemm(..., transpose_B=True, clear_accum=(ko==0))``).
AIV0 gathers the A tile into UB and stages it to L1 (and, off the tile grid,
stages W through a zero-filled UB buffer); AIV1 runs the fused silu/cast/scatter
epilogue from the L0C accumulator.  Leader grouping is governed by the
``is_leader`` block-level guard, but the guard gates *only the AIV1 epilogue*
(write-back): the Cube pipeline runs on every block because wrapping
``T.gemm``'s L0C accumulator in a runtime guard crashes the ``PrepareMultiBuffer``
storage-epoch analysis (see RETROSPECTIVE).  Non-leader blocks of a group
compute the identical tile and simply never write it back.

Call surface (wrapper-compatible, ``threads`` removed per K3/K9):

    _indexed_expert_gemm(T, top_k, E, n, k, dtype_str, activation,
                         route_input, dispatch_mode)(**config)(a, weights,
                         expert_ids, route_metadata, out)

``out_idx=[]`` -- in-place write of ``out``, no return value.

run: python _indexed_expert_gemm.py --level all
"""

import argparse
import functools
import importlib.util
import os

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

import tilelang
import tilelang.ascend.language as T

# Stage 4 tuned defaults (do_bench(msprof)-measured winners, round 3 +
# final verification): ONE uniform config for both instances and every
# benchmark workload (ds-v3 t1/t32/t64, qwen3-235b t32; fp16 + bf16; all
# on-grid at this config).  The T=1 direct down workloads measure -5.7% at
# block_k=512 (no leader grouping to amortize the longer k-step there, an
# absolute 150us against the +52-59% grouped wins) -- accepted for a single
# uniform config instead of a per-shape dispatch (see opt_log.md Final
# Summary).  The wrapper's perf_opt import switch passes these to the gemm
# factory instead of its own default_config() result.
TUNED_GATEUP_CONFIG = {"block_n": 128, "block_k": 512, "num_stages": 2}
TUNED_DOWN_CONFIG = {"block_n": 128, "block_k": 512, "num_stages": 2}

_BLOCK_M = 16  # Cube M granularity (DESIGN section 5.2).


def _route_metadata_size(num_experts: int, num_routes: int, num_tokens: int) -> int:
    """Per-expert route counts, per-route ranks, then up to num_tokens routes listed per expert."""
    return num_experts + num_routes + num_experts * num_tokens


# ---------- Golden (PyTorch reference, computed on NPU) ----------
def golden_indexed_expert_gemm(
    a: torch.Tensor,
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    activation: str,
    route_input: bool,
) -> torch.Tensor:
    """Per-route fp32 reference (DESIGN section 8.1), independent of grouping.

    Loops route by route without any leader/group/tile/staging structure and
    applies ``torch.sigmoid`` for the gated activation (independent of the
    kernel's direct-division form).  ``route_metadata`` is intentionally not
    read: grouping invariance (DESIGN E2) guarantees bitwise agreement with any
    self-consistent metadata ordering.
    """
    t, k_ = expert_ids.shape
    af = a.float()
    wf = weights.float()
    y = torch.empty((t, k_, wf.shape[1]), dtype=torch.float32, device=a.device)
    for ti in range(t):
        for si in range(k_):
            row = af[ti, si] if route_input else af[ti]
            y[ti, si] = row @ wf[int(expert_ids[ti, si])].t()
    if activation == "silu_and_mul":
        gate, up = y.chunk(2, dim=-1)
        o = gate * torch.sigmoid(gate) * up
    else:
        o = y
    return o.to(a.dtype)


# ---------- Synthetic route metadata (tests) ----------
def _make_route_metadata(
    expert_ids: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    *,
    random_bijection: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Build the three-segment metadata contract (counts / ranks / lists).

    Counts are exact; per-expert ranks form a bijection onto ``[0, counts[e])``;
    the canonical (stable) realization keeps routes in route order, while
    ``random_bijection=True`` permutes each expert's list (still a valid
    bijection).  List slots are written only for ``rank < num_tokens``.
    """
    tt, k_ = expert_ids.shape
    r = tt * k_
    size = num_experts + r + num_experts * num_tokens
    flat = expert_ids.reshape(-1).cpu().tolist()
    routes_by_expert = [[] for _ in range(num_experts)]
    for route in range(r):
        routes_by_expert[flat[route]].append(route)
    if random_bijection:
        g = torch.Generator().manual_seed(seed)
        for e in range(num_experts):
            idx = torch.randperm(len(routes_by_expert[e]), generator=g).tolist()
            routes_by_expert[e] = [routes_by_expert[e][i] for i in idx]
    metadata = [0] * size
    for e in range(num_experts):
        metadata[e] = len(routes_by_expert[e])
        for rank, route in enumerate(routes_by_expert[e]):
            metadata[num_experts + route] = rank
            if rank < num_tokens:
                metadata[num_experts + r + e * num_tokens + rank] = route
    return torch.tensor(metadata, dtype=torch.int32, device=expert_ids.device)


# ---------- Kernel ----------
@functools.lru_cache(maxsize=64)
def _indexed_expert_gemm(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    n: int,
    k: int,
    dtype: str,
    activation: str,
    route_input: bool,
    dispatch_mode: str,
):
    """Build the mixed Cube+Vector indexed expert GEMM factory (DESIGN 3.3).

    Returns the ``@tilelang.jit`` factory; the A5 wrapper calls
    ``_indexed_expert_gemm(...)(**config)(a, weights, expert_ids,
    route_metadata, out)`` with ``config == {"block_n", "block_k",
    "num_stages"}``.
    """
    num_routes = num_tokens * top_k
    gated = activation != "none"  # "silu_and_mul" / "none"
    weight_n = 2 * n if gated else n
    a_shape = (num_tokens, top_k, k) if route_input else (num_tokens, k)
    grouped = dispatch_mode == "grouped"  # iff T > 1 (DESIGN K-predicate)
    metadata_size = (num_experts + num_routes + num_experts * num_tokens) if grouped else 1
    rank_base = num_experts
    list_base = num_experts + num_routes
    block_m = _BLOCK_M

    @tilelang.jit(target="ascend", out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _kernel(block_n: int, block_k: int, num_stages: int):
        n_gemm = 2 * block_n if gated else block_n  # W tile N width (gated joins the halves)
        n_blocks = (n + block_n - 1) // block_n
        k_blocks = (k + block_k - 1) // block_k
        on_grid = (n % block_n == 0) and (k % block_k == 0)

        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            weights: T.Tensor((num_experts, weight_n, k), dtype),
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            route_metadata: T.Tensor((metadata_size,), "int32"),
            out: T.Tensor((num_tokens, top_k, n), dtype),
        ):
            with T.MixedKernel(num_routes * n_blocks, sids=2) as (pid, sid):
                # UB is per-AIV private; L1/L0C are common (DESIGN section 5.5).
                a_ub = T.alloc_shared((block_m, block_k), dtype)  # AIV0 private
                acc_ub = T.alloc_shared((block_m, n_gemm), "float32")  # AIV1 private
                st_ub = T.alloc_shared((block_m,), "int32")  # per-AIV route staging
                ss_ub = T.alloc_shared((block_m,), "int32")
                a_l1 = T.alloc_l1((block_m, block_k), dtype)
                w_l1 = T.alloc_l1((n_gemm, block_k), dtype)
                res = T.alloc_l0c((block_m, n_gemm), "float32")
                if not on_grid:  # Python-level: only emit W staging when off-grid
                    w_ub = T.alloc_shared((n_gemm, block_k), dtype)

                # ---- common scalar prologue (identical evaluation on all 3 sub-cores) ----
                route = pid // n_blocks
                nb = pid % n_blocks
                token = route // top_k
                slot = route % top_k
                expert = expert_ids[token, slot]
                route_count = T.alloc_var("int32", init=1)
                is_leader = T.alloc_var("int32", init=1)
                group_start = T.alloc_var("int32", init=0)
                if grouped:  # Python-level: no metadata code for direct dispatch
                    rank = route_metadata[rank_base + route]
                    if rank < num_tokens:
                        group_start = rank - rank % block_m
                        if rank != group_start:
                            is_leader = 0
                        route_count = T.min(
                            block_m, T.min(route_metadata[expert], num_tokens) - group_start
                        )

                # ---- AIV0: row-metadata staging (before the K pipeline) ----
                # Runs for *every* block (leader and non-leader alike): the
                # non-leader blocks of a group still gather the same rows, so
                # their (redundant) Cube result is identical to the leader's and
                # is simply never written back (see the epilogue guard below).
                if sid == 0:
                    if grouped:
                        for mi in T.serial(block_m):
                            if mi < route_count:
                                if route_count == 1:  # isolated leader / duplicate route -> self
                                    st_ub[mi] = route // top_k
                                    ss_ub[mi] = route % top_k
                                else:  # group row -> lists (in-bounds: mi < route_count)
                                    sel = route_metadata[
                                        list_base + expert * num_tokens + group_start + mi
                                    ]
                                    st_ub[mi] = sel // top_k
                                    ss_ub[mi] = sel % top_k
                    else:  # direct: no metadata
                        st_ub[0] = token
                        ss_ub[0] = slot

                # ---- K soft-pipeline (AIV0 produces A; AIC loads W + MAC) ----
                # The pipeline and the Cube MAC run unconditionally on every
                # block: a runtime `is_leader` guard wrapping `T.gemm` would put
                # the `res` L0C accumulator's write and read under differing
                # conditional guards, which crashes `PrepareMultiBuffer`'s
                # storage-epoch analysis (arith simplifier folds
                # `cast(bool, sub(...))` to an out-of-range bool literal).
                for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                    if sid == 0:  # AIV0: gather a_ub -> a_l1
                        with T.SimtVF(threads=128):
                            for mi in T.serial(block_m):
                                for ki in T.Parallel(block_k):
                                    offset = ko * block_k + ki
                                    if mi < route_count and (on_grid or offset < k):
                                        if route_input:
                                            a_ub[mi, ki] = a[st_ub[mi], ss_ub[mi], offset]
                                        else:
                                            a_ub[mi, ki] = a[st_ub[mi], offset]
                                    else:  # deterministic zero-fill for OOB rows/cols
                                        a_ub[mi, ki] = T.Cast(dtype, 0)
                        T.copy(a_ub, a_l1)  # UB -> L1 (MTE3)
                        if not on_grid:  # off-grid: per-element bounded W load (DESIGN 9.4 risk 5 fallback)
                            # A runtime-extent `T.copy` reads OOB past k/n (the
                            # copy engine rounds the clamped extent up to its
                            # granularity), leaking NaN into the padded W tile.
                            # Load every W element with per-element bounds checks,
                            # zero-filling the padded rows/cols deterministically.
                            rb = T.max(T.min(n - nb * block_n, block_n), 0)
                            kb = T.min(k - ko * block_k, block_k)
                            with T.SimtVF(threads=128):
                                if gated:
                                    for ri in T.serial(n_gemm):
                                        for ki in T.Parallel(block_k):
                                            ln = ri % block_n
                                            src_n = nb * block_n + ln + (ri // block_n) * n
                                            col = ko * block_k + ki
                                            if ln < rb and ki < kb:
                                                w_ub[ri, ki] = weights[expert, src_n, col]
                                            else:
                                                w_ub[ri, ki] = T.Cast(dtype, 0)
                                else:
                                    for ri in T.serial(n_gemm):
                                        for ki in T.Parallel(block_k):
                                            col = ko * block_k + ki
                                            if ri < rb and ki < kb:
                                                w_ub[ri, ki] = weights[expert, nb * block_n + ri, col]
                                            else:
                                                w_ub[ri, ki] = T.Cast(dtype, 0)
                            T.copy(w_ub, w_l1)  # full tile UB -> L1 (rows 16-divisible)
                    # ---- common Cube segment (AIC, outside sid guards) ----
                    # round3_leaderload: leader blocks only.  The guard wraps
                    # ONLY the W copies -- NOT ``T.gemm`` (a runtime guard around
                    # T.gemm puts the res L0C accumulator's write and read under
                    # differing guards and crashes PrepareMultiBuffer's
                    # storage-epoch analysis).  Non-leader blocks keep whatever
                    # stale w_l1 content is present; their tile is never written
                    # back, so the stale data is unobservable.
                    if on_grid and is_leader != 0:
                        T.copy(
                            weights[
                                expert,
                                nb * block_n : (nb + 1) * block_n,
                                ko * block_k : (ko + 1) * block_k,
                            ],
                            w_l1[0:block_n, :],
                        )
                        if gated:
                            T.copy(
                                weights[
                                    expert,
                                    n + nb * block_n : n + (nb + 1) * block_n,
                                    ko * block_k : (ko + 1) * block_k,
                                ],
                                w_l1[block_n : 2 * block_n, :],
                            )
                    T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))

                # ---- AIV1: epilogue (after the K loop, res fully accumulated) ----
                # Only the leader block of each group writes `out`; non-leader
                # blocks of a group would produce the identical tile and are
                # skipped here to avoid duplicate/racing writes.
                if sid == 1:
                    if is_leader != 0:
                        if grouped:  # AIV1-private staging copy (identical)
                            for mi in T.serial(block_m):
                                if mi < route_count:
                                    if route_count == 1:
                                        st_ub[mi] = route // top_k
                                        ss_ub[mi] = route % top_k
                                    else:
                                        sel = route_metadata[
                                            list_base + expert * num_tokens + group_start + mi
                                        ]
                                        st_ub[mi] = sel // top_k
                                        ss_ub[mi] = sel % top_k
                        else:
                            st_ub[0] = token
                            ss_ub[0] = slot
                        T.copy(res, acc_ub)  # L0C -> UB (Fixpipe, routes to AIV1)
                        with T.SimtVF(threads=128):
                            for mi in T.serial(block_m):
                                for j in T.Parallel(block_n):
                                    col = nb * block_n + j
                                    if mi < route_count and (on_grid or col < n):
                                        if gated:
                                            gate = acc_ub[mi, j]
                                            value = (
                                                gate / (1.0 + T.exp(-gate)) * acc_ub[mi, block_n + j]
                                            )
                                        else:
                                            value = acc_ub[mi, j]
                                        out[st_ub[mi], ss_ub[mi], col] = T.Cast(dtype, value)

        return main

    return _kernel


# ---------- Layered testing ----------
def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]  # "float16" / "bfloat16"


def _compile(num_tokens, top_k, num_experts, n, k, dtype, activation, route_input, dispatch_mode,
             block_n, block_k, num_stages):
    return _indexed_expert_gemm(
        num_tokens, top_k, num_experts, n, k, _dtype_str(dtype), activation,
        route_input, dispatch_mode,
    )(block_n=block_n, block_k=block_k, num_stages=num_stages)


def _run_case(num_tokens, top_k, num_experts, n, k, dtype, activation, route_input,
              dispatch_mode, block_n, block_k, num_stages, tag, *, seed=0,
              expert_ids=None, random_bijection=False, equal_nan=False):
    """Compile, run, and compare one configuration against the golden."""
    torch.manual_seed(seed)
    a_shape = (num_tokens, top_k, k) if route_input else (num_tokens, k)
    a = torch.randn(a_shape, dtype=dtype).to("npu")
    weight_n = 2 * n if activation != "none" else n
    weights = torch.randn(num_experts, weight_n, k, dtype=dtype).to("npu")
    if expert_ids is None:
        expert_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32).to("npu")
    if dispatch_mode == "grouped":
        route_metadata = _make_route_metadata(
            expert_ids, num_experts, num_tokens,
            random_bijection=random_bijection, seed=seed,
        )
    else:
        route_metadata = expert_ids.reshape(-1)[:1].contiguous()
    kernel = _compile(num_tokens, top_k, num_experts, n, k, dtype, activation, route_input,
                      dispatch_mode, block_n, block_k, num_stages)
    out = torch.empty(num_tokens, top_k, n, dtype=dtype).to("npu")
    kernel(a, weights, expert_ids, route_metadata, out)
    torch.npu.synchronize()
    golden = golden_indexed_expert_gemm(a, weights, expert_ids,
                                        activation=activation, route_input=route_input)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2, equal_nan=equal_nan)
    max_diff = (out.float() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: T={num_tokens} K={top_k} E={num_experts} n={n} k={k} "
          f"dtype={_dtype_str(dtype)} {activation} route_input={route_input} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    # 1. Compile smoke: manifest-scale on-grid specializations lower without capacity errors.
    # gate/up: n=F=2048, k=H=7168 (weights [E,2F,H]); down: n=H=7168, k=F=2048.
    _compile(32, 8, 256, 2048, 7168, torch.float16, "silu_and_mul", False, "grouped", 64, 128, 2)
    _compile(32, 8, 256, 7168, 2048, torch.float16, "none", True, "grouped", 128, 128, 2)
    _compile(32, 8, 256, 2048, 7168, torch.bfloat16, "silu_and_mul", False, "grouped", 64, 128, 2)
    print("[L0-compile-smoke] PASS: manifest-scale specializations lower")

    # 2. Main path (grouped, on-grid, balanced): ds-v3-t32-shaped (reduced dims).
    _run_case(8, 2, 8, 128, 512, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "L0-grouped-gateup", seed=0)
    _run_case(8, 2, 8, 256, 256, torch.float16, "none", True, "grouped",
              128, 128, 2, "L0-grouped-down", seed=1)

    # 3. Direct mode (T == 1, placeholder metadata).
    _run_case(1, 8, 8, 128, 512, torch.float16, "silu_and_mul", False, "direct",
              64, 128, 2, "L0-direct-gateup", seed=2)
    _run_case(1, 8, 8, 256, 256, torch.float16, "none", True, "direct",
              128, 128, 2, "L0-direct-down", seed=3)

    # 4. Off-grid (clamp + zero-fill path).
    _run_case(4, 2, 8, 96, 100, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "L0-offgrid-gateup", seed=4)
    _run_case(4, 2, 8, 100, 96, torch.float16, "none", True, "grouped",
              128, 128, 2, "L0-offgrid-down", seed=5)

    # 5. bf16 smoke.
    _run_case(8, 2, 8, 128, 512, torch.bfloat16, "silu_and_mul", False, "grouped",
              64, 128, 2, "L0-bf16-gateup", seed=6)


def _gen_expert_ids(pattern, num_tokens, top_k, num_experts, seed):
    g = torch.Generator().manual_seed(seed)
    r = num_tokens * top_k
    if pattern == "balanced":
        ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=g)
    elif pattern == "monopoly":
        ids = torch.zeros(num_tokens, top_k)
    elif pattern == "duplicate":  # > num_tokens routes to expert 0 -> rank >= T fallback
        flat = torch.zeros(r, dtype=torch.int64)
        flat[12:] = 2
        ids = flat.reshape(num_tokens, top_k)
    elif pattern == "sparse":  # only experts {0, 3, num_experts-1} active
        pool = torch.tensor([0, 3, num_experts - 1], dtype=torch.int64)
        ids = pool[torch.randint(0, pool.numel(), (num_tokens, top_k), generator=g)]
    elif pattern == "zipf":  # heavy skew toward low expert ids
        weights = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64) ** 1.1
        ids = torch.multinomial(weights, r, replacement=True, generator=g).reshape(
            num_tokens, top_k)
    else:  # random
        ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=g)
    return ids.to(torch.int32).to("npu")


def _run_pattern(pattern, tag, gateup=True, down=True):
    t, k_, e = 8, 2, 8
    for bijection, suffix in ((False, "stable"), (True, "random-bij")):
        ids = _gen_expert_ids(pattern, t, k_, e, seed=hash((pattern, suffix)) & 0xFFFF)
        if gateup:
            _run_case(t, k_, e, 128, 512, torch.float16, "silu_and_mul", False, "grouped",
                      64, 128, 2, f"{tag}-gateup-{suffix}", expert_ids=ids,
                      random_bijection=bijection)
        if down:
            _run_case(t, k_, e, 256, 256, torch.float16, "none", True, "grouped",
                      128, 128, 2, f"{tag}-down-{suffix}", expert_ids=ids,
                      random_bijection=bijection)


def run_L1():
    # Full routing-pattern matrix x stable/random bijection metadata.
    for pattern in ("balanced", "monopoly", "duplicate", "sparse", "zipf", "random"):
        _run_pattern(pattern, f"L1-{pattern}")

    # Degenerate shapes.
    ids0 = torch.zeros(8, 2, dtype=torch.int32).to("npu")  # E == 1
    _run_case(8, 2, 1, 128, 512, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "L1-E1", expert_ids=ids0)
    ids1 = torch.randint(0, 8, (8, 1), dtype=torch.int32).to("npu")  # K == 1
    _run_case(8, 1, 8, 128, 512, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "L1-K1", expert_ids=ids1)

    # Determinism: same input twice -> bitwise identical.
    torch.manual_seed(9)
    a = torch.randn(8, 512, dtype=torch.float16).to("npu")
    w = torch.randn(8, 256, 512, dtype=torch.float16).to("npu")
    ids = torch.randint(0, 8, (8, 2), dtype=torch.int32).to("npu")
    meta = _make_route_metadata(ids, 8, 8, seed=0)
    kern = _compile(8, 2, 8, 128, 512, torch.float16, "silu_and_mul", False, "grouped",
                    64, 128, 2)
    outs = []
    for _ in range(2):
        out = torch.empty(8, 2, 128, dtype=torch.float16).to("npu")
        kern(a, w, ids, meta, out)
        torch.npu.synchronize()
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1]), "non-bitwise-deterministic output"
    print("[L1-determinism] PASS: 2x bitwise identical")


def run_L2():
    # Op-level integration: real _route_stats metadata (loaded from the sibling
    # module, which ships no package __init__.py) drives a larger grouped case;
    # failures recorded as warnings (non-blocking).
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "_route_stats", "_route_stats.py",
    )
    try:
        spec = importlib.util.spec_from_file_location("_route_stats_impl", candidate)
        rs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs)
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record without blocking): cannot load _route_stats: {e}")
        return
    for t, k_, n, k, bn, bk, ns in ((16, 4, 256, 512, 64, 128, 2), (32, 8, 256, 512, 128, 128, 2)):
        try:
            e = 16
            ids = torch.randint(0, e, (t, k_), dtype=torch.int32).to("npu")
            meta = rs.golden_route_stats(ids, e)
            a = torch.randn(t, 512, dtype=torch.float16).to("npu")
            weights = torch.randn(e, 2 * 256, 512, dtype=torch.float16).to("npu")
            kern = _compile(t, k_, e, 256, 512, torch.float16, "silu_and_mul", False,
                            "grouped", bn, bk, ns)
            out = torch.empty(t, k_, 256, dtype=torch.float16).to("npu")
            kern(a, weights, ids, meta, out)
            torch.npu.synchronize()
            golden = golden_indexed_expert_gemm(a, weights, ids,
                                                activation="silu_and_mul", route_input=False)
            torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
            print(f"[L2] PASS: real metadata T={t} K={k_} n={n}")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (not blocking): T={t} K={k_}: {e}")


def run_boundary():
    # off-grid with block_n == n edge (single partial W row block).
    _run_case(4, 2, 4, 64, 128, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "Boundary-n-eq-bn", seed=10)
    # k smaller than a single block.
    _run_case(4, 2, 4, 128, 64, torch.float16, "none", True, "grouped",
              128, 128, 2, "Boundary-k-lt-bk", seed=11)
    # duplicate route overflow: every route maps to expert 0 (rank runs full R).
    ids = torch.zeros(4, 2, dtype=torch.int32).to("npu")
    _run_case(4, 2, 4, 128, 256, torch.float16, "silu_and_mul", False, "grouped",
              64, 128, 2, "Boundary-all-same", expert_ids=ids)


def _show_source():
    kernel = _compile(2, 2, 4, 128, 128, torch.float16, "silu_and_mul", False, "grouped",
                      64, 128, 2)
    print(kernel.get_kernel_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--show-source", action="store_true",
                        help="print the generated AscendC kernel source")
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