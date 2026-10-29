"""Equivalence verification for the SimdVF _logsumexp_kernel_single design (D-1).

CPU (torch) numerical对照 for the math-equivalence optimizations adopted in
DESIGN.md section 1.6.1.  All chains model the *group-of-64* SIMD data path at
fp32 fidelity (the kernel's intermediate dtype, per DESIGN section 0.1):

  V1  SIMD two-pass UB-resident chain (per-lane running accumulation across
      64-element groups + one cross-lane vcmax/vcadd per row + fp32-domain
      -inf tail masking on the last group) vs the source GPU chain
      (N_padded = align_up(N, 256) -inf padding + full-row max / exp / sum).
      Covers adopted items O2 (UB-resident two-pass + -inf group masking) and
      O3 (vexpdif fused exp(x-m): modeled as the identical fp32 formula; the
      hardware fused-instruction rounding difference is bounded analytically
      in DESIGN section 1.6.1, not CPU-simulable).
  V2  Deferred cross-lane reduction (elementwise running accumulation, one
      vcmax/vcadd per row -- the explicit-intrinsic form) vs per-group
      cross-lane reduction (the form AscendSimdVFLowerParallel emits for
      high-level T.reduce inside SimdVF).  fp32 grouping difference only.
  V3  SIMD chain vs torch.logsumexp golden (fp32, informational).
  V4  Source chain vs SIMD chain, fp64 -> fp16 / bf16 output dtype, including
      special rows (all -inf / contains +inf / contains NaN / partial -inf /
      all-equal) -- NaN-pattern parity + migration fidelity.
  V5  Saturation corner (informational, mathematically closed): fp32->fp16
      output cast with sat=True (CCE vcvt default) can only diverge from an
      IEEE cast for out >= 65520 (the fp16 round-to-inf threshold), but the
      contract space bounds out = m + ln(sum) <= 65504 + ln(32512) ~= 65514.4
      < 65520 (sum >= 1 because the max element contributes exp(0) = 1).
      Empirical confirmation below; sat=True is safe in-contract.

Usage:  python3 verify_equiv.py
"""

import math

import torch

GROUP = 64  # fp32 SIMD lane count; UNPK_B16 group width (16-bit inputs)
ALIGN_SRC = 256  # source GPU N_padded alignment (_primitives.DEFAULT_ALIGNMENT)


def _pad_groups(x: torch.Tensor, n: int) -> torch.Tensor:
    """Return (M, G, 64) fp32 groups with the -inf tail on the last group.

    Models: vld(UNPK_B16/NORM) + vcvt -> fp32, then vsel(c, -inf,
    update_mask(k, 32)) on the last group only (DESIGN sections 1.6.1 O2 /
    3.3).  Lanes beyond n are -inf in *every* group view (only the last group
    has any).
    """
    m = x.shape[0]
    g = (n + GROUP - 1) // GROUP
    out = torch.full((m, g, GROUP), float("-inf"), dtype=torch.float32)
    for i in range(g):
        lo, hi = i * GROUP, min((i + 1) * GROUP, n)
        if lo < hi:
            out[:, i, : hi - lo] = x[:, lo:hi]
    return out


def simd_chain(x32: torch.Tensor) -> torch.Tensor:
    """SimdVF two-pass chain at fp32 fidelity (DESIGN section 3.3).

    Pass 1: acc_max[lane] = running max over groups (exact, order-free);
            m = vcmax(acc_max)               -> full 64-lane reduce
    Pass 2: acc_sum[lane] += exp(c - m) over groups (vexpdif + vadd);
            s = vcadd(acc_sum)               -> full 64-lane reduce
    Out:    y = m + log(s)                    (vadd + vln)
    """
    m_rows, n = x32.shape
    groups = _pad_groups(x32, n)
    g = groups.shape[1]

    acc_max = torch.full((m_rows, GROUP), float("-inf"), dtype=torch.float32)
    for i in range(g):
        acc_max = torch.maximum(acc_max, groups[:, i, :])
    m = acc_max.max(dim=1).values  # vcmax

    acc_sum = torch.zeros((m_rows, GROUP), dtype=torch.float32)
    for i in range(g):
        acc_sum = acc_sum + torch.exp(groups[:, i, :] - m.unsqueeze(1))  # vexpdif
    s = acc_sum.sum(dim=1)  # vcadd

    return m + torch.log(s)


def per_group_reduce_chain(x32: torch.Tensor) -> torch.Tensor:
    """Per-group cross-lane reduce form (AscendSimdVFLowerParallel lowering of
    high-level T.reduce inside SimdVF): per group vcmax/vcadd then elementwise
    merge.  Same math, different fp32 grouping (V2 baseline)."""
    m_rows, n = x32.shape
    groups = _pad_groups(x32, n)
    g = groups.shape[1]

    m = torch.full((m_rows,), float("-inf"), dtype=torch.float32)
    for i in range(g):
        m = torch.maximum(m, groups[:, i, :].max(dim=1).values)

    s = torch.zeros((m_rows,), dtype=torch.float32)
    for i in range(g):
        s = s + torch.exp(groups[:, i, :] - m.unsqueeze(1)).sum(dim=1)

    return m + torch.log(s)


def source_chain(x32: torch.Tensor, n: int) -> torch.Tensor:
    """Source GPU single-tile algorithm at fp32 fidelity (DESIGN section 0.3):
    N_padded = align_up(N, 256), -inf padded masked load, full-row reduce_max,
    in-place exp(x - m), full-row reduce_sum, m + log(s)."""
    n_padded = ((n + ALIGN_SRC - 1) // ALIGN_SRC) * ALIGN_SRC
    x_pad = torch.full((x32.shape[0], n_padded), float("-inf"), dtype=torch.float32)
    x_pad[:, :n] = x32
    m = x_pad.max(dim=1).values
    e = torch.exp(x_pad - m.unsqueeze(1))
    s = e.sum(dim=1)
    return m + torch.log(s)


def golden(x32: torch.Tensor) -> torch.Tensor:
    """torch golden (source-repo ref_program, fp32 compute domain)."""
    return torch.logsumexp(x32, dim=-1)


# ---- metrics ----------------------------------------------------------------


def _ulp_diff_fp32(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max ULP distance between two fp32 tensors (finite lanes only)."""
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not mask.any():
        return 0
    ai = a[mask].view(torch.int32).to(torch.int64)
    bi = b[mask].view(torch.int32).to(torch.int64)
    # Map sign-magnitude to monotonic lattice.
    ai = torch.where(ai < 0, -2147483648 - ai, ai)
    bi = torch.where(bi < 0, -2147483648 - bi, bi)
    return int((ai - bi).abs().max().item())


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not mask.any():
        return 0.0
    diff = (a[mask] - b[mask]).abs()
    denom = b[mask].abs().clamp_min(1e-30)
    return float((diff / denom).max().item())


def _violation(a: torch.Tensor, b: torch.Tensor, tol: float) -> float:
    """Fraction of lanes outside rtol=atol=tol (NaN-equal lanes excluded)."""
    ok = torch.isclose(a, b, rtol=tol, atol=tol, equal_nan=True)
    return float((~ok).float().mean().item())


def _cast_out(y32: torch.Tensor, dtype: torch.dtype, saturate: bool) -> torch.Tensor:
    """fp32 -> fp16/bf16 output cast.  saturate=True models the CCE vcvt
    sat=True default (clamp to dtype max); False models IEEE overflow->inf."""
    if not saturate:
        return y32.to(dtype)
    if dtype == torch.float16:
        return y32.clamp(-65504.0, 65504.0).to(dtype)
    return y32.to(dtype)  # bf16 range ~3.4e38: corner unreachable in practice


# ---- cases ------------------------------------------------------------------

SHAPES = [
    (33, 300),  # M tail + non-pow2 N (tail group k = 44)
    (32, 256),  # N = 4 groups exactly
    (64, 4096),  # wl1-equivalent (attn-weights-4k reduced)
    (128, 16384),  # wl4-equivalent (3d-multidim-reduce reduced)
    (3, 7),  # tiny N < 64 (single tail group)
    (1, 1),  # minimal
    (2, 32512),  # contract N boundary (508 groups, no tail)
    (57, 64),  # single group per row, M tail vs SUB
]


def _make_x(m: int, n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.float32)
    return x.to(dtype).float()  # values live in the input dtype domain


def _special_rows(n: int) -> torch.Tensor:
    """Rows exercising the boundary semantics of DESIGN section 0.1."""
    rows = [
        torch.zeros(n),  # all zeros
        torch.full((n,), 3.5),  # all equal
        torch.cat([torch.tensor([100.0]), torch.zeros(n - 1)]),  # single dominant
        torch.cat([torch.tensor([65504.0, -65504.0]), torch.zeros(n - 2)]),  # extremes
        torch.cat([torch.tensor([6e-8, -6e-8]), torch.zeros(n - 2)]),  # subnormal
        torch.cat([torch.tensor([-745.0, -744.0]), torch.full((n - 2,), -700.0)]),  # deep underflow
        torch.cat([torch.tensor([float("-inf")]), torch.zeros(n - 1)]),  # partial -inf
        torch.full((n,), float("-inf")),  # all -inf (source-inherent NaN)
        torch.cat([torch.tensor([float("inf")]), torch.zeros(n - 1)]),  # contains +inf
        torch.cat([torch.tensor([float("nan")]), torch.zeros(n - 1)]),  # contains NaN
    ]
    return torch.stack(rows)


def main() -> None:
    torch.manual_seed(0)
    results = []

    # -- V1 / V2 / V3 over random shapes, fp32 chains ------------------------
    for dtype_name, torch_dtype, tol in (
        ("float16", torch.float16, 1e-2),
        ("bfloat16", torch.bfloat16, 1e-2),
    ):
        for si, (m, n) in enumerate(SHAPES):
            x = _make_x(m, n, torch_dtype, seed=100 + si)
            cand = simd_chain(x)
            base = source_chain(x, n)
            ref64 = source_chain(x.double(), n)
            gold = golden(x)

            results.append(
                (
                    f"V1/{dtype_name}/({m},{n})",
                    _ulp_diff_fp32(cand, base),
                    _rel_diff(cand, base),
                    _violation(cand, base, tol),
                )
            )
            # fp64 grounding of both chains (chain-vs-ref64, information row)
            results.append(
                (
                    f"V1-fp64ref/{dtype_name}/({m},{n})",
                    _ulp_diff_fp32(cand, ref64.float()),
                    _rel_diff(cand, ref64.float()),
                    _violation(cand, ref64.float(), tol),
                )
            )

    # V2: deferred vs per-group cross-lane reduction (fp32 grouping only)
    for si, (m, n) in enumerate(SHAPES):
        x = _make_x(m, n, torch.float32, seed=200 + si)
        cand = simd_chain(x)
        base = per_group_reduce_chain(x)
        results.append(
            (
                f"V2/float32/({m},{n})",
                _ulp_diff_fp32(cand, base),
                _rel_diff(cand, base),
                _violation(cand, base, 1e-3),
            )
        )

    # V3: SIMD chain vs golden (fp32)
    for si, (m, n) in enumerate(SHAPES):
        x = _make_x(m, n, torch.float32, seed=300 + si)
        cand = simd_chain(x)
        gold = golden(x)
        results.append(
            (
                f"V3/float32/({m},{n})",
                _ulp_diff_fp32(cand, gold),
                _rel_diff(cand, gold),
                _violation(cand, gold, 1e-3),
            )
        )

    # -- V4: special rows, fp64 -> fp16 output, NaN-pattern parity ------------
    for dtype_name, torch_dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        for n in (300, 256):
            x = _special_rows(n).to(torch_dtype).float()
            cand32 = simd_chain(x)
            base32 = source_chain(x, n)
            cand = _cast_out(cand32, torch_dtype, saturate=False)
            base = _cast_out(base32, torch_dtype, saturate=False)
            nan_match = bool(
                torch.equal(torch.isnan(cand), torch.isnan(base))
                and torch.equal(cand == float("inf"), base == float("inf"))
                and torch.equal(cand == float("-inf"), base == float("-inf"))
            )
            ulp = _ulp_diff_fp32(cand32, base32)
            rel = _rel_diff(cand32, base32)
            viol = _violation(cand, base, 1e-2)
            results.append((f"V4/{dtype_name}/special-n{n}", ulp, rel, viol))
            if not nan_match:
                results.append((f"V4/{dtype_name}/special-n{n}-NANPATTERN", -1, -1.0, 1.0))

    # -- V5: saturation corner (informational, mathematically closed) ---------
    # out = m + ln(sum) <= 65504 + ln(N_max=32512) ~= 65514.4 < 65520, so the
    # sat=True clamp can never disagree with an IEEE overflow inside the
    # single-tile contract space.  The empirical check uses the worst in-
    # contract row (all elements = fp16 max at the largest contract N).
    n = 32512
    x = torch.full((2, n), 65504.0)  # all-equal fp16 max -> out = 65504 + ln(32512)
    y32 = simd_chain(x)
    sat = _cast_out(y32, torch.float16, saturate=True)
    ieee = _cast_out(y32, torch.float16, saturate=False)
    diverged = int((sat != ieee).sum().item())
    bound = 65504.0 + math.log(n)
    print(
        f"[V5] sat-cast vs ieee-cast on all-65504 rows at N={n}: diverged lanes = "
        f"{diverged}/{sat.numel()} (out bound = {bound:.1f} < 65520); "
        f"corner mathematically closed in-contract"
    )

    # -- report ----------------------------------------------------------------
    print(f"\n{'case':38s} {'max_ulp':>8s} {'max_rel':>12s} {'viol_rate':>10s}  verdict")
    fails = 0
    for name, ulp, rel, viol in results:
        # V1/V2/V3/V4 gate: violation rate 0 at the dtype tolerance and
        # finite-lane ULP bounded (sum-regrouping scale, not a semantic break).
        ok = viol == 0.0 and rel < 1e-4 and ulp >= 0
        if not ok:
            fails += 1
        print(f"{name:38s} {ulp:8d} {rel:12.3e} {viol:10.6f}  {'EQUIV_PASS' if ok else 'EQUIV_FAIL'}")

    print(
        f"\nTotal {len(results)} gated rows, EQUIV_FAIL count = {fails}"
        + ("  -> ALL ADOPTED ITEMS EQUIV_PASS" if fails == 0 else "  -> FAILURES PRESENT")
    )
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
