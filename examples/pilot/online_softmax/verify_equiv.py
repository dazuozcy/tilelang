"""Equivalence machine verification for online_softmax DESIGN.md section 1.6.1 (D-1).

Adopted optimization items (vs the GPU source formula in
examples/online_softmax/online_softmax.py):
  V1 (O3, drop log2e rebase):  e = exp(x - m)                vs exp2(x*s - m*s), s = 1.44269504
  V2 (O2, reciprocal-multiply): y = e * (1/S)                vs e / S
  V3 (O1, resident single-eval chain, full): fp16(e * (1/S)) vs fp16(exp2(x*s - lse2))
      (source recomputes exp in pass 2; resident path reuses e; both vs exact softmax)
  V4 (streaming online update, e-base lse): 2-chunk online   vs direct full-row
  V5 (informational, NOT adopted - fp16-e alternate of section 1.6.4):
      simulated fp16-precision chain error vs exact (precision guard evidence)

Method: all chains evaluated in float64 on torch CPU (random tensors + corner
values), then rounded to float16 for tolerance assessment (atol=rtol=1e-2,
fp16 output contract). fp16 inputs are modeled by pre-rounding x to fp16.
"""

import numpy as np
import torch

torch.manual_seed(20261016)
SCALE = 1.44269504  # source literal (truncated log2(e)), source L14
ATOL = RTOL = 1e-2  # fp16 tolerance per AGENTS.md API Convention


def exact_softmax_f64(x: torch.Tensor) -> torch.Tensor:
    """fp64 exact softmax (ground truth)."""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def source_chain_f64(x: torch.Tensor) -> torch.Tensor:
    """Source formula evaluated in fp64: y = exp2(x*s - lse2), single chunk."""
    s = SCALE
    m = x.amax(dim=-1, keepdim=True)
    e2 = torch.exp2(x * s - m * s)
    lse = m * s + torch.log2(e2.sum(dim=-1, keepdim=True))
    return torch.exp2(x * s - lse)


def resident_chain_f64(x: torch.Tensor) -> torch.Tensor:
    """Adopted resident formula in fp64: m -> e=exp(x-m) -> S -> y=e*(1/S)."""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e * (1.0 / s)


def resident_chain_div_f64(x: torch.Tensor) -> torch.Tensor:
    """O2 baseline arm: y = e / S (direct division)."""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def exp2_rebase_f64(x: torch.Tensor) -> torch.Tensor:
    """O3 baseline arm: e = exp2(x*s - m*s) (rebased exp, no lse)."""
    s = SCALE
    m = x.amax(dim=-1, keepdim=True)
    return torch.exp2(x * s - m * s)


def exp_ebase_f64(x: torch.Tensor) -> torch.Tensor:
    """O3 candidate arm: e = exp(x - m)."""
    m = x.amax(dim=-1, keepdim=True)
    return torch.exp(x - m)


def online_2chunk_f64(x: torch.Tensor) -> torch.Tensor:
    """Streaming path (N > resident cap): ROBUST e-base online merge over 2
    chunks (flash-attention form + fp16 finite sentinel clamp), then pass-2
    normalize y = exp(x - lse).

    Robustness vs the source's naive merge (m_c-based lse update):
      - chunk max gap > ~88.7 would overflow exp() in fp32 (fp16 data range
        spans 131008) -> the max-based rescale form keeps every exp arg <= 0;
      - an all-(-inf) chunk yields e = exp(-inf - (-inf)) = NaN -> clamping
        the chunk max to the fp16 finite sentinel (-65504) makes those lanes
        exp(-inf) = 0 (sum-neutral), matching torch semantics.
    Degenerate N < 2 falls back to the direct single-chunk form."""
    n = x.shape[-1]
    if n < 2:
        return resident_chain_f64(x)
    c = n - n // 2  # uneven split to exercise the merge
    sentinel = -65504.0  # fp16 finite lower bound
    m1 = x[..., :c].amax(dim=-1, keepdim=True).clamp_min(sentinel)
    e1 = torch.exp(x[..., :c] - m1)
    s1 = e1.sum(dim=-1, keepdim=True)
    m2 = x[..., c:].amax(dim=-1, keepdim=True).clamp_min(sentinel)
    e2 = torch.exp(x[..., c:] - m2)
    s2 = e2.sum(dim=-1, keepdim=True)
    # flash-form running statistics merge (all exp args <= 0)
    m_run = torch.maximum(m1, m2)
    s_run = s1 * torch.exp(m1 - m_run) + s2 * torch.exp(m2 - m_run)
    lse = m_run + torch.log(s_run)
    return torch.exp(x - lse)


def online_2chunk_naive_f64(x: torch.Tensor) -> torch.Tensor:
    """The SOURCE's naive merge (lse = m_c + log(exp(lse_old - m_c) + s_c)),
    kept as diagnostic evidence: diverges on extreme corner rows (see V4b)."""
    n = x.shape[-1]
    if n < 2:
        return resident_chain_f64(x)
    c = n - n // 2
    m1 = x[..., :c].amax(dim=-1, keepdim=True)
    e1 = torch.exp(x[..., :c] - m1)
    lse = m1 + torch.log(e1.sum(dim=-1, keepdim=True))
    m2 = x[..., c:].amax(dim=-1, keepdim=True)
    e2 = torch.exp(x[..., c:] - m2)
    lse = m2 + torch.log(torch.exp(lse - m2) + e2.sum(dim=-1, keepdim=True))
    return torch.exp(x - lse)


def online_2chunk_naive_f32(x: torch.Tensor) -> torch.Tensor:
    """The SOURCE's naive merge with fp32 intermediates (as the kernel would
    compute them) - diagnostic for the chunk-max-gap overflow corner."""
    x32 = x.to(torch.float32)
    n = x32.shape[-1]
    if n < 2:
        return resident_chain_f64(x)
    c = n - n // 2
    m1 = x32[..., :c].amax(dim=-1, keepdim=True)
    e1 = torch.exp(x32[..., :c] - m1)
    lse = m1 + torch.log(e1.sum(dim=-1, keepdim=True))
    m2 = x32[..., c:].amax(dim=-1, keepdim=True)
    e2 = torch.exp(x32[..., c:] - m2)
    lse = m2 + torch.log(torch.exp(lse - m2) + e2.sum(dim=-1, keepdim=True))
    return torch.exp(x32 - lse).to(torch.float64)


def fp16e_chain_sim(x: torch.Tensor) -> torch.Tensor:
    """Section 1.6.4 alternate (fp16-e) simulated precision chain:
    fp16 x -> exact max -> fp16 arg rounding -> fp16 e -> y = e / S(S in fp64).
    Upper-bounds the arg/exp rounding; tree-sum error analyzed separately."""
    x16 = x.to(torch.float16).to(torch.float64)
    m = x16.amax(dim=-1, keepdim=True)
    arg16 = (x16 - m).to(torch.float16).to(torch.float64)
    e16 = torch.exp(arg16).to(torch.float16).to(torch.float64)
    return e16 / e16.sum(dim=-1, keepdim=True)


def make_cases() -> dict:
    cases = {}
    # random tensors (values pre-rounded to the fp16 input contract)
    for name, shape, std in [
        ("rand_64x8192_s1", (64, 8192), 1.0),
        ("rand_8x8192_s10", (8, 8192), 10.0),
        ("rand_3x1000_s3", (3, 1000), 3.0),
        ("rand_2x17_s1", (2, 17), 1.0),
        ("rand_1x1", (1, 1), 1.0),
        ("rand_5x2", (5, 2), 1.0),
    ]:
        cases[name] = (torch.randn(*shape, dtype=torch.float64) * std).to(torch.float16).to(torch.float64)
    # corner rows (fp16-representable values)
    corner = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],                      # all-equal
            [1.0e4, 1.0e-3, -1.0e4, 0.0],              # single dominant
            [65504.0, -65504.0, 6.0e-8, -6.0e-8],      # fp16 max / min subnormal
            [30000.0, 29999.0, 29998.0, -30000.0],     # near-tie large magnitudes
            [-745.0, -746.0, -750.0, -800.0],          # deep underflow of exp
            [1.0, 2.0, 3.0, float("inf")],             # mixed +inf
            [float("-inf"), float("-inf"), -1.0, 0.0], # -inf lanes
            [float("nan"), 1.0, 2.0, 3.0],             # NaN lane
            [5.0, 5.0, 5.0, 5.0],                      # constant row
            [0.1, 0.2, 0.3, 0.4],                      # benign
            [0.0, 0.0, -100.0, -100.0],                # chunk-max gap 100 > 88.7 (fp32 exp overflow in naive merge)
        ],
        dtype=torch.float64,
    )
    cases["corner_rows"] = corner
    cases["corner_single_col"] = torch.tensor([[3.0], [-4.0], [float("inf")], [float("nan")]])
    return cases


def ulp_diff_f64(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max ULP distance between two fp64 tensors (bit-level, NaN-safe).
    Uses the monotone key mapping on the raw IEEE bit patterns (numpy view)."""
    def keys(t: torch.Tensor) -> np.ndarray:
        arr = torch.nan_to_num(t, nan=0.0).detach().cpu().numpy().astype(np.float64)
        bits = arr.view(np.uint64)
        # monotone ordering of IEEE doubles: negatives reversed, positives offset
        key = np.where(bits >> np.uint64(63), ~bits, bits | np.uint64(0x8000000000000000))
        return key.astype(np.int64)

    both_nan = torch.isnan(a) & torch.isnan(b)
    a2 = torch.where(both_nan, torch.zeros_like(a), a)
    b2 = torch.where(both_nan, torch.zeros_like(b), b)
    db = np.abs(keys(a2) - keys(b2))
    return int(db.max().item()) if db.size else 0


def fp16_mismatch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of elements violating assert_close(a16, b16, 1e-2, 1e-2).
    NaN==NaN treated as equal (equal_nan=True)."""
    a16 = a.to(torch.float16)
    b16 = b.to(torch.float16)
    ok = torch.isclose(a16.float(), b16.float(), atol=ATOL, rtol=RTOL, equal_nan=True)
    return 1.0 - ok.float().mean().item()


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    both_nan = torch.isnan(a) & torch.isnan(b)
    a2 = torch.where(both_nan | torch.isinf(a), torch.zeros_like(a), a)
    b2 = torch.where(both_nan | torch.isinf(a), torch.zeros_like(b), b)
    denom = b2.abs().clamp_min(1e-300)
    return float(((a2 - b2).abs() / denom).max().item())


def main() -> None:
    cases = make_cases()
    results = []

    def run(vid, desc, fn_a, fn_b, guard_exact=None):
        worst_ulp, worst_mis, worst_rel = 0, 0.0, 0.0
        for name, x in cases.items():
            a, b = fn_a(x), fn_b(x)
            worst_ulp = max(worst_ulp, ulp_diff_f64(a, b))
            worst_mis = max(worst_mis, fp16_mismatch(a, b))
            worst_rel = max(worst_rel, max_rel(a, b))
            if guard_exact is not None:
                g = guard_exact(x)
                worst_mis = max(worst_mis, fp16_mismatch(a, g), fp16_mismatch(b, g))
        status = "EQUIV_PASS" if worst_mis == 0.0 else "EQUIV_FAIL"
        results.append((vid, desc, worst_ulp, worst_rel, worst_mis, status))
        print(f"{vid} | {desc} | max_ulp={worst_ulp} | max_rel={worst_rel:.3e} | "
              f"fp16_violation={worst_mis:.6f} | {status}")

    print("=== online_softmax verify_equiv (fp64 reference, torch CPU) ===")
    run("V1", "O3 exp(x-m) vs exp2(x*s-m*s)", exp_ebase_f64, exp2_rebase_f64)
    run("V2", "O2 e*(1/S) vs e/S (full chains)", resident_chain_f64, resident_chain_div_f64,
        guard_exact=exact_softmax_f64)
    run("V3", "O1 resident chain vs source chain (fp16 out)",
        resident_chain_f64, source_chain_f64, guard_exact=exact_softmax_f64)
    run("V4", "online robust merge (flash form) vs direct full-row", online_2chunk_f64,
        resident_chain_f64, guard_exact=exact_softmax_f64)

    # V4b: diagnostic only (NOT adopted) - the source's naive merge divergence
    # evidence on extreme corner rows; quantifies why the robust form is required.
    naive_mis, naive32_mis = 0.0, 0.0
    for name, x in cases.items():
        try:
            y = online_2chunk_naive_f64(x)
            g = exact_softmax_f64(x)
            naive_mis = max(naive_mis, fp16_mismatch(y, g))
            y32 = online_2chunk_naive_f32(x)
            naive32_mis = max(naive32_mis, fp16_mismatch(y32, g))
        except Exception:
            continue
    print(f"V4b | (info, NOT adopted) source naive merge vs exact | "
          f"fp16_violation(fp64)={naive_mis:.6f} | fp16_violation(fp32)={naive32_mis:.6f} | "
          f"{'diverges on corners - robust form required' if max(naive_mis, naive32_mis) > 0 else 'no divergence detected'}")

    # V5: informational only (alternate of section 1.6.4, NOT adopted here)
    worst_rel_exact, worst_mis_exact = 0.0, 0.0
    finite_cases = {k: v for k, v in cases.items()
                    if not torch.isnan(v).any() and not torch.isinf(v).any()}
    for name, x in finite_cases.items():
        y = fp16e_chain_sim(x)
        g = exact_softmax_f64(x)
        # relative error over the meaningful range only (|y| >= 1e-3);
        # subnormal-flushed lanes are absolutely negligible (<= 1e-24)
        mask = g.abs() >= 1e-3
        if mask.any():
            rel = ((y - g).abs() / g.abs().clamp_min(1e-300))[mask].max().item()
            worst_rel_exact = max(worst_rel_exact, float(rel))
        worst_mis_exact = max(worst_mis_exact, fp16_mismatch(y, g))
    print(f"V5 | (info, NOT adopted) fp16-e simulated chain vs exact | "
          f"max_rel(|y|>=1e-3)={worst_rel_exact:.3e} | fp16_violation={worst_mis_exact:.6f} | "
          f"{'within 5e-3 guard' if worst_mis_exact == 0.0 and worst_rel_exact < 5e-3 else 'GUARD REVIEW'}")

    print("=== summary ===")
    for vid, desc, ulp, rel, mis, status in results:
        print(f"{vid}: {status} (max_ulp={ulp}, max_rel={rel:.3e}, fp16_violation={mis:.6f})")
    if any(r[-1] != "EQUIV_PASS" for r in results):
        raise SystemExit("EQUIV verification failed - adopted items must be dropped or fixed")


if __name__ == "__main__":
    main()
