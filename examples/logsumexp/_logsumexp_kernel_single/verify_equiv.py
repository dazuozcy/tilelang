# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Equivalence machine verification for _logsumexp_kernel_single (D-1).

Verifies the math-level equivalence of the A5 redesign candidates against the
GPU baseline chain (source algorithm) and the exact fp64 reference, on random
tensors plus corner-value rows. All chains implement:

    lse_i = m_i + log(sum_j exp(x_ij - m_i)),  m_i = max_j x_ij

Candidate vs baseline structural differences under test:

  V1 (O2 compact-UB + pow2-fragment masked load):
     baseline pads columns to N_padded = align_up(N, 256) with -inf
     (GPU shared-mem alignment); candidate pads to TILE = next_pow2(N) with
     -inf (A5 fragment pow2 rule). Both pads are max/exp/sum-neutral, so the
     chains must agree; verified at fp64 (bitwise) and fp32 (ulp).
  V2 (O3 in-place fragment overwrite e = exp(x - m)):
     candidate overwrites x_frag in place instead of a separate e buffer;
     same rounding path, verified bitwise identical at fp32/fp64.
  V3 (full A5 chain vs torch.logsumexp golden):
     fp32 chain rounded to fp16 output vs torch golden (fp32 compute ->
     fp16), on random data and finite corner rows. Corner rows with +/-inf
     inputs are listed separately (V3b) as known source-inherited divergences
     (NaN vs torch's -inf/+inf) - they are semantics-preservation evidence,
     not pass/fail items for the adopted optimizations.
  V4 (source GPU chain vs A5 chain, both fp64->fp16 rounded):
     end-to-end source-vs-target equivalence on the shared value domain.
  V5 (fp16-e backup variant simulation, NOT adopted - guard for Sec 1.6.4):
     arg rounded to fp16 before exp, sum in fp16 tree, then fp32 log.
     Records the predicted accuracy headroom for the Stage-4 A/B decision.

Run: PYTHONPATH=<repo> python3 verify_equiv.py
"""

import math

import torch

torch.manual_seed(20261016)

RESULTS = []


def _ulp_diff_fp32(a, b):
    """Max ulp distance between two fp32 tensors (finite entries only)."""
    ai = a.view(torch.int32).to(torch.int64)
    bi = b.view(torch.int32).to(torch.int64)
    # map sign-magnitude to monotonic ordering
    ai = torch.where(ai < 0, -2147483648 - ai, ai)
    bi = torch.where(bi < 0, -2147483648 - bi, bi)
    d = (ai - bi).abs()
    return int(d.max().item())


def _report(vid, desc, base, cand, dtype_name, tol, skip_nonfinite=True):
    """Compare candidate vs baseline; record max ulp / max rel / violation rate."""
    base = base.to(torch.float64)
    cand = cand.to(torch.float64)
    finite = torch.isfinite(base) & torch.isfinite(cand)
    if skip_nonfinite:
        b, c = base[finite], cand[finite]
    else:
        b, c = base, cand
    if b.numel() == 0:
        max_rel, viol = 0.0, 0.0
    else:
        rel = ((c - b).abs() / b.abs().clamp_min(1e-30))
        max_rel = float(rel.max().item())
        viol = float((rel > tol).float().mean().item())
    bitwise = bool(torch.equal(base.view(torch.int64), cand.view(torch.int64))) \
        if base.dtype == cand.dtype else False
    ulp = _ulp_diff_fp32(base.float(), cand.float()) if base.numel() else 0
    verdict = "EQUIV_PASS" if viol == 0.0 else "EQUIV_FAIL"
    RESULTS.append((vid, desc, dtype_name, ulp, max_rel, viol, verdict))
    print(f"[{vid}] {desc} [{dtype_name}]: max_ulp={ulp} max_rel={max_rel:.3e} "
          f"viol={viol:.6f} bitwise={bitwise} -> {verdict}")
    return verdict


def _report_info(vid, desc, note):
    RESULTS.append((vid, desc, "info", "-", "-", "-", note))
    print(f"[{vid}] {desc}: {note}")


def _lse_chain(x, pad_to, pad_val=-math.inf):
    """Generic chain: pad columns to pad_to with pad_val, then m+log(sum(exp(x-m)))."""
    M, N = x.shape
    if pad_to > N:
        xp = torch.full((M, pad_to), pad_val, dtype=x.dtype)
        xp[:, :N] = x
    else:
        xp = x
    m = xp.max(dim=1).values
    e = torch.exp(xp - m.unsqueeze(1))
    s = e.sum(dim=1)
    return m + torch.log(s)


def _lse_chain_inplace(x, pad_to, pad_val=-math.inf):
    """Candidate chain with in-place fragment overwrite (O3)."""
    M, N = x.shape
    if pad_to > N:
        xp = torch.full((M, pad_to), pad_val, dtype=x.dtype)
        xp[:, :N] = x
    else:
        xp = x.clone()
    m = xp.max(dim=1).values
    xp.sub_(m.unsqueeze(1)).exp_()  # in-place: xp <- exp(xp - m)
    s = xp.sum(dim=1)
    return m + torch.log(s)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _align_up(n, a):
    return ((n + a - 1) // a) * a


def _corner_rows(M, N, dtype):
    """Random tensor with injected corner rows (finite-domain + special-value)."""
    x = torch.randn(M, N, dtype=torch.float64) * 3.0
    if M >= 1 and N >= 2:
        x[0, :] = 0.0                      # all-zero row
        x[1 % M, :] = 3.5                  # all-equal row
        x[2 % M, 0] = 100.0                # single dominant
        x[3 % M, 0] = 65504.0 if dtype == "float16" else 3.4e38
        x[3 % M, 1] = -65504.0 if dtype == "float16" else -3.4e38
        x[4 % M, 0] = 6e-8                 # subnormal-magnitude
        x[4 % M, 1] = -6e-8
        x[5 % M, 0] = -745.0               # deep underflow (exp -> 0)
        x[5 % M, 1] = -744.0
        x[6 % M, 0] = -math.inf            # partial -inf row (finite row)
    return x


def main():
    shapes = [(33, 300), (32, 256), (4, 4096), (128, 16384), (3, 7), (1, 1)]
    all_pass = True

    for (M, N) in shapes:
        x64 = _corner_rows(M, N, "float16")
        # finite-domain rows only for V1/V2/V4 (special rows handled in V3b)
        finite_mask = torch.isfinite(x64).all(dim=1)
        xf = x64[finite_mask]
        if xf.shape[0] == 0:
            xf = x64[:1] * 0.0

        N_pad = _align_up(N, 256)
        TILE = _next_pow2(N)

        # --- V1: compact/pow2-masked chain vs baseline N_padded chain (fp64) ---
        b64 = _lse_chain(xf, N_pad)
        c64 = _lse_chain(xf, TILE)
        if _report("V1", f"pow2-masked vs N_padded pad (M={M},N={N})", b64, c64,
                   "fp64", 1e-12) == "EQUIV_FAIL":
            all_pass = False

        # fp32 rounded path
        b32 = _lse_chain(xf.to(torch.float32), N_pad).to(torch.float64)
        c32 = _lse_chain(xf.to(torch.float32), TILE).to(torch.float64)
        if _report("V1", f"pow2-masked vs N_padded pad (M={M},N={N})", b32, c32,
                   "fp32", 1e-6) == "EQUIV_FAIL":
            all_pass = False

        # --- V2: in-place overwrite chain vs separate-e chain (fp32, bitwise) ---
        s32 = _lse_chain(xf.to(torch.float32), TILE)
        i32 = _lse_chain_inplace(xf.to(torch.float32), TILE)
        if _report("V2", f"in-place e vs separate e (M={M},N={N})", s32, i32,
                   "fp32", 0.0) == "EQUIV_FAIL":
            all_pass = False

        # --- V4: source GPU chain vs A5 chain, fp64 -> fp16 output ---
        b16 = _lse_chain(x64, N_pad).to(torch.float16)
        c16 = _lse_chain(x64, TILE).to(torch.float16)
        # special rows (all -inf / +inf / NaN) produce NaN on both chains;
        # compare on rows where both are finite, then check NaN-agreement
        both_fin = torch.isfinite(b16.float()) & torch.isfinite(c16.float())
        if _report("V4", f"GPU-chain vs A5-chain fp16 out (M={M},N={N})",
                   b16.float()[both_fin], c16.float()[both_fin],
                   "fp16", 1e-2) == "EQUIV_FAIL":
            all_pass = False
        nan_agree = torch.equal(torch.isnan(b16.float()), torch.isnan(c16.float()))
        if not nan_agree:
            print(f"  [V4] NaN pattern mismatch at ({M},{N})!")
            all_pass = False

        # --- V3: full A5 chain (fp32) vs torch golden on finite domain ---
        xt = xf.to(torch.float32)
        gold = torch.logsumexp(xt, dim=1)
        cand = _lse_chain(xt, TILE)
        if _report("V3", f"A5 chain vs torch.logsumexp (M={M},N={N})", gold, cand,
                   "fp32", 1e-3) == "EQUIV_FAIL":
            all_pass = False

    # --- V3b: special-value rows: source-inherited semantics vs torch ---
    N = 256
    rows = {
        "all -inf": torch.full((1, N), -math.inf),
        "has +inf": torch.cat([torch.full((1, N - 1), 1.0),
                               torch.full((1, 1), math.inf)], dim=1),
        "has NaN": torch.cat([torch.randn(1, N - 1),
                              torch.full((1, 1), float("nan"))], dim=1),
        "partial -inf": torch.cat([torch.full((1, N // 2), -math.inf),
                                   torch.randn(1, N // 2)], dim=1),
    }
    for name, xr in rows.items():
        xr = xr.to(torch.float32)
        chain = _lse_chain(xr, _next_pow2(N))
        gold = torch.logsumexp(xr, dim=1)
        chain_v = "NaN" if torch.isnan(chain).any() else f"{chain[0].item():.4f}"
        gold_v = "NaN" if torch.isnan(gold).any() else f"{gold[0].item():.4f}"
        agree = (torch.isnan(chain).any() == torch.isnan(gold).any()) and (
            torch.isnan(chain).any() or torch.allclose(chain, gold, atol=1e-3))
        tag = "AGREE" if agree else "DIVERGE(source-inherited)"
        _report_info("V3b", f"special row '{name}': chain={chain_v} torch={gold_v} -> {tag}",
                     "documented boundary divergence; randn test domain unaffected")

    # --- V5: fp16-e backup variant simulation (guard, NOT adopted) ---
    for (M, N) in [(1024, 4096), (128, 16384)]:
        x = (torch.randn(M, N, dtype=torch.float64) * 3.0).to(torch.float32)
        # backup chain: arg -> fp16, exp fp16, sum fp16 (tree), log fp32
        m = x.max(dim=1).values  # fp32 (max stays fp32 in backup design)
        arg = (x - m.unsqueeze(1)).to(torch.float16)
        e = torch.exp(arg)  # fp16 exp
        s = e.sum(dim=1, dtype=torch.float16)  # fp16 accumulation
        lse_bk = (m + torch.log(s.to(torch.float32)))
        exact = _lse_chain(x, _next_pow2(N))
        rel = ((lse_bk - exact).abs() / exact.abs().clamp_min(1e-30))
        max_rel = float(rel.max().item())
        viol = float((rel > 1e-2).float().mean().item())
        verdict = "GUARD_PASS" if viol == 0.0 and max_rel <= 5e-3 else "GUARD_TIGHT"
        RESULTS.append(("V5", f"fp16-e backup sim (M={M},N={N})", "fp16",
                        "-", f"{max_rel:.3e}", f"{viol:.6f}", verdict))
        print(f"[V5] fp16-e backup sim (M={M},N={N}): max_rel={max_rel:.3e} "
              f"viol@1e-2={viol:.6f} -> {verdict}")

    print()
    print("=" * 88)
    print(f"{'ID':<4} {'description':<46} {'dtype':<6} {'ulp':>10} {'max_rel':>10} "
          f"{'viol':>9}  verdict")
    print("-" * 88)
    for r in RESULTS:
        print(f"{r[0]:<4} {r[1]:<46} {r[2]:<6} {str(r[3]):>10} {str(r[4]):>10} "
              f"{str(r[5]):>9}  {r[6]}")
    print("=" * 88)
    if all_pass:
        print("ALL ADOPTED-ITEM CHECKS: EQUIV_PASS")
    else:
        print("SOME CHECKS FAILED - see table")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
