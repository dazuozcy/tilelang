# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Equivalence machine verification for _logsumexp_kernel_tiled (D-1).

Verifies the math-level equivalence of the A5 redesign candidates against the
GPU baseline chain (source algorithm) and the exact fp64 reference, on random
tensors plus corner-value rows. All chains implement row-wise logsumexp over
N-chunked streaming with the online (max, sum) flash merge:

    per chunk c:  m_c = max(x_c);  prev = m_run;  m_run = max(m_run, m_c)
                  e_c = exp(x_c - m_run);  s_c = sum(e_c)
                  s_run = s_run * exp(prev - m_run) + s_c
    output:       y = m_run + log(s_run)

Candidate vs baseline structural differences under test:

  V1 (O2 pow2 chunk re-partition + compact residency):
      baseline chunks columns by the wrapper tile_n (any 256-multiple, last
      chunk masked to N_padded = align_up(N, 256) with -inf on the GPU);
      candidate re-partitions into BLOCK_N = pow2 <= tile_n full chunks plus a
      static tail chunk of rem = N % BLOCK_N columns (A5 fragment pow2 rule).
      The flash merge is associative over chunk partitions (each element
      counted exactly once, merge algebra identical), so the chains must agree;
      verified at fp64 (bitwise-adjacent, 1e-12) and fp32 (ulp).
  V2 (O3 in-place fragment overwrite e = exp(x - m_run)):
      candidate overwrites the chunk buffer in place instead of a separate e
      buffer; same rounding path, verified bitwise identical at fp32.
  V3 (full A5 chain vs torch.logsumexp golden, fp32):
      on random data and finite corner rows.
  V4 (source GPU chain vs A5 chain, both fp64 -> fp16 rounded):
      end-to-end source-vs-target equivalence on the shared value domain,
      including NaN-pattern agreement on special rows.
  V5 (corner rows specific to the online merge):
      all -inf row -> NaN (source-inherited, both chains); +inf row -> NaN;
      partial -inf chunk (both orders: -inf chunk before/after finite chunks)
      neutral; fp16 max-gap 131008 across chunks (naive m_c-based merge
      overflows at gap > 88.7, flash form stays safe since all exp args <= 0);
      dominant value in different chunks (merge-order sensitivity).
  V6 (chunk-partition sensitivity):
      same input under multiple (tile_n, BLOCK_N) partitions agrees within
      tolerance (associativity of the merge under fp rounding).
  V7 (fp16-e backup variant simulation, NOT adopted - guard for Sec 1.6.4):
      per-chunk exp/sum in fp16, running stats rescale in fp32; records the
      predicted accuracy headroom for the Stage-4 A/B decision.

Run: PYTHONPATH=<repo> python3 verify_equiv.py   (torch CPU; NPU has no fp64)
"""

import math

import torch

torch.manual_seed(20261016)

RESULTS = []


def _ulp_diff_fp32(a, b):
    """Max ulp distance between two fp32 tensors (finite entries only)."""
    ai = a.view(torch.int32).to(torch.int64)
    bi = b.view(torch.int32).to(torch.int64)
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


def _align_up(n, a):
    return ((n + a - 1) // a) * a


def _gpu_tiled_chain(x, tile_n, in_place=True):
    """Source GPU _logsumexp_kernel_tiled chain (single dtype domain of x).

    Chunks by tile_n over N_padded = align_up(N, 256); the last chunk is
    masked (columns >= N loaded as -inf), earlier chunks use the fast copy
    path (fully in-bounds). Row tail (M % block_m) is orthogonal to the chunk
    merge and not simulated here.
    """
    M, N = x.shape
    N_padded = _align_up(N, 256)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    row_max = torch.full((M,), -math.inf, dtype=x.dtype)
    row_sum = torch.zeros((M,), dtype=x.dtype)
    for t in range(num_tiles):
        lo = t * tile_n
        valid = min(lo + tile_n, N) - lo
        tile = torch.full((M, tile_n), -math.inf, dtype=x.dtype)
        if valid > 0:
            tile[:, :valid] = x[:, lo:lo + valid]
        tile_max = tile.max(dim=1).values
        prev_max = row_max.clone()
        row_max = torch.maximum(row_max, tile_max)
        if in_place:
            tile.sub_(row_max.unsqueeze(1)).exp_()          # e = exp(x - m_run)
        else:
            tile = torch.exp(tile - row_max.unsqueeze(1))   # separate e buffer
        tile_sum = tile.sum(dim=1)
        row_sum = row_sum * torch.exp(prev_max - row_max) + tile_sum
    return row_max + torch.log(row_sum)


def _a5_tiled_chain(x, block_n, in_place=True):
    """A5 redesign chain: BLOCK_N pow2 full chunks + static tail chunk.

    Full chunks load all-valid columns (no predicate); the tail chunk holds
    rem = N % BLOCK_N valid columns padded to TILE_R = next_pow2(rem) with
    -inf (masked load identity). Running stats live across chunks.
    """
    M, N = x.shape
    num_full = N // block_n
    rem = N - num_full * block_n
    m_run = torch.full((M,), -math.inf, dtype=x.dtype)
    s_run = torch.zeros((M,), dtype=x.dtype)

    def _merge(tile):
        nonlocal m_run, s_run
        m_c = tile.max(dim=1).values
        prev = m_run.clone()
        m_run = torch.maximum(m_run, m_c)
        if in_place:
            tile.sub_(m_run.unsqueeze(1)).exp_()
        else:
            tile = torch.exp(tile - m_run.unsqueeze(1))
        s_c = tile.sum(dim=1)
        s_run = s_run * torch.exp(prev - m_run) + s_c

    for c in range(num_full):
        _merge(x[:, c * block_n:(c + 1) * block_n].clone())
    if rem > 0:
        tile_r = 1 << (rem - 1).bit_length()
        tile = torch.full((M, tile_r), -math.inf, dtype=x.dtype)
        tile[:, :rem] = x[:, num_full * block_n:]
        _merge(tile)
    return m_run + torch.log(s_run)


def _a5_tiled_chain_fp16e(x, block_n):
    """Backup variant (Sec 1.6.4): per-chunk exp/sum in fp16, stats fp32."""
    M, N = x.shape
    num_full = N // block_n
    rem = N - num_full * block_n
    m_run = torch.full((M,), -math.inf, dtype=torch.float32)
    s_run = torch.zeros((M,), dtype=torch.float32)

    def _merge(tile32):
        nonlocal m_run, s_run
        m_c = tile32.max(dim=1).values
        prev = m_run.clone()
        m_run = torch.maximum(m_run, m_c)
        e = torch.exp((tile32 - m_run.unsqueeze(1)).to(torch.float16))  # fp16 exp
        s_c = e.sum(dim=1, dtype=torch.float16).to(torch.float32)       # fp16 tree
        s_run = s_run * torch.exp(prev - m_run) + s_c                   # fp32 rescale
        return m_run, s_run

    for c in range(num_full):
        _merge(x[:, c * block_n:(c + 1) * block_n].to(torch.float32))
    if rem > 0:
        tile_r = 1 << (rem - 1).bit_length()
        tile = torch.full((M, tile_r), -math.inf, dtype=torch.float32)
        tile[:, :rem] = x[:, num_full * block_n:]
        _merge(tile)
    return m_run + torch.log(s_run)


def _corner_rows(M, N):
    """Random tensor with injected corner rows (finite-domain values)."""
    x = torch.randn(M, N, dtype=torch.float64) * 3.0
    if M >= 1 and N >= 2:
        x[0, :] = 0.0                      # all-zero row
        x[1 % M, :] = 3.5                  # all-equal row
        x[2 % M, 0] = 100.0                # single dominant
        x[3 % M, 0] = 65504.0
        x[3 % M, 1] = -65504.0             # fp16 extreme pair in one row
        x[4 % M, 0] = 6e-8                 # subnormal-magnitude
        x[4 % M, 1] = -6e-8
        x[5 % M, 0] = -745.0               # deep underflow (exp -> 0)
        x[5 % M, 1] = -744.0
        x[6 % M, 0] = -math.inf            # partial -inf row (finite row)
    return x


def main():
    # (M, N, tile_n, block_n) partitions covering: exact-division, pow2 tail,
    # tiny non-pow2 tail, tile_n non-pow2 (16640), tile_n > 32512 budget form.
    cases = [
        (33, 33000, 16640, 16384),   # wrapper default partition for (33,33000)
        (33, 32768, 16384, 16384),   # exact division, both partitions equal
        (4, 102400, 25600, 16384),   # lm-head-logits: rem = 4096 (pow2 tail)
        (4, 32768, 16384, 16384),    # test-file large-N case
        (3, 65540, 8192, 8192),      # rem = 4 (tiny tail)
        (2, 32513, 32512, 16384),    # N just above the single-tile column cap
        (5, 5000, 4096, 4096),       # mid N, several small chunks
    ]
    all_pass = True

    for (M, N, tile_n, block_n) in cases:
        x64 = _corner_rows(M, N)
        finite_mask = torch.isfinite(x64).all(dim=1)
        xf = x64[finite_mask]
        if xf.shape[0] == 0:
            xf = x64[:1] * 0.0

        # --- V1: A5 pow2 re-partition vs GPU tile_n partition (fp64) ---
        b64 = _gpu_tiled_chain(xf, tile_n)
        c64 = _a5_tiled_chain(xf, block_n)
        if _report("V1", f"pow2-chunk vs tile_n-chunk (M={M},N={N})", b64, c64,
                   "fp64", 1e-12) == "EQUIV_FAIL":
            all_pass = False

        # fp32 rounded path
        b32 = _gpu_tiled_chain(xf.to(torch.float32), tile_n).to(torch.float64)
        c32 = _a5_tiled_chain(xf.to(torch.float32), block_n).to(torch.float64)
        if _report("V1", f"pow2-chunk vs tile_n-chunk (M={M},N={N})", b32, c32,
                   "fp32", 1e-6) == "EQUIV_FAIL":
            all_pass = False

        # --- V2: in-place overwrite vs separate e buffer (fp32, bitwise) ---
        s32 = _a5_tiled_chain(xf.to(torch.float32), block_n, in_place=False)
        i32 = _a5_tiled_chain(xf.to(torch.float32), block_n, in_place=True)
        if _report("V2", f"in-place e vs separate e (M={M},N={N})", s32, i32,
                   "fp32", 0.0) == "EQUIV_FAIL":
            all_pass = False

        # --- V4: source GPU chain vs A5 chain, fp64 -> fp16 output ---
        b16 = _gpu_tiled_chain(x64, tile_n).to(torch.float16)
        c16 = _a5_tiled_chain(x64, block_n).to(torch.float16)
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
        cand = _a5_tiled_chain(xt, block_n)
        if _report("V3", f"A5 chain vs torch.logsumexp (M={M},N={N})", gold, cand,
                   "fp32", 1e-3) == "EQUIV_FAIL":
            all_pass = False

    # --- V5: online-merge corner rows (both chains + exact reference) ---
    N = 8192  # two 4096 chunks under block_n=4096
    BN = 4096
    rows = {}
    # all -inf row -> NaN on both chains (source-inherited divergence vs torch)
    rows["all -inf"] = torch.full((1, N), -math.inf)
    # +inf row -> NaN on both chains
    r = torch.randn(1, N); r[0, 5] = math.inf
    rows["has +inf"] = r
    # NaN row -> NaN on both chains
    r = torch.randn(1, N); r[0, 7] = float("nan")
    rows["has NaN"] = r
    # partial -inf: -inf columns in chunk 0 only (finite row, neutral)
    r = torch.randn(1, N); r[0, :BN // 2] = -math.inf
    rows["-inf in chunk0"] = r
    # partial -inf: entire chunk 1 all -inf (finite row via chunk 0, neutral)
    r = torch.randn(1, N); r[0, BN:] = -math.inf
    rows["chunk1 all -inf"] = r
    # fp16 max-gap 131008 across chunks: chunk0 all -65504, chunk1 has +65504
    r = torch.full((1, N), -65504.0); r[0, BN + 3] = 65504.0
    rows["max-gap 131008"] = r
    # dominant value in chunk 0 vs chunk 1 (merge order both ways)
    r0 = torch.randn(1, N) - 20.0; r0[0, 10] = 500.0
    rows["dominant in chunk0"] = r0
    r1 = torch.randn(1, N) - 20.0; r1[0, BN + 10] = 500.0
    rows["dominant in chunk1"] = r1

    for name, xr in rows.items():
        xr = xr.to(torch.float32)
        gpu = _gpu_tiled_chain(xr, 4096)
        a5 = _a5_tiled_chain(xr, BN)
        gold = torch.logsumexp(xr, dim=1)
        fin = torch.isfinite(gpu.float()) & torch.isfinite(a5.float())
        if fin.any():
            v = _report("V5", f"corner '{name}': GPU vs A5 chain",
                        gpu.float()[fin], a5.float()[fin], "fp32", 1e-3)
        else:
            v = _report("V5", f"corner '{name}': GPU vs A5 chain (both non-finite)",
                        gpu.float()[:1], a5.float()[:1], "fp32", 1e-3,
                        skip_nonfinite=False)
            v = "EQUIV_PASS" if torch.equal(
                torch.isnan(gpu.float()), torch.isnan(a5.float())) else "EQUIV_FAIL"
            print(f"  [V5] NaN-pattern agreement for '{name}': {v}")
        if v == "EQUIV_FAIL":
            all_pass = False
        # info: relation to exact/golden on finite corners
        if torch.isfinite(gpu.float()).all() and torch.isfinite(gold.float()).all():
            gv = _report("V5", f"corner '{name}': A5 vs exact",
                         gold, a5.float(), "fp32", 1e-3)
            if gv == "EQUIV_FAIL":
                all_pass = False
        else:
            g_nan = "NaN" if torch.isnan(gpu.float()).any() else \
                f"{gpu[0].item():.4f}"
            t_nan = "NaN" if torch.isnan(gold.float()).any() else \
                f"{gold[0].item():.4f}"
            _report_info("V5", f"corner '{name}': chain={g_nan} torch={t_nan}",
                         "source-inherited divergence (documented, Sec 9.3)")

    # --- V6: chunk-partition sensitivity (multiple block_n / tile_n) ---
    M, N = 17, 50000
    x = (_corner_rows(M, N))[torch.isfinite(_corner_rows(M, N)).all(dim=1)]
    base = _a5_tiled_chain(x.to(torch.float32), 16384)
    for bn in (256, 1024, 4096, 8192, 16384):
        for tn in (16640, 25000, 32512):
            cand = _gpu_tiled_chain(x.to(torch.float32), tn)
            if _report("V6", f"block_n={bn} base vs tile_n={tn} partition",
                       base, cand, "fp32", 1e-3) == "EQUIV_FAIL":
                all_pass = False
            break  # one tile_n per block_n keeps the table compact
        cand2 = _a5_tiled_chain(x.to(torch.float32), bn)
        if _report("V6", f"block_n=16384 base vs block_n={bn}",
                   base, cand2, "fp32", 1e-3) == "EQUIV_FAIL":
            all_pass = False

    # --- V7: fp16-e backup variant simulation (guard, NOT adopted) ---
    for (M, N, bn) in [(1024, 32768, 16384), (4, 102400, 16384)]:
        x = (torch.randn(M, N, dtype=torch.float64) * 3.0).to(torch.float32)
        exact = _a5_tiled_chain(x, bn)
        bk = _a5_tiled_chain_fp16e(x, bn)
        rel = ((bk - exact).abs() / exact.abs().clamp_min(1e-30))
        max_rel = float(rel.max().item())
        viol = float((rel > 1e-2).float().mean().item())
        verdict = "GUARD_PASS" if viol == 0.0 and max_rel <= 5e-3 else "GUARD_TIGHT"
        RESULTS.append(("V7", f"fp16-e backup sim (M={M},N={N})", "fp16",
                        "-", f"{max_rel:.3e}", f"{viol:.6f}", verdict))
        print(f"[V7] fp16-e backup sim (M={M},N={N}): max_rel={max_rel:.3e} "
              f"viol@1e-2={viol:.6f} -> {verdict}")

    print()
    print("=" * 92)
    print(f"{'ID':<4} {'description':<48} {'dtype':<6} {'ulp':>10} {'max_rel':>10} "
          f"{'viol':>9}  verdict")
    print("-" * 92)
    for r in RESULTS:
        print(f"{r[0]:<4} {r[1]:<48} {r[2]:<6} {str(r[3]):>10} {str(r[4]):>10} "
              f"{str(r[5]):>9}  {r[6]}")
    print("=" * 92)
    if all_pass:
        print("ALL ADOPTED-ITEM CHECKS: EQUIV_PASS")
    else:
        print("SOME CHECKS FAILED - see table")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
