# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 equivalence machine verification for _logsumexp_kernel_tiled (SimdVF variant).

Candidate chain (this DESIGN): per-row N-chunk streaming with the flash-form
online merge, chunk width BLOCK_N = tile_n (64-multiple, no pow2 repartition),
per-chunk two-pass group-64 lane accumulation + single deferred cross-lane
reduce + fused exp(x-m) (vexpdif semantics) + fp32-domain tail-group -inf mask.

Baselines:
  * source chain: the GPU `_logsumexp_kernel_tiled` recursion (tile_n chunks
    over N_padded=align256(N), masked last chunk, per-chunk tree reduce,
    in-place exp(x - m_run), s_run = s_run*exp(prev-m_run) + s_c);
  * torch.logsumexp (golden, fp32);
  * fp64 exact reference.

All chains simulated on torch CPU (NPU has no fp64: TRAP-NPU-DOUBLE-SILENT-FP32).
Run: PYTHONPATH=/home/z00875089/tilelang-fork python3 verify_equiv.py
"""

import math

import torch

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# Chain simulators (fp32 domain unless noted)
# ----------------------------------------------------------------------------


def _chunk_segments(width, group=64):
    """Split a chunk of `width` valid columns into (G, k): full groups + tail."""
    return width // group, width % group


def _group_max(row, G, k):
    """Pass-1 max accumulator: lane-serial group accumulation + ONE cross-lane
    reduce (SIMD deferred-reduction structure). row has width G*64+k <= len."""
    acc = torch.full((64,), float("-inf"), dtype=torch.float32)
    for g in range(G):
        acc = torch.maximum(acc, row[g * 64 : (g + 1) * 64])
    if k:
        tail = torch.full((64,), float("-inf"), dtype=torch.float32)
        tail[:k] = row[G * 64 : G * 64 + k]
        acc = torch.maximum(acc, tail)
    return acc.max()  # single cross-lane reduce


def _group_sum_exp(row, m_run, G, k):
    """Pass-2 sum accumulator: lane-serial accumulation of exp(x - m) + ONE
    cross-lane reduce. Tail group masked with -inf (exp(-inf-m)=0 neutral)."""
    accs = torch.zeros(64, dtype=torch.float32)
    for g in range(G):
        accs = accs + torch.exp(row[g * 64 : (g + 1) * 64] - m_run)
    if k:
        tail = torch.full((64,), float("-inf"), dtype=torch.float32)
        tail[:k] = row[G * 64 : G * 64 + k]
        accs = accs + torch.exp(tail - m_run)
    return accs.sum()  # single cross-lane reduce


def simd_tiled_chain(x, tile_n, block_n=None, dtype=torch.float32):
    """Candidate: SIMD tiled chain (per DESIGN section 1.4).

    BLOCK_N defaults to the largest 64-multiple <= tile_n (no pow2 repartition,
    no UB clamp here -- pure math). Chunks: num_full full chunks of BLOCK_N plus
    one tail chunk of rem = N % BLOCK_N columns (group-masked, pad=-inf).
    """
    if block_n is None:
        block_n = (tile_n // 64) * 64
    assert block_n > 0 and block_n <= tile_n
    M, N = x.shape
    xf = x.to(dtype)
    num_full = N // block_n
    rem = N % block_n
    y = torch.empty(M, dtype=dtype)
    m_run = torch.tensor(float("-inf"), dtype=dtype)
    s_run = torch.tensor(0.0, dtype=dtype)
    for i in range(M):
        m_run.fill_(float("-inf"))
        s_run.fill_(0.0)
        for c in range(num_full + (1 if rem else 0)):
            start = c * block_n
            width = block_n if c < num_full else rem
            row = xf[i, start : start + width]
            G, k = _chunk_segments(width)
            m_c = _group_max(row, G, k)
            prev = m_run.clone()
            m_run = torch.maximum(m_run, m_c)
            s_c = _group_sum_exp(row, m_run, G, k)
            # flash rescale merge (source form, single exp)
            s_run = s_run * torch.exp(prev - m_run) + s_c
        y[i] = m_run + torch.log(s_run)
    return y


def source_tiled_chain(x, tile_n, dtype=torch.float32):
    """Baseline: GPU source `_logsumexp_kernel_tiled` recursion.

    N_padded = align_up(N, 256); num_tiles = ceil(N_padded / tile_n); only the
    last chunk masked (cols >= N -> -inf); per-chunk torch tree reduce_max /
    reduce_sum; in-place exp(x - row_max); row_sum = row_sum*exp(prev-m)+s_c.
    """
    M, N = x.shape
    xf = x.to(dtype)
    N_padded = ((N + 255) // 256) * 256
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    y = torch.empty(M, dtype=dtype)
    for i in range(M):
        row_max = torch.tensor(float("-inf"), dtype=dtype)
        row_sum = torch.tensor(0.0, dtype=dtype)
        for t in range(num_tiles):
            lo, hi = t * tile_n, min((t + 1) * tile_n, N)
            cols = torch.arange(lo, max(hi, lo), dtype=torch.int64)
            tile = xf[i, cols] if hi > lo else torch.empty(0, dtype=dtype)
            if total_cols > N and t == num_tiles - 1 and len(tile) < tile_n:
                pad = torch.full((tile_n - len(tile),), float("-inf"), dtype=dtype)
                tile = torch.cat([tile, pad])
            tile_max = tile.max() if len(tile) else torch.tensor(float("-inf"), dtype=dtype)
            prev = row_max.clone()
            row_max = torch.maximum(row_max, tile_max)
            e = torch.exp(tile - row_max)
            tile_sum = e.sum()
            row_sum = row_sum * torch.exp(prev - row_max) + tile_sum
        y[i] = row_max + torch.log(row_sum)
    return y


def pergroup_reduce_chain(x, tile_n, dtype=torch.float32):
    """V2 baseline: per-group cross-lane reduce + elementwise merge (the
    auto-lowered T.reduce form the deferred reduction replaces). Both passes
    use the FINAL merged m_run (pass 1 completes before pass 2)."""
    block_n = (tile_n // 64) * 64
    M, N = x.shape
    xf = x.to(dtype)
    num_full = N // block_n
    rem = N % block_n
    y = torch.empty(M, dtype=dtype)
    for i in range(M):
        m_run = torch.tensor(float("-inf"), dtype=dtype)
        s_run = torch.tensor(0.0, dtype=dtype)
        for c in range(num_full + (1 if rem else 0)):
            start = c * block_n
            width = block_n if c < num_full else rem
            row = xf[i, start : start + width]
            G, k = _chunk_segments(width)
            # pass 1: per-group cross-lane max, elementwise merge, final max
            acc = torch.full((64,), float("-inf"), dtype=dtype)
            for g in range(G):
                acc = torch.maximum(acc, row[g * 64 : (g + 1) * 64].max())
            if k:
                tail = torch.full((64,), float("-inf"), dtype=dtype)
                tail[:k] = row[G * 64 : G * 64 + k]
                acc = torch.maximum(acc, tail.max())
            m_c = acc.max()
            prev = m_run.clone()
            m_run = torch.maximum(m_run, m_c)
            # pass 2: per-group cross-lane sum of exp(x - m_final), scalar merge
            s_c = torch.tensor(0.0, dtype=dtype)
            for g in range(G):
                s_c = s_c + torch.exp(row[g * 64 : (g + 1) * 64] - m_run).sum()
            if k:
                tail = torch.full((64,), float("-inf"), dtype=dtype)
                tail[:k] = row[G * 64 : G * 64 + k]
                s_c = s_c + torch.exp(tail - m_run).sum()
            s_run = s_run * torch.exp(prev - m_run) + s_c
        y[i] = m_run + torch.log(s_run)
    return y


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def ulp_diff_fp32(a, b):
    """Max ULP distance between two fp32 tensors (bit-level)."""
    ai = a.float().view(torch.int32).to(torch.int64)
    bi = b.float().view(torch.int32).to(torch.int64)
    # map sign-magnitude to monotonic ordering
    ai = torch.where(ai < 0, -2147483648 - ai, ai)
    bi = torch.where(bi < 0, -2147483648 - bi, bi)
    return (ai - bi).abs().max().item()


def bits_of(t):
    return t.view(torch.int16) if t.dtype == torch.float16 else t.view(torch.int16)


def report(tag, cand, base, tol, check_bits=False):
    """Compare candidate vs baseline; returns (pass, max_ulp, max_rel, viol)."""
    if check_bits:
        same = torch.equal(bits_of(cand), bits_of(base))
        nan_same = torch.equal(torch.isnan(cand), torch.isnan(base))
        ok = same and nan_same
        print(f"  [{tag}] {'PASS' if ok else 'FAIL'}: bitwise={'yes' if same else 'NO'} nan_pattern={'yes' if nan_same else 'NO'}")
        return ok, 0, 0.0, 0.0
    assert cand.shape == base.shape
    cf, bf = cand.float(), base.float()
    ulp = ulp_diff_fp32(cf, bf)
    finite = torch.isfinite(cf) & torch.isfinite(bf)
    diff = (cf - bf).abs()
    rel = (diff[finite] / bf[finite].abs().clamp_min(1e-30)).max().item() if finite.any() else 0.0
    viol = (~(diff <= tol * (1 + bf.abs()))).float().mean().item()
    ok = viol == 0.0
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}: max_ulp={ulp} max_rel={rel:.3e} viol_rate={viol * 100:.6f}% @tol={tol}")
    return ok, ulp, rel, viol


RESULTS = []


def gate(name, ok):
    RESULTS.append((name, ok))


# ----------------------------------------------------------------------------
# V1: candidate (BLOCK_N=tile_n) vs source tile_n chain, fp32 + fp64
# ----------------------------------------------------------------------------
print("== V1: SIMD chain (BLOCK_N=tile_n, group-64 deferred reduce) vs source tile_n chain ==")
V1_CASES = [
    (8, 32768, 16384),  # wl2 analog: 2 exact chunks
    (4, 102400, 25600),  # wl3 analog: 4 exact chunks (non-pow2 BLOCK_N)
    (33, 33000, 32512),  # masked-tail analog: 1 full + rem=488 tail (v1: corrected
    # wrapper landing -- compute_tile_n rejects divisor
    # 11008, returns tile_n_max=32512; REVIEW issue 1)
    (4, 32768, 32768),  # test L75: single chunk per row (BLOCK_N=32768)
    (8, 100, 64),  # N < BLOCK_N explicit: num_full=1 + rem=36
    (3, 257, 256),  # tiny N, rem=1
    (2, 1024000, 16384),  # 62 full + rem=8192: long fp32 rescale chain
    (57, 32768, 16384),  # M tail value
]
v1_all = True
for M, N, tn in V1_CASES:
    x = torch.randn(M, N, dtype=torch.float32) * 3
    for dt, tol in ((torch.float32, 1e-3), (torch.float64, 1e-12)):
        c = simd_tiled_chain(x, tn, dtype=dt)
        b = source_tiled_chain(x, tn, dtype=dt)
        ok, ulp, rel, viol = report(f"V1 ({M},{N},tn={tn},{str(dt).split('.')[-1]})", c, b, tol)
        v1_all &= ok
gate("V1 partition/grouping equivalence", v1_all)

# ----------------------------------------------------------------------------
# V2: deferred cross-lane reduce vs per-group cross-lane reduce
# ----------------------------------------------------------------------------
print("== V2: deferred cross-lane reduction vs per-group reduction (fp32) ==")
v2_all = True
for M, N, tn in [(8, 32768, 16384), (4, 102400, 25600), (33, 33000, 16640)]:
    x = torch.randn(M, N, dtype=torch.float32) * 3
    c = simd_tiled_chain(x, tn)
    b = pergroup_reduce_chain(x, tn)
    ok, ulp, rel, viol = report(f"V2 ({M},{N},tn={tn})", c, b, 1e-3)
    v2_all &= ok
gate("V2 reduction-grouping equivalence", v2_all)

# ----------------------------------------------------------------------------
# V3: candidate vs torch.logsumexp golden (fp32)
# ----------------------------------------------------------------------------
print("== V3: SIMD chain vs torch.logsumexp (fp32) ==")
v3_all = True
for M, N, tn in V1_CASES:
    x = torch.randn(M, N, dtype=torch.float32) * 3
    c = simd_tiled_chain(x, tn)
    g = torch.logsumexp(x, dim=-1)
    ok, ulp, rel, viol = report(f"V3 ({M},{N},tn={tn})", c, g, 1e-3)
    v3_all &= ok
gate("V3 golden equivalence", v3_all)

# ----------------------------------------------------------------------------
# V4: fp16/bf16 output bitwise, special rows (source-fidelity, fp64 chains)
# ----------------------------------------------------------------------------
print("== V4: fp64-chain -> 16-bit output bitwise vs source chain (special rows) ==")


def special_rows(N, tn, dtype_bits=torch.float16):
    """Build rows exercising every corner; chunk width = tn (64-mult here)."""
    rows = []
    rows.append(torch.zeros(N, dtype=torch.float64))  # all zero
    rows.append(torch.full((N,), 3.5, dtype=torch.float64))  # all equal
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = 100.0
    rows.append(r)  # single dominant
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = 65504.0
    r[1] = -65504.0
    rows.append(r)  # fp16 extremes
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = 6e-8
    r[1] = -6e-8
    rows.append(r)  # subnormal
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = -745.0
    r[1] = -744.0
    rows.append(r)  # deep underflow
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = float("-inf")
    rows.append(r)  # partial -inf
    r = torch.full((N,), float("-inf"), dtype=torch.float64)
    rows.append(r)  # all -inf
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = float("inf")
    rows.append(r)  # +inf
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = float("nan")
    rows.append(r)  # NaN
    # tiled-specific: FIRST chunk entirely -inf, later finite -> source NaN
    r = torch.zeros(N, dtype=torch.float64)
    r[:tn] = float("-inf")
    rows.append(r)
    # tiled-specific: LAST chunk entirely -inf (neutral) -> finite
    r = torch.zeros(N, dtype=torch.float64)
    r[-tn:] = float("-inf")
    rows.append(r)
    # cross-chunk max gap 131008: -65504 in chunk0, +65504 in chunk1
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = -65504.0
    r[tn] = 65504.0
    rows.append(r)
    # dominant in chunk0 / chunk1 (merge order both directions)
    r = torch.zeros(N, dtype=torch.float64)
    r[0] = 100.0
    rows.append(r)
    r = torch.zeros(N, dtype=torch.float64)
    r[tn] = 100.0
    rows.append(r)
    # equal values spanning the chunk boundary (associativity)
    rows.append(torch.ones(N, dtype=torch.float64))
    return torch.stack(rows)


v4_all = True
for N, tn in [(32768, 16384), (33000, 16640), (102400, 25600)]:
    x64 = special_rows(N, tn)
    for out_dt in (torch.float16, torch.bfloat16):
        c = simd_tiled_chain(x64, tn, dtype=torch.float64).to(out_dt)
        b = source_tiled_chain(x64, tn, dtype=torch.float64).to(out_dt)
        ok, _, _, _ = report(f"V4 (N={N},tn={tn},{str(out_dt).split('.')[-1]})", c, b, 0, check_bits=True)
        v4_all &= ok
gate("V4 source-fidelity bitwise (special rows)", v4_all)

# ----------------------------------------------------------------------------
# V5: corner semantics documentation (first-chunk-all-inf vs single/torch)
# ----------------------------------------------------------------------------
print("== V5: tiled-specific corner semantics (info) ==")
N, tn = 32768, 16384
r = torch.zeros(1, N, dtype=torch.float64)
r[0, :tn] = float("-inf")  # first chunk all -inf, second chunk finite zeros
cand = simd_tiled_chain(r, tn, dtype=torch.float64)
src = source_tiled_chain(r, tn, dtype=torch.float64)
single_form = r.max(dim=-1).values + torch.log(torch.exp(r - r.max(dim=-1, keepdim=True).values).sum(dim=-1))
tch = torch.logsumexp(r, dim=-1)
print(
    f"  first-chunk-all-inf row: candidate={cand[0].item()} source={src[0].item()} "
    f"single-form={single_form[0].item()} torch={tch[0].item()}"
)
ok = torch.equal(cand.view(torch.int64), src.view(torch.int64)) and torch.isnan(cand).all()
print(
    f"  -> candidate == source (NaN, migration fidelity): {'PASS' if ok else 'FAIL'}; "
    f"divergence vs single-form/torch (finite) is SOURCE-INHERENT (DESIGN 9.3)"
)
gate("V5 first-chunk-all-inf source fidelity", ok)

# last-chunk-all-inf must stay finite and match torch
r2 = torch.zeros(1, N, dtype=torch.float64)
r2[0, -tn:] = float("-inf")
cand2 = simd_tiled_chain(r2, tn, dtype=torch.float64)
ok2 = torch.allclose(cand2, torch.logsumexp(r2, dim=-1))
print(
    f"  last-chunk-all-inf row: candidate={cand2[0].item()} torch={torch.logsumexp(r2, dim=-1)[0].item()} "
    f"-> {'PASS' if ok2 else 'FAIL'} (neutral)"
)
gate("V5 last-chunk-all-inf neutrality", ok2)

# ----------------------------------------------------------------------------
# V6: partition sensitivity sweep (BLOCK_N variants incl. non-pow2)
# ----------------------------------------------------------------------------
print("== V6: partition sweep (BLOCK_N in {4096,8192,16384,16640,25600,32512,32768}) ==")
v6_all = True
for M, N, tn in [(4, 102400, 25600), (33, 33000, 32512), (8, 32768, 16384)]:
    x = torch.randn(M, N, dtype=torch.float32) * 3
    ref = source_tiled_chain(x, tn)
    for bn in (4096, 8192, 16384, 16640, 25600, 32512, 32768):
        if bn > ((tn // 64) * 64) or bn > N:
            continue
        c = simd_tiled_chain(x, tn, block_n=bn)
        ok, ulp, rel, viol = report(f"V6 ({M},{N}) BLOCK_N={bn}", c, ref, 1e-3)
        v6_all &= ok
gate("V6 partition invariance", v6_all)

# ----------------------------------------------------------------------------
# V7: fp16 output saturation closure (info, O6 analog)
# ----------------------------------------------------------------------------
print("== V7: fp16 output saturation closure (info) ==")
ok7 = True
for N in (102400, 1024000):
    row = torch.full((1, N), 65504.0, dtype=torch.float64)
    val = simd_tiled_chain(row, 16384, dtype=torch.float64)
    upper = 65504.0 + math.log(N)
    closed = val[0].item() < 65520.0 and upper < 65520.0
    print(
        f"  all-65504 row N={N}: y={val[0].item():.2f} bound=65504+ln(N)={upper:.2f} "
        f"< 65520 IEEE-inf threshold: {'closed' if closed else 'OPEN'}"
    )
    ok7 &= closed
print(
    "  closure holds for N <= e^16 ~= 8.86M (contract space); beyond is documented in DESIGN 9.3 (source shares the behaviour via T.cast)"
)
gate("V7 saturation closure (info)", ok7)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n===== SUMMARY =====")
all_pass = True
for name, ok in RESULTS:
    print(f"  {name}: {'EQUIV_PASS' if ok else 'EQUIV_FAIL'}")
    all_pass &= ok
print(f"\nVERDICT: {'ALL EQUIV_PASS' if all_pass else 'EQUIV_FAIL PRESENT'}")
raise SystemExit(0 if all_pass else 1)
