# DESIGN.md section 4.5 / D-2 single-source budget & roofline calculator
# (re-runs deterministically; numbers embedded in DESIGN.md must match this output)
# Constants: CONST-capacity-a5 (UB 248KB, status assumption, 90% discount),
#            CONST-valu-model (ALU 51 ops/cyc/AIV, special 8 ops/cyc/AIV, assumption),
#            CONST-mte-bandwidth-model (AIV MTE2 GM->UB 100 B/cyc wide-row, base 90;
#                                      MTE3 UB->GM 115 B/cyc, base 180; assumption),
#            CONST-arch-a5 (vector_core_num=56, measured on this host).

UB_PHYS = 248 * 1024
UB_BUDGET = int(UB_PHYS * 0.9)  # 228556
RESERVE = 8256  # lse_ub (<=64B) + alignment/scheduler margin
VLANES = 64  # 2048-bit / fp32
ALU_TPUT = 51  # lane-ops/cycle/AIV
SP_TPUT = 8  # special lane-ops/cycle/AIV
MTE2_BW = 100  # B/cycle (wide-row GM->UB)
MTE2_BASE = 90  # cycles per copy issue
MTE3_BW = 115
MTE3_BASE = 180
UB_VRF_BW = 256  # B/cycle VRF<->UB
VCORES = 56


def elem(dtype):
    return {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]


def block_n_cap(dt):
    e = elem(dt)
    b = (UB_BUDGET - RESERVE) // (2 * e)
    return (b // 64) * 64


def instr_chain(G, dt, tail=False, k_tail=0):
    """Per-chunk instruction counts (row granularity, BLOCK_N=64*G[+tail]).
    fp16/bf16: vld+vcvt per group; fp32: vld NORM only."""
    cvt = 0 if dt == "float32" else 1
    alu = 2 * cvt + 1 + 2 * cvt + 1 + 3  # pass1: G*(vcvt+vmax)+3 merge; pass2: G*(vcvt+vadd)+3 merge
    # per group: pass1 (cvt + vmax), pass2 (cvt + vadd) -> expand below instead
    alu_full_group = (cvt + 1) + (cvt + 1)
    sp_full_group = 1  # vexpdif in pass2
    vld_full_group = 2  # pass1 + pass2 re-read
    alu = G * alu_full_group + 3 + 3  # +3 pass1 chunk-level (vcmax,vdupv,vmax) +3 pass2 (vcadd,vmul,vadd)
    sp = G * sp_full_group + 1  # +1 rescale vexpdif
    vld = G * vld_full_group
    vsel = 0
    if tail and k_tail > 0:
        # partial group: +1 vsel per pass
        vsel = 2
        alu += 2 * cvt + 2  # the partial group's cvt/vmax + cvt/vadd
        sp += 1
        vld += 2
    return {
        "alu": alu,
        "sp": sp,
        "vld": vld,
        "vsel": vsel,
        "merge": 7,
    }  # 7 chunk-level merge instrs (vcmax,vdupv,vmax,vcadd,vexpdif,vmul,vadd)


def row_cycles(chunks, dt, G_full, G_tail=None, k_tail=0):
    """Cycles for one row: init(2 vdup) + chunks + epilogue."""
    cvt_ep = 0 if dt == "float32" else 1
    c = instr_chain(G_full, dt)
    alu_ops = chunks * c["alu"] * VLANES
    sp_ops = chunks * c["sp"] * VLANES
    n_instr = chunks * (c["alu"] + c["sp"] + c["vld"] + c["vsel"])
    if G_tail is not None:
        t = instr_chain(G_tail, dt, tail=True, k_tail=k_tail)
        alu_ops += t["alu"] * VLANES
        sp_ops += t["sp"] * VLANES
        n_instr += t["alu"] + t["sp"] + t["vld"] + t["vsel"]
    # init: 2 vdup; epilogue: vln(special) + vadd(ALU) [+ vcvt] (vsts = ONEPT store)
    alu_ops += (2 + 1 + cvt_ep) * VLANES
    sp_ops += 1 * VLANES
    return {"alu_ops": alu_ops, "sp_ops": sp_ops, "cyc": alu_ops / ALU_TPUT + sp_ops / SP_TPUT, "instr": n_instr + 2 + 1 + cvt_ep + 1 + 1}


def workload(name, M, N, dt, tile_n, block_m=1):
    e = elem(dt)
    cap = block_n_cap(dt)
    BN = min(tile_n, cap)
    BN = (BN // 64) * 64
    G_full = BN // 64
    num_full = N // BN
    rem = N % BN
    G_tail = rem // 64 if rem else 0
    k_tail = rem % 64 if rem else 0
    x_ub = 2 * BN * e  # double version
    lse_ub = block_m * e
    total = x_ub + RESERVE
    rows_core = M / VCORES
    # D-2 (mainstream block_m=1: one row-task per task)
    rc = row_cycles(num_full, dt, G_full, G_tail if rem else None, k_tail)
    cyc_row = rc["cyc"]
    comp_avg = rows_core * cyc_row
    comp_worst = -(-M // VCORES) * cyc_row
    gm_read = M * N * e
    mte2_bw_cyc = gm_read / VCORES / MTE2_BW
    copies_core = rows_core * (num_full + (1 if rem else 0))
    mte2_base = copies_core * MTE2_BASE
    mte3 = rows_core * (block_m * e) / MTE3_BW + rows_core * MTE3_BASE
    ub_reread = gm_read / VCORES / UB_VRF_BW
    sp_share = (rc["sp_ops"] / SP_TPUT) / cyc_row
    # merge counting fix (v1): a shape with rem>0 incurs the 7-instr merge in BOTH
    # the full chunks and the tail chunk (v0 counted num_full only -> (33,33000)
    # undercounted 2x); single-source with DESIGN 1.6.1 O5.
    merge_share = (num_full + (1 if rem else 0)) * 7 / (rc["instr"])
    print(f"--- {name}: (M={M}, N={N}, {dt}, tile_n={tile_n}, block_m={block_m}) ---")
    print(
        f"  BLOCK_N={BN} (cap {cap}, clamp {'ON' if tile_n > BN else 'off'})  G={G_full}  "
        f"chunks: {num_full} full + {'tail(G=%d,k=%d)' % (G_tail, k_tail) if rem else 'no tail'}"
    )
    print(
        f"  UB: x_ub(2ver)={x_ub}B + reserve={RESERVE}B = {total}B = {100 * total / UB_BUDGET:.1f}% of {UB_BUDGET}"
        f"  (lse_ub={lse_ub}B in reserve)"
    )
    print(f"  rows/core avg={rows_core:.2f} worst={-(-M // VCORES)}  active_cores={min(M, VCORES)}")
    print(f"  per-row: ALU lane-ops={rc['alu_ops']:.0f} SP lane-ops={rc['sp_ops']:.0f} cyc={cyc_row:.0f} instrs~{rc['instr']:.0f}")
    print(f"  D-2 compute: avg {comp_avg / 1000:.1f}K cyc/core (worst {comp_worst / 1000:.1f}K), special share {100 * sp_share:.0f}%")
    print(
        f"  D-2 MTE2: {mte2_bw_cyc / 1000:.2f}K cyc/core (bw) + {mte2_base / 1000:.2f}K (base x{copies_core:.1f} copies)"
        f"  MTE3: {mte3 / 1000:.2f}K  UB-reread(VRF): {ub_reread / 1000:.2f}K"
    )
    tot_mte = mte2_bw_cyc + mte2_base + mte3
    print(
        f"  ratio compute/MTE(bw-only)={comp_avg / mte2_bw_cyc:.1f}x  (with base {comp_avg / tot_mte:.1f}x)"
        f"  merge-instr share={100 * merge_share:.2f}%"
    )
    print()


print(f"UB_BUDGET_BYTES = {UB_BUDGET}   RESERVE = {RESERVE}")
for dt in ("float16", "bfloat16", "float32"):
    print(f"BLOCK_N cap {dt}: {block_n_cap(dt)}")
print()
# manifest workloads (wrapper heuristic: block_m=1 for tiled path)
workload("wl2 attn-weights-32k", 1024, 32768, "bfloat16", 16384)
workload("wl3 lm-head-logits", 4, 102400, "float16", 25600)
workload("wl3 lm-head-logits", 4, 102400, "bfloat16", 25600)
# precision-test shapes
workload("test (4,32768)", 4, 32768, "float16", 16384)
# v1 (REVIEW issue 1): (33,33000) tile_n corrected 16640 -> 32512.
# N_padded = align_up(33000, 256) = 33,024; compute_tile_n(1, 2, 33024,
# budget=min(221184, 32512*1*2)) -> tile_n_max=32512; best_dividing=11,008
# (129=3*43, largest factor <=127 is 43) -> div_tiles=3 > max_tiles=2 ->
# divisor rejected, returns tile_n_max = 32,512.
workload("test (33,33000)", 33, 33000, "float16", 32512)
workload("test (33,32768)", 33, 32768, "float16", 16384)
# fp32 explicit-config path at cap
workload("fp32@cap (1024,32768) tn=32512", 1024, 32768, "float32", 32512)
# bm>1 DMA-descriptor growth check (wl2 shape, block_m=4): copies/core identical,
# per-task loop overhead only -> same totals; documented in section 9.1 risk 7.
# VRF estimate: live 2048-bit regs at any instant.
names = ["neg_inf", "zero_f", "one_pt", "acc", "accs", "c", "e", "m_c_brc", "prev_brc", "m_run", "s_run", "s_c", "out"]
print(f"VRF live regs ~{len(names)} x 256B = {len(names) * 256}B (streaming, zero row-data residency)")
