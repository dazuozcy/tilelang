"""Reviewer probe (Stage 2, tiled logsumexp): reduce over a fragment whose
element count (SUB*TILE_R = 256) is SMALLER than the SimtVF thread count
(1024) -- the tail-chunk configuration of (33, 33000): SUB=1, rem=232,
TILE_R=256, BLOCK_N=16384 -> threads=min(1024, 1*16384)=1024.

Mirrors the DESIGN tail-chunk body exactly: static rem-slice copy -> masked
fragment load (pad=-inf) -> reduce_max(dim=1, clear=True) -> exp in-place ->
reduce_sum(dim=1, clear=True) -> lse/sum writeback.

Detects: (a) compile/lowering rejection of replicated-layout reduce;
(b) replication double-count in SUM (ratio would be 4.0);
(c) max/sum corruption from idle-thread partials.
"""

import torch
import tilelang
import tilelang.ascend.language as T

SUB, TILE_R, rem, rows, threads = 1, 256, 232, 1, 1024


@tilelang.jit(target="ascend", out_idx=[1])
def _probe():
    @T.prim_func
    def main(x: T.Tensor((rows, rem), "float16"), y: T.Tensor((2,), "float32")):
        with T.Kernel(1) as core_id:
            x_ub = T.alloc_shared((SUB, TILE_R), "float16")
            y_ub = T.alloc_shared((2,), "float32")
            T.copy(x[0:rows, 0:rem], x_ub[0:rows, 0:rem])
            with T.SimtVF(threads=threads):
                x_frag = T.alloc_fragment((SUB, TILE_R), "float32")
                m_frag = T.alloc_fragment((SUB,), "float32")
                s_frag = T.alloc_fragment((SUB,), "float32")
                out_frag = T.alloc_fragment((SUB,), "float32")
                for r, i in T.Parallel(SUB, TILE_R):
                    x_frag[r, i] = T.if_then_else(
                        T.And(r < rows, i < rem),
                        T.cast(x_ub[r, i], "float32"),
                        -T.infinity(T.float32),
                    )
                T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                for r, i in T.Parallel(SUB, TILE_R):
                    x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                for r in T.Parallel(SUB):
                    out_frag[r] = m_frag[r] + T.log(s_frag[r])
                T.copy(out_frag, y_ub[0:1])
                T.copy(s_frag, y_ub[1:2])
            T.copy(y_ub, y)

    return main


def main():
    torch.manual_seed(0)
    k = _probe()
    x = torch.randn(rows, rem, dtype=torch.float16, device="npu")
    y = k(x)
    torch.npu.synchronize()
    xf = x.to(torch.float64)
    m_ref = xf.max(dim=1).values
    s_ref = torch.exp(xf - m_ref.unsqueeze(1)).sum(dim=1)
    lse_ref = m_ref + torch.log(s_ref)
    lse_k, s_k = y[0].item(), y[1].item()
    ratio = s_k / s_ref[0].item()
    print(f"kernel lse={lse_k:.8f} sum={s_k:.8f}")
    print(f"ref    lse={lse_ref[0].item():.8f} sum={s_ref[0].item():.8f}")
    print(f"sum ratio={ratio:.6f} (1.0 => no replication double-count; 4.0 => BUG)")
    assert abs(lse_k - lse_ref[0].item()) < 1e-3, "LSE mismatch"
    assert abs(ratio - 1.0) < 1e-4, f"SUM replication double-count: ratio={ratio}"
    print(
        "PROBE_PASS: reduce_max/reduce_sum(dim=1) over 256-elem fragment "
        "@ SimtVF(threads=1024) correct (extent<threads, replication=4)"
    )


if __name__ == "__main__":
    main()
