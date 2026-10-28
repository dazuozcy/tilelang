"""LogSumExp operator — SIMD-mode variant (Ascend A5).

Independent SIMD-mode variant of the LogSumExp operator, ported from the GPU
TileOPs ``tileops/ops/reduction/softmax.py`` ``LogSumExpFwdOp`` class.  This
module does not modify the already-integrated SIMT op at
``tileops/ops/reduction/logsumexp.py``; it reuses that module's A5-adapted
``_SoftmaxBaseOp`` (device check via ``get_device_backend()``, no ``tune``
param, kernel constructor ``(M, N, op_kind, dtype, config=None,
device_index=None)``) and binds the SIMD kernel class
``LogSumExpSimdKernel``.  The Stage 1-3 conductor re-implementation of the
underlying kernel builders must use the **SimdVF** programming mode
(explicit 2048-bit ``T.simd.*`` vector intrinsics); ``auto_schedule`` is
``yes``.

Op-specific flow (O6 — preserved unchanged from the GPU op):
validate → dim resolution → flatten/transpose → reshape to (M, N) → kernel
dispatch → output reshape; multi-dim reduction via ``_multidim`` helpers;
dynamic ``_static_axes`` binding; roofline formula ``4*M*N`` flops /
``(M*N + M)*elem_bytes`` bytes.
"""

from typing import Dict, List, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.logsumexp_simd import LogSumExpSimdKernel

# A5-adapted softmax-family base (O1-O6 already applied in the SIMT port);
# shared to keep the family flow in one place without touching the SIMT op.
from tileops.ops.reduction.logsumexp import _SoftmaxBaseOp

__all__ = ["LogSumExpFwdOpSimd"]


class LogSumExpFwdOpSimd(_SoftmaxBaseOp):
    """LogSumExp operator (SIMD variant): y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension
    (or with size-1 if keepdim=True).  Dispatches the SIMD-mode kernel
    ``LogSumExpSimdKernel``; results are expected to match the SIMT
    variant within tolerance.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension (default False).
        kernel_map: Optional override for kernel dispatch.
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpSimdKernel
    _supports_multidim = True

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        super().__init__(dtype=dtype, dim=dim, kernel_map=kernel_map)
        self.keepdim = keepdim
