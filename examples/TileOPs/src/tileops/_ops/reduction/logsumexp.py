"""LogSumExp operator (Ascend A5).

Ported from the GPU TileOPs ``src/tileops/_ops/reduction/softmax.py`` — the
softmax-family base class (``_SoftmaxBaseOp``) and ``LogSumExpFwdOp``.  The
softmax/log_softmax ops are not yet migrated to A5; they will reuse
``_SoftmaxBaseOp`` (including its ``eval_roofline`` branches) when added.

A5 adaptations (O1-O6 of ``docs/gpu_to_ascend_adaptation.md``):

- Device check uses ``backend.is_device_tensor(x)`` via
  ``get_device_backend()`` instead of ``x.is_cuda``.
- ``tune`` parameter removed from ``__init__`` and ``_get_or_create_kernel``;
  the kernel constructor takes ``(M, N, op_kind, dtype, config=None,
  device_index=None)``.
- No ``compile_boundary`` registration and no ``get_sm_version`` import
  (arch-compat checking lives in the Op base via the device backend).
- Op-specific flow (validate → dim resolution → flatten/transpose → reshape
  → kernel dispatch → output reshape; multi-dim reduction via
  ``_multidim`` helpers; dynamic ``_static_axes`` binding) preserved
  unchanged.
"""

import warnings
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.reduction.logsumexp import LogSumExpKernel
from tileops._ops._multidim import (
    EmptyDimPolicy,
    flatten_for_multidim,
    normalize_dim,
    restore_multidim_shape,
)
from tileops._ops.op_base import Op
from tileops.backend.device import get_device_backend

__all__ = ["LogSumExpFwdOp", "_SoftmaxBaseOp"]


def _resolve_implicit_softmax_dim(name: str, ndim: int) -> int:
    """Mirror ``torch.nn.functional._get_softmax_dim``."""
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        UserWarning,
        stacklevel=3,
    )
    if ndim in (0, 1, 3):
        return 0
    return 1


class _SoftmaxBaseOp(Op):
    """Base class for softmax-family ops.

    Handles shared validation and reshape logic. Subclasses only
    need to set ``_op_kind``, ``_kernel_key``, ``_kernel_class`` and
    override output reshaping if needed.

    Args:
        dtype: Optional committed dtype for subclasses that still expose it.
        dim: Reduction dimension (default -1).
        N: Optional committed reduction dim for subclasses that still expose it.
        kernel_map: Optional override for kernel dispatch.
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass
    _supports_multidim: bool = False  # override to True in reduced-dim ops (e.g. LogSumExpFwdOp)
    _empty_dim_policy: EmptyDimPolicy = "reject"

    # `static_dims.N = x.shape[dim]` is param-dependent (depends on `dim`),
    # so the static-axis frozenset is bound at forward time after dim
    # normalization, not at the class level.
    _static_axes: frozenset = frozenset()

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        dim: Union[int, List[int]] = -1,
        N: Optional[int] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.dtype = dtype
        self._committed_dtype = dtype
        self.dim = dim
        self.N = N
        self.keepdim = False
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self.kernel: object | None = None
        self._last_roofline_spec: tuple[int, int, torch.dtype] | None = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    # Validation

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"x.dtype must be float16, bfloat16, or float32, got {x.dtype}")
        if self._committed_dtype is not None and x.dtype != self._committed_dtype:
            raise ValueError(f"Expected x.dtype {self._committed_dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")
        self.dtype = x.dtype

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Accepts arbitrary-dim input along the configured dim.
        Supports ``dim=list[int]`` for multi-dim reduction (logsumexp).
        """
        self._validate(x)
        orig_shape = x.shape

        # Resolve dim=None per call (don't mutate self.dim) so the same op
        # instance accepts inputs of different ranks, matching F.softmax.
        effective_dim: Union[int, List[int], Tuple[int, ...], None] = self.dim
        if effective_dim is None and not self._supports_multidim:
            effective_dim = _resolve_implicit_softmax_dim(self._op_kind, x.ndim)

        if isinstance(effective_dim, (list, tuple)) or effective_dim is None:
            if not self._supports_multidim:
                raise ValueError(
                    f"{type(self).__name__} does not support multi-dim reduction. Use a scalar dim."
                )
            dims = normalize_dim(
                effective_dim,
                x.ndim,
                empty_dim_policy=self._empty_dim_policy,
            )
            # Bind the dynamic static-axes (param-dependent reduction axes) so
            # the Op-layer cache-key / introspection consumers see the
            # committed axes. Mirrors the single-dim path below.
            self._static_axes = frozenset((0, d) for d in dims)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            dtype = x.dtype
            self._last_roofline_spec = (M, N, dtype)
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(
                M,
                N,
                dtype=dtype,
                device_index=x.device.index,
            )
            self.kernel = kernel
            # Alignment padding is handled by the kernel's forward().
            y = kernel(x)
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        # --- single-dim path ---
        # Validate and normalize dim (match PyTorch IndexError behavior).
        assert isinstance(effective_dim, int)
        if effective_dim < -x.ndim or effective_dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {effective_dim})"
            )
        dim = effective_dim % x.ndim

        # N = size along reduction dim, M = product of all other dims.
        N = x.shape[dim]
        if self.N is not None and N != self.N:
            raise ValueError(
                f"{type(self).__name__}: committed N={self.N} does not match "
                f"x.shape[{effective_dim}]={N}"
            )
        # Bind the dynamic static-axis (param-dependent N axis) so the
        # Op-layer cache-key / introspection consumers see the committed axis.
        self._static_axes = frozenset({(0, dim)})
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        dtype = x.dtype
        self._last_roofline_spec = (M, N, dtype)

        # If reduction dim is not the last, move it to the end.
        needs_transpose = dim != x.ndim - 1
        if needs_transpose:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N, device).
        kernel = self._get_or_create_kernel(
            M,
            N,
            dtype=dtype,
            device_index=x.device.index,
        )
        self.kernel = kernel

        # Alignment padding is handled by the kernel's forward().
        y = kernel(x)

        return self._reshape_output(y, orig_shape, dim, needs_transpose)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        if self._op_kind == "softmax":
            return 5 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "log_softmax":
            return 5 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "logsumexp":
            return 4 * M * N, (M * N + M) * elem_bytes
        raise NotImplementedError(
            f"{type(self).__name__} has unknown roofline op kind {self._op_kind!r}"
        )

    def _get_or_create_kernel(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device_index: int | None = None,
    ) -> object:
        """Return a cached kernel for (M, N, dtype, device), creating one if needed.

        The kernel constructor takes ``(M, N, op_kind, dtype, config=None,
        device_index=None)`` — no ``tune`` argument (A5: heuristic config
        selection only).
        """
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M,
                N,
                self._op_kind,
                dtype,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    # Output reshaping

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        needs_transpose: bool,
    ) -> torch.Tensor:
        """Restore original shape.

        Default (softmax/log_softmax): output same shape as input.
        Reduced-dim ops (logsumexp): remove or keep dim based on keepdim.
        """
        # y is (M, N) or (M,) depending on op kind.
        if y.ndim == 2:
            # Same-shape ops: rebuild the transposed shape, then move dim back.
            if needs_transpose:
                transposed_shape = list(orig_shape)
                transposed_shape.append(transposed_shape.pop(dim))
                y = y.reshape(transposed_shape)
                y = y.movedim(-1, dim)
            else:
                y = y.reshape(orig_shape)
        else:
            # Reduced-dim ops (logsumexp): (M,) -> remove or keep dim.
            if self.keepdim:
                kept_shape = list(orig_shape)
                kept_shape[dim] = 1
                y = y.reshape(kept_shape)
            else:
                reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
                y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y


class LogSumExpFwdOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension
    (or with size-1 if keepdim=True).

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension (default False).
        kernel_map: Optional override for kernel dispatch.
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel
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
