"""Kernel base class — TileLang-based, shared by GPU and Ascend backends.

Since TileLang supports both CUDA and Ascend backends, this module preserves
the TileLang kernel integration. The ``supported_archs`` is typed as
``Optional[list]`` to accommodate both CUDA SM integers (``[80, 86, 89, 90]``)
and Ascend architecture name strings (``["Ascend950"]`` for dav-3510);
``None`` means all architectures are supported.

Autotune has been removed: Ascend TileLang kernels use heuristic config
selection only. The ``init_config`` method resolves config from
``default_config`` or a user-provided override.

The programming-mode surface (simt / simd / mixed, plus the orthogonal
auto_schedule pass) is a property of each kernel's source form, not of this
base class: kernels are compiled through the canonical
``@tilelang.jit(target="ascend")`` entry and declare their mode inside the
kernel body (``T.SimtVF`` / ``T.SimdVF`` / ``T.MixedKernel``).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class Kernel(ABC):
    """Abstract base class for TileLang-based operator kernels.

    Subclasses set:
    - ``supported_archs``: list of supported architectures (SM ints or
      Ascend name strings, e.g. ``["Ascend950"]``); ``None`` means all
      supported.
    - ``default_config``: dict of default kernel parameters.
    - ``kernel``: the TileLang JIT-compiled kernel callable.
    """

    dtype: Optional[torch.dtype] = None
    config: Dict[str, Any]
    supported_archs: Optional[list] = None
    kernel: Optional[Any] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = {}

    def init_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            merged = dict(self.default_config)
            for k in merged:
                if config.get(k) is not None:
                    merged[k] = config[k]
            self.config = merged
        else:
            self.config = dict(self.default_config)

        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def dtype_str(self) -> str:
        """Convert dtype to str for tl kernels"""
        return self.dtype_to_str(self.dtype)

    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        """Convert a torch dtype to the TileLang dtype string."""
        return str(dtype).split(".")[-1]

    @property
    def default_config(self) -> Dict[str, Any]:
        """Return the default config for the kernel"""
        return {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the kernel"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
