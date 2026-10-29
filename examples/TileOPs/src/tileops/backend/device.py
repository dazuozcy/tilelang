"""Device backend abstraction — the single device adaptation surface.

Every device-specific operation in the framework (device detection, tensor
placement, synchronization, event timing, memory cache, device properties,
environment metadata) is funneled through :class:`DeviceBackend`.

Three backends are supported:

- **NPU** (Ascend A5 via ``torch_npu``) — the primary target.
- **CUDA** (NVIDIA via ``torch.cuda``) — retained for cross-validation.
- **CPU** — fallback when no accelerator is available.

The framework code never calls ``torch.cuda.*`` or ``torch.npu.*`` directly;
it always goes through :func:`get_device_backend`.  Adding a new backend
(e.g. a different NPU vendor) means implementing one class here — the rest
of the framework stays unchanged.

This module is also the single source of truth for the Ascend A5 (dav-3510)
on-chip Unified Buffer (UB) capacity constants: the physical UB is 248 KiB,
and a SimtVF kernel additionally pays a 32 KiB per-core thread context,
leaving 216 KiB for allocations (mirrors
``src/ascend/transform/auto_schedule.cc`` ``GetSharedMemoryLimit``).
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import Optional

import torch

__all__ = [
    "DEFAULT_DEVICE",
    "UB_PHYSICAL_BYTES",
    "UB_SIMTVF_BUDGET_BYTES",
    "UB_SIMTVF_CONTEXT_BYTES",
    "CPUBackend",
    "CUDABackend",
    "DeviceBackend",
    "NPUBackend",
    "get_device_backend",
]

# Ascend A5 (dav-3510, Ascend950) Unified Buffer constants — the single
# source of truth for the on-chip UB budget in this framework.  Mirrors
# ``src/ascend/transform/auto_schedule.cc`` (``GetSharedMemoryLimit``): the
# physical UB is 248 KiB, and a SimtVF kernel additionally pays a 32 KiB
# per-core thread context, leaving 216 KiB for allocations.
UB_PHYSICAL_BYTES: int = 248 * 1024
UB_SIMTVF_CONTEXT_BYTES: int = 32 * 1024
UB_SIMTVF_BUDGET_BYTES: int = UB_PHYSICAL_BYTES - UB_SIMTVF_CONTEXT_BYTES


class DeviceBackend(ABC):
    """Abstract device backend.

    Subclasses wrap the vendor-specific runtime so that the rest of the
    framework is fully device-agnostic.
    """

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def current_device(self) -> int: ...

    @abstractmethod
    def synchronize(self, device: Optional[int] = None) -> None: ...

    @abstractmethod
    def empty_cache(self) -> None: ...

    @abstractmethod
    def get_device_name(self, device: Optional[int] = None) -> str: ...

    @abstractmethod
    def get_device_properties(self, device: Optional[int] = None) -> object: ...

    @abstractmethod
    def is_device_tensor(self, tensor: torch.Tensor) -> bool: ...

    @abstractmethod
    def Event(self, enable_timing: bool = False) -> object: ...

    @abstractmethod
    def manual_seed_all(self, seed: int) -> None: ...

    @abstractmethod
    def env_metadata(self) -> list[str]:
        """Return markdown lines describing the device environment."""
        ...

    def cache_flush(self) -> None:
        """Flush the device cache before benchmark iterations.

        NPU and CPU have no direct equivalent of the CUDA L2-flush; the
        default is a no-op.  CUDA overrides this with an L2-sized buffer
        fill.
        """
        return

    def supports_profiler(self) -> bool:
        """Whether ``torch.profiler`` CUPTI-style kernel tracing is available."""
        return False

    def shared_memory_budget(self, device: Optional[int] = None) -> int:
        """Return the on-chip memory budget in bytes for tile sizing.

        Subclasses should override to return the actual device-specific
        budget: the Ascend Unified Buffer (UB), CUDA shared memory, etc.
        The default returns a conservative 48 KiB which is safe for all
        backends.
        """
        return 48 * 1024


class NPUBackend(DeviceBackend):
    """Ascend NPU backend via ``torch_npu``."""

    name = "npu"

    def __init__(self) -> None:
        import torch_npu  # noqa: F401  — import side-effect registers NPU ops

    def is_available(self) -> bool:
        return torch.npu.is_available()

    def device_count(self) -> int:
        return torch.npu.device_count()

    def current_device(self) -> int:
        return torch.npu.current_device()

    def synchronize(self, device: Optional[int] = None) -> None:
        torch.npu.synchronize(device)

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def get_device_name(self, device: Optional[int] = None) -> str:
        return torch.npu.get_device_name(device)

    def get_device_properties(self, device: Optional[int] = None) -> object:
        return torch.npu.get_device_properties(device)

    def is_device_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_npu

    def Event(self, enable_timing: bool = False) -> object:
        return torch.npu.Event(enable_timing=enable_timing)

    def manual_seed_all(self, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    def shared_memory_budget(self, device: Optional[int] = None) -> int:
        """Return the Unified Buffer (UB) budget in bytes for tile sizing.

        A5 (dav-3510, Ascend950) UB capacity is a per-chip-model constant,
        not a per-device property, so the *device* argument is accepted only
        for API parity with :meth:`DeviceBackend.shared_memory_budget`.  The
        returned budget mirrors ``src/ascend/transform/auto_schedule.cc``
        (``GetSharedMemoryLimit``): the physical UB is 248 KiB
        (:data:`UB_PHYSICAL_BYTES`), and a SimtVF kernel additionally pays
        a 32 KiB per-core thread context, leaving 216 KiB
        (:data:`UB_SIMTVF_BUDGET_BYTES`).  SimtVF is the default programming
        mode, so the conservative SimtVF budget is returned; kernels that
        are known not to use SimtVF may size against
        :data:`UB_PHYSICAL_BYTES` instead.
        """
        return UB_SIMTVF_BUDGET_BYTES

    def env_metadata(self) -> list[str]:
        lines = [
            f"- **Torch version**: {torch.__version__}",
        ]
        try:
            import torch_npu

            lines.append(f"- **torch_npu version**: {torch_npu.__version__}")
        except ImportError:
            lines.append("- **torch_npu version**: N/A")

        if self.is_available():
            lines.append(f"- **NPU model**: {self.get_device_name(0)}")
            lines.append(f"- **NPU count**: {self.device_count()}")
        else:
            lines.append("- **NPU model**: N/A (no NPU device)")

        npu_info = self._query_npu_smi()
        if npu_info:
            lines.append(f"- **Driver version**: {npu_info.get('driver', 'N/A')}")
            lines.append(
                f"- **NPU clocks**: AI core {npu_info.get('aicore', 'N/A')} MHz, "
                f"memory {npu_info.get('memory', 'N/A')} MHz"
            )
        return lines

    @staticmethod
    def _query_npu_smi() -> dict:
        """Query ``npu-smi`` for driver and clock info."""
        try:
            result = subprocess.run(
                [
                    "npu-smi",
                    "info",
                    "-t",
                    "board",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            info: dict = {}
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "Driver Version" in line:
                        info["driver"] = line.split(":")[-1].strip()
                    elif "AI Core" in line and "Freq" in line:
                        info["aicore"] = line.split(":")[-1].strip().split()[0]
                    elif "Memory" in line and "Freq" in line:
                        info["memory"] = line.split(":")[-1].strip().split()[0]
            return info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}


class CUDABackend(DeviceBackend):
    """NVIDIA CUDA backend (retained for cross-validation)."""

    name = "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def synchronize(self, device: Optional[int] = None) -> None:
        torch.cuda.synchronize(device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def get_device_name(self, device: Optional[int] = None) -> str:
        return torch.cuda.get_device_name(device)

    def get_device_properties(self, device: Optional[int] = None) -> object:
        return torch.cuda.get_device_properties(device)

    def is_device_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def Event(self, enable_timing: bool = False) -> object:
        return torch.cuda.Event(enable_timing=enable_timing)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def shared_memory_budget(self, device: Optional[int] = None) -> int:
        """Return the actual opt-in shared memory budget in bytes.

        Queries ``torch.cuda.get_device_properties`` for the real
        ``shared_memory_per_block_optin`` (falling back to
        ``shared_memory_per_block``).  When *device* is explicitly
        requested but CUDA is unavailable, a ``RuntimeError`` is raised
        rather than silently returning the default.
        """
        explicit = device is not None
        try:
            if not torch.cuda.is_available():
                if explicit:
                    raise RuntimeError(
                        f"CUDA is not available but explicit device={device} was requested"
                    )
                return 48 * 1024

            if device is None:
                device = torch.cuda.current_device()

            props = torch.cuda.get_device_properties(device)
            smem_optin = getattr(props, "shared_memory_per_block_optin", 0)
            if smem_optin > 0:
                return smem_optin
            return getattr(props, "shared_memory_per_block", 48 * 1024)
        except (RuntimeError, AssertionError):
            if explicit:
                raise
            return 48 * 1024

    def supports_profiler(self) -> bool:
        return True

    _l2_flush_cache: Optional[torch.Tensor] = None

    def cache_flush(self) -> None:
        """Flush L2 cache by zeroing an L2-sized buffer (CUDA-specific)."""
        if self._l2_flush_cache is None:
            l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
            if l2_bytes <= 0:
                l2_bytes = int(256e6)
            self._l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
        self._l2_flush_cache.zero_()

    def env_metadata(self) -> list[str]:
        lines = [
            f"- **Torch version**: {torch.__version__}",
            f"- **CUDA version (torch)**: {torch.version.cuda or 'N/A'}",
        ]
        if self.is_available():
            lines.append(f"- **GPU model**: {self.get_device_name(0)}")
        else:
            lines.append("- **GPU model**: N/A (no CUDA device)")
        return lines


class CPUBackend(DeviceBackend):
    """CPU fallback backend (for smoke testing without a device)."""

    name = "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    def synchronize(self, device: Optional[int] = None) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_device_name(self, device: Optional[int] = None) -> str:
        return "CPU"

    def get_device_properties(self, device: Optional[int] = None) -> object:
        class _CPUProps:
            name = "CPU"
            total_memory = 0

        return _CPUProps()

    def is_device_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cpu"

    def Event(self, enable_timing: bool = False) -> object:
        class _CPUEvent:
            def record(self) -> None:
                import time

                self._t = time.time()

            def elapsed_time(self, other: "_CPUEvent") -> float:
                return (other._t - self._t) * 1000.0

        return _CPUEvent()

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def env_metadata(self) -> list[str]:
        return [
            f"- **Torch version**: {torch.__version__}",
            "- **Device**: CPU (no accelerator)",
        ]


def _detect_backend() -> DeviceBackend:
    """Auto-detect the best available backend (NPU > CUDA > CPU)."""
    try:
        backend = NPUBackend()
        if backend.is_available():
            return backend
    except ImportError:
        pass

    cuda = CUDABackend()
    if cuda.is_available():
        return cuda

    return CPUBackend()


_backend: Optional[DeviceBackend] = None


def get_device_backend() -> DeviceBackend:
    """Return the singleton :class:`DeviceBackend` for the current environment."""
    global _backend
    if _backend is None:
        _backend = _detect_backend()
    return _backend


def reset_device_backend() -> None:
    """Reset the cached backend (useful for testing)."""
    global _backend
    _backend = None


DEFAULT_DEVICE: str = get_device_backend().name
