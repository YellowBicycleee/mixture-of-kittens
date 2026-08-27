"""Host gate for the first CUDA-shaped CuTe DSL BF16 FC1 slice.

This is deliberately *not* a persistent-forward backend.  It has one current
purpose: compile and eventually correctness-test the cluster-2 Gate/Up MMA
topology used by ``csrc/megakernel/fused_gate_up.cuh`` without packing the two
weight tensors.

The fixed slice is one logical ``M256 x N128 x K4096`` Gate/Up tile.  CTA 0
loads its local ``M128`` A rows and the Gate ``N128`` weight tile; CTA 1 loads
its different local ``M128`` A rows and the Up ``N128`` tile.  The cooperative
MMA exposes ``[Gate128 | Up128]`` in each CTA's local TMEM accumulator.  The
current device body writes that raw packed accumulator for a later numerical
probe.  SwiGLU, Down, communication workers, readiness scheduling, and the
148-CTA persistent launch are N/A in this slice.

Imports of torch/CUTLASS remain lazy so CPU-only source tests can validate the
contract.  There is no fallback to the host-wavefront forward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


TARGET_ARCH: Final = "sm_103a"
FC1_TILE_M: Final = 256
FC1_LOGICAL_N: Final = 128
FC1_PACKED_N: Final = 2 * FC1_LOGICAL_N
FC1_TILE_K: Final = 4096
CTA_M: Final = FC1_TILE_M // 2
THREADS_PER_CTA: Final = 256
CLUSTER_SHAPE: Final = (2, 1, 1)

# This flag records source presence only.  GPU compile and numerical
# correctness remain separate acceptance gates.
FC1_SLICE_SOURCE_BODY_PRESENT: Final = True
FULL_PERSISTENT_FORWARD_COMPLETE: Final = False


@dataclass(frozen=True)
class Fc1SlicePlan:
    """The only specialization accepted by the private compile probe."""

    arch: str = TARGET_ARCH
    m: int = FC1_TILE_M
    n: int = FC1_LOGICAL_N
    k: int = FC1_TILE_K
    cluster_shape: tuple[int, int, int] = CLUSTER_SHAPE
    threads_per_cta: int = THREADS_PER_CTA

    def validate(self) -> None:
        wanted = (
            TARGET_ARCH,
            FC1_TILE_M,
            FC1_LOGICAL_N,
            FC1_TILE_K,
            CLUSTER_SHAPE,
            THREADS_PER_CTA,
        )
        actual = (
            self.arch,
            self.m,
            self.n,
            self.k,
            self.cluster_shape,
            self.threads_per_cta,
        )
        if actual != wanted:
            raise NotImplementedError(
                "the BF16 FC1 slice is fixed to "
                "SM103 M256xN128xK4096, cluster=(2,1,1), block=256"
            )


def validate_fc1_slice_tensors(x, gate, up, packed_output) -> None:
    """Validate four torch tensors without importing torch at module import."""

    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    device = x.device
    tensors = {
        "x": (x, (FC1_TILE_M, FC1_TILE_K)),
        "gate": (gate, (FC1_LOGICAL_N, FC1_TILE_K)),
        "up": (up, (FC1_LOGICAL_N, FC1_TILE_K)),
        "packed_output": (packed_output, (FC1_TILE_M, FC1_PACKED_N)),
    }
    for name, (tensor, shape) in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} data pointer must be 16-byte aligned")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")


def compile_fc1_slice(
    x,
    gate,
    up,
    packed_output,
    *,
    stream=None,
    plan: Fc1SlicePlan = Fc1SlicePlan(),
):
    """Compile the private raw Gate/Up slice; do not run a full Forward.

    The returned executor is intentionally separate from the public backend.
    Calling this function on a non-SM103 device or without exact CuTe DSL
    dependencies is an explicit error, never a fallback.
    """

    import torch

    plan.validate()
    validate_fc1_slice_tensors(x, gate, up, packed_output)
    if torch.cuda.get_device_capability(x.device) != (10, 3):
        raise NotImplementedError("the BF16 FC1 slice requires B300/SM103")

    from ._persistent_bf16_gemm import compile_fc1_slice as _compile

    return _compile(x, gate, up, packed_output, stream=stream)


__all__ = [
    "CLUSTER_SHAPE",
    "FC1_LOGICAL_N",
    "FC1_PACKED_N",
    "FC1_SLICE_SOURCE_BODY_PRESENT",
    "FC1_TILE_K",
    "FC1_TILE_M",
    "FULL_PERSISTENT_FORWARD_COMPLETE",
    "Fc1SlicePlan",
    "compile_fc1_slice",
    "validate_fc1_slice_tensors",
]
