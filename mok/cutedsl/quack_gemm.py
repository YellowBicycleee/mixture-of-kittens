"""Thin QuACK GEMM bridge for MoK's CuTe DSL forward.

QuACK 0.6.4 supplies the SM100/SM103 ``tcgen05`` GEMM kernels.  MoK keeps
ownership of dispatch, SwiGLU, combine, macro ordering, and ring-buffer state;
this module only adapts MoK's weight layout to QuACK's public GEMM interface.
"""

from __future__ import annotations

from functools import cache

import torch


_REQUIRED_QUACK_VERSION = "0.6.4"


@cache
def _gemm():
    try:
        import quack
        from quack.gemm_interface import gemm
    except ImportError as error:
        raise ImportError(
            "MoK's CuTe DSL forward requires the 'cutedsl' extra "
            "(quack-kernels[cu13]==0.6.4)"
        ) from error

    if quack.__version__ != _REQUIRED_QUACK_VERSION:
        raise RuntimeError(
            "MoK's CuTe DSL forward requires "
            f"quack-kernels=={_REQUIRED_QUACK_VERSION}; got {quack.__version__}"
        )
    return gemm


def shared_gemm(
    activations: torch.Tensor,
    weights_nk: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Compute ``[M,K] @ [N,K].T`` into a caller-owned BF16 output."""

    _gemm()(
        activations,
        weights_nk.transpose(0, 1),
        out=output,
        tuned=False,
    )


def routed_gemm(
    activations: torch.Tensor,
    weights_enk: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> None:
    """Run one 64-expert variable-M GEMM on the current PyTorch stream.

    ``weights_enk`` is MoK's contiguous ``[E,N,K]`` allocation.  The transpose
    is a zero-copy ``[E,K,N]`` view matching QuACK's public varlen-M ABI.
    Empty experts are represented by repeated entries in ``cu_seqlens_m``.
    """

    _gemm()(
        activations,
        weights_enk.transpose(-2, -1),
        out=output,
        cu_seqlens_m=cu_seqlens_m,
        dynamic_scheduler=False,
        tuned=False,
    )
