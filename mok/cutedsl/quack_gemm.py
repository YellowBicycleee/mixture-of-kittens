"""Thin QuACK GEMM bridge for MoK's CuTe DSL BF16 forward.

QuACK 0.6.4 supplies SM100/SM103 ``tcgen05`` GEMMs and its public epilogue
authoring surface. Pipeline-v2 fuses shared and routed Gate, Up, and SwiGLU
while preserving MoK's BF16 preactivation roundtrip exactly. MoK keeps
dispatch, Down, combine, reverse macro order, and the replay ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref

import torch

from .forward_contract import packed_gate_up_shape, packed_weight_cache_key


_REQUIRED_QUACK_VERSION = "0.6.4"

try:
    import quack
    from cutlass import BFloat16, Float32
    from quack.activation import swiglu
    from quack.epilogue.frontend import gemm_epilogue, unpack
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


# Deliberately direct tune seam for the fixed BF16 routed FC1. The first
# correctness round pins the topology; later iterations may sweep this small
# set without changing the dispatch/combine or ABI code.
GATED_TILE_M = 128
GATED_TILE_N = 256
GATED_CLUSTER_M = 1
GATED_CLUSTER_N = 1
GATED_MAX_SWIZZLE = 8


def _bf16_roundtrip(value):
    """Match CUDA: accumulator -> BF16 Gate/Up storage -> F32 SwiGLU input."""

    return value.to(BFloat16).to(Float32)


@gemm_epilogue(outputs=("gate", "up", "hidden"), mode="acc_pair")
def _mok_swiglu_save(acc):
    gate, up = unpack(acc)
    gate = _bf16_roundtrip(gate)
    up = _bf16_roundtrip(up)
    return {"gate": gate, "up": up, "hidden": swiglu(gate, up)}


@gemm_epilogue(outputs=("hidden",), mode="acc_pair")
def _mok_swiglu_replay(acc):
    gate, up = unpack(acc)
    gate = _bf16_roundtrip(gate)
    up = _bf16_roundtrip(up)
    return {"hidden": swiglu(gate, up)}


@dataclass(slots=True)
class _PackedGateUpEntry:
    gate_ref: weakref.ReferenceType[torch.Tensor]
    up_ref: weakref.ReferenceType[torch.Tensor]
    key: tuple[int, int, int, int]
    packed: torch.Tensor
    ready_event: torch.cuda.Event


_PACKED_GATE_UP_BY_SLOT: dict[
    tuple[str, str, int | None], _PackedGateUpEntry
] = {}
_PACKED_GATE_UP_LOCK = threading.Lock()


def _packed_key(
    gate_weights_enk: torch.Tensor,
    up_weights_enk: torch.Tensor,
) -> tuple[int, int, int, int]:
    return packed_weight_cache_key(
        id(gate_weights_enk),
        int(gate_weights_enk._version),
        id(up_weights_enk),
        int(up_weights_enk._version),
    )


def _packed_entry_matches(
    entry: _PackedGateUpEntry | None,
    gate_weights_enk: torch.Tensor,
    up_weights_enk: torch.Tensor,
    key: tuple[int, int, int, int],
) -> bool:
    return (
        entry is not None
        and entry.gate_ref() is gate_weights_enk
        and entry.up_ref() is up_weights_enk
        and entry.key == key
    )


def _wait_for_packed_entry(
    entry: _PackedGateUpEntry,
    device: torch.device,
) -> torch.Tensor:
    """Order this stream after a mirror created on another CUDA stream."""

    stream = torch.cuda.current_stream(device)
    stream.wait_event(entry.ready_event)
    # The cache can replace this tensor after a weight-version change while a
    # different stream is still consuming it. Tell the caching allocator about
    # that use; Event ordering alone protects the start, not the lifetime.
    entry.packed.record_stream(stream)
    return entry.packed


def _packed_gate_up_weights(
    slot: str,
    gate_weights_enk: torch.Tensor,
    up_weights_enk: torch.Tensor,
) -> torch.Tensor:
    """Return one cached contiguous BF16 Gate+Up mirror for a fixed slot.

    The mirror is materialized only on the first call or after either source
    tensor's PyTorch version changes. The cache keeps exactly one ``shared``
    and one ``routed`` entry per device, so model replacement cannot grow an
    unbounded list of 1 GiB Qwen mirrors.
    """

    if slot not in ("shared", "routed"):
        raise ValueError("packed Gate+Up slot must be 'shared' or 'routed'")
    if gate_weights_enk.shape != up_weights_enk.shape:
        raise ValueError("Gate and Up weights must have identical shapes")
    expected_shape = packed_gate_up_shape(slot, tuple(gate_weights_enk.shape))
    cache_slot = (
        slot,
        gate_weights_enk.device.type,
        gate_weights_enk.device.index,
    )
    key = _packed_key(gate_weights_enk, up_weights_enk)
    entry = _PACKED_GATE_UP_BY_SLOT.get(cache_slot)
    if _packed_entry_matches(entry, gate_weights_enk, up_weights_enk, key):
        assert entry is not None
        return _wait_for_packed_entry(entry, gate_weights_enk.device)

    with _PACKED_GATE_UP_LOCK:
        entry = _PACKED_GATE_UP_BY_SLOT.get(cache_slot)
        key = _packed_key(gate_weights_enk, up_weights_enk)
        if _packed_entry_matches(entry, gate_weights_enk, up_weights_enk, key):
            assert entry is not None
            return _wait_for_packed_entry(entry, gate_weights_enk.device)
        cat_dim = 0 if slot == "shared" else 1
        with torch.no_grad():
            packed = torch.cat(
                (gate_weights_enk, up_weights_enk),
                dim=cat_dim,
            )
        if tuple(packed.shape) != expected_shape:
            raise RuntimeError("packed Gate+Up weights have an invalid shape")
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(gate_weights_enk.device))
        _PACKED_GATE_UP_BY_SLOT[cache_slot] = _PackedGateUpEntry(
            gate_ref=weakref.ref(gate_weights_enk),
            up_ref=weakref.ref(up_weights_enk),
            key=key,
            packed=packed,
            ready_event=ready_event,
        )
        return packed


def packed_shared_gate_up_weights(
    gate_weights_nk: torch.Tensor,
    up_weights_nk: torch.Tensor,
) -> torch.Tensor:
    """Return cached contiguous shared weights with shape ``[2I,H]``."""

    return _packed_gate_up_weights("shared", gate_weights_nk, up_weights_nk)


def packed_routed_gate_up_weights(
    gate_weights_enk: torch.Tensor,
    up_weights_enk: torch.Tensor,
) -> torch.Tensor:
    """Return cached contiguous routed weights with shape ``[E,2I,H]``."""

    return _packed_gate_up_weights("routed", gate_weights_enk, up_weights_enk)


def shared_gemm(
    activations: torch.Tensor,
    weights_nk: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Compute ``[M,K] @ [N,K].T`` into a caller-owned BF16 output."""

    gemm(
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

    gemm(
        activations,
        weights_enk.transpose(-2, -1),
        out=output,
        cu_seqlens_m=cu_seqlens_m,
        dynamic_scheduler=False,
        tuned=False,
    )


def _gated_swiglu(
    epilogue,
    activations: torch.Tensor,
    packed_weights_e2ik: torch.Tensor,
    *,
    outputs: dict[str, torch.Tensor],
    cu_seqlens_m: torch.Tensor | None,
) -> None:
    # QuACK's concat-layout contract consumes the packed physical B as
    # [..., N, K].  Its v0.6.4 canonical frontend deliberately disables the
    # alternative KxN (`b_kn`) path whenever concat layout is active.
    # Passing the transposed view plus b_kn=True reaches a low-level validation
    # hole and is not a supported layout combination.
    epilogue.gemm(
        activations,
        packed_weights_e2ik,
        None,
        epi_args=outputs,
        tile_M=GATED_TILE_M,
        tile_N=GATED_TILE_N,
        cluster_M=GATED_CLUSTER_M,
        cluster_N=GATED_CLUSTER_N,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=True,
        max_swizzle_size=GATED_MAX_SWIZZLE,
        cu_seqlens_m=cu_seqlens_m,
        concat_layout=("B",),
    )


def shared_gated_swiglu(
    activations: torch.Tensor,
    packed_weights_2ik: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    hidden_output: torch.Tensor,
) -> None:
    """Fuse shared Gate+Up+SwiGLU and save all three ``[M,I]`` tensors."""

    _gated_swiglu(
        _mok_swiglu_save,
        activations,
        packed_weights_2ik,
        outputs={"gate": gate_output, "up": up_output, "hidden": hidden_output},
        cu_seqlens_m=None,
    )


def routed_gated_swiglu(
    activations: torch.Tensor,
    packed_weights_e2ik: torch.Tensor,
    hidden_output: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    store_preact: bool,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
) -> None:
    """Fuse routed Gate+Up+SwiGLU with replay-only Gate/Up store elision.

    ``acc_pair`` pairs the concat-layout Gate/Up columns. The save epilogue
    directly writes three contiguous ``[M,I]`` tensors; the replay epilogue
    declares only Hidden, so macro>0 has no Gate/Up global-memory stores.
    """

    outputs = {"hidden": hidden_output}
    epilogue = _mok_swiglu_replay
    if store_preact:
        outputs.update(gate=gate_output, up=up_output)
        epilogue = _mok_swiglu_save
    _gated_swiglu(
        epilogue,
        activations,
        packed_weights_e2ik,
        outputs=outputs,
        cu_seqlens_m=cu_seqlens_m,
    )
