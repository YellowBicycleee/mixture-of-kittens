"""Dependency-free contract for the first real CuTe DSL MoK forward.

The implementation deliberately supports one production target while the
kernel topology is being brought up: Qwen BF16, EP8, H=4096, I=1024,
E=512, top-k=10.  Keeping the shape checks here lets the contract be tested on
a CPU-only development machine without importing torch or CUTLASS.
"""

from __future__ import annotations

from collections.abc import Sequence


EP_SIZE = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
NUM_GLOBAL_EXPERTS = 512
NUM_LOCAL_EXPERTS = NUM_GLOBAL_EXPERTS // EP_SIZE
TOPK = 10
ROW_ALIGNMENT = 256


def macro_offsets(num_tokens: int, macrobatch_size: int) -> tuple[int, ...]:
    """Return real padded-token offsets in reverse order, leaving macro 0 last."""

    if type(num_tokens) is not int or num_tokens < 0:
        raise ValueError("num_tokens must be a non-negative integer")
    if num_tokens % ROW_ALIGNMENT:
        raise ValueError("num_tokens must be divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0:
        raise ValueError("macrobatch_size must be a positive integer")
    if macrobatch_size % ROW_ALIGNMENT:
        raise ValueError("macrobatch_size must be divisible by 256")

    if num_tokens == 0:
        return ()
    last = ((num_tokens - 1) // macrobatch_size) * macrobatch_size
    return tuple(range(last, -1, -macrobatch_size))


def decode_schedule_entry(peer_rank: int, route_idx: int) -> tuple[int, int] | None:
    """Decode one scheduler row into ``(source_token, return_route)``.

    The scheduler stores ``route_idx = source_token * topk + k``.  Padding is
    represented by ``peer_rank == -1`` and must neither read x nor write the
    remote combine buffer.
    """

    if int(peer_rank) == -1:
        return None
    if not 0 <= int(peer_rank) < EP_SIZE:
        raise ValueError(f"peer_rank must be -1 or in [0, {EP_SIZE})")
    if int(route_idx) < 0:
        raise ValueError("route_idx must be non-negative for a real route")
    return int(route_idx) // TOPK, int(route_idx)


def validate_fixed_forward_contract(
    *,
    ep_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    num_local_tokens: int,
    schedule_capacity: int,
    macrobatch_size: int,
    minibatch_size: int,
    x_ptrs: Sequence[int],
    combine_ptrs: Sequence[int],
) -> None:
    """Validate scalar metadata and raw symmetric pointers for the fixed path."""

    expected = {
        "ep_size": (ep_size, EP_SIZE),
        "hidden_size": (hidden_size, HIDDEN_SIZE),
        "intermediate_size": (intermediate_size, INTERMEDIATE_SIZE),
        "num_local_experts": (num_local_experts, NUM_LOCAL_EXPERTS),
        "topk": (topk, TOPK),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise NotImplementedError(
                f"CuTe DSL forward requires {name}={wanted}; got {actual}"
            )

    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % ROW_ALIGNMENT:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(minibatch_size) is not int or minibatch_size <= 0:
        raise ValueError("minibatch_size must be a positive integer")
    if minibatch_size % ROW_ALIGNMENT:
        raise ValueError("minibatch_size must be divisible by 256")
    if macrobatch_size % minibatch_size:
        raise ValueError("macrobatch_size must be a multiple of minibatch_size")

    if type(schedule_capacity) is not int or schedule_capacity <= 0:
        raise ValueError("schedule_capacity must be a positive integer")
    if schedule_capacity % ROW_ALIGNMENT:
        raise ValueError("schedule_capacity must be divisible by 256")
    # Reuse the macrobatch validation from the iterator.
    macro_offsets(schedule_capacity, macrobatch_size)
    for name, pointers in (("x_ptrs", x_ptrs), ("combine_ptrs", combine_ptrs)):
        if len(pointers) != EP_SIZE:
            raise ValueError(f"{name} must contain exactly {EP_SIZE} pointers")
        if any(type(pointer) is not int or pointer <= 0 for pointer in pointers):
            raise ValueError(f"{name} must contain positive integer pointers")
