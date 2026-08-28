import torch

from . import _C


def _validate_union_schedule_args(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[int, int, int]:
    if not isinstance(topk_all, torch.Tensor):
        raise TypeError("topk_all must be a torch.Tensor")
    if not topk_all.is_cuda:
        raise ValueError("topk_all must be a CUDA tensor")
    if topk_all.dtype != torch.int32:
        raise TypeError("topk_all must have dtype torch.int32")
    if not topk_all.is_contiguous():
        raise ValueError("topk_all must be contiguous")
    if topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")

    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 1, 4, 8, 16, 32, 64")
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError(
            "topk_all num_local_tokens must be at least 512 and divisible by 256"
        )
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if ep_size * num_local_tokens >= 2**31:
        raise ValueError("dense (peer_rank, original_token_idx) key space exceeds int32")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (
        type(schedule_capacity) is not int
        or schedule_capacity <= 0
        or schedule_capacity % 256 != 0
    ):
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    return ep_size, num_local_tokens, topk


@torch.library.custom_op("mok::union_schedule", mutates_args=())
def union_schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the legacy route schedule plus compact rank-token union metadata.

    ``route_to_union[row]`` is the compact id of
    ``(schedule_peer_rank[row], schedule_peer_token_idx[row] // topk)`` for
    every valid route row.  Valid ids are in ``[0, num_union[0])``.  Padding
    rows in ``route_to_union`` are ``-1``.
    """
    _validate_union_schedule_args(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    return _C.union_schedule(
        topk_all, num_local_experts, schedule_capacity, rank
    )


@torch.library.register_fake("mok::union_schedule")
def _union_schedule_fake(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_union_schedule_args(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    return (
        topk_all.new_empty((schedule_capacity,), dtype=torch.int32),
        topk_all.new_empty((schedule_capacity,), dtype=torch.int32),
        topk_all.new_empty((1,), dtype=torch.int32),
        topk_all.new_empty((num_local_experts,), dtype=torch.int32),
        topk_all.new_empty((schedule_capacity,), dtype=torch.int32),
        topk_all.new_empty((1,), dtype=torch.int32),
    )


def _validate_union_x_forward_args(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    route_to_union: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[int, int, int]:
    tensors = (
        ("x", x),
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("route_to_union", route_to_union),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")

    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be BF16 [T, 4096]")
    num_local_tokens, hidden_size = x.shape
    if (
        num_local_tokens < 512
        or num_local_tokens % 256
        or hidden_size != 4096
    ):
        raise ValueError(
            "Union-X forward requires T >= 512, T divisible by 256, "
            "and H=4096"
        )
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if (
        swiglu_limit is not None
        and (
            type(swiglu_limit) not in (int, float)
            or swiglu_limit < 0
        )
    ):
        raise ValueError("swiglu_limit must be None or non-negative")
    if (
        type(num_comm_sms) is not int
        or num_comm_sms <= 0
        or num_comm_sms % 2
    ):
        raise ValueError("num_comm_sms must be a positive even integer")
    if num_comm_sms >= torch.cuda.get_device_properties(
        x.device
    ).multi_processor_count:
        raise ValueError("num_comm_sms must leave at least one compute SM")
    if (
        type(minibatch_size) is not int
        or minibatch_size <= 0
        or minibatch_size % 256
    ):
        raise ValueError(
            "minibatch_size must be positive and divisible by 256"
        )
    if (
        type(macrobatch_size) is not int
        or macrobatch_size <= 0
        or macrobatch_size % minibatch_size
    ):
        raise ValueError(
            "macrobatch_size must be a positive multiple of minibatch_size"
        )

    for name, pointers in (
        ("x_ptrs", x_ptrs),
        ("combine_buffer_ptrs", combine_buffer_ptrs),
    ):
        if (
            not isinstance(pointers, list)
            or len(pointers) != 8
            or any(type(pointer) is not int or pointer <= 0
                   for pointer in pointers)
        ):
            raise ValueError(f"{name} must contain exactly 8 positive pointers")

    if (
        schedule_peer_rank.dtype != torch.int32
        or schedule_peer_rank.ndim != 1
        or schedule_peer_rank.numel() == 0
        or schedule_peer_rank.numel() % 256
    ):
        raise ValueError(
            "schedule_peer_rank must be nonempty int32 [capacity], "
            "capacity divisible by 256"
        )
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule capacity must be at least T * topk")
    for name, tensor in (
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("route_to_union", route_to_union),
    ):
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != (
            schedule_capacity,
        ):
            raise ValueError(
                f"{name} must be int32 [{schedule_capacity}]"
            )
    if num_tokens.dtype != torch.int32 or tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must be int32 [1]")

    if (
        w_shared_gate.dtype != torch.bfloat16
        or w_shared_gate.ndim != 2
        or w_shared_gate.shape[1] != hidden_size
        or w_shared_gate.shape[0] <= 0
        or w_shared_gate.shape[0] % 256
    ):
        raise ValueError("w_shared_gate must be BF16 [I, 4096], I % 256 == 0")
    intermediate_size = w_shared_gate.shape[0]
    if w_shared_up.dtype != torch.bfloat16 or tuple(w_shared_up.shape) != (
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("w_shared_up must match w_shared_gate")
    if (
        w_routed_gate.dtype != torch.bfloat16
        or w_routed_gate.ndim != 3
        or w_routed_gate.shape[0] <= 0
        or tuple(w_routed_gate.shape[1:])
        != (intermediate_size, hidden_size)
    ):
        raise ValueError("w_routed_gate must be BF16 [E, I, 4096]")
    num_local_experts = w_routed_gate.shape[0]
    if w_routed_up.dtype != torch.bfloat16 or tuple(w_routed_up.shape) != (
        num_local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("w_routed_up must match w_routed_gate")
    if w_shared_down.dtype != torch.bfloat16 or tuple(
        w_shared_down.shape
    ) != (hidden_size, intermediate_size):
        raise ValueError("w_shared_down must be BF16 [4096, I]")
    if w_routed_down.dtype != torch.bfloat16 or tuple(
        w_routed_down.shape
    ) != (num_local_experts, hidden_size, intermediate_size):
        raise ValueError("w_routed_down must be BF16 [E, 4096, I]")
    if tokens_per_expert.dtype != torch.int32 or tuple(
        tokens_per_expert.shape
    ) != (num_local_experts,):
        raise ValueError("tokens_per_expert must be int32 [E]")
    if combine_buffer.dtype != torch.bfloat16 or tuple(
        combine_buffer.shape
    ) != (num_local_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must be BF16 [T * topk, 4096]")

    device = x.device
    for name, tensor in tensors[1:]:
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
    return num_local_tokens, hidden_size, intermediate_size


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_fwd_bf16_union_x",
    mutates_args=("combine_buffer",),
)
def dispatch_mlp_swiglu_combine_fwd_bf16_union_x(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    route_to_union: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the private EP8 BF16 parallel forward with persistent Union-X."""
    _validate_union_x_forward_args(
        x,
        x_ptrs,
        combine_buffer,
        combine_buffer_ptrs,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        schedule_peer_rank,
        schedule_peer_token_idx,
        route_to_union,
        num_tokens,
        tokens_per_expert,
        topk,
        swiglu_limit,
        num_comm_sms,
        macrobatch_size,
        minibatch_size,
    )
    return _C.dispatch_mlp_swiglu_combine_fwd_bf16_union_x(
        x,
        x_ptrs,
        combine_buffer,
        combine_buffer_ptrs,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        schedule_peer_rank,
        schedule_peer_token_idx,
        route_to_union,
        num_tokens,
        tokens_per_expert,
        topk,
        swiglu_limit,
        num_comm_sms,
        macrobatch_size,
        minibatch_size,
    )


@torch.library.register_fake(
    "mok::dispatch_mlp_swiglu_combine_fwd_bf16_union_x"
)
def _dispatch_mlp_swiglu_combine_fwd_bf16_union_x_fake(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    route_to_union: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_local_tokens, hidden_size, intermediate_size = (
        _validate_union_x_forward_args(
            x,
            x_ptrs,
            combine_buffer,
            combine_buffer_ptrs,
            w_shared_gate,
            w_routed_gate,
            w_shared_up,
            w_routed_up,
            w_shared_down,
            w_routed_down,
            schedule_peer_rank,
            schedule_peer_token_idx,
            route_to_union,
            num_tokens,
            tokens_per_expert,
            topk,
            swiglu_limit,
            num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
    )
    return (
        x.new_empty((8 * num_local_tokens, hidden_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty((num_local_tokens, hidden_size)),
        x.new_empty((macrobatch_size, hidden_size)),
    )
