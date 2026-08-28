from dataclasses import dataclass

import torch

from .functional import (
    MoKConfig,
    MoKSchedule,
    MoKWorkspace,
    validate_inputs,
)
from .ops import all_gather_top_experts, barrier_all, fwd_epilogue
from .union_ops import (
    dispatch_mlp_swiglu_combine_fwd_bf16_union_x,
    union_schedule,
)


@dataclass(frozen=True, slots=True)
class MoKUnionSchedule:
    peer_rank: torch.Tensor
    peer_token_idx: torch.Tensor
    num_tokens: torch.Tensor
    tokens_per_expert: torch.Tensor
    route_to_union: torch.Tensor
    num_union: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoKUnionXForwardContext:
    union_x: torch.Tensor
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor
    up_shared: torch.Tensor
    up_routed: torch.Tensor
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor
    schedule: MoKUnionSchedule


def _validate_union_config(
    workspace: MoKWorkspace,
    config: MoKConfig,
) -> None:
    """Reject Union-X configurations that cannot make forward progress."""
    device_properties = torch.cuda.get_device_properties(workspace.device)
    if (
        type(config.fwd_num_comm_sms) is not int
        or config.fwd_num_comm_sms <= 0
        or config.fwd_num_comm_sms % 2
        or config.fwd_num_comm_sms >= device_properties.multi_processor_count
    ):
        raise ValueError(
            "fwd_num_comm_sms must be positive, even, and leave a compute SM"
        )
    if (
        type(config.minibatch_size) is not int
        or config.minibatch_size <= 0
        or config.minibatch_size % 256
    ):
        raise ValueError(
            "minibatch_size must be positive and divisible by 256"
        )
    if (
        type(config.macrobatch_size) is not int
        or config.macrobatch_size <= 0
        or config.macrobatch_size % config.minibatch_size
    ):
        raise ValueError(
            "macrobatch_size must be a positive multiple of minibatch_size"
        )
    chunk_bytes = config.all_gather_top_experts_chunk_bytes
    if (
        type(chunk_bytes) is not int
        or chunk_bytes <= 0
        or chunk_bytes % 16
        or chunk_bytes + 1024
            > device_properties.shared_memory_per_block_optin
        or workspace.num_local_tokens * workspace.topk * 4 % chunk_bytes
    ):
        raise ValueError(
            "all_gather_top_experts_chunk_bytes must be an aligned divisor "
            "of one rank's route buffer and fit in shared memory"
        )


def build_union_schedule(
    workspace: MoKWorkspace,
    config: MoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
) -> MoKUnionSchedule:
    """Build the legacy route schedule plus compact Union-X row ids."""
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    _validate_union_config(workspace, config)
    if not top_experts.is_cuda or top_experts.device != workspace.device:
        raise ValueError("top_experts must be on the workspace CUDA device")
    if top_experts.dtype != torch.int64:
        raise TypeError("top_experts must have dtype torch.int64")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if tuple(top_experts.shape) != (
        workspace.num_local_tokens,
        workspace.topk,
    ):
        raise ValueError(
            "top_experts must have shape (num_local_tokens, topk)"
        )
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")

    top_experts_int32 = top_experts.to(torch.int32)
    all_gather_top_experts(
        top_experts_int32,
        workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr,
        workspace.ep_rank,
        config.all_gather_top_experts_chunk_bytes,
    )
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    outputs = union_schedule(
        workspace.all_gather_top_experts_buffer,
        num_local_experts,
        workspace.schedule_capacity,
        workspace.ep_rank,
    )
    return MoKUnionSchedule(*outputs)


def forward_union_x(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKUnionSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor,
    routed_up_weights: torch.Tensor,
    routed_down_weights: torch.Tensor,
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, MoKUnionXForwardContext]:
    """Run the private BF16 EP8 Union-X forward path.

    This is intentionally forward-only.  The returned context is a distinct
    type and is rejected by the current MoK backward implementation.
    """
    if not isinstance(schedule, MoKUnionSchedule):
        raise TypeError("schedule must be a MoKUnionSchedule")
    legacy_schedule = MoKSchedule(
        peer_rank=schedule.peer_rank,
        peer_token_idx=schedule.peer_token_idx,
        num_tokens=schedule.num_tokens,
        tokens_per_expert=schedule.tokens_per_expert,
    )
    validate_inputs(
        config,
        workspace,
        legacy_schedule,
        x,
        router_weights,
    )
    _validate_union_config(workspace, config)
    for name, weight in (
        ("shared_gate_weights", shared_gate_weights),
        ("shared_up_weights", shared_up_weights),
        ("shared_down_weights", shared_down_weights),
        ("routed_gate_weights", routed_gate_weights),
        ("routed_up_weights", routed_up_weights),
        ("routed_down_weights", routed_down_weights),
    ):
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"{name} must be a BF16 tensor")

    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    (
        union_x,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        y_shared,
        _y_routed,
    ) = dispatch_mlp_swiglu_combine_fwd_bf16_union_x(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.route_to_union,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        workspace.topk,
        swiglu_limit,
        config.fwd_num_comm_sms,
        config.macrobatch_size,
        config.minibatch_size,
    )
    context = MoKUnionXForwardContext(
        union_x=union_x,
        gate_shared=gate_shared,
        gate_routed=gate_routed,
        up_shared=up_shared,
        up_routed=up_routed,
        hidden_shared=hidden_shared,
        hidden_routed=hidden_routed,
        schedule=schedule,
    )
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    output = fwd_epilogue(
        y_shared,
        workspace.combine_buffer,
        workspace.router_weight_buffer,
    )
    return output, context


__all__ = [
    "MoKUnionSchedule",
    "MoKUnionXForwardContext",
    "build_union_schedule",
    "forward_union_x",
]
