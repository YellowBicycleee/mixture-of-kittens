import pytest
import torch
import torch.distributed as dist

from mok import _C, functional, union_functional
from mok.union_ops import dispatch_mlp_swiglu_combine_fwd_bf16_union_x

from .utils import (
    BF16_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
)


def test_union_x_forward_matches_legacy_and_reference(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("Union-X forward currently requires EP8")

    num_local_tokens = 512
    hidden_size = 4096
    intermediate_size = 256
    num_local_experts = 4
    num_experts = world_size * num_local_experts
    topk = 4
    config = functional.MoKConfig(
        fwd_num_comm_sms=40,
        bwd_num_comm_sms=28,
        minibatch_size=256,
        macrobatch_size=4096,
    )

    (
        x,
        _random_top_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _d_output,
    ) = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_size,
        intermediate_size,
    )

    tokens = torch.arange(num_local_tokens, device=device)
    destination_rank = (rank + tokens) % world_size
    top_experts = (
        destination_rank[:, None] * num_local_experts
        + torch.arange(num_local_experts, device=device)[None, :]
    )
    if rank == 0:
        top_experts[0] = num_local_experts + torch.arange(
            num_local_experts, device=device
        )
    top_experts = top_experts.to(torch.int64).contiguous()

    (
        reference_combine,
        reference_gate_shared,
        reference_up_shared,
        reference_hidden_shared,
        reference_y_shared,
    ) = run_forward_reference_bf16(
        x,
        top_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    reference_output = run_fwd_epilogue_reference(
        reference_y_shared, reference_combine, router_weights
    )

    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    legacy_schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
    )
    union_schedule = union_functional.build_union_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
    )
    assert torch.equal(union_schedule.peer_rank, legacy_schedule.peer_rank)
    valid_schedule_rows = legacy_schedule.peer_rank >= 0
    assert torch.equal(
        union_schedule.peer_token_idx[valid_schedule_rows],
        legacy_schedule.peer_token_idx[valid_schedule_rows],
    )
    assert torch.equal(
        union_schedule.tokens_per_expert,
        legacy_schedule.tokens_per_expert,
    )

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    invalid_comm_sms = sm_count + sm_count % 2
    low_level_args = (
        x,
        workspace.x_buffer_ptrs,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        union_schedule.peer_rank,
        union_schedule.peer_token_idx,
        union_schedule.route_to_union,
        union_schedule.num_tokens,
        union_schedule.tokens_per_expert,
        topk,
        None,
        invalid_comm_sms,
        config.macrobatch_size,
        config.minibatch_size,
    )
    with pytest.raises(ValueError, match="leave at least one compute SM"):
        dispatch_mlp_swiglu_combine_fwd_bf16_union_x(*low_level_args)
    with pytest.raises(RuntimeError, match="leave at least one compute SM"):
        _C.dispatch_mlp_swiglu_combine_fwd_bf16_union_x(*low_level_args)

    legacy_output, legacy_context = functional.forward(
        config,
        workspace,
        legacy_schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    legacy_combine = workspace.combine_buffer.clone()
    legacy_tensors = {
        "output": legacy_output.clone(),
        "combine": legacy_combine,
        "gate_shared": legacy_context.gate_shared.clone(),
        "up_shared": legacy_context.up_shared.clone(),
        "hidden_shared": legacy_context.hidden_shared.clone(),
        "gate_routed": legacy_context.gate_routed.clone(),
        "up_routed": legacy_context.up_routed.clone(),
        "hidden_routed": legacy_context.hidden_routed.clone(),
        "x_routed": legacy_context.x_routed.clone(),
    }

    for name, reference, actual in (
        ("legacy output", reference_output, legacy_tensors["output"]),
        ("legacy combine", reference_combine, legacy_tensors["combine"]),
        (
            "legacy gate_shared",
            reference_gate_shared,
            legacy_tensors["gate_shared"],
        ),
        (
            "legacy up_shared",
            reference_up_shared,
            legacy_tensors["up_shared"],
        ),
        (
            "legacy hidden_shared",
            reference_hidden_shared,
            legacy_tensors["hidden_shared"],
        ),
    ):
        check_correctness(name, reference, actual, BF16_TOLERANCE)

    union_output, union_context = union_functional.forward_union_x(
        config,
        workspace,
        union_schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )

    active_rows = int(union_schedule.num_tokens.item())
    route_to_union = union_schedule.route_to_union[:active_rows]
    valid_rows = route_to_union >= 0
    assert int(union_schedule.num_union.item()) < int(valid_rows.sum().item())

    exact_pairs = (
        ("output", union_output, legacy_tensors["output"]),
        ("combine", workspace.combine_buffer, legacy_tensors["combine"]),
        (
            "gate_shared",
            union_context.gate_shared,
            legacy_tensors["gate_shared"],
        ),
        (
            "up_shared",
            union_context.up_shared,
            legacy_tensors["up_shared"],
        ),
        (
            "hidden_shared",
            union_context.hidden_shared,
            legacy_tensors["hidden_shared"],
        ),
        (
            "gate_routed",
            union_context.gate_routed[:active_rows],
            legacy_tensors["gate_routed"][:active_rows],
        ),
        (
            "up_routed",
            union_context.up_routed[:active_rows],
            legacy_tensors["up_routed"][:active_rows],
        ),
        (
            "hidden_routed",
            union_context.hidden_routed[:active_rows],
            legacy_tensors["hidden_routed"][:active_rows],
        ),
        (
            "union_x",
            union_context.union_x[route_to_union[valid_rows].long()],
            legacy_tensors["x_routed"][:active_rows][valid_rows],
        ),
    )
    for name, actual, expected in exact_pairs:
        assert torch.equal(actual, expected), f"Union-X {name} differs from legacy"
