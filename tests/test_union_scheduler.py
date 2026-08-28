import math

import torch

from mok.ops import schedule as legacy_schedule
from mok.union_ops import union_schedule
from tests.union_schedule_reference import build_union_schedule_reference


def test_union_schedule_matches_cpu_reference(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    num_local_tokens = 512
    num_local_experts = 4
    num_experts = world_size * num_local_experts
    topk = min(4, num_experts)
    schedule_capacity = (
        num_local_tokens
        * topk
        * max(2, math.ceil(world_size * 1.5))
    )

    peer = torch.arange(world_size, device=device).view(-1, 1, 1)
    token = torch.arange(num_local_tokens, device=device).view(1, -1, 1)
    slot = torch.arange(topk, device=device).view(1, 1, -1)
    topk_all = (
        token * 5 + slot * 3 + peer * 7
    ).remainder(num_experts).to(torch.int32).contiguous()

    actual = union_schedule(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    legacy = legacy_schedule(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    reference = build_union_schedule_reference(
        topk_all.cpu().tolist(),
        num_local_experts,
        schedule_capacity,
        rank,
    )

    expected = (
        torch.tensor(reference.peer_rank, dtype=torch.int32, device=device),
        torch.tensor(reference.peer_token_idx, dtype=torch.int32, device=device),
        torch.tensor([reference.num_tokens], dtype=torch.int32, device=device),
        torch.tensor(reference.tokens_per_expert, dtype=torch.int32, device=device),
        torch.tensor(reference.route_to_union, dtype=torch.int32, device=device),
        torch.tensor([reference.num_union], dtype=torch.int32, device=device),
    )
    valid = expected[0] >= 0
    legacy_valid = legacy[0] >= 0
    assert torch.equal(actual[0], legacy[0])
    assert torch.equal(valid, legacy_valid)
    assert torch.equal(actual[1][valid], legacy[1][legacy_valid])
    assert torch.equal(actual[2], legacy[2])
    assert torch.equal(actual[3], legacy[3])
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1][valid], expected[1][valid])
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
    assert torch.equal(actual[4], expected[4])
    assert torch.equal(actual[5], expected[5])

    valid_union_ids = actual[4][valid]
    assert bool(torch.all(valid_union_ids >= 0))
    assert bool(torch.all(valid_union_ids < actual[5][0]))
    assert torch.equal(
        torch.unique(valid_union_ids),
        torch.arange(reference.num_union, dtype=torch.int32, device=device),
    )


def test_union_schedule_fake_shapes(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    rank, world_size, device = context
    topk_all = torch.empty(
        world_size, 512, 2, dtype=torch.int32, device=device
    )
    with FakeTensorMode() as mode:
        fake_topk = mode.from_tensor(topk_all)
        outputs = union_schedule(fake_topk, 2, 4096, rank)
    assert [tuple(output.shape) for output in outputs] == [
        (4096,),
        (4096,),
        (1,),
        (2,),
        (4096,),
        (1,),
    ]
    assert all(output.dtype == torch.int32 for output in outputs)


def test_union_schedule_respects_nondefault_stream(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    num_local_tokens = 512
    num_local_experts = 2
    num_experts = world_size * num_local_experts
    topk = min(2, num_experts)
    schedule_capacity = num_local_tokens * topk * max(2, world_size)

    peer = torch.arange(world_size, device=device).view(-1, 1, 1)
    token = torch.arange(num_local_tokens, device=device).view(1, -1, 1)
    slot = torch.arange(topk, device=device).view(1, 1, -1)
    source = (token + peer * 3 + slot * 5).remainder(num_experts).to(torch.int32)
    topk_all = torch.empty_like(source)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        topk_all.copy_(source)
        actual = union_schedule(
            topk_all, num_local_experts, schedule_capacity, rank
        )
        legacy = legacy_schedule(
            topk_all, num_local_experts, schedule_capacity, rank
        )
    stream.synchronize()

    valid = actual[0] >= 0
    assert torch.equal(actual[0], legacy[0])
    assert torch.equal(actual[1][valid], legacy[1][valid])
    assert torch.equal(actual[2], legacy[2])
    assert torch.equal(actual[3], legacy[3])


def test_union_schedule_empty_destination_and_repeated_call_reset(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    if world_size == 1:
        return
    num_local_experts = 1
    other_rank = (rank + 1) % world_size
    topk_all = torch.full(
        (world_size, 512, 1),
        other_rank,
        dtype=torch.int32,
        device=device,
    )
    schedule_capacity = world_size * 512
    empty = union_schedule(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    assert int(empty[2][0]) == 0
    assert int(empty[5][0]) == 0
    assert bool(torch.all(empty[0] == -1))
    assert bool(torch.all(empty[4] == -1))

    topk_all.fill_(rank)
    populated = union_schedule(
        topk_all, num_local_experts, schedule_capacity, rank
    )
    assert int(populated[2][0]) > 0
    assert int(populated[5][0]) == world_size * 512
    assert bool(torch.all(populated[4][populated[0] >= 0] >= 0))


def test_union_schedule_has_no_mutated_arguments() -> None:
    operation = torch.ops.mok.union_schedule.default
    mutated = {
        argument.name
        for argument in operation._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert mutated == set()
