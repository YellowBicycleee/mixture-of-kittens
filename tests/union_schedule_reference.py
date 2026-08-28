from dataclasses import dataclass


@dataclass(frozen=True)
class UnionScheduleReference:
    peer_rank: list[int]
    peer_token_idx: list[int]
    num_tokens: int
    tokens_per_expert: list[int]
    route_to_union: list[int]
    num_union: int


def build_union_schedule_reference(
    topk_all: list[list[list[int]]],
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> UnionScheduleReference:
    """Pure-CPU reference for the CUDA union scheduler.

    The route order intentionally matches ``csrc/scheduler.cuh``: routes owned
    by CUDA thread 0 precede routes owned by thread 1, and peer routes are then
    interleaved by their within-peer ordinal for each local expert.
    """
    world_size = len(topk_all)
    if world_size == 0:
        raise ValueError("topk_all must contain at least one peer")
    num_local_tokens = len(topk_all[0])
    if num_local_tokens == 0:
        raise ValueError("topk_all must contain at least one token")
    topk = len(topk_all[0][0])
    if topk == 0:
        raise ValueError("topk_all must contain at least one route per token")
    if any(
        len(peer) != num_local_tokens
        or any(len(token) != topk for token in peer)
        for peer in topk_all
    ):
        raise ValueError("topk_all must be rectangular")

    first_expert = rank * num_local_experts
    num_routes_per_peer = num_local_tokens * topk
    route_order = [
        route
        for thread in range(1024)
        for route in range(thread, num_routes_per_peer, 1024)
    ]

    peer_rank = [-1] * schedule_capacity
    peer_token_idx = [-1] * schedule_capacity
    route_to_union = [-1] * schedule_capacity
    tokens_per_expert: list[int] = []
    offset = 0

    scheduled: list[tuple[int, int, int]] = []
    for local_expert in range(num_local_experts):
        expert = first_expert + local_expert
        routes_by_peer: list[list[int]] = []
        for peer in range(world_size):
            routes_by_peer.append([
                route
                for route in route_order
                if topk_all[peer][route // topk][route % topk] == expert
            ])

        num_expert_routes = sum(len(routes) for routes in routes_by_peer)
        padded = (num_expert_routes + 255) // 256 * 256
        if offset + padded > schedule_capacity:
            raise ValueError("schedule_capacity is too small")
        tokens_per_expert.append(padded)

        max_peer_routes = max((len(routes) for routes in routes_by_peer), default=0)
        for ordinal in range(max_peer_routes):
            for peer, routes in enumerate(routes_by_peer):
                if ordinal < len(routes):
                    route = routes[ordinal]
                    scheduled.append((offset, peer, route))
                    offset += 1
        offset += padded - num_expert_routes

    keys = sorted({
        peer * num_local_tokens + route // topk
        for _, peer, route in scheduled
    })
    key_to_union = {key: union for union, key in enumerate(keys)}

    for row, peer, route in scheduled:
        peer_rank[row] = peer
        peer_token_idx[row] = route
        key = peer * num_local_tokens + route // topk
        route_to_union[row] = key_to_union[key]

    return UnionScheduleReference(
        peer_rank=peer_rank,
        peer_token_idx=peer_token_idx,
        num_tokens=sum(tokens_per_expert),
        tokens_per_expert=tokens_per_expert,
        route_to_union=route_to_union,
        num_union=len(keys),
    )
