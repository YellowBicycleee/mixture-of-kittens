"""Pure-Python contract for the CuTe DSL Stage-0 risk spike.

The contract is deliberately independent of torch and CUTLASS so it can be
checked on a CPU-only development machine.  The GPU implementation consumes
the existing ``MoKWorkspace.x_buffer_ptrs`` and schedule tensor ABI.
"""

from __future__ import annotations

from collections.abc import Sequence


EP_SIZE = 8
QWEN_HIDDEN_SIZE = 4096
ROW_TILE_ELEMENTS = 256


def validate_stage0_contract(
    peer_ptrs: Sequence[int],
    schedule_peer_rank: Sequence[int],
    schedule_peer_token_idx: Sequence[int],
    *,
    num_peer_tokens: int,
    hidden_size: int = QWEN_HIDDEN_SIZE,
    row_tile_elements: int = ROW_TILE_ELEMENTS,
) -> None:
    """Validate the small, fixed Stage-0 dispatch contract.

    Stage 0 assigns one routed row to each CTA.  Two CTAs form a Blackwell
    cluster, so the schedule length must be even.  The implementation copies
    the complete BF16 row in ``row_tile_elements``-wide TMA transactions.
    """

    if len(peer_ptrs) != EP_SIZE:
        raise ValueError(f"Stage 0 requires exactly {EP_SIZE} peer pointers")
    if any(type(pointer) is not int or pointer <= 0 for pointer in peer_ptrs):
        raise ValueError("peer pointers must be positive integers")
    if len(schedule_peer_rank) != len(schedule_peer_token_idx):
        raise ValueError("peer-rank and peer-token schedules must have equal length")
    if not schedule_peer_rank or len(schedule_peer_rank) % 2:
        raise ValueError("the Stage-0 schedule must contain a positive even row count")
    if num_peer_tokens <= 0:
        raise ValueError("num_peer_tokens must be positive")
    if hidden_size != QWEN_HIDDEN_SIZE:
        raise ValueError(
            f"Stage 0 is fixed to Qwen hidden_size={QWEN_HIDDEN_SIZE}; got {hidden_size}"
        )
    if row_tile_elements <= 0 or hidden_size % row_tile_elements:
        raise ValueError("row_tile_elements must divide hidden_size")

    for index, (peer_rank, peer_token_idx) in enumerate(
        zip(schedule_peer_rank, schedule_peer_token_idx)
    ):
        if not 0 <= int(peer_rank) < EP_SIZE:
            raise ValueError(f"schedule_peer_rank[{index}] is outside [0, {EP_SIZE})")
        if not 0 <= int(peer_token_idx) < num_peer_tokens:
            raise ValueError(
                f"schedule_peer_token_idx[{index}] is outside [0, {num_peer_tokens})"
            )


def reference_dispatch_rows(
    peer_rows: Sequence[Sequence[Sequence[float]]],
    schedule_peer_rank: Sequence[int],
    schedule_peer_token_idx: Sequence[int],
) -> list[list[float]]:
    """Return the obvious routed-row reference used by the CPU smoke."""

    return [
        list(peer_rows[int(peer_rank)][int(peer_token_idx)])
        for peer_rank, peer_token_idx in zip(
            schedule_peer_rank, schedule_peer_token_idx
        )
    ]


def expected_role_log(num_comm_clusters: int, num_compute_clusters: int) -> list[int]:
    """Return the expected two-CTA role stamp for every logical CLC task."""

    if num_comm_clusters <= 0:
        raise ValueError("num_comm_clusters must be positive")
    if num_compute_clusters <= 0:
        raise ValueError("num_compute_clusters must be positive")
    return [100, 101] * num_comm_clusters + [200, 201] * num_compute_clusters
