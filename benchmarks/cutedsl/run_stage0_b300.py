"""Eight-rank B300 correctness runner for the CuTe DSL Stage-0 risk spike."""

from __future__ import annotations

import importlib.metadata
import json
import os

import torch
import torch.distributed as dist

from mok.functional import MoKConfig, create_workspace
from mok.cutedsl.experimental.contract import expected_role_log
from mok.cutedsl.experimental.stage0 import run_stage0


EP_SIZE = 8
NUM_LOCAL_TOKENS = 512
HIDDEN_SIZE = 4096
TOPK = 10
NUM_SCHEDULE_ROWS = 16
NUM_COMPUTE_CLUSTERS = 2
SEED = 20260820


def _make_rows(rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED + rank)
    return torch.randn(
        NUM_LOCAL_TOKENS,
        HIDDEN_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )


def _make_schedule(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(NUM_SCHEDULE_ROWS, device=device, dtype=torch.int32)
    peer_rank = (rank + index * 3) % EP_SIZE
    peer_token_idx = (rank * 11 + index * 29) % NUM_LOCAL_TOKENS
    return peer_rank.contiguous(), peer_token_idx.contiguous()


def _reference(
    all_peer_rows: torch.Tensor,
    peer_rank: torch.Tensor,
    peer_token_idx: torch.Tensor,
) -> torch.Tensor:
    return all_peer_rows[peer_rank.long(), peer_token_idx.long()].contiguous()


def main() -> None:
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.6.2":
        raise RuntimeError("Stage 0 is pinned to nvidia-cutlass-dsl==4.6.2")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != EP_SIZE:
        raise RuntimeError(f"Stage 0 requires exactly {EP_SIZE} ranks")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the Stage-0 runner is intentionally restricted to B300/SM103")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        torch.manual_seed(SEED + rank)
        workspace = create_workspace(
            MoKConfig(),
            dist.group.WORLD,
            device=device,
            num_local_tokens=NUM_LOCAL_TOKENS,
            hidden_size=HIDDEN_SIZE,
            topk=TOPK,
        )
        workspace.x_buffer.copy_(_make_rows(rank, device))
        peer_rank, peer_token_idx = _make_schedule(rank, device)
        all_peer_rows_flat = torch.empty(
            EP_SIZE * NUM_LOCAL_TOKENS,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
        )
        dist.all_gather_into_tensor(all_peer_rows_flat, workspace.x_buffer)
        reference = _reference(
            all_peer_rows_flat.view(EP_SIZE, NUM_LOCAL_TOKENS, HIDDEN_SIZE),
            peer_rank,
            peer_token_idx,
        )

        num_comm_clusters = NUM_SCHEDULE_ROWS // 2
        num_logical_clusters = num_comm_clusters + NUM_COMPUTE_CLUSTERS
        output = torch.empty(
            NUM_SCHEDULE_ROWS,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
        )
        system_counters = torch.zeros(
            num_logical_clusters, dtype=torch.int32, device=device
        )
        gpu_counters = torch.zeros_like(system_counters)
        role_log = torch.full(
            (num_logical_clusters * 2,), -1, dtype=torch.int32, device=device
        )
        completion_log = torch.full(
            (num_logical_clusters,), -1, dtype=torch.int32, device=device
        )

        # The symmetric producer rows must be visible before any peer TMA load.
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[local_rank])
        run_stage0(
            workspace,
            peer_rank,
            peer_token_idx,
            output,
            system_counters,
            gpu_counters,
            role_log,
            completion_log,
            num_compute_clusters=NUM_COMPUTE_CLUSTERS,
        )
        torch.cuda.synchronize(device)

        expected_roles = torch.tensor(
            expected_role_log(num_comm_clusters, NUM_COMPUTE_CLUSTERS),
            dtype=torch.int32,
            device=device,
        )
        expected_system = torch.tensor(
            [2] * num_comm_clusters + [0] * NUM_COMPUTE_CLUSTERS,
            dtype=torch.int32,
            device=device,
        )
        expected_gpu = torch.tensor(
            [0] * num_comm_clusters + [2] * NUM_COMPUTE_CLUSTERS,
            dtype=torch.int32,
            device=device,
        )
        expected_completion = torch.tensor(
            [1] * num_comm_clusters + [2] * NUM_COMPUTE_CLUSTERS,
            dtype=torch.int32,
            device=device,
        )

        checks = {
            "dynamic_peer_tma": bool(torch.equal(output, reference)),
            "clc_role_log": bool(torch.equal(role_log, expected_roles)),
            "system_counter": bool(torch.equal(system_counters, expected_system)),
            "gpu_counter": bool(torch.equal(gpu_counters, expected_gpu)),
            "completion_log": bool(torch.equal(completion_log, expected_completion)),
        }
        local_ok = torch.tensor(
            [int(all(checks.values()))], dtype=torch.int32, device=device
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": "mok-cutedsl-stage0",
                        "cutedsl": "4.6.2",
                        "device": torch.cuda.get_device_name(device),
                        "config": {
                            "ep": EP_SIZE,
                            "dtype": "bf16",
                            "hidden_size": HIDDEN_SIZE,
                            "topk": TOPK,
                            "num_local_tokens": NUM_LOCAL_TOKENS,
                            "schedule_rows": NUM_SCHEDULE_ROWS,
                            "compute_clusters": NUM_COMPUTE_CLUSTERS,
                            "cluster_shape": [2, 1, 1],
                        },
                        "checks_rank0": checks,
                        "all_ranks_pass": bool(local_ok.item()),
                    },
                    sort_keys=True,
                )
            )
        if not bool(local_ok.item()):
            raise AssertionError(f"Stage-0 correctness failed on rank {rank}: {checks}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
