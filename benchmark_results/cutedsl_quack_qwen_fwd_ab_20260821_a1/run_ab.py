#!/usr/bin/env python3
"""One-allocation EP8 FWD A/B for MoK CUDA and CuTe DSL + QuACK.

Run on one eight-B300 node with::

    torchrun --standalone --nproc-per-node=8 run_ab.py

T=2048 checks both backends against the PyTorch/NCCL forward reference.
T=20480 checks finite output and CUDA-vs-CuTe agreement, then runs 50 warmups
and 30 CUDA-Event samples.  Each case uses one input set, workspace, and
schedule for both backends.

Events bracket only ``functional.forward``.  This includes input D2D copies,
MoK barriers, backend kernels, epilogue, GPU-visible Python launch gaps, and
CuTe's synchronous ``schedule.num_tokens.item()`` D2H read.  Allocation,
schedule construction, first-call JIT, sample aggregation, and output writing
are outside the boundary.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist

# Load the wheel-installed package before exposing archive-only benchmark/test
# helpers.  The compiled mok._C extension lives in this venv package, not in
# the clean git-archive source tree used by sbatch.sh.
from mok import functional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import get_tflops, init_distributed
from tests.utils import (
    BF16_TOLERANCE,
    generate_inputs,
    get_error_stats,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
)


EP_SIZE = 8
HIDDEN = 4096
INTERMEDIATE = 1024
EXPERTS = 512
LOCAL_EXPERTS = EXPERTS // EP_SIZE
TOPK = 10
CORRECTNESS_TOKENS = 2048
TIMING_TOKENS = 20480
WARMUPS = 50
SAMPLES = 30
DESIGN_BASE_COMMIT = "6f89aab04b092a3b3f8695735b84a62b2f05e5d9"

# Prior Qwen MoK selections.  Backward SMs only satisfy the shared config
# contract; this runner never calls backward.
POINTS = {
    CORRECTNESS_TOKENS: dict(
        fwd_num_comm_sms=28,
        bwd_num_comm_sms=36,
        minibatch_size=4096,
        macrobatch_size=131072,
    ),
    TIMING_TOKENS: dict(
        fwd_num_comm_sms=36,
        bwd_num_comm_sms=44,
        minibatch_size=4096,
        macrobatch_size=262144,
    ),
}


def global_bool(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def metric(
    name: str, reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, object]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError(
            f"{name}: {reference.shape}/{reference.dtype} != "
            f"{actual.shape}/{actual.dtype}"
        )
    mean, maximum, relative = get_error_stats(reference, actual)
    atol, rtol = BF16_TOLERANCE
    passed = (
        all(math.isfinite(value) for value in (mean, maximum, relative))
        and maximum <= atol
        and relative <= rtol
    )
    return {
        "name": name,
        "abs_error_mean": mean,
        "abs_error_max": maximum,
        "relative_l1_error": relative,
        "atol": atol,
        "relative_l1_tolerance": rtol,
        "pass": passed,
    }


def configs(tokens: int) -> tuple[functional.MoKConfig, functional.MoKConfig]:
    point = POINTS[tokens]
    return (
        functional.MoKConfig(**point, fwd_backend="cuda"),
        functional.MoKConfig(**point, fwd_backend="cutedsl"),
    )


def forward(
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    schedule: functional.MoKSchedule,
    inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    output, _ = functional.forward(
        config,
        workspace,
        schedule,
        inputs[0],
        inputs[2],
        *inputs[3:9],
    )
    return output


def checked_forward(
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    schedule: functional.MoKSchedule,
    inputs: tuple[torch.Tensor, ...],
    local_rank: int,
) -> torch.Tensor:
    # Prevent the second backend from inheriting a missed routed write.
    workspace.combine_buffer.fill_(float("nan"))
    dist.barrier(device_ids=[local_rank])
    output = forward(config, workspace, schedule, inputs)
    torch.cuda.synchronize(workspace.device)
    return output


def time_forward(
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    schedule: functional.MoKSchedule,
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    for _ in range(WARMUPS):
        output = forward(config, workspace, schedule, inputs)
        output = None

    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(SAMPLES)
    ]
    for start, end in events:
        start.record()
        output = forward(config, workspace, schedule, inputs)
        end.record()
        output = None
    torch.cuda.synchronize(workspace.device)
    dist.barrier()

    local = torch.tensor(
        [start.elapsed_time(end) for start, end in events],
        dtype=torch.float64,
        device=workspace.device,
    )
    gathered = [torch.empty_like(local) for _ in range(EP_SIZE)]
    dist.all_gather(gathered, local)
    rank_max = torch.stack(gathered).amax(dim=0)
    return {
        "median_rank_max_ms": float(torch.quantile(rank_max, 0.5).item()),
        "min_rank_max_ms": float(rank_max.min().item()),
        "max_rank_max_ms": float(rank_max.max().item()),
        "rank_max_samples_ms": rank_max.cpu().tolist(),
    }


def make_case(
    tokens: int, rank: int, local_rank: int, device: torch.device
) -> dict[str, object]:
    inputs = generate_inputs(
        rank,
        device,
        EXPERTS,
        LOCAL_EXPERTS,
        TOPK,
        tokens,
        HIDDEN,
        INTERMEDIATE,
    )
    cuda_config, cutedsl_config = configs(tokens)
    workspace = functional.get_workspace(
        cuda_config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=tokens,
        hidden_size=HIDDEN,
        topk=TOPK,
    )
    schedule = functional.build_schedule(
        workspace,
        cuda_config,
        inputs[1],
        num_local_experts=LOCAL_EXPERTS,
    )

    # This stats read is outside timing.  The CuTe backend still performs its
    # own one-item D2H read inside every measured functional.forward call.
    local_rows = torch.tensor(
        [int(schedule.num_tokens.item())], dtype=torch.int64, device=device
    )
    gathered_rows = torch.empty(EP_SIZE, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_rows, local_rows)
    padded_rows = [int(value) for value in gathered_rows.cpu().tolist()]

    cuda_output = checked_forward(
        cuda_config, workspace, schedule, inputs, local_rank
    )
    cutedsl_output = checked_forward(
        cutedsl_config, workspace, schedule, inputs, local_rank
    )
    cuda_finite = global_bool(
        cuda_output.shape == (tokens, HIDDEN)
        and cuda_output.dtype == torch.bfloat16
        and bool(torch.isfinite(cuda_output).all().item()),
        device,
    )
    cutedsl_finite = global_bool(
        cutedsl_output.shape == (tokens, HIDDEN)
        and cutedsl_output.dtype == torch.bfloat16
        and bool(torch.isfinite(cutedsl_output).all().item()),
        device,
    )
    direct = metric("cutedsl_vs_cuda", cuda_output, cutedsl_output)
    correctness: dict[str, Any] = {
        "mode": "finite_plus_cross_backend",
        "cuda_finite": cuda_finite,
        "cutedsl_quack_finite": cutedsl_finite,
        "cutedsl_vs_cuda": direct,
    }

    checks = [cuda_finite, cutedsl_finite, bool(direct["pass"])]
    if tokens == CORRECTNESS_TOKENS:
        combine, gate, up, hidden, shared = run_forward_reference_bf16(
            inputs[0], inputs[1], *inputs[3:9]
        )
        reference = run_fwd_epilogue_reference(shared, combine, inputs[2])
        cuda_reference = metric("cuda_vs_reference", reference, cuda_output)
        cutedsl_reference = metric(
            "cutedsl_quack_vs_reference", reference, cutedsl_output
        )
        correctness.update(
            mode="full_forward_output_against_pytorch_nccl_reference",
            cuda_vs_reference=cuda_reference,
            cutedsl_quack_vs_reference=cutedsl_reference,
        )
        checks += [bool(cuda_reference["pass"]), bool(cutedsl_reference["pass"])]
        del combine, gate, up, hidden, shared, reference

    correctness["pass"] = global_bool(all(checks), device)
    if not correctness["pass"]:
        raise AssertionError(f"T={tokens} forward correctness failed")

    result: dict[str, object] = {
        "tokens_per_rank": tokens,
        "config": {
            "dtype": "BF16",
            "ep": EP_SIZE,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "topk": TOPK,
            "macrobatch": cuda_config.macrobatch_size,
            "backend_controls": {
                "mok_cuda": {
                    "fwd_comm_sms": cuda_config.fwd_num_comm_sms,
                    "minibatch": cuda_config.minibatch_size,
                },
                "mok_cutedsl_quack": {
                    "fwd_comm_sms": cutedsl_config.fwd_num_comm_sms,
                    "minibatch": "N/A (contract validation only)",
                },
            },
        },
        "same_inputs_workspace_schedule": True,
        "padded_routed_rows_by_rank": padded_rows,
        "macrobatches_by_rank": [
            (rows + cuda_config.macrobatch_size - 1)
            // cuda_config.macrobatch_size
            for rows in padded_rows
        ],
        "correctness": correctness,
    }

    del cuda_output, cutedsl_output
    if tokens == TIMING_TOKENS:
        cuda_timing = time_forward(cuda_config, workspace, schedule, inputs)
        cutedsl_timing = time_forward(cutedsl_config, workspace, schedule, inputs)
        cuda_ms = float(cuda_timing["median_rank_max_ms"])
        cutedsl_ms = float(cutedsl_timing["median_rank_max_ms"])
        cuda_timing["effective_tflops_per_gpu"] = get_tflops(
            cuda_ms, tokens, TOPK, HIDDEN, INTERMEDIATE
        )
        cutedsl_timing["effective_tflops_per_gpu"] = get_tflops(
            cutedsl_ms, tokens, TOPK, HIDDEN, INTERMEDIATE
        )
        result["timing"] = {
            "warmups": WARMUPS,
            "samples": SAMPLES,
            "aggregation": "per-sample EP8 rank maximum, then median",
            "backend_order": ["mok_cuda", "mok_cutedsl_quack"],
            "interpretation": (
                "initial fixed-order A/B; reverse-order confirmation is "
                "required before treating a small speedup as final"
            ),
            "mok_cuda": cuda_timing,
            "mok_cutedsl_quack": cutedsl_timing,
            "cutedsl_speedup_vs_cuda": cuda_ms / cutedsl_ms,
        }
    return result


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    cache_root = os.environ.get("MOK_CUTEDSL_CACHE_ROOT")
    if cache_root:
        rank_cache = Path(cache_root).resolve() / f"rank-{local_rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        os.environ["CUTE_DSL_CACHE_DIR"] = str(rank_cache)

    rank, world_size, device = init_distributed()
    if world_size != EP_SIZE:
        raise RuntimeError(f"runner requires EP{EP_SIZE}; got EP{world_size}")
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("runner requires B300/SM103")
    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    quack_version = importlib.metadata.version("quack-kernels")
    if (cutlass_version, quack_version) != ("4.6.2", "0.6.4"):
        raise RuntimeError(
            "requires nvidia-cutlass-dsl==4.6.2 and quack-kernels==0.6.4; "
            f"got {cutlass_version}/{quack_version}"
        )

    try:
        with torch.no_grad():
            cases = [
                make_case(tokens, rank, local_rank, device)
                for tokens in (CORRECTNESS_TOKENS, TIMING_TOKENS)
            ]
        payload = {
            "status": "PASS",
            "comparison": "MoK CUDA FWD vs MoK CuTe DSL + QuACK FWD",
            "cutedsl_variant": "tiled peer TMA + QuACK tcgen05 GEMMs",
            "design_base_commit": DESIGN_BASE_COMMIT,
            "runtime_source_commit": os.environ.get(
                "MOK_SOURCE_COMMIT", "not provided"
            ),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "software": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "nvidia_cutlass_dsl": cutlass_version,
                "quack_kernels": quack_version,
            },
            "device": torch.cuda.get_device_name(device),
            "input_seed": "1234 + EP rank (tests.utils.generate_inputs)",
            "timing_boundary": {
                "clock": "CUDA Events around functional.forward",
                "aggregation": "per-sample EP8 rank maximum, then median",
                "cutedsl_num_tokens_item_inside_boundary": True,
                "gpu_visible_python_launch_gaps_inside_boundary": True,
                "pure_cpu_work_before_start_or_after_end_inside_boundary": False,
                "schedule_build_and_first_call_jit_inside_boundary": False,
            },
            "cases": cases,
        }
        if rank == 0:
            result_path = os.environ.get("MOK_AB_RESULT_JSON")
            if result_path:
                path = Path(result_path).expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(
                "RESULT_JSON="
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
        dist.barrier()
        functional.clear_workspace_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
