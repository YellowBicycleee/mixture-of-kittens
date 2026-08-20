#!/usr/bin/env python3
"""CUDA-Event stage profile of one formal T=20K MoK CuTe forward.

The CUDA/CuTe correctness gate also warms every kernel.  Five later calls to
``functional.forward`` are profiled by temporarily wrapping the existing CuTe
launch seams; no stage is replayed in isolation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.distributed as dist

# The compiled mok._C extension comes from the wheel installed by sbatch.sh.
from mok import functional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import init_distributed
from tests.utils import BF16_TOLERANCE, generate_inputs, get_error_stats


EP_SIZE = 8
TOKENS = 20480
HIDDEN = 4096
INTERMEDIATE = 1024
EXPERTS = 512
LOCAL_EXPERTS = 64
TOPK = 10
MACROBATCH = 262144
MINIBATCH = 4096
SAMPLES = 5

STAGES = (
    "shared_gate_gemm",
    "shared_up_gemm",
    "shared_swiglu",
    "shared_down_gemm",
    "dispatch",
    "routed_gate_gemm",
    "routed_up_gemm",
    "routed_swiglu",
    "routed_down_gemm",
    "combine",
)


def config(backend: str) -> functional.MoKConfig:
    return functional.MoKConfig(
        fwd_num_comm_sms=36,
        bwd_num_comm_sms=44,  # Validation only: this runner is FWD-only.
        minibatch_size=MINIBATCH,
        macrobatch_size=MACROBATCH,
        fwd_backend=backend,
    )


def forward(config_, workspace, schedule, inputs) -> torch.Tensor:
    output, _ = functional.forward(
        config_, workspace, schedule, inputs[0], inputs[2], *inputs[3:9]
    )
    return output


def checked_forward(config_, workspace, schedule, inputs, local_rank):
    workspace.combine_buffer.fill_(float("nan"))
    torch.cuda.synchronize(workspace.device)
    dist.barrier(device_ids=[local_rank])
    output = forward(config_, workspace, schedule, inputs)
    torch.cuda.synchronize(workspace.device)
    return output


def global_bool(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def cross_backend_metric(reference, actual, device) -> dict[str, object]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError("CUDA and CuTe outputs have different shape or dtype")
    values = torch.tensor(
        get_error_stats(reference, actual), dtype=torch.float64, device=device
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    mean, maximum, relative = (float(value) for value in values.cpu().tolist())
    atol, rtol = BF16_TOLERANCE
    passed = (
        all(math.isfinite(value) for value in (mean, maximum, relative))
        and maximum <= atol
        and relative <= rtol
    )
    return {
        "aggregation": "maximum metric across EP8 ranks",
        "abs_error_mean": mean,
        "abs_error_max": maximum,
        "relative_l1_error": relative,
        "atol": atol,
        "relative_l1_tolerance": rtol,
        "pass": global_bool(passed, device),
    }


def profile_one_forward(module, config_, workspace, schedule, inputs):
    """Patch five launch symbols for exactly one single-macro forward."""

    stage_events = {
        stage: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for stage in STAGES
    }
    total_events = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    calls = {stage: 0 for stage in STAGES}
    sequence = {"shared": 0, "routed": 0}
    originals = {
        name: getattr(module, name)
        for name in ("_DISPATCH", "shared_gemm", "routed_gemm", "_SWIGLU", "_COMBINE")
    }

    def timed(stage: str, original: Callable[..., Any], *args, **kwargs):
        if calls[stage] != 0:
            raise RuntimeError(f"single-macro profile saw an extra {stage} call")
        calls[stage] += 1
        start, end = stage_events[stage]
        start.record()
        result = original(*args, **kwargs)
        end.record()
        return result

    def fixed(stage: str, original: Callable[..., Any]):
        return lambda *args, **kwargs: timed(stage, original, *args, **kwargs)

    shared_names = ("shared_gate_gemm", "shared_up_gemm", "shared_down_gemm")
    routed_names = ("routed_gate_gemm", "routed_up_gemm", "routed_down_gemm")

    def shared(*args, **kwargs):
        index = sequence["shared"]
        if index >= len(shared_names):
            raise RuntimeError("unexpected extra shared_gemm call")
        sequence["shared"] += 1
        return timed(shared_names[index], originals["shared_gemm"], *args, **kwargs)

    def routed(*args, **kwargs):
        index = sequence["routed"]
        if index >= len(routed_names):
            raise RuntimeError("unexpected extra routed_gemm call")
        sequence["routed"] += 1
        return timed(routed_names[index], originals["routed_gemm"], *args, **kwargs)

    def swiglu(*args, **kwargs):
        num_tokens = args[3] if len(args) > 3 else kwargs.get("num_tokens")
        stage = "shared_swiglu" if num_tokens is None else "routed_swiglu"
        return timed(stage, originals["_SWIGLU"], *args, **kwargs)

    replacements = {
        "_DISPATCH": fixed("dispatch", originals["_DISPATCH"]),
        "shared_gemm": shared,
        "routed_gemm": routed,
        "_SWIGLU": swiglu,
        "_COMBINE": fixed("combine", originals["_COMBINE"]),
    }
    try:
        for name, replacement in replacements.items():
            setattr(module, name, replacement)
        total_events[0].record()
        output = forward(config_, workspace, schedule, inputs)
        total_events[1].record()
        output = None
    finally:
        for name, original in originals.items():
            setattr(module, name, original)

    torch.cuda.synchronize(workspace.device)
    if any(count != 1 for count in calls.values()):
        raise RuntimeError(f"stage call counts are not all one: {calls}")
    values = {
        stage: start.elapsed_time(end)
        for stage, (start, end) in stage_events.items()
    }
    values["attributed_sum"] = sum(values.values())
    values["total"] = total_events[0].elapsed_time(total_events[1])
    values["unattributed_gap"] = values["total"] - values["attributed_sum"]
    return values


def aggregate_rank_max(local_samples, device):
    metrics = (*STAGES, "attributed_sum", "unattributed_gap", "total")
    local = torch.tensor(
        [[sample[metric] for metric in metrics] for sample in local_samples],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(EP_SIZE)]
    dist.all_gather(gathered, local)
    rank_max = torch.stack(gathered).amax(dim=0)
    return {
        metric: {
            "median_rank_max_ms": float(torch.quantile(rank_max[:, column], 0.5)),
            "rank_max_samples_ms": rank_max[:, column].cpu().tolist(),
        }
        for column, metric in enumerate(metrics)
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    cache_root = os.environ.get("MOK_CUTEDSL_CACHE_ROOT")
    if cache_root:
        rank_cache = Path(cache_root).resolve() / f"rank-{local_rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        os.environ["CUTE_DSL_CACHE_DIR"] = str(rank_cache)

    rank, world_size, device = init_distributed()
    if world_size != EP_SIZE or torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("runner requires one EP8 B300/SM103 node")
    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    quack_version = importlib.metadata.version("quack-kernels")
    if (cutlass_version, quack_version) != ("4.6.2", "0.6.4"):
        raise RuntimeError(
            "requires nvidia-cutlass-dsl==4.6.2 and quack-kernels==0.6.4; "
            f"got {cutlass_version}/{quack_version}"
        )

    cuda_config, cutedsl_config = config("cuda"), config("cutedsl")
    try:
        with torch.no_grad():
            inputs = generate_inputs(
                rank, device, EXPERTS, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, INTERMEDIATE
            )
            workspace = functional.get_workspace(
                cuda_config,
                dist.group.WORLD,
                device=device,
                num_local_tokens=TOKENS,
                hidden_size=HIDDEN,
                topk=TOPK,
            )
            schedule = functional.build_schedule(
                workspace, cuda_config, inputs[1], num_local_experts=LOCAL_EXPERTS
            )
            routed_rows = torch.tensor(
                [int(schedule.num_tokens.item())], dtype=torch.int64, device=device
            )
            routed_rows_by_rank = torch.empty(EP_SIZE, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(routed_rows_by_rank, routed_rows)
            if bool(
                ((routed_rows_by_rank <= 0) | (routed_rows_by_rank > MACROBATCH))
                .any()
                .item()
            ):
                raise RuntimeError("T=20480 profile requires one routed macro on every rank")

            cuda_output = checked_forward(
                cuda_config, workspace, schedule, inputs, local_rank
            )
            cutedsl_output = checked_forward(
                cutedsl_config, workspace, schedule, inputs, local_rank
            )
            cuda_finite = global_bool(bool(torch.isfinite(cuda_output).all()), device)
            cutedsl_finite = global_bool(bool(torch.isfinite(cutedsl_output).all()), device)
            metric = cross_backend_metric(cuda_output, cutedsl_output, device)
            correctness_pass = cuda_finite and cutedsl_finite and bool(metric["pass"])
            if not correctness_pass:
                raise AssertionError("T=20480 CUDA/CuTe correctness gate failed")
            del cuda_output, cutedsl_output

            module = importlib.import_module("mok.cutedsl.forward")
            local_samples = []
            for _ in range(SAMPLES):
                workspace.combine_buffer.fill_(float("nan"))
                torch.cuda.synchronize(device)
                dist.barrier(device_ids=[local_rank])
                local_samples.append(
                    profile_one_forward(
                        module, cutedsl_config, workspace, schedule, inputs
                    )
                )
            profile = aggregate_rank_max(local_samples, device)
            payload = {
                "status": "PASS",
                "subject": "MoK CuTe DSL + QuACK T=20K forward stage profile",
                "runtime_source_commit": os.environ.get(
                    "MOK_SOURCE_COMMIT", "not provided"
                ),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "device": torch.cuda.get_device_name(device),
                "software": {
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "nvidia_cutlass_dsl": cutlass_version,
                    "quack_kernels": quack_version,
                },
                "config": {
                    "tokens_per_rank": TOKENS,
                    "dtype": "BF16",
                    "ep": EP_SIZE,
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "experts": EXPERTS,
                    "local_experts": LOCAL_EXPERTS,
                    "topk": TOPK,
                    "macrobatch": MACROBATCH,
                    "minibatch": MINIBATCH,
                    "fwd_comm_sms": "N/A (not used by CuTe backend)",
                },
                "input_seed": "1234 + EP rank (tests.utils.generate_inputs)",
                "padded_routed_rows_by_rank": routed_rows_by_rank.cpu().tolist(),
                "macrobatches_by_rank": [1] * EP_SIZE,
                "correctness_gate": {
                    "same_inputs_workspace_schedule": True,
                    "cuda_finite": cuda_finite,
                    "cutedsl_finite": cutedsl_finite,
                    "cutedsl_vs_cuda": metric,
                    "pass": correctness_pass,
                },
                "profile": {
                    "samples": SAMPLES,
                    "aggregation": "per-sample EP8 rank maximum, then median",
                    "calls_per_forward": {stage: 1 for stage in STAGES},
                    "stage_order": list(STAGES),
                    "timing_boundary": (
                        "CUDA Events around existing launch seams inside one "
                        "functional.forward; no isolated replay"
                    ),
                    "attribution_identity": (
                        "sum and gap are calculated per rank/sample before rank max"
                    ),
                    "unattributed_includes": [
                        "input copies and MoK barriers",
                        "schedule.num_tokens.item() and other host-visible gaps",
                        "buffer allocation and DLPack setup",
                        "forward epilogue and Event insertion overhead",
                    ],
                    "metrics": profile,
                },
                "prior_uninstrumented_reference": {
                    "job_id": 3809313,
                    "runtime_source_commit": (
                        "4c1f24127aee4ce1e604f7ff41b3b97e92078cb0"
                    ),
                    "samples": 30,
                    "median_rank_max_ms": 43.625152587890625,
                    "artifact": (
                        "benchmark_results/cutedsl_quack_qwen_fwd_ab_20260821_a1/"
                        "runs/job-3809313/result.json"
                    ),
                },
            }

        if rank == 0:
            path = Path(os.environ["MOK_STAGE_PROFILE_RESULT_JSON"]).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print("RESULT_JSON=" + json.dumps(payload, separators=(",", ":")), flush=True)
        dist.barrier()
        functional.clear_workspace_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
