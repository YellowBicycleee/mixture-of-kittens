"""One-shot B300 EP8 correctness gate for the fixed Qwen CuTe DSL forward."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from mok import functional


EP_SIZE = 8
NUM_LOCAL_TOKENS = 512
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 512
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
TOPK = 10
MINIBATCH_SIZE = 4096
MACROBATCH_SIZE = 4096
SEED = 20260821
BF16_ATOL = 0.5
BF16_RELATIVE_TOL = 0.01

LOW_LEVEL_NAMES = (
    "x_routed",
    "gate_shared",
    "gate_routed",
    "up_shared",
    "up_routed",
    "hidden_shared",
    "hidden_routed",
    "y_shared",
    "y_routed",
)
CONTEXT_NAMES = LOW_LEVEL_NAMES[:7]
ROUTED_LOW_LEVEL_INDICES = (0, 2, 4, 6, 8)


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = torch.randn(
        shape, generator=generator, dtype=torch.bfloat16, device=device
    )
    if scale != 1.0:
        result.mul_(scale)
    return result


def _make_inputs(rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    # Every rank contributes exactly ten routes to every global expert.  Thus
    # every local expert receives 80 real rows and is padded to 256 rows by the
    # production scheduler: 64 * 256 = 16384 rows, or four 4096-row macros.
    route = torch.arange(
        NUM_LOCAL_TOKENS * TOPK, dtype=torch.int64, device=device
    )
    top_experts = (
        rank * NUM_LOCAL_TOKENS * TOPK + route
    ).remainder(NUM_EXPERTS).view(NUM_LOCAL_TOKENS, TOPK).contiguous()

    router_generator = torch.Generator(device=device).manual_seed(SEED + 10 + rank)
    router_weights = torch.softmax(
        torch.randn(
            NUM_LOCAL_TOKENS,
            TOPK,
            generator=router_generator,
            dtype=torch.float32,
            device=device,
        ),
        dim=-1,
    ).contiguous()

    return {
        "x": _randn(
            (NUM_LOCAL_TOKENS, HIDDEN_SIZE),
            seed=SEED + 100 + rank,
            device=device,
        ),
        "top_experts": top_experts,
        "router_weights": router_weights,
        # Shared weights are identical on all EP ranks; routed weights use a
        # rank-specific seed because each rank owns a different expert shard.
        "shared_gate": _randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            seed=SEED + 200,
            device=device,
            scale=HIDDEN_SIZE**-0.5,
        ),
        "shared_up": _randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            seed=SEED + 201,
            device=device,
            scale=HIDDEN_SIZE**-0.5,
        ),
        "shared_down": _randn(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            seed=SEED + 202,
            device=device,
            scale=INTERMEDIATE_SIZE**-0.5,
        ),
        "routed_gate": _randn(
            (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
            seed=SEED + 300 + rank,
            device=device,
            scale=HIDDEN_SIZE**-0.5,
        ),
        "routed_up": _randn(
            (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
            seed=SEED + 400 + rank,
            device=device,
            scale=HIDDEN_SIZE**-0.5,
        ),
        "routed_down": _randn(
            (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            seed=SEED + 500 + rank,
            device=device,
            scale=INTERMEDIATE_SIZE**-0.5,
        ),
    }


def _context_tensors(context: functional.MoKForwardContext) -> tuple[torch.Tensor, ...]:
    values = (
        context.x_routed,
        context.gate_shared,
        context.gate_routed,
        context.up_shared,
        context.up_routed,
        context.hidden_shared,
        context.hidden_routed,
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("BF16 forward_context must contain seven tensors")
    return values  # type: ignore[return-value]


def _run_backend(
    backend: str,
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    schedule: functional.MoKSchedule,
    tensors: dict[str, torch.Tensor],
    local_rank: int,
) -> tuple[
    torch.Tensor,
    functional.MoKForwardContext,
    tuple[torch.Tensor, ...],
    torch.Tensor,
]:
    captured: dict[str, tuple[torch.Tensor, ...]] = {}
    if backend == "cuda":
        owner: Any = functional
        attribute = "dispatch_mlp_swiglu_combine_fwd_bf16"
    elif backend == "cutedsl":
        owner = importlib.import_module("mok.cutedsl.forward")
        attribute = "forward_bf16"
    else:
        raise ValueError(f"unknown backend {backend!r}")

    original = getattr(owner, attribute)

    def capture(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != len(LOW_LEVEL_NAMES):
            raise RuntimeError(
                f"{backend} low-level forward returned {type(result)!r} with "
                f"length {len(result) if isinstance(result, tuple) else 'N/A'}"
            )
        if not all(isinstance(value, torch.Tensor) for value in result):
            raise TypeError(f"{backend} low-level forward returned a non-tensor")
        captured["values"] = result
        return result

    setattr(owner, attribute, capture)
    # Poison every local return slot.  A missed remote combine store therefore
    # cannot accidentally reuse the previous backend's valid result.
    workspace.combine_buffer.fill_(float("nan"))
    dist.barrier(device_ids=[local_rank])
    try:
        output, context = functional.forward(
            config,
            workspace,
            schedule,
            tensors["x"],
            tensors["router_weights"],
            tensors["shared_gate"],
            tensors["shared_up"],
            tensors["shared_down"],
            tensors["routed_gate"],
            tensors["routed_up"],
            tensors["routed_down"],
        )
        torch.cuda.synchronize(workspace.device)
    finally:
        setattr(owner, attribute, original)

    low_level = captured.get("values")
    if low_level is None:
        raise RuntimeError(f"{backend} low-level forward hook was not reached")
    return output, context, low_level, workspace.combine_buffer.clone()


def _global_error(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float = BF16_ATOL,
    relative_tolerance: float = BF16_RELATIVE_TOL,
) -> dict[str, object]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError(
            f"{name}: shape/dtype mismatch: {reference.shape}/{reference.dtype} "
            f"vs {actual.shape}/{actual.dtype}"
        )
    reference_float = reference.float()
    actual_float = actual.float()
    diff = (reference_float - actual_float).abs()
    diff_sum = diff.sum()
    diff_max = diff.max()
    reference_sum = reference_float.abs().sum()
    element_count = torch.tensor(
        diff.numel(), dtype=torch.float64, device=diff.device
    )
    finite = torch.tensor(
        [
            int(
                bool(torch.isfinite(reference_float).all().item())
                and bool(torch.isfinite(actual_float).all().item())
            )
        ],
        dtype=torch.int32,
        device=diff.device,
    )
    for value, operation in (
        (diff_sum, dist.ReduceOp.SUM),
        (diff_max, dist.ReduceOp.MAX),
        (reference_sum, dist.ReduceOp.SUM),
        (element_count, dist.ReduceOp.SUM),
        (finite, dist.ReduceOp.MIN),
    ):
        dist.all_reduce(value, op=operation)

    mean = float((diff_sum / element_count).item())
    maximum = float(diff_max.item())
    denominator = float(reference_sum.item())
    relative = float(diff_sum.item()) / denominator if denominator else float(diff_sum.item())
    passed = (
        bool(finite.item())
        and all(math.isfinite(value) for value in (mean, maximum, relative))
        and maximum <= atol
        and relative <= relative_tolerance
    )
    return {
        "name": name,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "abs_error_mean": mean,
        "abs_error_max": maximum,
        "relative_l1_error": relative,
        "atol": atol,
        "relative_l1_tolerance": relative_tolerance,
        "pass": passed,
    }


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _write_result(payload: dict[str, object]) -> None:
    destination = Path(os.environ["MOK_CORRECTNESS_RESULT_JSON"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != EP_SIZE:
        raise RuntimeError(f"correctness gate requires WORLD_SIZE={EP_SIZE}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("correctness gate requires B300/SM103")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.6.2":
        raise RuntimeError("correctness gate is pinned to nvidia-cutlass-dsl==4.6.2")

    cache_root = os.environ.get("MOK_CUTEDSL_CACHE_ROOT")
    if cache_root:
        os.environ["CUTE_DSL_CACHE_DIR"] = str(
            Path(cache_root).resolve() / f"rank-{local_rank}"
        )

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    succeeded = False
    try:
        with torch.no_grad():
            tensors = _make_inputs(rank, device)
            common_config = {
                "fwd_num_comm_sms": 40,
                "bwd_num_comm_sms": 28,
                "minibatch_size": MINIBATCH_SIZE,
                "macrobatch_size": MACROBATCH_SIZE,
                "schedule_capacity_multiplier": 0.5,
                "all_gather_top_experts_chunk_bytes": 2048,
            }
            cuda_config = functional.MoKConfig(**common_config, fwd_backend="cuda")
            cutedsl_config = functional.MoKConfig(
                **common_config, fwd_backend="cutedsl"
            )
            workspace = functional.get_workspace(
                cuda_config,
                dist.group.WORLD,
                device=device,
                num_local_tokens=NUM_LOCAL_TOKENS,
                hidden_size=HIDDEN_SIZE,
                topk=TOPK,
            )
            schedule = functional.build_schedule(
                workspace,
                cuda_config,
                tensors["top_experts"],
                num_local_experts=NUM_LOCAL_EXPERTS,
            )

            num_tokens = int(schedule.num_tokens.item())
            prefix_rank = schedule.peer_rank[:num_tokens]
            prefix_route = schedule.peer_token_idx[:num_tokens]
            real_mask = prefix_rank >= 0
            padding_mask = ~real_mask
            real_routes = int(real_mask.sum().item())
            padding_rows = int(padding_mask.sum().item())
            remote_schedule_rows = int(
                ((prefix_rank != rank) & real_mask).sum().item()
            )
            peer_counts = torch.bincount(
                prefix_rank[real_mask].long(), minlength=EP_SIZE
            ).cpu().tolist()
            schedule_ok = (
                num_tokens == 4 * MACROBATCH_SIZE
                and real_routes == EP_SIZE * NUM_LOCAL_TOKENS * TOPK
                and padding_rows == num_tokens - real_routes
                and bool(torch.all(schedule.tokens_per_expert == 256).item())
                and peer_counts == [NUM_LOCAL_TOKENS * TOPK // EP_SIZE] * EP_SIZE
                and remote_schedule_rows > 0
                and bool(torch.any(prefix_route[real_mask].remainder(TOPK) != 0).item())
                and int(prefix_route[real_mask].min().item()) >= 0
                and int(prefix_route[real_mask].max().item())
                < NUM_LOCAL_TOKENS * TOPK
            )
            if not _all_ranks_true(schedule_ok, device):
                raise AssertionError(
                    f"rank {rank}: balanced padded schedule invariant failed"
                )

            all_x_flat = torch.empty(
                EP_SIZE * NUM_LOCAL_TOKENS,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
            )
            dist.all_gather_into_tensor(all_x_flat, tensors["x"])
            all_x = all_x_flat.view(EP_SIZE, NUM_LOCAL_TOKENS, HIDDEN_SIZE)
            macro_rank = schedule.peer_rank[:MACROBATCH_SIZE]
            macro_route = schedule.peer_token_idx[:MACROBATCH_SIZE]
            macro_real = macro_rank >= 0
            expected_macro0_x = torch.zeros(
                MACROBATCH_SIZE,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
            )
            expected_macro0_x[macro_real] = all_x[
                macro_rank[macro_real].long(),
                macro_route[macro_real].long() // TOPK,
            ]

            cuda_output, cuda_context, cuda_low, cuda_combine = _run_backend(
                "cuda",
                cuda_config,
                workspace,
                schedule,
                tensors,
                local_rank,
            )
            cutedsl_output, cutedsl_context, cutedsl_low, cutedsl_combine = (
                _run_backend(
                    "cutedsl",
                    cutedsl_config,
                    workspace,
                    schedule,
                    tensors,
                    local_rank,
                )
            )

            cuda_context_values = _context_tensors(cuda_context)
            cutedsl_context_values = _context_tensors(cutedsl_context)
            context_abi_wiring = all(
                context_value is low_value
                for context_value, low_value in zip(
                    (*cuda_context_values, *cutedsl_context_values),
                    (*cuda_low[:7], *cutedsl_low[:7]),
                    strict=True,
                )
            )

            final_metric = _global_error(
                "final_output", cuda_output, cutedsl_output
            )
            low_metrics: dict[str, dict[str, object]] = {}
            for index, name in enumerate(LOW_LEVEL_NAMES):
                exact = name == "x_routed"
                low_metrics[name] = _global_error(
                    f"low_level.{name}",
                    cuda_low[index],
                    cutedsl_low[index],
                    atol=0.0 if exact else BF16_ATOL,
                    relative_tolerance=0.0 if exact else BF16_RELATIVE_TOL,
                )
            combine_metric = _global_error(
                "combine_buffer", cuda_combine, cutedsl_combine
            )
            macro0_cuda_metric = _global_error(
                "macro0.cuda_x_routed",
                expected_macro0_x,
                cuda_low[0],
                atol=0.0,
                relative_tolerance=0.0,
            )
            macro0_cutedsl_metric = _global_error(
                "macro0.cutedsl_x_routed",
                expected_macro0_x,
                cutedsl_low[0],
                atol=0.0,
                relative_tolerance=0.0,
            )

            macro_padding = ~macro_real
            padding_zero_local = all(
                bool(torch.count_nonzero(backend_low[index][macro_padding]).item() == 0)
                for backend_low in (cuda_low, cutedsl_low)
                for index in ROUTED_LOW_LEVEL_INDICES
            )
            original_remote_routes = (
                tensors["top_experts"].view(-1) // NUM_LOCAL_EXPERTS != rank
            )
            remote_combine_local = (
                bool(original_remote_routes.any().item())
                and bool(torch.isfinite(cuda_combine[original_remote_routes]).all().item())
                and bool(
                    torch.isfinite(cutedsl_combine[original_remote_routes]).all().item()
                )
            )
            edge_checks = {
                "route_idx_divided_by_topk_for_dispatch": _all_ranks_true(
                    bool(macro0_cutedsl_metric["pass"]), device
                ),
                "padding_peer_rank_minus_one_zero_filled": _all_ranks_true(
                    padding_zero_local, device
                ),
                "remote_combine_overwrote_nan_poison": _all_ranks_true(
                    remote_combine_local, device
                ),
                "macro0_resident_after_four_reverse_macros": _all_ranks_true(
                    bool(macro0_cuda_metric["pass"])
                    and bool(macro0_cutedsl_metric["pass"]),
                    device,
                ),
                "forward_context_is_low_level_first_seven": _all_ranks_true(
                    context_abi_wiring, device
                ),
            }

            comparisons_pass = (
                bool(final_metric["pass"])
                and all(bool(metric["pass"]) for metric in low_metrics.values())
                and bool(combine_metric["pass"])
                and bool(macro0_cuda_metric["pass"])
                and bool(macro0_cutedsl_metric["pass"])
                and all(edge_checks.values())
            )
            all_pass = _all_ranks_true(comparisons_pass, device)
            payload: dict[str, object] = {
                "status": "PASS" if all_pass else "FAIL",
                "backend_under_test": "mok-cutedsl-forward",
                "reference_backend": "mok-cuda-forward",
                "performance": "N/A (correctness-only job)",
                "seed": SEED,
                "software": {
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "nvidia_cutlass_dsl": importlib.metadata.version(
                        "nvidia-cutlass-dsl"
                    ),
                    "quack_kernels": importlib.metadata.version("quack-kernels"),
                },
                "device": torch.cuda.get_device_name(device),
                "config": {
                    "ep": EP_SIZE,
                    "dtype": "bf16",
                    "num_local_tokens": NUM_LOCAL_TOKENS,
                    "hidden_size": HIDDEN_SIZE,
                    "intermediate_size": INTERMEDIATE_SIZE,
                    "num_experts": NUM_EXPERTS,
                    "num_local_experts": NUM_LOCAL_EXPERTS,
                    "topk": TOPK,
                    "minibatch_size": MINIBATCH_SIZE,
                    "macrobatch_size": MACROBATCH_SIZE,
                    "fwd_num_comm_sms": 40,
                },
                "schedule_rank0": {
                    "num_tokens": num_tokens,
                    "num_macros": num_tokens // MACROBATCH_SIZE,
                    "real_routes": real_routes,
                    "padding_rows": padding_rows,
                    "remote_schedule_rows": remote_schedule_rows,
                    "real_rows_per_source_peer": peer_counts,
                    "tokens_per_expert": schedule.tokens_per_expert.cpu().tolist(),
                },
                "comparisons": {
                    "final_output": final_metric,
                    # The hook captures all nine low-level outputs.  Identity
                    # checks above prove these first seven are exactly the
                    # tensors exported in each MoKForwardContext.
                    "forward_context": {
                        name: low_metrics[name] for name in CONTEXT_NAMES
                    },
                    "low_level_nine_tensor_abi": low_metrics,
                    "combine_buffer": combine_metric,
                    "macro0_cuda_x_routed": macro0_cuda_metric,
                    "macro0_cutedsl_x_routed": macro0_cutedsl_metric,
                },
                "edge_checks": edge_checks,
                "all_ranks_pass": all_pass,
            }
            if rank == 0:
                _write_result(payload)
                print(json.dumps(payload, sort_keys=True), flush=True)
            dist.barrier(device_ids=[local_rank])
            if not all_pass:
                raise AssertionError("CuTe DSL Qwen forward correctness gate failed")

        succeeded = True
    finally:
        if succeeded:
            functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
