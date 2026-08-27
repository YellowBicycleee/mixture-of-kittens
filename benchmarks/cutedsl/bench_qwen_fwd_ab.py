#!/usr/bin/env python3
"""BF16-only EP8 Qwen forward A/B: CUDA megakernel versus CuTe DSL.

Run on one eight-B300 node with ``torchrun --standalone --nproc-per-node=8``.
The two small cases check one- and two-macro correctness.  The 100K/rank case
uses a 32K macrobatch and captures the raw nine-tensor forward ABI before BWD
can rewrite its ring buffers.  A parity failure is fsynced as per-rank JSON
before raising.  Only after parity, six CUDA communication-SM counts are
screened and the winner runs in CUDA-CuTe-CuTe-CUDA W10/N30 order.  CUDA Events
bracket only ``functional.forward``; schedule construction and the first CuTe
compilation call are outside the measured boundary.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


EP, HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 8, 4096, 1024, 512, 10
LOCAL_EXPERTS = EXPERTS // EP
CORRECTNESS_CASES = ((2048, 1), (4096, 2))
TIMING_TOKENS = 102400
WARMUPS, SAMPLES = 10, 30
SCREEN_WARMUPS, SCREEN_SAMPLES = 3, 10
FORMAL_ORDER = ("cuda", "cutedsl", "cutedsl", "cuda")
MACROBATCH, MINIBATCH = 32768, 4096
CUDA_COMM_CANDIDATES = (36, 40, 44, 48, 52, 56)
RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)
FORWARD_CONTEXT_NAMES = (
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


def static_self_test() -> dict[str, object]:
    assert EP == 8 and EXPERTS // EP == 64
    assert FORMAL_ORDER == ("cuda", "cutedsl", "cutedsl", "cuda")
    assert WARMUPS == 10 and SAMPLES == 30
    assert SCREEN_WARMUPS == 3 and SCREEN_SAMPLES == 10
    assert MACROBATCH == 2**15 and MINIBATCH == 2**12
    assert MACROBATCH % MINIBATCH == 0
    assert CUDA_COMM_CANDIDATES == (36, 40, 44, 48, 52, 56)
    assert len(RESULT_NAMES) == 9
    assert len(FORWARD_CONTEXT_NAMES) == 9
    assert json_number(float("nan")) is None and json_number(1.25) == 1.25
    return {
        "status": "PASS",
        "dtype": "BF16",
        "shape": {"ep": EP, "hidden": HIDDEN, "intermediate": INTERMEDIATE,
                  "experts": EXPERTS, "local_experts": LOCAL_EXPERTS, "topk": TOPK},
        "correctness_tokens": [case[0] for case in CORRECTNESS_CASES],
        "full_scale_cuda_parity_tokens": TIMING_TOKENS,
        "full_scale_expected_generations": 32,
        "timing_tokens": TIMING_TOKENS,
        "macrobatch": MACROBATCH,
        "minibatch": MINIBATCH,
        "cuda_comm_screen": list(CUDA_COMM_CANDIDATES),
        "formal_order": list(FORMAL_ORDER),
        "warmups": WARMUPS,
        "samples": SAMPLES,
    }


def load_runtime() -> None:
    global torch, dist, functional, contract
    global BF16_TOLERANCE, generate_inputs, get_error_stats, run_reference_bf16
    import torch as _torch
    import torch.distributed as _dist
    from mok import functional as _functional
    from mok.cutedsl import forward_contract as _contract
    from tests.utils import (
        BF16_TOLERANCE as _tolerance,
        generate_inputs as _generate_inputs,
        get_error_stats as _get_error_stats,
        run_reference_bf16 as _run_reference_bf16,
    )
    torch, dist, functional, contract = _torch, _dist, _functional, _contract
    BF16_TOLERANCE, generate_inputs = _tolerance, _generate_inputs
    get_error_stats, run_reference_bf16 = _get_error_stats, _run_reference_bf16


def make_config(macro: int, mini: int, comm: int, backend: str):
    return functional.MoKConfig(
        fwd_num_comm_sms=comm,
        bwd_num_comm_sms=36,
        minibatch_size=mini,
        macrobatch_size=macro,
        bwd_schedule="minibatch",
        fwd_backend=backend,
    )


def forward(config, workspace, schedule, inputs):
    return functional.forward(
        config, workspace, schedule, inputs[0], inputs[2], *inputs[3:9]
    )


def backward(config, workspace, schedule, context, inputs):
    return functional.backward(
        config, workspace, schedule, context, inputs[9], inputs[0], inputs[2],
        *inputs[3:9],
    )


def metric(name: str, reference, actual) -> dict[str, object]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError(
            f"{name}: {reference.shape}/{reference.dtype} != "
            f"{actual.shape}/{actual.dtype}"
        )
    mean, maximum, relative = get_error_stats(reference, actual)
    atol, rtol = BF16_TOLERANCE
    passed = (
        all(math.isfinite(value) for value in (mean, maximum, relative))
        and maximum <= atol and relative <= rtol
    )
    return {
        "name": name,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "abs_error_mean": mean,
        "abs_error_max": maximum,
        "relative_l1_error": relative,
        "atol": atol,
        "relative_l1_tolerance": rtol,
        "pass": passed,
    }


def all_ranks_true(value: bool, device) -> bool:
    flag = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def json_number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def parity_metric(name: str, category: str, reference, actual,
                  device) -> tuple[dict[str, object], dict[str, object]]:
    """Return the unchanged global acceptance metric plus a local-rank metric.

    Unlike ``metric``, this never asserts on a shape or dtype mismatch.  That
    lets the full-scale case persist every rank's evidence before raising.
    """

    reference_shape = list(reference.shape)
    actual_shape = list(actual.shape)
    reference_dtype = str(reference.dtype)
    actual_dtype = str(actual.dtype)
    shape_match = reference_shape == actual_shape
    dtype_match = reference_dtype == actual_dtype
    reference_finite = bool(torch.isfinite(reference).all().item())
    actual_finite = bool(torch.isfinite(actual).all().item())
    shapes_match_everywhere = all_ranks_true(shape_match, device)
    dtypes_match_everywhere = all_ranks_true(dtype_match, device)
    atol, rtol = BF16_TOLERANCE

    local_mean = local_maximum = local_relative = None
    global_mean = global_maximum = global_relative = None
    global_finite = False
    if shapes_match_everywhere:
        reference_float = reference.float()
        actual_float = actual.float()
        difference = (reference_float - actual_float).abs()
        local_sum = difference.sum()
        local_count = torch.tensor(
            difference.numel(), dtype=torch.float64, device=device
        )
        local_max = difference.max()
        local_reference_sum = reference_float.abs().sum()
        local_mean = float((local_sum / local_count).item())
        local_maximum = float(local_max.item())
        local_relative = float((local_sum / local_reference_sum).item())

        global_sum = local_sum.clone()
        global_count = local_count.clone()
        global_max = local_max.clone()
        global_reference_sum = local_reference_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(global_reference_sum, op=dist.ReduceOp.SUM)
        global_mean = float((global_sum / global_count).item())
        global_maximum = float(global_max.item())
        global_relative = float((global_sum / global_reference_sum).item())
        global_finite = all_ranks_true(
            reference_finite
            and actual_finite
            and all(math.isfinite(value) for value in (
                local_mean, local_maximum, local_relative,
            )),
            device,
        )

    local_finite = (
        reference_finite
        and actual_finite
        and local_mean is not None
        and all(math.isfinite(value) for value in (
            local_mean, local_maximum, local_relative,
        ))
    )
    local_pass = bool(
        shape_match
        and dtype_match
        and local_finite
        and local_maximum <= atol
        and local_relative <= rtol
    )
    global_pass = bool(
        shapes_match_everywhere
        and dtypes_match_everywhere
        and global_finite
        and global_maximum <= atol
        and global_relative <= rtol
    )
    local = {
        "name": name,
        "category": category,
        "reference_shape": reference_shape,
        "actual_shape": actual_shape,
        "shape_match": shape_match,
        "reference_dtype": reference_dtype,
        "actual_dtype": actual_dtype,
        "dtype_match": dtype_match,
        "reference_finite": reference_finite,
        "actual_finite": actual_finite,
        "finite": local_finite,
        "abs_error_mean": json_number(local_mean),
        "max_abs_error": json_number(local_maximum),
        "relative_l1_error": json_number(local_relative),
        "atol": atol,
        "relative_l1_tolerance": rtol,
        "pass": local_pass,
    }
    global_metric = {
        "name": name,
        "category": category,
        "reference_shape": reference_shape,
        "actual_shape": actual_shape,
        "shape_match_on_all_ranks": shapes_match_everywhere,
        "reference_dtype": reference_dtype,
        "actual_dtype": actual_dtype,
        "dtype_match_on_all_ranks": dtypes_match_everywhere,
        "finite_on_all_ranks": global_finite,
        "abs_error_mean": json_number(global_mean),
        "abs_error_max": json_number(global_maximum),
        "relative_l1_error": json_number(global_relative),
        "atol": atol,
        "relative_l1_tolerance": rtol,
        "pass": global_pass,
    }
    return global_metric, local


def unavailable_local_metric(name: str, category: str,
                             reason: str) -> dict[str, object]:
    return {
        "name": name,
        "category": category,
        "reference_shape": None,
        "actual_shape": None,
        "shape_match": False,
        "reference_dtype": None,
        "actual_dtype": None,
        "dtype_match": False,
        "reference_finite": False,
        "actual_finite": False,
        "finite": False,
        "abs_error_mean": None,
        "max_abs_error": None,
        "relative_l1_error": None,
        "atol": BF16_TOLERANCE[0],
        "relative_l1_tolerance": BF16_TOLERANCE[1],
        "pass": False,
        "unavailable_reason": reason,
    }


def routed_rows(schedule, device) -> list[int]:
    if type(schedule.num_tokens_host) is not int:
        raise AssertionError("cached CuTe schedule lacks num_tokens_host")
    local = torch.tensor([schedule.num_tokens_host], dtype=torch.int64, device=device)
    gathered = torch.empty(EP, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local)
    return [int(value) for value in gathered.cpu().tolist()]


def make_case(tokens: int, config, rank: int, device):
    inputs = generate_inputs(
        rank, device, EXPERTS, LOCAL_EXPERTS, TOPK, tokens, HIDDEN, INTERMEDIATE
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=tokens,
        hidden_size=HIDDEN, topk=TOPK,
    )
    schedule = functional.build_schedule(
        workspace, config, inputs[1], num_local_experts=LOCAL_EXPERTS
    )
    return inputs, workspace, schedule


def checked_forward(config, workspace, schedule, inputs, local_rank: int):
    workspace.combine_buffer.fill_(float("nan"))
    dist.barrier(device_ids=[local_rank])
    output, context = forward(config, workspace, schedule, inputs)
    torch.cuda.synchronize(workspace.device)
    return output, context


def checked_forward_with_abi(config, workspace, schedule, inputs,
                             local_rank: int):
    """Run ``functional.forward`` while retaining its raw nine-tensor ABI."""

    captured: list[tuple[object, ...]] = []
    if config.fwd_backend == "cuda":
        owner = functional
        attribute = "dispatch_mlp_swiglu_combine_fwd_bf16"
    else:
        import importlib

        owner = importlib.import_module("mok.cutedsl.forward")
        attribute = "forward_bf16"
    original = getattr(owner, attribute)

    def capture(*args, **kwargs):
        tensors = original(*args, **kwargs)
        captured.append(tuple(tensors))
        return tensors

    workspace.combine_buffer.fill_(float("nan"))
    dist.barrier(device_ids=[local_rank])
    setattr(owner, attribute, capture)
    try:
        output, context = forward(config, workspace, schedule, inputs)
    finally:
        setattr(owner, attribute, original)
    torch.cuda.synchronize(workspace.device)
    raw_abi = captured[0] if len(captured) == 1 else ()
    return output, context, raw_abi


def local_schedule_diagnostic(schedule, rank: int) -> dict[str, object]:
    host_tokens = schedule.num_tokens_host
    num_tokens = (
        host_tokens if type(host_tokens) is int else int(schedule.num_tokens.item())
    )
    valid_tokens = int(
        schedule.peer_rank[:num_tokens].ge(0).sum(dtype=torch.int64).item()
    )
    return {
        "rank": rank,
        "num_tokens": num_tokens,
        "num_tokens_kind": "scheduler_padded_routed_rows",
        "non_padding_tokens": valid_tokens,
        "padding": num_tokens - valid_tokens,
        "generations": (num_tokens + MACROBATCH - 1) // MACROBATCH,
        "macrobatch": MACROBATCH,
        "schedule_capacity": int(schedule.peer_rank.numel()),
    }


def diagnostic_path() -> Path:
    explicit = os.environ.get("MOK_QWEN_FWD_PARITY_DIAGNOSTIC_JSON")
    if explicit:
        return Path(explicit).expanduser().resolve()
    result = os.environ.get("MOK_QWEN_FWD_AB_JSON")
    if result:
        result_path = Path(result).expanduser().resolve()
        suffix = result_path.suffix or ".json"
        return result_path.with_name(
            f"{result_path.stem}.full-parity-failure{suffix}"
        )
    return (Path.cwd() / "qwen_fwd_full_parity_failure.json").resolve()


def atomic_write_json_fsync(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def persist_full_scale_failure(local_diagnostic: dict[str, object]) -> str:
    gathered: list[dict[str, object] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_diagnostic)
    rank = dist.get_rank()
    outcome: list[dict[str, str] | None] = [None]
    if rank == 0:
        path = diagnostic_path()
        payload = {
            "status": "FAIL",
            "case": "T=102400 BF16 CUDA-vs-CuTe full parity",
            "tokens_per_rank": TIMING_TOKENS,
            "macrobatch": MACROBATCH,
            "minibatch": MINIBATCH,
            "expected_generations": 32,
            "acceptance_tensors": list(RESULT_NAMES),
            "forward_context_tensors_diagnostic_only": list(
                FORWARD_CONTEXT_NAMES
            ),
            "rank_diagnostics": gathered,
            "timing_started": False,
        }
        try:
            atomic_write_json_fsync(path, payload)
            outcome[0] = {"path": str(path)}
            print(json.dumps({
                "full_parity_diagnostic": str(path), "status": "WRITTEN",
            }, sort_keys=True), flush=True)
        except Exception as error:  # synchronize the persistence error to all ranks
            outcome[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is None or "error" in outcome[0]:
        detail = "missing rank-0 outcome" if outcome[0] is None else outcome[0]["error"]
        raise RuntimeError(f"failed to persist full parity diagnostic: {detail}")
    return outcome[0]["path"]


def correctness_case(tokens: int, expected_generations: int, rank: int,
                     local_rank: int, device) -> dict[str, object]:
    cute_config = make_config(MACROBATCH, MINIBATCH, 40, "cutedsl")
    cuda_config = make_config(MACROBATCH, MINIBATCH, 40, "cuda")
    inputs, workspace, schedule = make_case(tokens, cute_config, rank, device)
    reference = run_reference_bf16(*inputs)
    with torch.no_grad():
        cuda_output, cuda_context = checked_forward(
            cuda_config, workspace, schedule, inputs, local_rank
        )
        cuda_gradients = backward(
            cuda_config, workspace, schedule, cuda_context, inputs
        )
        cute_output, cute_context = checked_forward(
            cute_config, workspace, schedule, inputs, local_rank
        )
        cute_gradients = backward(
            cute_config, workspace, schedule, cute_context, inputs
        )
        torch.cuda.synchronize(device)

    comparisons = [
        metric(f"{backend}/{name}", expected, actual)
        for backend, actuals in (
            ("cuda", (cuda_output, *cuda_gradients)),
            ("cutedsl", (cute_output, *cute_gradients)),
        )
        for name, expected, actual in zip(
            RESULT_NAMES, reference, actuals, strict=True
        )
    ]
    rows = routed_rows(schedule, device)
    generations = [(value + MACROBATCH - 1) // MACROBATCH for value in rows]
    generation_pass = all(value == expected_generations for value in generations)
    passed = all_ranks_true(
        generation_pass and all(bool(item["pass"]) for item in comparisons), device
    )
    if not passed:
        raise AssertionError(f"T={tokens} BF16 correctness failed")
    return {
        "tokens_per_rank": tokens,
        "fixed_seed": "1234 + EP rank (tests.utils.generate_inputs)",
        "macrobatch": MACROBATCH,
        "minibatch_bwd": True,
        "padded_routed_rows_by_rank": rows,
        "generations_by_rank": generations,
        "expected_generations": expected_generations,
        "comparisons": comparisons,
        "pass": passed,
    }


def full_scale_parity_case(inputs, workspace, schedule, local_rank: int,
                           device) -> dict[str, object]:
    """Check all 32 replay generations against the trusted CUDA backend."""

    cuda_config = make_config(MACROBATCH, MINIBATCH, 40, "cuda")
    cute_config = make_config(MACROBATCH, MINIBATCH, 40, "cutedsl")
    with torch.no_grad():
        cuda_output, cuda_context, cuda_forward_abi = checked_forward_with_abi(
            cuda_config, workspace, schedule, inputs, local_rank
        )
        cute_output, cute_context, cute_forward_abi = checked_forward_with_abi(
            cute_config, workspace, schedule, inputs, local_rank
        )

        output_comparison, local_output = parity_metric(
            "cutedsl_vs_cuda/output", "forward_output",
            cuda_output, cute_output, device,
        )
        cuda_abi_count = len(cuda_forward_abi)
        cute_abi_count = len(cute_forward_abi)
        local_abi_ok = (
            cuda_abi_count == len(FORWARD_CONTEXT_NAMES)
            and cute_abi_count == len(FORWARD_CONTEXT_NAMES)
        )
        abi_ok = all_ranks_true(local_abi_ok, device)
        forward_context_comparisons = []
        local_forward_contexts = []
        if abi_ok:
            for name, expected, actual in zip(
                FORWARD_CONTEXT_NAMES,
                cuda_forward_abi,
                cute_forward_abi,
                strict=True,
            ):
                comparison, local = parity_metric(
                    f"cutedsl_vs_cuda/{name}",
                    "forward_context_diagnostic_only",
                    expected,
                    actual,
                    device,
                )
                forward_context_comparisons.append(comparison)
                local_forward_contexts.append(local)
        else:
            local_forward_contexts = [
                unavailable_local_metric(
                    f"cutedsl_vs_cuda/{name}",
                    "forward_context_diagnostic_only",
                    "raw forward ABI did not contain exactly nine tensors",
                )
                for name in FORWARD_CONTEXT_NAMES
            ]

        # Backward replay mutates routed context ring buffers.  Their CUDA-vs-
        # CuTe diagnostic above must therefore finish before either BWD starts.
        cuda_forward_abi = cute_forward_abi = None
        cuda_gradients = backward(
            cuda_config, workspace, schedule, cuda_context, inputs
        )
        torch.cuda.synchronize(device)
        cute_gradients = backward(
            cute_config, workspace, schedule, cute_context, inputs
        )
        torch.cuda.synchronize(device)

    gradient_count_ok = all_ranks_true(
        len(cuda_gradients) == len(RESULT_NAMES) - 1
        and len(cute_gradients) == len(RESULT_NAMES) - 1,
        device,
    )
    gradient_comparisons = []
    local_gradients = []
    if gradient_count_ok:
        for name, expected, actual in zip(
            RESULT_NAMES[1:], cuda_gradients, cute_gradients, strict=True
        ):
            comparison, local = parity_metric(
                f"cutedsl_vs_cuda/{name}", "backward_gradient",
                expected, actual, device,
            )
            gradient_comparisons.append(comparison)
            local_gradients.append(local)
    else:
        local_gradients = [
            unavailable_local_metric(
                f"cutedsl_vs_cuda/{name}",
                "backward_gradient",
                "backward did not return exactly eight gradients",
            )
            for name in RESULT_NAMES[1:]
        ]
    comparisons = [output_comparison, *gradient_comparisons]
    rank = dist.get_rank()
    schedule_diagnostic = local_schedule_diagnostic(schedule, rank)
    generation_pass = schedule_diagnostic["generations"] == 32
    passed = all_ranks_true(
        generation_pass
        and gradient_count_ok
        and all(bool(item["pass"]) for item in comparisons),
        device,
    )
    if not passed:
        diagnostic = {
            "rank": rank,
            "schedule": schedule_diagnostic,
            "forward": {
                "output": local_output,
                "contexts": {
                    "diagnostic_only": True,
                    "affects_acceptance": False,
                    "cuda_abi_tensor_count": cuda_abi_count,
                    "cutedsl_abi_tensor_count": cute_abi_count,
                    "comparisons": local_forward_contexts,
                },
            },
            "backward": {"gradients": local_gradients},
            "acceptance": {
                "generation_pass": generation_pass,
                "output_and_eight_gradients_pass": all(
                    bool(item["pass"]) for item in comparisons
                ) and gradient_count_ok,
                "cuda_gradient_tensor_count": len(cuda_gradients),
                "cutedsl_gradient_tensor_count": len(cute_gradients),
                "pass": False,
            },
        }
        path = persist_full_scale_failure(diagnostic)
        raise AssertionError(
            "T=102400 BF16 CUDA-vs-CuTe parity failed; "
            f"diagnostic={path}"
        )
    rows = routed_rows(schedule, device)
    generations = [(value + MACROBATCH - 1) // MACROBATCH for value in rows]
    return {
        "tokens_per_rank": TIMING_TOKENS,
        "reference": "trusted CUDA backend at the same source revision",
        "fixed_seed": "1234 + EP rank (tests.utils.generate_inputs)",
        "macrobatch": MACROBATCH,
        "minibatch_bwd": True,
        "padded_routed_rows_by_rank": rows,
        "generations_by_rank": generations,
        "expected_generations": 32,
        "comparisons": comparisons,
        "forward_context_diagnostic_only": {
            "affects_acceptance": False,
            "captured_nine_tensor_abi_on_all_ranks": abi_ok,
            "comparisons": forward_context_comparisons,
            "pass": (
                abi_ok
                and all(bool(item["pass"])
                        for item in forward_context_comparisons)
            ),
        },
        "pass": passed,
    }


def measure_launch(backend: str, run, device, *, warmups: int = WARMUPS,
                   samples: int = SAMPLES) -> dict[str, object]:
    for _ in range(warmups):
        output, context = run()
        output = context = None
    dist.barrier(async_op=True).block_current_stream()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(samples)
    ]
    for start, end in events:
        start.record()
        output, context = run()
        end.record()
        output = context = None
    torch.cuda.synchronize(device)
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in events],
        dtype=torch.float64, device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(EP)]
    dist.all_gather(gathered, local)
    rank_max = torch.stack(gathered).amax(dim=0)
    return {
        "backend": backend,
        "warmups": warmups,
        "samples": samples,
        "median_rank_max_ms": float(torch.quantile(rank_max, 0.5).item()),
        "min_rank_max_ms": float(rank_max.min().item()),
        "max_rank_max_ms": float(rank_max.max().item()),
        "rank_max_samples_ms": rank_max.cpu().tolist(),
    }


def screen_cuda_comm(inputs, workspace, schedule, device) -> dict[str, object]:
    cells = []
    for comm in CUDA_COMM_CANDIDATES:
        config = make_config(MACROBATCH, MINIBATCH, comm, "cuda")
        run = lambda config=config: forward(config, workspace, schedule, inputs)
        output, context = run()
        torch.cuda.synchronize(device)
        output = context = None
        cells.append(measure_launch(
            "cuda", run, device,
            warmups=SCREEN_WARMUPS, samples=SCREEN_SAMPLES,
        ) | {"fwd_num_comm_sms": comm})
    winner = min(cells, key=lambda cell: cell["median_rank_max_ms"])
    return {
        "macrobatch": MACROBATCH,
        "minibatch": MINIBATCH,
        "cells": cells,
        "selected_fwd_num_comm_sms": winner["fwd_num_comm_sms"],
        "selection_rule": "minimum quick-screen median rank-max CUDA Event time",
    }


def timing_case(comm: int, inputs, workspace, schedule, device) -> dict[str, object]:
    macro, mini = MACROBATCH, MINIBATCH
    configs = {
        backend: make_config(macro, mini, comm, backend)
        for backend in ("cuda", "cutedsl")
    }
    runs = {
        backend: (
            lambda backend=backend: forward(
                configs[backend], workspace, schedule, inputs
            )
        )
        for backend in configs
    }
    # Materialize allocations and every CuTe specialization before any event.
    for backend in ("cuda", "cutedsl"):
        output, context = runs[backend]()
        torch.cuda.synchronize(device)
        output = context = None
        dist.barrier()

    launches = [measure_launch(backend, runs[backend], device)
                for backend in FORMAL_ORDER]
    medians = {
        backend: sorted(
            launch["median_rank_max_ms"] for launch in launches
            if launch["backend"] == backend
        )
        for backend in ("cuda", "cutedsl")
    }
    summary_ms = {
        backend: sum(values) / len(values) for backend, values in medians.items()
    }
    flops = 6 * TIMING_TOKENS * (TOPK + 1) * HIDDEN * INTERMEDIATE
    rows = routed_rows(schedule, device)
    return {
        "macrobatch": macro,
        "public_config": {"minibatch": mini, "fwd_num_comm_sms": comm},
        "generations_by_rank": [(value + macro - 1) // macro for value in rows],
        "formal_order": list(FORMAL_ORDER),
        "launches": launches,
        "summary": {
            backend: {
                "mean_of_two_launch_medians_ms": summary_ms[backend],
                "effective_tflops_per_gpu": flops / 1e9 / summary_ms[backend],
            }
            for backend in ("cuda", "cutedsl")
        } | {"cutedsl_speedup_vs_cuda": summary_ms["cuda"] / summary_ms["cutedsl"]},
    }


def write_result(payload: dict[str, object]) -> None:
    destination = os.environ.get("MOK_QWEN_FWD_AB_JSON")
    if destination:
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main() -> None:
    load_runtime()
    local_rank = int(os.environ["LOCAL_RANK"])
    cache_root = os.environ.get("MOK_CUTEDSL_CACHE_ROOT")
    if cache_root:
        os.environ["CUTE_DSL_CACHE_DIR"] = str(
            Path(cache_root).resolve() / f"rank-{local_rank}"
        )
    from benchmarks.utils import init_distributed
    rank, world_size, device = init_distributed()
    if world_size != EP or torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("benchmark requires EP8 on B300/SM103")
    versions = {
        "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "quack_kernels": importlib.metadata.version("quack-kernels"),
        "torch": torch.__version__,
    }
    if (versions["nvidia_cutlass_dsl"], versions["quack_kernels"]) != ("4.6.2", "0.6.4"):
        raise RuntimeError(f"unexpected CuTe DSL/QuACK versions: {versions}")

    correctness = [
        correctness_case(tokens, generations, rank, local_rank, device)
        for tokens, generations in CORRECTNESS_CASES
    ]
    base = make_config(MACROBATCH, MINIBATCH, 40, backend="cutedsl")
    inputs, workspace, schedule = make_case(TIMING_TOKENS, base, rank, device)
    rows = routed_rows(schedule, device)
    correctness.append(
        full_scale_parity_case(inputs, workspace, schedule, local_rank, device)
    )
    properties = torch.cuda.get_device_properties(device)
    dispatch_ctas, combine_ctas = contract.standalone_comm_worker_grids(
        properties.multi_processor_count
    )
    cuda_screen = screen_cuda_comm(inputs, workspace, schedule, device)
    selected_comm = int(cuda_screen["selected_fwd_num_comm_sms"])
    timing = [timing_case(selected_comm, inputs, workspace, schedule, device)]
    payload = {
        "status": "PASS",
        "comparison": "MoK BF16 CUDA FWD vs BF16 CuTe DSL FWD",
        "shape": static_self_test()["shape"],
        "software": versions,
        "device": properties.name,
        "source_commit": os.environ.get("MOK_SOURCE_COMMIT", "not provided"),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "correctness": correctness,
        "timing": {
            "tokens_per_rank": TIMING_TOKENS,
            "cached_schedule_builds": 1,
            "cuda_comm_screen": cuda_screen,
            "deferred_macrobatches": [131072, 262144, 524288, 1048576],
            "padded_routed_rows_by_rank": rows,
            "boundary": "CUDA Events around functional.forward only",
            "aggregation": "per-iteration EP8 rank maximum, then launch median",
            "first_backend_call_inside_events": False,
            "cutedsl_actual_fixed_grid": {
                "physical_sms": properties.multi_processor_count,
                "dispatch_ctas": dispatch_ctas,
                "dispatch_ctas_per_sm": contract.DISPATCH_CTAS_PER_SM,
                "combine_ctas": combine_ctas,
                "combine_ctas_per_sm": 1,
                "public_fwd_num_comm_sms_tunes_grid": False,
            },
            "points": timing,
        },
    }
    if rank == 0:
        write_result(payload)
    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        print(json.dumps(static_self_test(), sort_keys=True))
    elif sys.argv[1:]:
        raise SystemExit("usage: bench_qwen_fwd_ab.py [--self-test]")
    else:
        main()
