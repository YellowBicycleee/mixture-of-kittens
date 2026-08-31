#!/usr/bin/env python3
"""T=102400 private CuTe macrobatch screen against accepted CuTe B=32768.

Correctness uses two fresh CUDA legs before each candidate.  Timing is a
rank-max CUDA-Event ABBA diagnostic; it only screens a CuTe macrobatch for a
later formal timing node and cannot change the public CUDA default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
TOKENS = 102400
MACROBATCHES = (4096, 8192, 16384, 32768)
ACCEPTED_MACROBATCH = 32768
MINIBATCH = 4096
COMM_SMS = 40
WARMUP_PAIRS = 2
MEASURED_PAIRS = 10
SCHEMA = "mok_cutedsl_fwd_macrobatch_screen_v1"
FROZEN_SHA256 = {
    "mok/functional.py": "a1a1b5f392b8c42223d1e183a801c7c4f31941d806861de9c6af359b74714c2b",
    "mok/cutedsl/persistent_bf16.py": "fae66138682e1577e8a8760c4dedff2ba7d80244c21794f07651c86203e1084c",
    "mok/cutedsl/_persistent_bf16_mega.py": "9eadbf44eb775f470cda000072a233168a931011c57022e1ffe909b24d92b3ce",
    "csrc/megakernel/forward.cuh": "8809cff7fe2e4cac59e9cf8cde717f77b4ee0741bc0766994133a3dfd9745e81",
    "csrc/megakernel/entrypoints.cuh": "ba7cf15ba6c21b2787c522d73ee471fe54d48107649635653d5a2332acdece6b",
    "benchmarks/cutedsl/bench_qwen_fwd_ab.py": "51af8a77f0883074cd99827fb417d78ffad6eab48abb107fe1c851a9f7713b2b",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_gate() -> None:
    for relative, expected in FROZEN_SHA256.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")


def static_self_test() -> dict[str, object]:
    _frozen_gate()
    assert MACROBATCHES == (MINIBATCH, 2 * MINIBATCH, 4 * MINIBATCH, 8 * MINIBATCH)
    assert ACCEPTED_MACROBATCH == MACROBATCHES[-1]
    assert WARMUP_PAIRS == 2 and MEASURED_PAIRS == 10
    return {
        "status": "PASS",
        "schema": SCHEMA,
        "shape": {
            "tokens_per_rank": TOKENS,
            "ep": 8,
            "experts": 512,
            "topk": 10,
            "hidden": 4096,
            "intermediate": 1024,
            "dtype": "BF16",
        },
        "candidate_macrobatches": MACROBATCHES,
        "fixed": {"minibatch": MINIBATCH, "fwd_num_comm_sms": COMM_SMS},
        "timing": {
            "status": "N/A",
            "reason": "local static mode does not run a GPU",
            "boundary": "CUDA Events around complete forward entry",
            "warmup_pairs": WARMUP_PAIRS,
            "measured_pairs": MEASURED_PAIRS,
            "reduction": "same-index EP8 rank maximum",
        },
        "decision_scope": (
            "diagnostic CuTe macro screen only; formal W10/N30 with CV, "
            "CUDA-vs-CuTe, and CUDA own-best are separate later nodes"
        ),
    }


def _load_runtime() -> None:
    global torch, dist, accepted_bench, functional, candidate
    import torch as _torch
    import torch.distributed as _dist
    from benchmarks.cutedsl import bench_qwen_fwd_ab as _accepted_bench
    from mok import functional as _functional
    from mok.cutedsl import _persistent_bf16_macrobatch_experimental as _candidate

    _accepted_bench.load_runtime()
    torch, dist = _torch, _dist
    accepted_bench, functional, candidate = (
        _accepted_bench,
        _functional,
        _candidate,
    )


def _config(macro: int, backend: str):
    return accepted_bench.make_config(macro, MINIBATCH, COMM_SMS, backend)


def _candidate_forward(config, workspace, schedule, inputs):
    """Mirror only the frozen functional.forward shell around the private body."""

    functional.validate_inputs(
        config,
        workspace,
        schedule,
        inputs[0],
        inputs[2],
    )
    workspace.x_buffer.copy_(inputs[0])
    workspace.router_weight_buffer.copy_(inputs[2])
    functional.barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    values = candidate.forward_bf16(
        workspace,
        schedule,
        *inputs[3:9],
        macrobatch_size=config.macrobatch_size,
        minibatch_size=config.minibatch_size,
        swiglu_limit=None,
        num_comm_sms=config.fwd_num_comm_sms,
    )
    (
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        y_shared,
        _y_routed,
    ) = values
    functional.barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    output = functional.fwd_epilogue(
        y_shared,
        workspace.combine_buffer,
        workspace.router_weight_buffer,
    )
    context = functional.MoKForwardContext(
        x_routed=x_routed,
        gate_shared=gate_shared,
        gate_routed=gate_routed,
        up_shared=up_shared,
        up_routed=up_routed,
        hidden_shared=hidden_shared,
        hidden_routed=hidden_routed,
    )
    return output, context


def _candidate_leg(config, workspace, schedule, inputs, local_rank: int):
    workspace.combine_buffer.fill_(float("nan"))
    dist.barrier(device_ids=[local_rank])
    output, context = _candidate_forward(config, workspace, schedule, inputs)
    gradients = accepted_bench.backward(config, workspace, schedule, context, inputs)
    torch.cuda.synchronize(workspace.device)
    forward_values = (
        output.clone(),
        *(getattr(context, name).clone() for name in accepted_bench.FORWARD_CONTEXT_NAMES),
    )
    gradients = tuple(value.clone() for value in gradients)
    torch.cuda.synchronize(workspace.device)
    return forward_values, gradients


def _compare_legs(name: str, reference, actual, device) -> dict[str, object]:
    records = []
    reference_values = (*reference[0], *reference[1])
    actual_values = (*actual[0], *actual[1])
    names = (*accepted_bench.FORWARD_NAMES, *accepted_bench.RESULT_NAMES[1:])
    for tensor_name, expected, observed in zip(
        names,
        reference_values,
        actual_values,
        strict=True,
    ):
        record, _ = accepted_bench.parity_metric(
            f"{name}/{tensor_name}",
            "forward_or_backward",
            expected,
            observed,
            device,
        )
        bitwise = accepted_bench.all_ranks_true(
            expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and bool(torch.equal(expected, observed)),
            device,
        )
        record["bitwise_equal_on_all_ranks"] = bitwise
        record["pass"] = bool(record["pass"] and bitwise)
        records.append(record)
    return {"records": records, "pass": all(record["pass"] for record in records)}


def _rank_max_event(run: Callable[[], object], device) -> float:
    dist.barrier(async_op=True).block_current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, context = run()
    end.record()
    end.synchronize()
    local = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    output = context = None
    return float(local.item())


def _abba(accepted_run, candidate_run, device) -> dict[str, object]:
    for pair in range(WARMUP_PAIRS):
        order = (accepted_run, candidate_run) if pair % 2 == 0 else (
            candidate_run,
            accepted_run,
        )
        for run in order:
            run()
    torch.cuda.synchronize(device)

    accepted_ms, candidate_ms, orders = [], [], []
    for pair in range(MEASURED_PAIRS):
        order = ("A", "B") if pair % 2 == 0 else ("B", "A")
        orders.append("".join(order))
        values = {}
        for label in order:
            values[label] = _rank_max_event(
                accepted_run if label == "A" else candidate_run,
                device,
            )
        accepted_ms.append(values["A"])
        candidate_ms.append(values["B"])
    ratios = [a / b for a, b in zip(accepted_ms, candidate_ms, strict=True)]
    positive = sum(value > 1.0 for value in ratios)
    return {
        "order": orders,
        "accepted_rank_max_ms": accepted_ms,
        "candidate_rank_max_ms": candidate_ms,
        "accepted_median_ms": statistics.median(accepted_ms),
        "candidate_median_ms": statistics.median(candidate_ms),
        "paired_speedup_samples": ratios,
        "paired_median_speedup": statistics.median(ratios),
        "positive_pairs": positive,
        "screen_positive": statistics.median(ratios) > 1.0 and positive >= 8,
    }


def run_device() -> dict[str, object]:
    _frozen_gate()
    _load_runtime()
    from benchmarks.utils import init_distributed

    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world_size, device = init_distributed()
    if world_size != 8 or torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("device mode requires EP8 on B300/SM103")
    if os.environ.get("MOK_CUTEDSL_CACHE_ROOT"):
        os.environ["CUTE_DSL_CACHE_DIR"] = str(
            Path(os.environ["MOK_CUTEDSL_CACHE_ROOT"]).resolve()
            / f"rank-{local_rank}"
        )

    accepted_config = _config(ACCEPTED_MACROBATCH, "cutedsl")
    inputs, workspace, schedule = accepted_bench.make_case(
        TOKENS,
        accepted_config,
        rank,
        device,
    )
    rows = accepted_bench.routed_rows(schedule, device)
    correctness = {}
    for macro in MACROBATCHES:
        cuda_config = _config(macro, "cuda")
        candidate_config = _config(macro, "cutedsl")
        cuda_a = accepted_bench.checked_forward_backward(
            cuda_config, workspace, schedule, inputs, local_rank
        )
        cuda_b = accepted_bench.checked_forward_backward(
            cuda_config, workspace, schedule, inputs, local_rank
        )
        experimental = _candidate_leg(
            candidate_config, workspace, schedule, inputs, local_rank
        )
        self_control = _compare_legs(f"B{macro}/cuda_a_vs_cuda_b", cuda_a, cuda_b, device)
        candidate_control = _compare_legs(
            f"B{macro}/cuda_a_vs_candidate", cuda_a, experimental, device
        )
        passed = self_control["pass"] and candidate_control["pass"]
        if not passed:
            raise AssertionError(f"B={macro} correctness failed")
        correctness[str(macro)] = {
            "cuda_self_control": self_control,
            "candidate": candidate_control,
            "pass": True,
        }

    public_b32 = accepted_bench.checked_forward_backward(
        accepted_config, workspace, schedule, inputs, local_rank
    )
    private_b32 = _candidate_leg(
        accepted_config, workspace, schedule, inputs, local_rank
    )
    b32_private_self_control = _compare_legs(
        "B32768/public_accepted_vs_private_wrapper",
        public_b32,
        private_b32,
        device,
    )
    if (
        len(b32_private_self_control["records"]) != 16
        or not b32_private_self_control["pass"]
    ):
        raise AssertionError("B=32768 private wrapper self-control failed")

    accepted_run = lambda: accepted_bench.forward(
        accepted_config, workspace, schedule, inputs
    )
    accepted_run()
    torch.cuda.synchronize(device)
    timing = {}
    for macro in MACROBATCHES:
        config = _config(macro, "cutedsl")
        candidate_run = lambda config=config: _candidate_forward(
            config, workspace, schedule, inputs
        )
        candidate_run()
        torch.cuda.synchronize(device)
        timing[str(macro)] = _abba(accepted_run, candidate_run, device)

    screen_candidates = [
        macro
        for macro in MACROBATCHES
        if timing[str(macro)]["screen_positive"]
    ]
    screen_candidate = min(
        screen_candidates,
        key=lambda macro: timing[str(macro)]["candidate_median_ms"],
        default=ACCEPTED_MACROBATCH,
    )

    result = {
        "status": "PASS",
        "schema": SCHEMA,
        "provenance": {
            "frozen_sha256": FROZEN_SHA256,
            "candidate_module_sha256": _sha256(
                ROOT / "mok/cutedsl/_persistent_bf16_macrobatch_experimental.py"
            ),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "shape": static_self_test()["shape"],
        "padded_routed_rows_by_rank": rows,
        "fixed": {"minibatch": MINIBATCH, "fwd_num_comm_sms": COMM_SMS},
        "correctness": correctness,
        "b32_private_self_control": b32_private_self_control,
        "timing": {
            "accepted_A": "frozen public CuTe B32768",
            "candidate_B": "private CuTe variable macrobatch",
            "boundary": "CUDA Events around complete forward entry",
            "aggregation": "same-index EP8 rank maximum",
            "warmup_pairs": WARMUP_PAIRS,
            "measured_pairs": MEASURED_PAIRS,
            "points": timing,
        },
        "diagnostic_decision": {
            "screen_candidate": screen_candidate,
            "screen_differs_from_accepted_32768": (
                screen_candidate != ACCEPTED_MACROBATCH
            ),
            "screen_positive_candidates": screen_candidates,
            "formal_selection": "N/A_requires_W10_N30_and_CV",
            "next_required_node": (
                "formal W10/N30 with CV, then same-macro CUDA-vs-CuTe plus "
                "CUDA own-best before any base decision"
            ),
        },
        "decision_scope": (
            "diagnostic CuTe macro screen only; no CUDA default or "
            "development-base change"
        ),
    }
    functional.clear_workspace_cache()
    candidate.clear_experimental_macrobatch_cache()
    dist.destroy_process_group()
    return result


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.run_device:
        print(json.dumps(static_self_test(), sort_keys=True))
        return
    if not args.output:
        raise SystemExit("--output is required with --run-device")
    payload = run_device()
    if int(os.environ.get("RANK", "0")) == 0:
        _atomic_write(Path(args.output).expanduser().resolve(), payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
