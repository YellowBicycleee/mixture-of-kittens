#!/usr/bin/env python3
"""Formal 100K/rank CUDA-versus-private-CuTe macrobatch sweep.

The accepted CUDA and CuTe backends stay frozen.  This runner reuses the
private macrobatch fixture, requires three-way bitwise FWD/BWD parity at every
point, then measures CUDA-CuTe-CuTe-CUDA blocks with CUDA Events around only
the complete forward entry.
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


ROOT = Path(__file__).resolve().parents[2]
TOKENS = 102400
MACROBATCHES = (4096, 8192, 16384, 32768)
MINIBATCH = 4096
COMM_SMS = 40
ORDER = ("cuda", "cutedsl", "cutedsl", "cuda")
WARMUPS = 10
SAMPLES = 30
CV_LIMIT = 0.01
BOOKEND_DRIFT_LIMIT = 0.01
EQUIVALENCE_BAND = 0.01
SCHEMA = "mok_cutedsl_fwd_macrobatch_formal_v1"
EXACT3_SHA256 = {
    "mok/cutedsl/_persistent_bf16_macrobatch_experimental.py": (
        "4cba6b5f0f5eb2529dba7c963db1f27c88cfc085d9a97b53ba03e92e864d2f43"
    ),
    "benchmarks/cutedsl/bench_qwen_fwd_macrobatch_experimental.py": (
        "5cafd234d7ff779ba656e0688d3a8b901f01310f79de38e87d5fd323b625cf0c"
    ),
    "tests/test_cutedsl_persistent_bf16_macrobatch_experimental.py": (
        "9f23985bc18d409a036a96a471933617b716098f45c4fba4d6aaa71888279704"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact3_fixture():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarks.cutedsl import (
        bench_qwen_fwd_macrobatch_experimental as fixture,
    )

    return fixture


def _frozen_gate():
    for relative, expected in EXACT3_SHA256.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"exact3 source mismatch: {relative}")
    fixture = _exact3_fixture()
    fixture._frozen_gate()
    return fixture


def _sample_summary(samples: list[float]) -> dict[str, float]:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be positive finite milliseconds")
    mean = statistics.fmean(samples)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "cv": statistics.pstdev(samples) / mean,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _paired_summary(numerator: list[float], denominator: list[float]):
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired timing blocks must have equal nonzero lengths")
    samples = [
        left / right
        for left, right in zip(numerator, denominator)
    ]
    summary = _sample_summary(samples)
    return {"samples": samples, "median_speedup": summary.pop("median_ms"), **summary}


def _summarize_point(blocks: list[dict[str, object]]) -> dict[str, object]:
    if [block["backend"] for block in blocks] != list(ORDER):
        raise ValueError("timing order must be CUDA-CuTe-CuTe-CUDA")
    samples = [block["rank_max_samples_ms"] for block in blocks]
    if any(len(values) != SAMPLES for values in samples):
        raise ValueError(f"each timing block must contain {SAMPLES} samples")

    first_pair = _paired_summary(samples[0], samples[1])
    second_pair = _paired_summary(samples[3], samples[2])
    cuda_drift = abs(blocks[3]["median_ms"] / blocks[0]["median_ms"] - 1.0)
    checks = {
        "all_block_cv_le_1pct": all(block["cv"] <= CV_LIMIT for block in blocks),
        "cuda_bookend_drift_le_1pct": cuda_drift <= BOOKEND_DRIFT_LIMIT,
        "both_paired_ratio_cv_le_1pct": (
            first_pair["cv"] <= CV_LIMIT and second_pair["cv"] <= CV_LIMIT
        ),
    }
    valid = all(checks.values())
    ratios = (first_pair["median_speedup"], second_pair["median_speedup"])
    if not valid:
        classification = "INVALID_TIMING_STABILITY"
    elif all(abs(value - 1.0) <= EQUIVALENCE_BAND for value in ratios):
        classification = "EQUIVALENT"
    elif all(value > 1.0 + EQUIVALENCE_BAND for value in ratios):
        classification = "CUTEDSL_FASTER"
    elif all(value < 1.0 - EQUIVALENCE_BAND for value in ratios):
        classification = "CUDA_FASTER"
    else:
        classification = "INCONCLUSIVE"

    return {
        "valid": valid,
        "classification": classification,
        "stability_checks": checks,
        "cuda_bookend_abs_drift": cuda_drift,
        "paired_cuda_over_cutedsl": {
            "first_bookend": first_pair,
            "second_bookend": second_pair,
        },
        "backend_median_ms": {
            "cuda": statistics.median([*samples[0], *samples[3]]),
            "cutedsl": statistics.median([*samples[1], *samples[2]]),
        },
    }


def _select_own_best(points: dict[str, dict[str, object]]) -> dict[str, object]:
    complete = set(points) == {str(value) for value in MACROBATCHES}
    if not complete or not all(point["summary"]["valid"] for point in points.values()):
        return {
            "status": "N/A_incomplete_or_invalid_four_point_sweep",
            "cuda": "N/A",
            "cutedsl": "N/A",
        }

    result: dict[str, object] = {"status": "AVAILABLE"}
    for backend in ("cuda", "cutedsl"):
        medians = {
            macro: points[str(macro)]["summary"]["backend_median_ms"][backend]
            for macro in MACROBATCHES
        }
        best = min(medians, key=medians.__getitem__)
        threshold = medians[best] * (1.0 + EQUIVALENCE_BAND)
        result[backend] = {
            "best_macrobatch": best,
            "co_best_macrobatches_within_1pct": [
                macro for macro in MACROBATCHES if medians[macro] <= threshold
            ],
            "median_ms_by_macrobatch": medians,
        }
    return result


def static_self_test() -> dict[str, object]:
    _frozen_gate()
    assert MACROBATCHES == (MINIBATCH, 2 * MINIBATCH, 4 * MINIBATCH, 8 * MINIBATCH)
    assert ORDER == ("cuda", "cutedsl", "cutedsl", "cuda")
    assert WARMUPS == 10 and SAMPLES == 30
    assert CV_LIMIT == BOOKEND_DRIFT_LIMIT == EQUIVALENCE_BAND == 0.01
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
        "correctness": {
            "status": "N/A_local_static_mode",
            "required_records_per_macrobatch": 48,
            "comparisons": ("cuda_a_vs_cuda_b", "cuda_a_vs_cutedsl", "cuda_b_vs_cutedsl"),
        },
        "timing": {
            "status": "N/A_local_static_mode",
            "order": ORDER,
            "boundary": "CUDA Events around complete forward entry only",
            "warmups_per_block": WARMUPS,
            "samples_per_block": SAMPLES,
            "aggregation": "same-index EP8 rank maximum",
        },
        "own_best": "N/A_until_all_four_device_points_are_valid",
    }


def _load_runtime():
    global torch, dist, fixture, accepted_bench, functional
    fixture = _frozen_gate()
    fixture._load_runtime()
    torch, dist = fixture.torch, fixture.dist
    accepted_bench, functional = fixture.accepted_bench, fixture.functional


def _correctness_point(macro, workspace, schedule, inputs, local_rank, device):
    cuda_config = fixture._config(macro, "cuda")
    cutedsl_config = fixture._config(macro, "cutedsl")
    cuda_a = accepted_bench.checked_forward_backward(
        cuda_config, workspace, schedule, inputs, local_rank
    )
    cuda_b = accepted_bench.checked_forward_backward(
        cuda_config, workspace, schedule, inputs, local_rank
    )
    self_control = fixture._compare_legs(
        f"B{macro}/cuda_a_vs_cuda_b", cuda_a, cuda_b, device
    )
    if len(self_control["records"]) != 16 or not self_control["pass"]:
        raise AssertionError(f"B={macro} CUDA self-control failed")
    cutedsl = fixture._candidate_leg(
        cutedsl_config, workspace, schedule, inputs, local_rank
    )
    comparisons = {
        "cuda_a_vs_cuda_b": self_control,
        "cuda_a_vs_cutedsl": fixture._compare_legs(
            f"B{macro}/cuda_a_vs_cutedsl", cuda_a, cutedsl, device
        ),
        "cuda_b_vs_cutedsl": fixture._compare_legs(
            f"B{macro}/cuda_b_vs_cutedsl", cuda_b, cutedsl, device
        ),
    }
    record_count = sum(len(item["records"]) for item in comparisons.values())
    passed = record_count == 48 and all(item["pass"] for item in comparisons.values())
    if not passed:
        raise AssertionError(f"B={macro} 48-record correctness gate failed")
    return {"record_count": record_count, "comparisons": comparisons, "pass": True}


def _measure_block(backend, run, device, order_index):
    for _ in range(WARMUPS):
        output, context = run()
        output = context = None
    dist.barrier(async_op=True).block_current_stream()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(SAMPLES)
    ]
    for start, end in events:
        start.record()
        output, context = run()
        end.record()
        output = context = None
    torch.cuda.synchronize(device)
    rank_max = torch.tensor(
        [start.elapsed_time(end) for start, end in events],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    samples = rank_max.cpu().tolist()
    return {
        "order_index": order_index,
        "backend": backend,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "aggregation": "same-index EP8 rank maximum",
        "rank_max_samples_ms": samples,
    } | _sample_summary(samples)


def _timing_point(macro, workspace, schedule, inputs, device):
    cuda_config = fixture._config(macro, "cuda")
    cutedsl_config = fixture._config(macro, "cutedsl")
    runs = {
        "cuda": lambda: accepted_bench.forward(
            cuda_config, workspace, schedule, inputs
        ),
        "cutedsl": lambda: fixture._candidate_forward(
            cutedsl_config, workspace, schedule, inputs
        ),
    }
    # Both paths are compiled/materialized before any Event is recorded.
    for backend in ("cuda", "cutedsl"):
        output, context = runs[backend]()
        output = context = None
    torch.cuda.synchronize(device)
    blocks = [
        _measure_block(backend, runs[backend], device, order_index)
        for order_index, backend in enumerate(ORDER)
    ]
    return {"blocks": blocks, "summary": _summarize_point(blocks)}


def run_device() -> dict[str, object]:
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

    base = fixture._config(MACROBATCHES[-1], "cutedsl")
    inputs, workspace, schedule = accepted_bench.make_case(TOKENS, base, rank, device)
    rows = accepted_bench.routed_rows(schedule, device)
    correctness = {
        str(macro): _correctness_point(
            macro, workspace, schedule, inputs, local_rank, device
        )
        for macro in MACROBATCHES
    }
    points = {
        str(macro): _timing_point(macro, workspace, schedule, inputs, device)
        for macro in MACROBATCHES
    }
    result = {
        "status": "PASS",
        "schema": SCHEMA,
        "provenance": {
            "exact3_sha256": EXACT3_SHA256,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "source_commit": os.environ.get("MOK_SOURCE_COMMIT", "not provided"),
        },
        "shape": static_self_test()["shape"],
        "padded_routed_rows_by_rank": rows,
        "fixed": {"minibatch": MINIBATCH, "fwd_num_comm_sms": COMM_SMS},
        "correctness": correctness,
        "timing": {
            "boundary": "CUDA Events around complete forward entry only",
            "excluded": "compile, materialization, schedule construction, and backward",
            "order_per_macrobatch": ORDER,
            "warmups_per_block": WARMUPS,
            "samples_per_block": SAMPLES,
            "aggregation": "same-index EP8 rank maximum",
            "points": points,
        },
        "own_best": _select_own_best(points),
        "decision_scope": "formal four-point evidence only; CUDA remains the default backend",
    }
    functional.clear_workspace_cache()
    fixture.candidate.clear_experimental_macrobatch_cache()
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
        destination = Path(args.output).expanduser().resolve()
        _atomic_write(destination, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
