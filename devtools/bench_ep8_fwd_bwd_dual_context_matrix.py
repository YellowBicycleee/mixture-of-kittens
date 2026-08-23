#!/usr/bin/env python3
"""Bounded EP8 baseline/OLD-tuned/NEW-tuned FWD+BWD matrix.

Baseline and OLD-tuned run forward with capacity C=row B.  NEW retains one
workload-sized forward context (kernel B=C).  Every backward uses ring B=row B.
Each final number is a rank-max CUDA-Event sample.  The bandwidth is a derived
useful-payload rate, not an NCU or NVLink-counter measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


OLD = "old_macrobatch"
NEW = "new_ep8_full_context"
BASELINE = "old_default_baseline"
IMPLEMENTATIONS = (OLD, NEW)
FWD, BWD, FWD_BWD = "fwd", "bwd", "fwd_bwd"
SCOPES = (FWD, BWD, FWD_BWD)

NUM_EXPERTS, TOPK = 512, 10
HIDDEN_DIM, INTERMEDIATE_DIM = 4_096, 1_024
DEFAULT_OTHER_COMM_SMS = 36
BASELINE_MINIBATCH_SIZE, BASELINE_FWD_COMM_SMS, BASELINE_BWD_COMM_SMS = 4_096, 40, 28
MINIBATCH_ALIGNMENT = 256
MINIBATCH_TARGETS = (4_096, 16_384, 45_056, 94_208)
FALLBACK_MINIBATCH_TARGETS = (1_024, 2_048, 8_192, 32_768)
COARSE_COMM_SMS = {FWD: (16, 24, 32, 40, 48), BWD: (28, 40, 50, 54, 56, 62)}
REFINE_COMM_OFFSETS = (-4, -2, 0, 2, 4)
TOP_PHASE_CANDIDATES = 2

SCREEN_WARMUP, SCREEN_SAMPLES = 1, 2
REFINE_WARMUP, REFINE_SAMPLES = 1, 3
JOINT_WARMUP, JOINT_SAMPLES = 1, 2
FINAL_WARMUP_PAIRS, FINAL_TIMED_PAIRS = 5, 20

GRADIENT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)
SCHEMA = "mok_ep8_dual_context_fwd_bwd_matrix_v4"
DRY_RUN_SOURCE_OVERLAY_SENTINEL = "synthetic-dry-run-no-formal-checkpoint"
EXPECTED_RUNTIME_IMPORTS = (
    "mok.functional", "mok.ops", "mok.ops._C", "benchmarks.utils", "tests.utils",
)


@dataclass(frozen=True)
class Workload:
    key: str
    tokens_per_rank: int
    context_size: int
    macrobatch_sizes: tuple[int, ...]


WORKLOADS = {
    "20k": Workload(
        "20k",
        20_480,
        217_088,
        (
            4_096, 8_192, 12_288, 16_384, 20_480, 28_672,
            32_768, 36_864, 40_960, 61_440, 65_536, 69_632,
            81_920, 102_400, 106_496, 110_592, 114_688, 126_976,
            131_072, 135_168, 163_840, 196_608, 212_992, 217_088,
        ),
    ),
    "100k": Workload(
        "100k",
        102_400,
        1_036_288,
        (
            4_096, 8_192, 12_288, 16_384, 20_480, 28_672,
            32_768, 36_864, 61_440, 65_536, 69_632, 126_976,
            131_072, 135_168, 163_840, 196_608, 229_376, 258_048,
            262_144, 516_096, 520_192, 1_028_096, 1_032_192, 1_036_288,
        ),
    ),
}
WORKLOAD_SEED_INDEX = {"20k": 0, "100k": 1}


@dataclass(frozen=True)
class Cell:
    scope: str
    implementation: str
    row_macrobatch_size: int
    kernel_macrobatch_size: int
    minibatch_size: int
    num_comm_sms: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Cell":
        return cls(
            str(value["scope"]),
            str(value["implementation"]),
            int(value["row_macrobatch_size"]),
            int(value["kernel_macrobatch_size"]),
            int(value["minibatch_size"]),
            int(value["num_comm_sms"]),
        )


def legal_minibatches(macrobatch_size: int) -> tuple[int, ...]:
    if macrobatch_size <= 0 or macrobatch_size % MINIBATCH_ALIGNMENT:
        raise ValueError("B must be positive and 256-row aligned")
    return tuple(
        value
        for value in range(MINIBATCH_ALIGNMENT, macrobatch_size + 1, MINIBATCH_ALIGNMENT)
        if macrobatch_size % value == 0
    )


def short_minibatches(macrobatch_size: int) -> tuple[int, ...]:
    """At most four representative production-legal divisors of B."""

    legal = tuple(value for value in legal_minibatches(macrobatch_size) if value >= 1_024)
    if not legal:
        legal = legal_minibatches(macrobatch_size)
    selected: set[int] = set()
    for target in MINIBATCH_TARGETS + FALLBACK_MINIBATCH_TARGETS:
        selected.add(min(legal, key=lambda value: (abs(value - target), value)))
        if len(selected) == min(4, len(legal)):
            break
    return tuple(sorted(selected))


def phase_screen_cells(scope: str, implementation: str, row_b: int, kernel_b: int, comm_sms_max: int) -> list[Cell]:
    return [Cell(scope, implementation, row_b, kernel_b, mini, comm) for mini in short_minibatches(kernel_b) for comm in COARSE_COMM_SMS[scope] if comm <= comm_sms_max]
def phase_refine_cells(scope: str, implementation: str, row_b: int, kernel_b: int, winners: Iterable[Cell], comm_sms_max: int) -> list[Cell]:
    cells = (Cell(scope, implementation, row_b, kernel_b, winner.minibatch_size, comm) for winner in winners for comm in sorted({winner.num_comm_sms + offset for offset in REFINE_COMM_OFFSETS if 2 <= winner.num_comm_sms + offset <= comm_sms_max}))
    return list(dict.fromkeys(cells))
def abba_order(pair_index: int) -> tuple[str, str]:
    if pair_index < 0:
        raise ValueError("pair index must be nonnegative")
    return IMPLEMENTATIONS if pair_index % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))


def final_order(pair_index: int) -> tuple[str, str, str]:
    """Keep OLD/NEW paired while giving the default baseline equal sampling."""

    old_or_new, new_or_old = abba_order(pair_index)
    return old_or_new, BASELINE, new_or_old


def useful_payload_bytes(remote_routes: int, hidden_dim: int, scope: str) -> int:
    if remote_routes < 0 or hidden_dim <= 0:
        raise ValueError("invalid route count or hidden size")
    fwd = remote_routes * 4 * hidden_dim
    bwd = remote_routes * (4 * hidden_dim + 8)
    if scope == FWD:
        return fwd
    if scope == BWD:
        return bwd
    if scope == FWD_BWD:
        return fwd + bwd
    raise ValueError(f"unknown scope {scope}")


def effective_payload_gbps(payload_bytes: int, latency_ms: float) -> float:
    if payload_bytes < 0 or not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("payload/latency must be nonnegative and finite/positive")
    return payload_bytes / (latency_ms * 1_000_000.0)


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("quantile requires samples and q in [0,1]")
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def sample_summary(samples: list[float], keep_samples: bool = True) -> dict[str, Any]:
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError(f"invalid latency samples: {samples}")
    result = {
        "median_ms": statistics.median(samples),
        "p10_ms": quantile(samples, 0.1),
        "p90_ms": quantile(samples, 0.9),
    }
    if keep_samples:
        result["rank_max_samples_ms"] = samples
    return result


def stable_paired_win(delta_percent: list[float]) -> bool:
    return statistics.median(delta_percent) < 0 and quantile(delta_percent, 0.9) < 0


def selected_workloads(name: str) -> tuple[Workload, ...]:
    return (WORKLOADS["20k"], WORKLOADS["100k"]) if name == "all" else (WORKLOADS[name],)


def selected_macrobatches(workload: Workload, raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return workload.macrobatch_sizes
    requested = {int(value) for value in raw.split(",") if value.strip()}
    unknown = sorted(requested - set(workload.macrobatch_sizes))
    if not requested or unknown:
        raise ValueError(f"invalid {workload.key} --macrobatches; unknown={unknown}")
    return tuple(value for value in workload.macrobatch_sizes if value in requested)


def workload_row_seed(base_seed: int, workload_key: str, row_b: int) -> int:
    return base_seed + WORKLOAD_SEED_INDEX[workload_key] * 10_000_000 + row_b


def json_normalized(value: Any) -> Any:
    """Use the exact representation produced by a JSON checkpoint round trip."""

    return json.loads(json.dumps(value, sort_keys=True))


def validate_source_overlay_sha256(value: str | None, dry_run: bool) -> str:
    if dry_run:
        return DRY_RUN_SOURCE_OVERLAY_SENTINEL
    if not (
        isinstance(value, str) and len(value) == 64
        and value == value.lower() and set(value) <= set("0123456789abcdef")
    ):
        raise ValueError("MOK_SOURCE_OVERLAY_SHA256 must be lowercase 64-hex")
    return value


def source_overlay_sha256(args: argparse.Namespace) -> str:
    return validate_source_overlay_sha256(
        os.environ.get("MOK_SOURCE_OVERLAY_SHA256"), args.dry_run
    )


def validate_gpu_entry(args: argparse.Namespace) -> str:
    if args.dry_run:
        raise ValueError("dry-run cannot enter the GPU/checkpoint path")
    return source_overlay_sha256(args)


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return json_normalized({
        "minibatch_targets": MINIBATCH_TARGETS,
        "max_minibatches_per_tune": 4,
        "coarse_comm_sms_by_scope": COARSE_COMM_SMS,
        "refine_comm_offsets": REFINE_COMM_OFFSETS,
        "screen": {"warmup": args.screen_warmup, "samples": args.screen_samples},
        "refine": {"warmup": args.refine_warmup, "samples": args.refine_samples},
        "phase_finalists": TOP_PHASE_CANDIDATES,
        "combined_joint": {
            "candidates": "phase top2 FWD x phase top2 BWD per implementation",
            "warmup": args.joint_warmup,
            "samples": args.joint_samples,
        },
        "final_abba": {
            "warmup_pairs": args.final_warmup_pairs,
            "timed_pairs": args.final_timed_pairs,
            "order": "OLD/NEW ABBA with baseline between each pair",
            "sample_reduction": "per-sample maximum across EP8 ranks",
        },
        "old_default_baseline": {
            "schedule": "old macrobatch", "minibatch_size": BASELINE_MINIBATCH_SIZE,
            "fwd_num_comm_sms": BASELINE_FWD_COMM_SMS,
            "bwd_num_comm_sms": BASELINE_BWD_COMM_SMS,
        },
        "new_terminal_policy": {
            "condition": "row macrobatch B equals retained context C",
            "selected_path": "legacy_terminal_fallback",
            "candidate_evidence": "fine EP8 phase and combined tuning remain in raw JSON",
        },
        "bounded": True,
        "edge_expansion": False,
    })


def finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Pure, idempotent aggregate finalization for normal and resume paths."""

    result = dict(report)
    rows = result.get("rows", [])
    workload = WORKLOADS[str(result["shape"]["workload"])]
    complete = len(rows) == 24 and {
        int(row["macrobatch_size"]) for row in rows
    } == set(workload.macrobatch_sizes)
    result["state"] = "complete" if complete else "partial"
    result["all_rows_stable_new_win"] = {
        scope: complete and all(row["metrics"][scope]["stable_new_win"] for row in rows)
        for scope in SCOPES
    }
    result["all_rows_stable_new_vs_baseline"] = {
        scope: complete and all(
            row["metrics"][scope]["new_vs_baseline_stable_win"] for row in rows
        ) for scope in SCOPES
    }
    return result


def call_budget(workload: Workload, args: argparse.Namespace) -> dict[str, int]:
    """Exact default/B300 plan; counts complete forward/backward API calls."""

    def tune_trials(kernel_b: int, scope: str) -> int:
        screen_cells = len(short_minibatches(kernel_b)) * len(COARSE_COMM_SMS[scope])
        refine_cells = TOP_PHASE_CANDIDATES * len(REFINE_COMM_OFFSETS)
        return screen_cells * (args.screen_warmup + args.screen_samples) + refine_cells * (
            args.refine_warmup + args.refine_samples
        )

    new_fwd = tune_trials(workload.context_size, FWD)
    old_fwd = sum(tune_trials(value, FWD) for value in workload.macrobatch_sizes)
    # Every BWD trial includes one preparation FWD outside the event plus BWD.
    backward = 2 * sum(2 * tune_trials(value, BWD) for value in workload.macrobatch_sizes)
    joint = len(workload.macrobatch_sizes) * 2 * 4 * (
        args.joint_warmup + args.joint_samples
    ) * 2
    correctness = 2 + len(workload.macrobatch_sizes) * 2 * 2 * 2
    final_per_row = (
        2 * (args.final_warmup_pairs + args.final_timed_pairs)
        + 4 * (args.final_warmup_pairs + args.final_timed_pairs)
        + 4 * (args.final_warmup_pairs + args.final_timed_pairs)
    )
    final = len(workload.macrobatch_sizes) * final_per_row
    baseline_final = len(workload.macrobatch_sizes) * (
        args.final_warmup_pairs + args.final_timed_pairs
    ) * 5  # FWD=1 call, BWD=2, direct combined=2.
    result = {
        "forward_tuning": new_fwd + old_fwd,
        "backward_tuning_including_preparation_forward": backward,
        "combined_top2x2": joint,
        "correctness_reference_phase_and_joint": correctness,
        "old_tuned_new_tuned_final_w5_n20": final,
        "old_default_baseline_final_w5_n20": baseline_final,
    }
    result["total"] = sum(result.values())
    return result


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    workloads = []
    for workload in selected_workloads(args.workload):
        selected = selected_macrobatches(workload, args.macrobatches)
        workloads.append(
            {
                "workload": asdict(workload),
                "selected_macrobatch_sizes": selected,
                "matrix_row_count": len(selected),
                "minibatch_candidates_by_B": {
                    str(value): short_minibatches(value) for value in selected
                },
                "row_seed_base_by_B": {
                    str(value): workload_row_seed(args.seed, workload.key, value)
                    for value in selected
                },
                "new_fwd_kernelB_C": workload.context_size,
                "old_default_baseline": "old schedule, mini4096, fwdSM40, bwdSM28",
                "new_fwd_minibatch_candidates": short_minibatches(workload.context_size),
                "full_24_row_default_call_budget": call_budget(workload, args),
            }
        )
    return {
        "mode": "dry_run_no_torch_no_gpu",
        "source_overlay_sha256": source_overlay_sha256(args),
        "shape": {
            "model": "Qwen-shaped synthetic",
            "ep": 8,
            "experts": NUM_EXPERTS,
            "topk": TOPK,
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "precision": "MXFP8",
        },
        "protocol": protocol(args),
        "timing": {
            FWD: "CUDA Events around complete functional.forward",
            BWD: "CUDA Events around functional.backward; fresh forward excluded",
            FWD_BWD: "one CUDA-Event interval around direct forward then backward",
        },
        "bandwidth": "derived aggregate and per-GPU useful payload; not link counters",
        "checkpoint": "disabled: dry-run never writes a formal checkpoint",
        "workloads": workloads,
    }


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_import_provenance(
    repo_root: Path, resolved_paths: dict[str, Any]
) -> dict[str, Any]:
    """Fail if any benchmark/runtime module resolves outside this checkout."""

    if set(resolved_paths) != set(EXPECTED_RUNTIME_IMPORTS):
        raise RuntimeError("runtime import inventory mismatch")
    root = repo_root.resolve(strict=True)
    verified = {}
    for label in EXPECTED_RUNTIME_IMPORTS:
        raw_path = resolved_paths[label]
        if not isinstance(raw_path, (str, os.PathLike)):
            raise RuntimeError(f"{label} has no resolved file")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(f"{label} did not resolve to a file: {path}")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"{label} resolved outside repo: {path}") from error
        verified[label] = str(path)
    extension = Path(verified["mok.ops._C"])
    return {
        "runtime_resolved_paths": verified,
        "extension_absolute_path": str(extension),
        "extension_sha256": sha256(extension.read_bytes()),
    }


def source_provenance(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root).strip().decode()
        diff = subprocess.check_output(("git", "diff", "--binary", "HEAD"), cwd=root)
        dirty = bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=root).strip())
    except (OSError, subprocess.CalledProcessError):
        head, diff, dirty = "N/A", b"", "N/A"
    digest = hashlib.sha256()
    paths = [root / name for name in (
        "Makefile", "setup.py", "pyproject.toml", "benchmarks/utils.py", "tests/utils.py",
    )] + [Path(__file__).resolve()]
    for directory in (root / "csrc", root / "mok"):
        paths.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return {
        "argv": sys.argv,
        "tuning_seed": args.seed,
        "input_seed": "tests.utils.generate_inputs fixed 1234 + global_rank",
        "git_head": head,
        "git_dirty": dirty,
        "git_tracked_diff_sha256": sha256(diff),
        "source_tree_sha256": digest.hexdigest(),
        "source_overlay_sha256": source_overlay_sha256(args),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    device_keys = ("name", "compute_capability", "sm_count")
    return json_normalized(
        {
            "source_tree_sha256": provenance["source_tree_sha256"],
            "source_overlay_sha256": provenance["source_overlay_sha256"],
            "torch_version": provenance["torch_version"],
            "cuda_version": provenance["cuda_version"],
            "extension_absolute_path": provenance.get("extension_absolute_path", "N/A"),
            "extension_sha256": provenance.get("extension_sha256", "N/A"),
            "devices": [
                {key: device[key] for key in device_keys} for device in provenance["devices"]
            ],
        }
    )


def load_or_create_checkpoint(
    path: Path,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    restart: bool,
) -> dict[str, Any]:
    expected_protocol = json_normalized(protocol(args))
    if path.exists() and not restart:
        loaded = json.loads(path.read_text())
        if loaded.get("schema") != SCHEMA:
            raise ValueError("checkpoint schema mismatch; use --restart")
        if checkpoint_identity(loaded.get("provenance", {})) != checkpoint_identity(provenance):
            raise ValueError("checkpoint source/runtime/device identity mismatch; use --restart")
        if json_normalized(loaded.get("protocol")) != expected_protocol:
            raise ValueError("checkpoint timing protocol mismatch; use --restart")
        if loaded.get("provenance", {}).get("tuning_seed") != args.seed:
            raise ValueError("checkpoint tuning seed mismatch; use --restart")
        return loaded
    return {
        "schema": SCHEMA,
        "provenance": provenance,
        "protocol": expected_protocol,
        "reports": [],
    }


def run_gpu_suite(args: argparse.Namespace) -> dict[str, Any]:
    validate_gpu_entry(args)
    import torch
    import torch.distributed as dist

    import benchmarks.utils as benchmark_utils
    import tests.utils as test_utils
    from mok import functional, ops

    get_num_local_experts, init_distributed = (
        benchmark_utils.get_num_local_experts, benchmark_utils.init_distributed,
    )
    MXFP8_TOLERANCE, check_correctness, generate_inputs = (
        test_utils.MXFP8_TOLERANCE, test_utils.check_correctness, test_utils.generate_inputs,
    )
    runtime_imports = runtime_import_provenance(
        Path(__file__).resolve().parents[1],
        {
            "mok.functional": functional.__file__, "mok.ops": ops.__file__,
            "mok.ops._C": ops._C.__file__, "benchmarks.utils": benchmark_utils.__file__,
            "tests.utils": test_utils.__file__,
        },
    )

    rank = -1
    output_path = Path(args.output)
    suite: dict[str, Any] = {}

    def rank0_action(action: Callable[[], Any]) -> Any:
        message: list[Any] = [None]
        if rank == 0:
            try:
                message[0] = {"ok": True, "value": action()}
            except Exception as error:  # propagate I/O failures to every rank
                message[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(message, src=0)
        if not message[0]["ok"]:
            raise RuntimeError(f"rank-0 checkpoint I/O failed: {message[0]['error']}")
        return message[0].get("value")

    def checkpoint() -> None:
        rank0_action(lambda: atomic_write_json(output_path, suite))

    try:
        rank, world_size, device = init_distributed()
        if world_size != 8:
            raise ValueError(f"dual-context matrix requires EP8, got EP{world_size}")
        properties = torch.cuda.get_device_properties(device)
        comm_sms_max = ((properties.multi_processor_count - 1) // 2) * 2
        device_local = {
            "rank": rank,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "sm_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "runtime_imports": runtime_imports,
        }
        devices: list[Any] = [None] * world_size
        dist.all_gather_object(devices, device_local)
        if any(item["runtime_imports"] != runtime_imports for item in devices):
            raise RuntimeError("runtime import provenance differs across ranks")
        def build_provenance() -> dict[str, Any]:
            value = source_provenance(args)
            value.update({
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda), "devices": devices,
                **runtime_imports,
            })
            return value

        current_provenance = rank0_action(build_provenance)

        suite = rank0_action(
            lambda: load_or_create_checkpoint(
                output_path, args, current_provenance, args.restart
            )
        )
        checkpoint()

        def make_config(cell: Cell) -> functional.MoKConfig:
            return functional.MoKConfig(
                fwd_num_comm_sms=cell.num_comm_sms if cell.scope == FWD else DEFAULT_OTHER_COMM_SMS,
                bwd_num_comm_sms=cell.num_comm_sms if cell.scope == BWD else DEFAULT_OTHER_COMM_SMS,
                minibatch_size=cell.minibatch_size,
                macrobatch_size=cell.kernel_macrobatch_size,
                bwd_schedule=(
                    "macrobatch" if cell.implementation in (BASELINE, OLD) else "minibatch"
                ),
            )

        for workload in selected_workloads(args.workload):
            selected_b = selected_macrobatches(workload, args.macrobatches)
            report = next(
                (item for item in suite["reports"] if item["shape"]["workload"] == workload.key),
                None,
            )
            completed = {int(row["macrobatch_size"]) for row in report.get("rows", [])} if report else set()
            if report and set(selected_b) <= completed:
                report.update(finalize_report(report))
                checkpoint()
                continue

            num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
            inputs = generate_inputs(
                rank, device, NUM_EXPERTS, num_local_experts, TOPK,
                workload.tokens_per_rank, HIDDEN_DIM, INTERMEDIATE_DIM,
            )
            (
                x, topk_experts, router_weights, w_shared_gate, w_shared_up,
                w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output,
            ) = inputs
            w_gate_mxfp8 = ops.mxfp8_quantize(w_routed_gate, True, True)
            w_up_mxfp8 = ops.mxfp8_quantize(w_routed_up, True, True)
            w_down_mxfp8 = ops.mxfp8_quantize(w_routed_down, True, True)
            setup = Cell(FWD, OLD, workload.context_size, workload.context_size, 4_096, 36)
            workspace = functional.get_workspace(
                make_config(setup), dist.group.WORLD, device=device,
                num_local_tokens=workload.tokens_per_rank, hidden_size=HIDDEN_DIM, topk=TOPK,
            )
            schedule = functional.build_schedule(
                workspace, make_config(setup), topk_experts,
                num_local_experts=num_local_experts,
            )

            padded_rows = int(schedule.num_tokens.item())
            peers = schedule.peer_rank[:padded_rows]
            local_counts = torch.tensor(
                [padded_rows, int((peers >= 0).sum()), int(((peers >= 0) & (peers != rank)).sum())],
                dtype=torch.int64, device=device,
            )
            gathered = [torch.empty_like(local_counts) for _ in range(world_size)]
            dist.all_gather(gathered, local_counts)
            counts = torch.stack(gathered).cpu().tolist()
            padded_by_rank = [int(value[0]) for value in counts]
            valid_by_rank = [int(value[1]) for value in counts]
            remote_by_rank = [int(value[2]) for value in counts]
            aggregate_remote = sum(remote_by_rank)
            if sum(valid_by_rank) != world_size * workload.tokens_per_rank * TOPK:
                raise RuntimeError("valid route count mismatch")
            if max(padded_by_rank) > workload.context_size:
                raise RuntimeError("retained context capacity is below padded rows")

            def run_forward(cell: Cell) -> tuple[Any, Any]:
                return functional.forward(
                    make_config(cell), workspace, schedule, x, router_weights,
                    w_shared_gate, w_shared_up, w_shared_down,
                    w_gate_mxfp8[:2], w_up_mxfp8[:2], w_down_mxfp8[:2],
                )

            def run_backward(cell: Cell, context: Any) -> tuple[Any, ...]:
                return functional.backward(
                    make_config(cell), workspace, schedule, context, d_output, x,
                    router_weights, w_shared_gate, w_shared_up, w_shared_down,
                    w_gate_mxfp8, w_up_mxfp8, w_down_mxfp8[2:],
                )

            def run_once(scope: str, fwd_cell: Cell, bwd_cell: Cell | None) -> None:
                if scope == FWD:
                    run_forward(fwd_cell)
                elif scope == BWD:
                    _, context = run_forward(fwd_cell)
                    run_backward(bwd_cell, context)
                elif scope == FWD_BWD:
                    _, context = run_forward(fwd_cell)
                    run_backward(bwd_cell, context)
                else:
                    raise ValueError(scope)

            def pre_event_barrier() -> None:
                dist.barrier(async_op=True).block_current_stream()

            def rank_max(local_samples: list[float]) -> list[float]:
                tensor = torch.tensor(local_samples, dtype=torch.float64, device=device)
                all_samples = [torch.empty_like(tensor) for _ in range(world_size)]
                dist.all_gather(all_samples, tensor)
                return torch.stack(all_samples).max(dim=0).values.cpu().tolist()

            def measure(
                scope: str, fwd_cell: Cell, bwd_cell: Cell | None, warmup: int, timed: int
            ) -> list[float]:
                for _ in range(warmup):
                    run_once(scope, fwd_cell, bwd_cell)
                local_samples = []
                for _ in range(timed):
                    context = None
                    if scope == BWD:
                        _, context = run_forward(fwd_cell)
                    pre_event_barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    if scope == FWD:
                        run_forward(fwd_cell)
                    elif scope == BWD:
                        run_backward(bwd_cell, context)
                    else:
                        _, context = run_forward(fwd_cell)
                        run_backward(bwd_cell, context)
                    end.record()
                    end.synchronize()
                    local_samples.append(start.elapsed_time(end))
                return rank_max(local_samples)

            def tune(
                scope: str,
                implementation: str,
                row_b: int,
                kernel_b: int,
                preparation_fwd: Cell | None,
            ) -> tuple[Cell, list[Cell], dict[str, Any]]:
                candidates = phase_screen_cells(scope, implementation, row_b, kernel_b, comm_sms_max)
                if not candidates:
                    raise RuntimeError(f"no legal coarse comm-SM value for {scope}")
                seed = workload_row_seed(args.seed, workload.key, row_b)
                seed += (1_000_000 if implementation == NEW else 0) + (2_000_000 if scope == BWD else 0)
                random.Random(seed).shuffle(candidates)

                def evaluate(cell: Cell, warmup: int, samples: int) -> dict[str, Any]:
                    fwd_cell = cell if scope == FWD else preparation_fwd
                    values = measure(scope, fwd_cell, cell if scope == BWD else None, warmup, samples)
                    return {"cell": asdict(cell), **sample_summary(values, keep_samples=False)}

                screen = [evaluate(cell, args.screen_warmup, args.screen_samples) for cell in candidates]
                best_by_mini: dict[int, dict[str, Any]] = {}
                for item in screen:
                    mini = int(item["cell"]["minibatch_size"])
                    if mini not in best_by_mini or item["median_ms"] < best_by_mini[mini]["median_ms"]:
                        best_by_mini[mini] = item
                mini_winners = sorted(best_by_mini.values(), key=lambda item: item["median_ms"])
                winner_cells = [Cell.from_dict(item["cell"]) for item in mini_winners[:TOP_PHASE_CANDIDATES]]
                refine_cells = phase_refine_cells(scope, implementation, row_b, kernel_b, winner_cells, comm_sms_max)
                random.Random(seed + 1).shuffle(refine_cells)
                refine = [evaluate(cell, args.refine_warmup, args.refine_samples) for cell in refine_cells]
                ordered = sorted(refine, key=lambda item: item["median_ms"])
                finalists = [Cell.from_dict(item["cell"]) for item in ordered[:TOP_PHASE_CANDIDATES]]
                result = {
                    "winner": asdict(finalists[0]),
                    "top2": [asdict(cell) for cell in finalists],
                    "screen": screen,
                    "refine": refine,
                    "bounded_no_edge_expansion": True,
                }
                if rank == 0:
                    print("DUAL_CONTEXT_TUNE " + json.dumps(result, sort_keys=True), flush=True)
                return finalists[0], finalists, result

            def tune_combined(
                implementation: str, fwd_top2: list[Cell], bwd_top2: list[Cell], seed: int
            ) -> tuple[Cell, Cell, dict[str, Any]]:
                pairs = [(fwd, bwd) for fwd in fwd_top2 for bwd in bwd_top2]
                random.Random(seed).shuffle(pairs)
                summaries = []
                for fwd_cell, bwd_cell in pairs:
                    values = measure(
                        FWD_BWD, fwd_cell, bwd_cell, args.joint_warmup, args.joint_samples
                    )
                    summaries.append(
                        {"fwd": asdict(fwd_cell), "bwd": asdict(bwd_cell), **sample_summary(values, False)}
                    )
                winner = min(summaries, key=lambda item: item["median_ms"])
                return Cell.from_dict(winner["fwd"]), Cell.from_dict(winner["bwd"]), {
                    "winner": winner, "candidates": summaries,
                }

            reference_fwd = Cell(FWD, OLD, workload.context_size, workload.context_size, 4_096, 36)
            reference_bwd = Cell(BWD, OLD, workload.context_size, workload.context_size, 4_096, 36)
            reference_output, reference_context = run_forward(reference_fwd)
            reference_output = reference_output.clone()
            reference_gradients = tuple(
                value.clone() for value in run_backward(reference_bwd, reference_context)
            )
            reference_context = None

            if report is None:
                new_fwd, new_fwd_top2, new_fwd_tuning = tune(
                    FWD, NEW, workload.context_size, workload.context_size, None
                )
                report = {
                    "state": "in_progress",
                    "protocol": protocol(args),
                    "provenance": current_provenance,
                    "provenance_runs": [current_provenance],
                    "shape": {
                        "model": "Qwen-shaped synthetic", "workload": workload.key,
                        "tokens_per_rank": workload.tokens_per_rank,
                        "retained_context_C": workload.context_size,
                        "ep": world_size, "experts": NUM_EXPERTS,
                        "local_experts": num_local_experts, "topk": TOPK,
                        "hidden_dim": HIDDEN_DIM, "intermediate_dim": INTERMEDIATE_DIM,
                        "precision": "MXFP8",
                    },
                    "padded_rows_by_rank": padded_by_rank,
                    "valid_routes_by_rank": valid_by_rank,
                    "remote_valid_routes_by_rank": remote_by_rank,
                    "aggregate_remote_valid_routes": aggregate_remote,
                    "timing_boundary": {
                        FWD: "CUDA Events around complete functional.forward",
                        BWD: "CUDA Events around backward; fresh forward and schedule excluded",
                        FWD_BWD: "one CUDA-Event interval around direct forward then backward",
                    },
                    "bandwidth_definition": {
                        "label": "derived useful cross-GPU payload GB/s",
                        "per_gpu": "aggregate payload divided by EP8 (average per GPU)",
                        "excludes": "local/padded routes, replay, and protocol overhead",
                        "not_measured_by": "NCU or NVLink hardware counters",
                    },
                    "new_forward_tuned_once": asdict(new_fwd),
                    "new_forward_top2": [asdict(cell) for cell in new_fwd_top2],
                    "new_forward_tuning": new_fwd_tuning,
                    "rows": [],
                }
                suite["reports"].append(report)
                checkpoint()
            else:
                if int(report["shape"]["retained_context_C"]) != workload.context_size:
                    raise RuntimeError("checkpoint workload shape mismatch")
                runs = report.setdefault("provenance_runs", [report["provenance"]])
                if runs[-1]["argv"] != current_provenance["argv"]:
                    runs.append(current_provenance)
                report["provenance"] = current_provenance
                new_fwd = Cell.from_dict(report["new_forward_tuned_once"])
                new_fwd_top2 = [Cell.from_dict(value) for value in report["new_forward_top2"]]

            completed = {int(row["macrobatch_size"]) for row in report["rows"]}
            for row_b in selected_b:
                if row_b in completed:
                    continue
                if row_b % BASELINE_MINIBATCH_SIZE:
                    raise RuntimeError(f"baseline mini4096 is illegal for B={row_b}")
                baseline_fwd = Cell(
                    FWD, BASELINE, row_b, row_b,
                    BASELINE_MINIBATCH_SIZE, BASELINE_FWD_COMM_SMS,
                )
                baseline_bwd = Cell(
                    BWD, BASELINE, row_b, row_b,
                    BASELINE_MINIBATCH_SIZE, BASELINE_BWD_COMM_SMS,
                )
                old_fwd, old_fwd_top2, old_fwd_tuning = tune(FWD, OLD, row_b, row_b, None)
                old_bwd, old_bwd_top2, old_bwd_tuning = tune(BWD, OLD, row_b, row_b, old_fwd)
                new_bwd, new_bwd_top2, new_bwd_tuning = tune(BWD, NEW, row_b, row_b, new_fwd)
                candidate_phase_fwd = {OLD: old_fwd, NEW: new_fwd}
                candidate_phase_bwd = {OLD: old_bwd, NEW: new_bwd}
                candidate_combined, joint = {}, {}
                for impl, fwd_top2, bwd_top2 in (
                    (OLD, old_fwd_top2, old_bwd_top2),
                    (NEW, new_fwd_top2, new_bwd_top2),
                ):
                    selected_fwd, selected_bwd, details = tune_combined(
                        impl, fwd_top2, bwd_top2,
                        workload_row_seed(args.seed, workload.key, row_b)
                        + (3_000_000 if impl == NEW else 0),
                    )
                    candidate_combined[impl] = {FWD: selected_fwd, BWD: selected_bwd}
                    joint[impl] = details

                terminal_fallback = row_b == workload.context_size
                selected_path = (
                    "legacy_terminal_fallback" if terminal_fallback
                    else "ep8_full_context_fine"
                )
                phase_fwd = dict(candidate_phase_fwd)
                phase_bwd = dict(candidate_phase_bwd)
                combined = dict(candidate_combined)
                if terminal_fallback:
                    phase_fwd[NEW] = phase_fwd[OLD]
                    phase_bwd[NEW] = phase_bwd[OLD]
                    combined[NEW] = dict(combined[OLD])

                correctness = {}
                for impl in IMPLEMENTATIONS:
                    for label, fwd_cell, bwd_cell in (
                        ("phase", phase_fwd[impl], phase_bwd[impl]),
                        ("combined", combined[impl][FWD], combined[impl][BWD]),
                    ):
                        output, context = run_forward(fwd_cell)
                        gradients = run_backward(bwd_cell, context)
                        check_correctness(
                            f"{workload.key}/B{row_b}/{impl}/{label}/output",
                            reference_output, output, MXFP8_TOLERANCE, print_stats=False,
                        )
                        for name, expected, actual in zip(
                            GRADIENT_NAMES, reference_gradients, gradients, strict=True
                        ):
                            check_correctness(
                                f"{workload.key}/B{row_b}/{impl}/{label}/{name}",
                                expected, actual, MXFP8_TOLERANCE, print_stats=False,
                            )
                        output = context = gradients = None
                    correctness[impl] = "passed_phase_and_combined_output_plus_8_gradients"

                def final_pair(scope: str) -> dict[str, Any]:
                    if scope == FWD_BWD:
                        fwd_cells = {
                            BASELINE: baseline_fwd,
                            **{impl: combined[impl][FWD] for impl in IMPLEMENTATIONS},
                        }
                        bwd_cells = {
                            BASELINE: baseline_bwd,
                            **{impl: combined[impl][BWD] for impl in IMPLEMENTATIONS},
                        }
                    else:
                        fwd_cells = {BASELINE: baseline_fwd, **phase_fwd}
                        bwd_cells = {BASELINE: baseline_bwd, **phase_bwd}
                    for pair in range(args.final_warmup_pairs):
                        for impl in final_order(pair):
                            run_once(scope, fwd_cells[impl], bwd_cells.get(impl))
                    local = {impl: [] for impl in (BASELINE, OLD, NEW)}
                    order = []
                    for pair in range(args.final_timed_pairs):
                        current_order = final_order(pair)
                        order.append(">".join(current_order))
                        for impl in current_order:
                            values = measure(scope, fwd_cells[impl], bwd_cells.get(impl), 0, 1)
                            # measure already rank-reduces one sample; keep the same value on every rank.
                            local[impl].append(values[0])
                    payload = useful_payload_bytes(aggregate_remote, HIDDEN_DIM, scope)
                    rows = {}
                    for impl in (BASELINE, OLD, NEW):
                        summary = sample_summary(local[impl])
                        latency = summary["median_ms"]
                        rows[impl] = {
                            **summary,
                            "aggregate_useful_payload_bytes": payload,
                            "per_gpu_useful_payload_bytes": payload / world_size,
                            "derived_aggregate_effective_payload_gbps": effective_payload_gbps(payload, latency),
                            "derived_per_gpu_effective_payload_gbps": effective_payload_gbps(payload, latency) / world_size,
                        }
                    old_tuned_deltas = [
                        100 * (new / old - 1)
                        for old, new in zip(local[OLD], local[NEW], strict=True)
                    ]
                    baseline_deltas = [
                        100 * (new / baseline - 1)
                        for baseline, new in zip(local[BASELINE], local[NEW], strict=True)
                    ]
                    return {
                        **rows,
                        "new_vs_old_tuned_speedup_x": (
                            rows[OLD]["median_ms"] / rows[NEW]["median_ms"]
                        ),
                        "new_vs_baseline_speedup_x": (
                            rows[BASELINE]["median_ms"] / rows[NEW]["median_ms"]
                        ),
                        "paired_delta_percent_samples": old_tuned_deltas,
                        "paired_median_delta_percent": statistics.median(old_tuned_deltas),
                        "paired_p90_delta_percent": quantile(old_tuned_deltas, 0.9),
                        "stable_new_win": stable_paired_win(old_tuned_deltas),
                        "new_vs_baseline_paired_delta_percent_samples": baseline_deltas,
                        "new_vs_baseline_paired_median_delta_percent": statistics.median(
                            baseline_deltas
                        ),
                        "new_vs_baseline_paired_p90_delta_percent": quantile(
                            baseline_deltas, 0.9
                        ),
                        "new_vs_baseline_stable_win": stable_paired_win(baseline_deltas),
                        "measure_order": order,
                    }

                metrics = {scope: final_pair(scope) for scope in SCOPES}
                row = {
                    "macrobatch_size": row_b,
                    "kernel_contract": {
                        "old_fwd_C_equals_B": row_b,
                        "new_fwd_kernelB_equals_retained_C": workload.context_size,
                        "old_bwd_ringB": row_b,
                        "new_bwd_ringB": row_b,
                        "new_selected_path": selected_path,
                        "new_full_context_active": not terminal_fallback,
                    },
                    "pipeline_generations_by_rank": [
                        (rows + row_b - 1) // row_b for rows in padded_by_rank
                    ],
                    "baseline_configuration": {
                        FWD: asdict(baseline_fwd), BWD: asdict(baseline_bwd),
                    },
                    "phase_configurations": {
                        impl: {FWD: asdict(phase_fwd[impl]), BWD: asdict(phase_bwd[impl])}
                        for impl in IMPLEMENTATIONS
                    },
                    "combined_configurations": {
                        impl: {FWD: asdict(combined[impl][FWD]), BWD: asdict(combined[impl][BWD])}
                        for impl in IMPLEMENTATIONS
                    },
                    "tuning": {
                        OLD: {FWD: old_fwd_tuning, BWD: old_bwd_tuning, FWD_BWD: joint[OLD]},
                        NEW: {
                            BWD: new_bwd_tuning,
                            FWD_BWD: joint[NEW],
                            "candidate_phase_configuration": {
                                FWD: asdict(candidate_phase_fwd[NEW]),
                                BWD: asdict(candidate_phase_bwd[NEW]),
                            },
                            "candidate_combined_configuration": {
                                FWD: asdict(candidate_combined[NEW][FWD]),
                                BWD: asdict(candidate_combined[NEW][BWD]),
                            },
                            "selected_path": selected_path,
                        },
                    },
                    "correctness": correctness,
                    "metrics": metrics,
                }
                report["rows"].append(row)
                report["rows"].sort(key=lambda item: int(item["macrobatch_size"]))
                report["state"] = "complete" if len(report["rows"]) == 24 else "partial"
                checkpoint()
                if rank == 0:
                    print("DUAL_CONTEXT_ROW " + json.dumps(row, sort_keys=True), flush=True)
                torch.cuda.empty_cache()

            report.update(finalize_report(report))
            checkpoint()
            if rank == 0:
                print("DUAL_CONTEXT_MATRIX " + json.dumps(report, sort_keys=True), flush=True)

            reference_output = reference_gradients = inputs = None
            functional.clear_workspace_cache()
            torch.cuda.empty_cache()

        required_workloads = {workload.key for workload in selected_workloads(args.workload)}
        suite["state"] = "complete" if all(
            len(report["rows"]) == 24 for report in suite["reports"]
        ) and {
            report["shape"]["workload"] for report in suite["reports"]
        } == required_workloads else "partial"
        checkpoint()
        return suite
    finally:
        try:
            functional.clear_workspace_cache()
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("20k", "100k", "all"), default="all")
    parser.add_argument("--macrobatches", help="comma-separated canonical row subset")
    parser.add_argument("--output", help="atomic JSON checkpoint/result path")
    parser.add_argument("--restart", action="store_true", help="replace an existing checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20_260_823)
    parser.add_argument("--screen-warmup", type=int, default=SCREEN_WARMUP)
    parser.add_argument("--screen-samples", type=int, default=SCREEN_SAMPLES)
    parser.add_argument("--refine-warmup", type=int, default=REFINE_WARMUP)
    parser.add_argument("--refine-samples", type=int, default=REFINE_SAMPLES)
    parser.add_argument("--joint-warmup", type=int, default=JOINT_WARMUP)
    parser.add_argument("--joint-samples", type=int, default=JOINT_SAMPLES)
    parser.add_argument("--final-warmup-pairs", type=int, default=FINAL_WARMUP_PAIRS)
    parser.add_argument("--final-timed-pairs", type=int, default=FINAL_TIMED_PAIRS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    source_overlay_sha256(args)
    if not args.dry_run and not args.output:
        raise ValueError("--output is required for atomic checkpoint/resume")
    for name in (
        "screen_warmup", "screen_samples", "refine_warmup", "refine_samples",
        "joint_warmup", "joint_samples", "final_warmup_pairs", "final_timed_pairs",
    ):
        minimum = 0 if "warmup" in name else 1
        if getattr(args, name) < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.dry_run:
        print("DUAL_CONTEXT_DRY_RUN " + json.dumps(dry_run_plan(args), sort_keys=True))
    else:
        run_gpu_suite(args)


if __name__ == "__main__":
    main()
