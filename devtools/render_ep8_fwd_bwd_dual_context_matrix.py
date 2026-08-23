#!/usr/bin/env python3
"""Render checkpoint JSON or DUAL_CONTEXT_MATRIX logs as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


BASELINE, OLD, NEW = "old_default_baseline", "old_macrobatch", "new_ep8_full_context"
IMPLEMENTATIONS = (OLD, NEW)
SCOPES = ("fwd", "bwd", "fwd_bwd")
PREFIX = "DUAL_CONTEXT_MATRIX "

def reports_from_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        if isinstance(value.get("reports"), list):
            return list(value["reports"])
        if isinstance(value.get("rows"), list):
            return [value]
    return [json.loads(line[len(PREFIX) :]) for line in text.splitlines() if line.startswith(PREFIX)]

def load_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = [report for path in paths for report in reports_from_text(path.read_text())]
    if not reports:
        raise ValueError("no dual-context report found")
    return reports


def protocol_label(report: dict[str, Any]) -> str:
    value = report["protocol"]
    coarse, screen, refine = value["coarse_comm_sms_by_scope"], value["screen"], value["refine"]
    joint, final = value["combined_joint"], value["final_abba"]
    baseline = value["old_default_baseline"]
    return (
        f"coarse F{coarse['fwd']}/B{coarse['bwd']}; screen W{screen['warmup']}/N{screen['samples']}; "
        f"top{value['phase_finalists']} refine W{refine['warmup']}/N{refine['samples']} "
        f"at offsets {value['refine_comm_offsets']}; combined top2x2 "
        f"W{joint['warmup']}/N{joint['samples']}; final "
        f"W{final['warmup_pairs']}/N{final['timed_pairs']} ABBA rank-max; "
        f"baseline mini{baseline['minibatch_size']}/"
        f"F{baseline['fwd_num_comm_sms']}/B{baseline['bwd_num_comm_sms']}SM"
    )


def validate_contract(report: dict[str, Any], row: dict[str, Any]) -> None:
    row_b = int(row["macrobatch_size"])
    retained_c = int(report["shape"]["retained_context_C"])
    terminal = row_b == retained_c
    expected_path = "legacy_terminal_fallback" if terminal else "ep8_full_context_fine"
    contract = row["kernel_contract"]
    if (
        contract.get("new_selected_path") != expected_path
        or bool(contract.get("new_full_context_active")) == terminal
    ):
        raise ValueError(f"invalid NEW selected path at B={row_b}")
    policy = report["protocol"]["old_default_baseline"]
    terminal_policy = report["protocol"].get("new_terminal_policy", {})
    if (
        row_b % 4096
        or int(policy["minibatch_size"]) != 4096
        or int(policy["fwd_num_comm_sms"]) != 40
        or int(policy["bwd_num_comm_sms"]) != 28
        or terminal_policy.get("selected_path") != "legacy_terminal_fallback"
    ):
        raise ValueError("invalid baseline or terminal selection policy")
    baseline = row["baseline_configuration"]
    for scope, comm in (("fwd", 40), ("bwd", 28)):
        cell = baseline[scope]
        if (
            cell["implementation"] != BASELINE
            or int(cell["kernel_macrobatch_size"]) != row_b
            or int(cell["minibatch_size"]) != 4096
            or int(cell["num_comm_sms"]) != comm
        ):
            raise ValueError(f"invalid OLD-default {scope} configuration at B={row_b}")
    for configurations in (row["phase_configurations"], row["combined_configurations"]):
        if int(configurations[OLD]["fwd"]["kernel_macrobatch_size"]) != row_b:
            raise ValueError(f"OLD FWD must use C=B={row_b}")
        if int(configurations[NEW]["fwd"]["kernel_macrobatch_size"]) != retained_c:
            raise ValueError(f"NEW FWD must use kernelB=C={retained_c}")
        for implementation in IMPLEMENTATIONS:
            if int(configurations[implementation]["bwd"]["kernel_macrobatch_size"]) != row_b:
                raise ValueError(f"{implementation} BWD must use ringB={row_b}")
        new_impl = OLD if terminal else NEW
        if any(
            configurations[NEW][scope]["implementation"] != new_impl
            for scope in SCOPES[:2]
        ):
            raise ValueError(f"NEW selected configuration/path mismatch at B={row_b}")
        if terminal and configurations[NEW] != configurations[OLD]:
            raise ValueError("terminal NEW must reuse the exact tuned OLD configuration")


def config_label(cell: dict[str, Any]) -> str:
    return f"{cell['minibatch_size']}/{cell['num_comm_sms']}"


def rate_label(metric: dict[str, Any]) -> str:
    return (
        f"{metric['derived_aggregate_effective_payload_gbps']:.2f}/"
        f"{metric['derived_per_gpu_effective_payload_gbps']:.2f}"
    )


def selected_config(row: dict[str, Any], implementation: str, scope: str) -> str:
    if implementation == BASELINE:
        configurations = row["baseline_configuration"]
    else:
        configurations = (
            row["combined_configurations"] if scope == "fwd_bwd"
            else row["phase_configurations"]
        )[implementation]
    fwd, bwd = configurations["fwd"], configurations["bwd"]
    path = (
        f"path={row['kernel_contract']['new_selected_path']}; "
        if implementation == NEW else ""
    )
    if scope == "fwd":
        return f"{path}F kernelB={fwd['kernel_macrobatch_size']} mini/SM={config_label(fwd)}"
    if scope == "bwd":
        return f"{path}B ringB={bwd['kernel_macrobatch_size']} mini/SM={config_label(bwd)}"
    return (
        f"{path}F kernelB={fwd['kernel_macrobatch_size']} mini/SM={config_label(fwd)}; "
        f"B ringB={bwd['kernel_macrobatch_size']} mini/SM={config_label(bwd)}"
    )


def render(reports: list[dict[str, Any]]) -> str:
    merged = {
        (str(report["shape"]["workload"]), int(row["macrobatch_size"])): (report, row)
        for report in reports for row in report["rows"]
    }
    lines = ["# Qwen-shaped EP8 MXFP8 bounded tuned matrix"]
    ordering = lambda key: ({"20k": 0, "100k": 1}.get(key[0], 2), key[1])
    for scope, title in (("fwd", "FWD"), ("bwd", "BWD"), ("fwd_bwd", "FWD+BWD")):
        lines += [
            "", f"## {title}", "",
            "| T/rank | retained C | row macroB | OLD-default config; latency; agg/per-GPU GB/s | OLD-tuned config; latency; agg/per-GPU GB/s | NEW-tuned config; latency; agg/per-GPU GB/s | NEW/base speedup; paired med/p90/stable | NEW/OLD-tuned speedup; paired med/p90/stable |",
            "|---:|---:|---:|:---|:---|:---|:---:|:---:|",
        ]
        for key in sorted(merged, key=ordering):
            report, row = merged[key]
            validate_contract(report, row)
            pair = row["metrics"][scope]
            implementations = []
            for implementation in (BASELINE, OLD, NEW):
                metric = pair[implementation]
                implementations.append(
                    f"{selected_config(row, implementation, scope)}; "
                    f"{metric['median_ms']:.3f} ms; {rate_label(metric)}"
                )
            base_pair = (
                f"{pair['new_vs_baseline_speedup_x']:.3f}x; "
                f"{pair['new_vs_baseline_paired_median_delta_percent']:+.2f}%/"
                f"{pair['new_vs_baseline_paired_p90_delta_percent']:+.2f}%/"
                f"{'YES' if pair['new_vs_baseline_stable_win'] else 'NO'}"
            )
            old_pair = (
                f"{pair['new_vs_old_tuned_speedup_x']:.3f}x; "
                f"{pair['paired_median_delta_percent']:+.2f}%/"
                f"{pair['paired_p90_delta_percent']:+.2f}%/"
                f"{'YES' if pair['stable_new_win'] else 'NO'}"
            )
            lines.append(
                f"| {report['shape']['tokens_per_rank']} | {report['shape']['retained_context_C']} | "
                f"{row['macrobatch_size']} | {' | '.join(implementations)} | {base_pair} | {old_pair} |"
            )

    lines += [
        "",
        "FWD, BWD, and FWD+BWD are directly timed with CUDA Events. BWD excludes its fresh forward; FWD+BWD is one event around direct forward then backward. Schedule construction is excluded.",
        "",
        "Bandwidth cells show aggregate/per-GPU derived useful cross-GPU payload GB/s. They exclude local/padded routes, replay, and protocol overhead; they are not NCU or NVLink link-counter bandwidth. Speedup is reference median latency divided by NEW median latency. Both baseline and OLD-tuned paired comparisons report median/p90 delta and stable win; stable requires both deltas to be negative.",
        "",
        "| workload | state | measured rows | dynamic protocol | source tree SHA256 |",
        "|:---|:---:|---:|:---|:---|",
    ]
    seen = set()
    for report, _ in merged.values():
        key = str(report["shape"]["workload"])
        if key in seen:
            continue
        seen.add(key)
        digest = report.get("provenance", {}).get("source_tree_sha256", "N/A")
        lines.append(
            f"| {key} | {report.get('state', 'N/A')} | {len(report['rows'])} | "
            f"{protocol_label(report)} | {digest} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    markdown = render(load_reports(args.artifacts))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
