"""CPU-only source contract for the EP8 BF16 warp-specialized forward path.

The production fast path is intentionally not imported here: importing ``mok``
would require a CUDA extension.  These tests instead keep the first bring-up
honest about its compile-time scope, warp ownership, storage budget, and
training-context publication order.

The contract is expected to fail until
``csrc/megakernel/forward_ep8_bf16_warp_specialized.cuh`` is implemented and
wired into the BF16 EP8 entrypoint.  MXFP8 and the legacy BF16 launch must stay
available as fallbacks.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASTPATH = ROOT / "csrc/megakernel/forward_ep8_bf16_warp_specialized.cuh"
LAYOUT = ROOT / "csrc/megakernel/forward_ep8_bf16_ws_layout.cuh"
ENTRYPOINTS = ROOT / "csrc/megakernel/entrypoints.cuh"
FORWARD = ROOT / "csrc/megakernel/forward.cuh"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing warp-specialized source: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def _constexpr_int(source: str, name: str) -> int:
    match = re.search(
        rf"\b(?:static\s+)?constexpr\s+(?:int|unsigned|uint32_t|size_t)\s+"
        rf"{re.escape(name)}\s*=\s*(\d+)\s*;",
        source,
    )
    assert match is not None, f"missing integer source contract {name}"
    return int(match.group(1))


def _ordered(source: str, *tokens: str) -> None:
    cursor = -1
    for token in tokens:
        cursor = source.find(token, cursor + 1)
        assert cursor >= 0, f"missing or out-of-order source token: {token}"


def _slice(source: str, begin: str, end: str) -> str:
    begin_idx = source.find(begin)
    assert begin_idx >= 0, f"missing section start: {begin}"
    end_idx = source.find(end, begin_idx + len(begin))
    assert end_idx >= 0, f"missing section end: {end}"
    return source[begin_idx:end_idx]


def test_fastpath_is_compile_time_limited_to_ep8_bf16() -> None:
    source = _read(FASTPATH)

    # Keep the new implementation out of every generic/MXFP8 instantiation.
    assert re.search(r"static_assert\s*\(\s*NUM_DEVICES\s*==\s*8\b", source)
    assert re.search(r"static_assert\s*\(\s*!\s*USE_MXFP8\b", source)


def test_comm_tma_mma_scheduler_and_epilogue_warps_do_not_overlap() -> None:
    source = _read(LAYOUT) + "\n" + _read(FASTPATH)

    assert _constexpr_int(source, "EP8_BF16_WS_NUM_WARPGROUPS") == 3
    assert _constexpr_int(source, "EP8_BF16_WS_NUM_THREADS") == 384

    comm = set(
        range(
            _constexpr_int(source, "EP8_BF16_WS_COMM_WARP_BEGIN"),
            _constexpr_int(source, "EP8_BF16_WS_COMM_WARP_END"),
        )
    )
    tma = {
        _constexpr_int(source, "EP8_BF16_WS_TMA_A_WARP"),
        _constexpr_int(source, "EP8_BF16_WS_TMA_B_WARP"),
    }
    mma = {_constexpr_int(source, "EP8_BF16_WS_MMA_WARP")}
    scheduler = {_constexpr_int(source, "EP8_BF16_WS_SCHEDULER_WARP")}
    epilogue = set(
        range(
            _constexpr_int(source, "EP8_BF16_WS_EPI_WARP_BEGIN"),
            _constexpr_int(source, "EP8_BF16_WS_EPI_WARP_END"),
        )
    )

    roles = {
        "COMM": comm,
        "TMA": tma,
        "MMA": mma,
        "SCHEDULER": scheduler,
        "EPI": epilogue,
    }
    for (left_name, left), (right_name, right) in combinations(roles.items(), 2):
        assert left.isdisjoint(right), f"{left_name}/{right_name} warp overlap: {left & right}"
    assert set().union(*roles.values()) == set(range(12))


def test_concurrent_smem_regions_are_disjoint_and_fit_sm103() -> None:
    source = _read(LAYOUT) + "\n" + _read(FASTPATH)

    load_stages = _constexpr_int(source, "EP8_BF16_WS_LOAD_STAGES")
    a_stage_bytes = _constexpr_int(source, "EP8_BF16_WS_A_STAGE_BYTES")
    b_stage_bytes = _constexpr_int(source, "EP8_BF16_WS_B_STAGE_BYTES")
    comm_warps = _constexpr_int(source, "EP8_BF16_WS_COMM_WARPS")
    comm_buffer_bytes = _constexpr_int(source, "EP8_BF16_WS_COMM_BUFFER_BYTES")
    gate_bytes = _constexpr_int(source, "EP8_BF16_WS_GATE_BYTES")
    up_bytes = _constexpr_int(source, "EP8_BF16_WS_UP_BYTES")
    hidden_bytes = _constexpr_int(source, "EP8_BF16_WS_HIDDEN_BYTES")
    capacity = _constexpr_int(source, "EP8_BF16_WS_SMEM_CAPACITY_BYTES")

    assert load_stages == 3
    assert (a_stage_bytes, b_stage_bytes) == (16 * 1024, 16 * 1024)
    assert (comm_warps, comm_buffer_bytes) == (4, 4 * 1024)
    assert (gate_bytes, up_bytes, hidden_bytes) == (32 * 1024,) * 3
    assert capacity == 231424

    # Distinct struct members are the no-alias contract for regions that are
    # simultaneously live: the A/B ring, four communication pull buffers, and
    # Gate/Up/Hidden epilogue tiles.
    storage = re.search(
        r"struct\s+ep8_bf16_ws_smem_storage\s*\{(?P<body>.*?)\n\s*\};",
        source,
        flags=re.DOTALL,
    )
    assert storage is not None, "missing ep8_bf16_ws_smem_storage"
    body = storage.group("body")
    for member in (
        "a_smem",
        "b_smem",
        "comm_smem",
        "gate_smem",
        "up_smem",
        "hidden_smem",
    ):
        assert re.search(rf"\b{member}\b", body), f"missing disjoint SMEM member {member}"
    assert re.search(r"\ba_smem\s*\[\s*EP8_BF16_WS_LOAD_STAGES\s*\]", body)
    assert re.search(r"\bb_smem\s*\[\s*EP8_BF16_WS_LOAD_STAGES\s*\]", body)
    assert re.search(r"\bcomm_smem\s*\[\s*EP8_BF16_WS_COMM_WARPS\s*\]", body)

    payload_bytes = (
        load_stages * (a_stage_bytes + b_stage_bytes)
        + comm_warps * comm_buffer_bytes
        + gate_bytes
        + up_bytes
        + hidden_bytes
    )
    assert payload_bytes == 208 * 1024
    assert payload_bytes <= capacity
    assert re.search(
        r"static_assert\s*\(\s*sizeof\s*\(\s*ep8_bf16_ws_smem_storage\s*\)"
        r"\s*<=\s*EP8_BF16_WS_SMEM_CAPACITY_BYTES\b",
        source,
    ), "production layout must include padding/barriers in its compile-time capacity check"


def test_two_tmem_accumulator_stages_use_exactly_512_columns() -> None:
    source = _read(LAYOUT) + "\n" + _read(FASTPATH)

    stages = _constexpr_int(source, "EP8_BF16_WS_TMEM_STAGES")
    columns = _constexpr_int(source, "EP8_BF16_WS_TMEM_COLS_PER_STAGE")
    assert (stages, columns, stages * columns) == (2, 256, 512)

    # Full/empty ownership is per accumulator stage; BF16 must not reserve the
    # generic MXFP8 scale-TMEM ranges at columns 256 and 384.
    for name in (
        "d_tt",
        "tmem_full",
        "tmem_empty",
    ):
        assert re.search(
            rf"\b{re.escape(name)}\s*\[\s*EP8_BF16_WS_TMEM_STAGES\s*\]",
            source,
        )
    assert "a_sc_tt" not in source
    assert "b_sc_tt" not in source


def test_training_context_stores_complete_before_hidden_ready_publish() -> None:
    source = _read(FASTPATH)

    # Shared expert context and routed macrobatch zero are training inputs to
    # backward.  Down readiness must not become visible until Gate, Up, and
    # Hidden TMA stores have all released their SMEM sources.
    _ordered(
        source,
        "const bool save_context = IS_SHARED || macrobatch_idx == 0",
        "tma::store_async(gate_context_gmem",
        "tma::store_async(up_context_gmem",
        "tma::store_async(hidden_gmem",
        "tma::store_async_wait()",
        "barrier_arrive(hidden_row_block_ready",
    )


def test_entrypoint_keeps_legacy_and_mxfp8_fallbacks() -> None:
    source = _read(ENTRYPOINTS)
    forward = _read(FORWARD)
    fastpath_symbol = "dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized"
    opt_in = "MOK_FWD_EP8_BF16_WARP_SPECIALIZED"

    mxfp8 = _slice(
        source,
        "dispatch_mlp_swiglu_combine_fwd_mxfp8_entrypoint(",
        "dispatch_mlp_swiglu_combine_fwd_bf16_entrypoint(",
    )
    bf16 = _slice(
        source,
        "dispatch_mlp_swiglu_combine_fwd_bf16_entrypoint(",
        "dispatch_mlp_swiglu_combine_bwd_mxfp8_entrypoint(",
    )

    assert fastpath_symbol not in mxfp8
    assert opt_in not in mxfp8
    assert opt_in in bf16
    assert fastpath_symbol in bf16
    # The current EP8 launch selector is the required correctness fallback.
    assert "launch_ep8(I1{}, I1{}, I1{})" in bf16
    # Only the canonical legacy host specialization may instantiate the two
    # clamped/unclamped fastpath kernels.  The CLC2 grouping variants retain
    # their legacy launches without producing duplicate device symbols.
    for token in (
        "FWD_CLC_PIPE_DEPTH == 1",
        "FWD_GATE_GROUP_SIZE == 1",
        "FWD_DOWN_GROUP_SIZE == 1",
    ):
        assert token in forward
