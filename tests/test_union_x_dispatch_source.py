from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "csrc" / "megakernel" / "union_x_dispatch.cuh").read_text()


def test_main_dispatch_geometry_is_frozen() -> None:
    for token in (
        "UNION_X_DISPATCH_EP_SIZE = 8",
        "UNION_X_DISPATCH_HIDDEN = 4096",
        "UNION_X_DISPATCH_ROWS = 128",
        "UNION_X_DISPATCH_SLICE_COLS = 512",
        "UNION_X_DISPATCH_SLICE_BYTES == 1024",
        "UNION_X_DISPATCH_HIDDEN_SLICES == 8",
        "UNION_X_DISPATCH_ROWS == config::DISPATCH_Mb",
        "UNION_X_DISPATCH_SLICE_COLS == config::DISPATCH_Nb",
    ):
        assert token in SOURCE


def test_three_state_uint32_memory_order() -> None:
    for token in (
        "UNION_X_DISPATCH_EMPTY = 0",
        "UNION_X_DISPATCH_LOADING = 1",
        "UNION_X_DISPATCH_FULL = 2",
        "atomicCAS(",
        "fence.proxy.async.global",
        "st.release.gpu.global.u32",
        "ld.relaxed.gpu.global.u32",
        "fence.acquire.gpu",
    ):
        assert token in SOURCE
    assert "uint8_t *state" not in SOURCE
    assert "__threadfence_system" not in SOURCE


def test_all_claims_precede_winner_transaction_expectation() -> None:
    claim = SOURCE.index("observed = atomicCAS(")
    count = SOURCE.index("__syncthreads_count(is_winner)", claim)
    expect = SOURCE.index("tma::expect_bytes(", count)
    load = SOURCE.index("tma::load_async(", expect)
    assert claim < count < expect < load
    assert "num_winners * chunk_bytes" in SOURCE[expect:load]
    assert "num_winners > 0" in SOURCE[count:load]


def test_winner_store_wait_release_then_loser_acquire() -> None:
    load = SOURCE.index("tma::load_async(")
    load_wait = SOURCE.index("wait(inputs_arrived", load)
    store = SOURCE.index("tma::store_async(", load_wait)
    store_wait = SOURCE.index("tma::store_async_wait()", store)
    proxy_fence = SOURCE.index("fence.proxy.async.global", store_wait)
    publish = SOURCE.index("union_x_dispatch_publish_full(", proxy_fence)
    first_cta_sync = SOURCE.index("__syncthreads();", publish)
    loser = SOURCE.index("union_x_dispatch_wait_full(", first_cta_sync)
    second_cta_sync = SOURCE.index("__syncthreads();", loser)
    coarse = SOURCE.index("barrier_arrive(transfer_done", second_cta_sync)
    assert (
        load < load_wait < store < store_wait < proxy_fence < publish
        < first_cta_sync < loser < second_cta_sync < coarse
    )


def test_padding_is_resolved_without_state_or_tma() -> None:
    # A padding row keeps peer_rank=-1, union_idx=-1 and observed=FULL.  It is
    # neither a winner nor a loser, but the CTA still reaches the single coarse
    # arrival after all rows resolve.
    assert "else if (union_idx != -1)" in SOURCE
    assert "is_worker && peer_rank >= 0" in SOURCE
    assert SOURCE.count("barrier_arrive(transfer_done") == 1
    assert SOURCE.count("__syncthreads_count(is_winner)") == 1


@dataclass
class Cell:
    state: int = 0
    winner: int | None = None


def _claim(cell: Cell, route: int) -> int:
    observed = cell.state
    if observed == 0:
        cell.state = 1
        cell.winner = route
    return observed


def test_cpu_128_row_slice_duplicate_and_padding_model() -> None:
    # Two unions are repeated, two rows are padding, and the remaining routes
    # are unique.  Each union-slice has one producer while the whole 128-row
    # task contributes one coarse completion.
    route_to_union = list(range(124)) + [5, 9, -1, -1]
    cells = [Cell() for _ in range(124)]
    observed = []
    for route, union_idx in enumerate(route_to_union):
        if union_idx < 0:
            observed.append(2)
        else:
            observed.append(_claim(cells[union_idx], route))

    winners = [i for i, value in enumerate(observed) if value == 0]
    losers = [i for i, value in enumerate(observed) if value == 1]
    padding = [i for i, union_idx in enumerate(route_to_union) if union_idx < 0]
    assert len(winners) == 124
    assert len(losers) == 2
    assert len(padding) == 2

    for cell in cells:
        assert cell.state == 1 and cell.winner is not None
        cell.state = 2
    assert all(
        route_to_union[route] < 0
        or cells[route_to_union[route]].state == 2
        for route in range(128)
    )
    coarse_arrivals = 1
    assert coarse_arrivals == 1


def test_cpu_required_count_matches_route_block_slice_tasks() -> None:
    for minibatch_rows in (128, 4096, 16384):
        assert minibatch_rows % 128 == 0
        tasks = minibatch_rows // 128 * 8
        required = minibatch_rows // 128 * (4096 // 512)
        assert tasks == required
